import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.benchmark_throughput import DecoderTransformer
from data.dataset import get_real_dataloader
from tokenizers import Tokenizer

def generate_text(model, tokenizer, prompt_text, max_new_tokens=50, device='cuda'):
    model.eval()
    tokens = tokenizer.encode(prompt_text).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return decoded

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, choices=['10M', '100M', '200M', 'test'], required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--total_tokens', type=int, default=250_000_000)
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
    parser.add_argument('--kill_at_step', type=int, default=-1, help='For testing: kill process after this step')
    args = parser.parse_args()

    vocab_size = 32000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Configurations
    if args.model_size == '10M':
        d_model, n_layers, n_heads, d_ff = 256, 4, 4, 680
    elif args.model_size == '100M':
        d_model, n_layers, n_heads, d_ff = 768, 11, 12, 2048
    elif args.model_size == '200M':
        d_model, n_layers, n_heads, d_ff = 1024, 14, 16, 2752
    else:  # test
        d_model, n_layers, n_heads, d_ff = 64, 1, 4, 256
        
    model = DecoderTransformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layers=n_layers, 
        n_heads=n_heads, 
        d_ff=d_ff
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model_size} Pilot] Initialized model with {total_params / 1e6:.2f}M parameters")
    
    tokens_per_step = args.batch_size * args.seq_len
    max_steps = args.total_tokens // tokens_per_step

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)
    
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = int(max_steps * 0.1)
    
    if WANDB_AVAILABLE:
        wandb.init(
            project="slm-scale-pilot",
            name=f"pilot-{args.model_size}",
            config={
                "model_size": args.model_size,
                "params": total_params,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "total_tokens": args.total_tokens,
                "max_steps": max_steps,
            }
        )
    
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resuming from step {start_step} with previous loss: {checkpoint.get('loss', 'unknown')}")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    torch.manual_seed(1337)
    
    # Initialize Data Loader
    if args.model_size == 'test':
        class DummyDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                while True:
                    inputs = torch.randint(0, vocab_size, (args.seq_len,), dtype=torch.long)
                    targets = torch.randint(0, vocab_size, (args.seq_len,), dtype=torch.long)
                    yield inputs, targets
        dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=args.batch_size)
        data_iter = iter(dataloader)
    else:
        # Initialize Real Data Loader (offset seed on resume to avoid replaying step-0 data)
        print(f"Loading REAL dataset mixture (70% FineWeb-Edu, 20% Code, 10% Wikipedia)...", flush=True)
        dataloader = get_real_dataloader(batch_size=args.batch_size, seq_len=args.seq_len, seed=1337 + start_step)
        data_iter = iter(dataloader)
    
    print(f"Starting real data training for {max_steps} steps...", flush=True)
    start_time = time.time()
    
    from tqdm import tqdm
    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc=f"Training {args.model_size} Real Data")
    
    for step in pbar:
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (step + 1) % args.checkpoint_every == 0:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'step': step,
                'loss': loss.item(),
                'args': args
            }
            # Rotate checkpoints: keep only latest and prev to avoid filling Kaggle's 20GB disk
            latest_path = f"checkpoints/pilot_{args.model_size}_latest.pt"
            prev_path = f"checkpoints/pilot_{args.model_size}_prev.pt"
            tmp_path = f"checkpoints/pilot_{args.model_size}_tmp.pt"
            
            torch.save(ckpt, tmp_path)
            if os.path.exists(latest_path):
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                os.rename(latest_path, prev_path)
            os.rename(tmp_path, latest_path)
            print(f"\n[Checkpoint] Saved step {step+1} cleanly (Loss: {loss.item():.4f})", flush=True)
        
        if args.kill_at_step > 0 and step == args.kill_at_step:
            print(f"TEST: Killing at step {step} with loss: {loss.item()}")
            import sys
            sys.exit(0)
            
        if step % 10 == 0:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0
            if WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "system/peak_vram_mb": peak_vram,
                    "step": step
                })
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{lr:.2e}", 'VRAM': f"{peak_vram:.1f}MB"})

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    
    peak_vram_final = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0
    print(f"FINAL PEAK VRAM: {peak_vram_final:.2f} MB")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/pilot_{args.model_size}_step{max_steps}.pt"
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': max_steps - 1,
        'loss': loss.item() if 'loss' in locals() else 0.0,
        'args': args
    }
    torch.save(ckpt, checkpoint_path)
    torch.save(ckpt, f"checkpoints/pilot_{args.model_size}_latest.pt")
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluation suite against real prompts
    print("\n--- REAL EVALUATION PROMPT SUITE ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_dir, "tokenizer", "slm_v2_tokenizer.json")
    if os.path.exists(tokenizer_path):
        eval_tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Warning: Custom tokenizer not found, evaluation skipped.")
        eval_tokenizer = None

    prompts = {
        "Narrative Continuation": "Once upon a time in a distant galaxy, an explorer discovered a hidden planet that",
        "Factual Continuation": "The capital of France is Paris, and the country is known for its landmark the",
        "Python Code Completion": "def calculate_factorial(n):\n    # Return factorial of n\n    if n == 0 or n == 1:\n        return 1\n    "
    }
    
    if eval_tokenizer:
        for name, prompt_text in prompts.items():
            output = generate_text(model, eval_tokenizer, prompt_text, max_new_tokens=40, device=device)
            print(f"\n[{name}]\nPROMPT: {prompt_text}\nGENERATED: {output}\n")
        
    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    main()
