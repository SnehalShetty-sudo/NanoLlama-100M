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

# Import the model architecture from the benchmark script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.benchmark_throughput import DecoderTransformer

def generate_text(model, vocab_size, prompt_tokens, max_new_tokens=50, device='cuda'):
    model.eval()
    # Extremely basic greedy generation for evaluation
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    return input_ids[0].tolist()

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # Cosine learning rate schedule with linear warmup
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
    parser.add_argument('--model_size', type=str, choices=['100M', '200M'], required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--total_tokens', type=int, default=250_000_000)
    parser.add_argument('--dry_run', action='store_true', help='Run for a few steps to profile VRAM')
    args = parser.parse_args()

    vocab_size = 32000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Configurations
    if args.model_size == '100M':
        d_model, n_layers, n_heads, d_ff = 768, 11, 12, 2048
    else:  # 200M
        d_model, n_layers, n_heads, d_ff = 1024, 14, 16, 2752
        
    model = DecoderTransformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layers=n_layers, 
        n_heads=n_heads, 
        d_ff=d_ff
    )
    model.to(device)
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model_size} Pilot] Initialized model with {total_params / 1e6:.2f}M parameters")
    
    # Training setup
    tokens_per_step = args.batch_size * args.seq_len
    max_steps = args.total_tokens // tokens_per_step
    
    if args.dry_run:
        print("DRY RUN MODE: Limiting to 20 steps to profile VRAM.")
        max_steps = 20
        os.environ['WANDB_MODE'] = 'offline'

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)
    
    # LR Schedule parameters
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = int(max_steps * 0.1) # 10% warmup
    
    # WandB initialization
    if WANDB_AVAILABLE and not args.dry_run:
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
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    # Fixed seed for data ordering
    torch.manual_seed(1337)
    
    print(f"Starting training for {max_steps} steps...", flush=True)
    start_time = time.time()
    
    from tqdm import tqdm
    pbar = tqdm(range(max_steps), desc=f"Training {args.model_size}")
    
    for step in pbar:
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Dummy data for now (since we don't have the 250M token dataset locally)
        # In a real run, this would be `batch = next(data_iterator)`
        inputs = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 5 == 0:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0
            if WANDB_AVAILABLE and not args.dry_run:
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
    checkpoint_path = f"checkpoints/pilot_{args.model_size}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluation suite
    print("\n--- EVALUATION SUITE ---")
    prompts = {
        "Narrative": [12, 45, 89, 102], # Dummy token IDs representing a prompt
        "Factual": [500, 120, 13, 8],
        "Python Code": [34, 99, 10, 50, 44]
    }
    
    results = {}
    for name, prompt_tokens in prompts.items():
        gen_tokens = generate_text(model, vocab_size, prompt_tokens, max_new_tokens=20, device=device)
        # We don't have a tokenizer loaded, so we just return the raw tokens as a string
        results[name] = str(gen_tokens)
        print(f"[{name}] Output: {gen_tokens}")
        
    if WANDB_AVAILABLE and not args.dry_run:
        wandb.finish()

if __name__ == '__main__':
    main()
