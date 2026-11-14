import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from scripts.benchmark_throughput import DecoderTransformer
from tokenizers import Tokenizer

def load_generator(checkpoint_path, model_size='200M', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if model_size == '200M':
        d_model, n_layers, n_heads, d_ff = 1024, 14, 16, 2752
    elif model_size == '10M':
        d_model, n_layers, n_heads, d_ff = 256, 4, 4, 680
    elif model_size == '100M':
        d_model, n_layers, n_heads, d_ff = 768, 11, 12, 2048
    else:
        d_model, n_layers, n_heads, d_ff = 64, 1, 4, 256

    model = DecoderTransformer(vocab_size=32000, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    
    tokenizer_path = os.path.join(base_dir, "tokenizer", "slm_v2_tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    step = ckpt.get('step', 'N/A')
    loss = ckpt.get('loss', 'N/A')
    loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else str(loss)
    print(f"\n[NanoLlama] Loaded {model_size} Model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"[NanoLlama] Step: {step} | Loss: {loss_str} | Device: {device}\n")
    return model, tokenizer, device

def generate(model, tokenizer, prompt, device, max_new_tokens=60, temp=0.75, top_k=40, top_p=0.9, rep_penalty=1.15):
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated_ids = []
    start_t = time.time()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / max(temp, 1e-5)
            
            # Repetition penalty
            if rep_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= rep_penalty
                    else:
                        next_logits[token_id] *= rep_penalty
            
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')
            
            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
    total_time = time.time() - start_t
    full_text = tokenizer.decode(input_ids[0].tolist())
    new_text = tokenizer.decode(generated_ids)
    return full_text, new_text, total_time, len(generated_ids) / max(total_time, 1e-5)

def main():
    parser = argparse.ArgumentParser(description="NanoLlama Inference & Text Generation")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pilot_200M_step61035.pt')
    parser.add_argument('--model_size', type=str, default='200M', choices=['10M', '100M', '200M'])
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=60)
    parser.add_argument('--temp', type=float, default=0.75)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.90)
    parser.add_argument('--rep_penalty', type=float, default=1.15)
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        alt = os.path.join(base_dir, ckpt_path)
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            print(f"Error: Checkpoint file not found: {ckpt_path}")
            return
            
    model, tokenizer, device = load_generator(ckpt_path, args.model_size)
    
    if args.prompt:
        full, new, t, speed = generate(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty
        )
        print(f"\n{'='*50}\nPROMPT: {args.prompt}\nGENERATED: {new}\n{'='*50}")
        print(f"Speed: {speed:.1f} tokens/s ({t:.2f}s)\n")
    else:
        print("=== Interactive NanoLlama Console (Type 'exit' to quit) ===")
        while True:
            try:
                p = input("\nEnter prompt > ")
                if p.strip().lower() in ['exit', 'quit', 'q']:
                    break
                if not p.strip():
                    continue
                full, new, t, speed = generate(
                    model, tokenizer, p, device,
                    max_new_tokens=args.max_tokens,
                    temp=args.temp,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    rep_penalty=args.rep_penalty
                )
                print(f"\n{new}\n[Speed: {speed:.1f} tok/s | Time: {t:.2f}s]")
            except KeyboardInterrupt:
                print("\nExiting.")
                break

if __name__ == '__main__':
    main()
