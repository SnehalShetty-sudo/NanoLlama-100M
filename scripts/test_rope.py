import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.benchmark_throughput import DecoderTransformer

def main():
    model = DecoderTransformer(
        vocab_size=100,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=256,
        max_seq_len=128
    )
    model.eval()

    # If a transformer has NO position encoding, it acts as a bag-of-words.
    # The output for a query token should be identical regardless of the order 
    # of the context tokens before it.
    # We test this by passing the same query token (5) with context (6, 7) vs (7, 6).
    seq1 = torch.tensor([[6, 7, 5]])
    seq2 = torch.tensor([[7, 6, 5]])
    
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    # Output for token '5' at the last position
    out1 = logits1[0, 2, :]
    out2 = logits2[0, 2, :]

    diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"Max difference between swapped context positions: {diff:.6f}")

    if torch.allclose(out1, out2, atol=1e-5):
        print("FAIL: The outputs for swapped contexts are the same.")
        print("This means the model is acting as a bag-of-words (positional encoding is missing).")
        sys.exit(1)
    else:
        print("SUCCESS: The outputs differ when context order changes.")
        print("RoPE is successfully applying position-dependent rotations.")

if __name__ == "__main__":
    main()
