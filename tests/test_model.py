import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.benchmark_throughput import DecoderTransformer

def test_100M_parameter_count():
    model = DecoderTransformer(
        vocab_size=32000, 
        d_model=768, 
        n_layers=11, 
        n_heads=12, 
        d_ff=2048
    )
    total_params = sum(p.numel() for p in model.parameters())
    # Should be ~102.5M
    assert 102_000_000 < total_params < 103_000_000, f"Expected ~102.5M, got {total_params}"

def test_200M_parameter_count():
    model = DecoderTransformer(
        vocab_size=32000, 
        d_model=1024, 
        n_layers=14, 
        n_heads=16, 
        d_ff=2752
    )
    total_params = sum(p.numel() for p in model.parameters())
    # Should be ~209.9M
    assert 209_000_000 < total_params < 210_000_000, f"Expected ~209.9M, got {total_params}"

def test_rope_symmetry_breaking():
    model = DecoderTransformer(
        vocab_size=100,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=256,
        max_seq_len=128
    )
    model.eval()

    # Pass identical query token (5) with context (6, 7) vs (7, 6).
    seq1 = torch.tensor([[6, 7, 5]])
    seq2 = torch.tensor([[7, 6, 5]])
    
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    out1 = logits1[0, 2, :]
    out2 = logits2[0, 2, :]

    diff = torch.max(torch.abs(out1 - out2)).item()
    
    # The output MUST differ if positional encoding is working
    assert not torch.allclose(out1, out2, atol=1e-5), f"RoPE failed: output identical (diff {diff}), model is bag-of-words"
