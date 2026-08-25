"""
Standalone Throughput Benchmarking Script for NanoLlama-100M / slm-v2

Measures training throughput (tokens/sec), per-step latency (ms), and peak VRAM (MB)
for a 100M parameter decoder-only transformer with FP16 mixed precision.

Dimensions:
  - d_model: 768
  - n_layers: 11
  - n_heads: 12
  - d_ff: 2048
  - max_seq_len: 512
  - vocab_size: 32000
  - Tied embeddings (lm_head.weight == embed_tokens.weight)

Usage:
    python scripts/benchmark_throughput.py [--warmup 50] [--steps 200]
"""

import time
import argparse
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Model Architecture
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit Feed-Forward Network."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with PyTorch Scaled Dot-Product Attention."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Decoder Transformer Layer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderTransformer(nn.Module):
    """Decoder-only Transformer with tied embeddings."""
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 11,
        n_heads: int = 12,
        d_ff: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        return self.lm_head(x)


# ==============================================================================
# Benchmarking Engine
# ==============================================================================

def run_benchmark_for_config(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    use_compile: bool,
) -> dict:
    """Runs full training step benchmark for a specific configuration."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    is_cuda = device.type == "cuda"
    use_amp = is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    gc.collect()

    try:
        # 1. Warmup steps (discarded to purge allocation / compilation overhead)
        print(f"   [Warmup] Running {warmup_steps} warmup steps (batch_size={batch_size})...", flush=True)
        for step in range(warmup_steps):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if is_cuda:
            torch.cuda.synchronize(device)

        # 2. Timed steps
        print(f"   [Benchmarking] Running {timed_steps} timed steps...", flush=True)
        start_time = time.perf_counter()

        for step in range(timed_steps):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if is_cuda:
            torch.cuda.synchronize(device)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time

        total_tokens = batch_size * seq_len * timed_steps
        tokens_per_sec = total_tokens / elapsed_sec
        ms_per_step = (elapsed_sec / timed_steps) * 1000.0

        max_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else 0.0
        )

        return {
            "status": "SUCCESS",
            "batch_size": batch_size,
            "tokens_per_sec": tokens_per_sec,
            "ms_per_step": ms_per_step,
            "max_mem_mb": max_mem_mb,
            "elapsed_sec": elapsed_sec,
        }

    except torch.cuda.OutOfMemoryError:
        if is_cuda:
            torch.cuda.empty_cache()
        return {"status": "OOM", "batch_size": batch_size}
    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "batch_size": batch_size}


def main():
    parser = argparse.ArgumentParser(description="NanoLlama-100M / slm-v2 Throughput Benchmark")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup steps (default: 50)")
    parser.add_argument("--steps", type=int, default=200, help="Number of timed benchmark steps (default: 200)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32], help="Batch sizes to benchmark")
    args = parser.parse_args()

    # Architecture specs
    d_model = 768
    n_layers = 11
    n_heads = 12
    d_ff = 2048
    vocab_size = 32000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("NanoLlama-100M / slm-v2 Throughput & Memory Benchmark")
    print("=" * 80)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Specs          : d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"Sequence & Vocab     : max_seq_len={args.seq_len}, vocab_size={vocab_size}")
    print(f"Precision            : FP16 Mixed Precision (autocast + GradScaler)")
    print(f"Warmup / Timed Steps : {args.warmup} warmup / {args.steps} timed steps")
    print("=" * 80)

    # Calculate total parameter count
    raw_model = DecoderTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f}M")
    print("=" * 80)

    compile_modes = [False]
    # Check if torch.compile is supported on platform
    if hasattr(torch, "compile"):
        compile_modes.append(True)

    results = []

    for use_compile in compile_modes:
        compile_str = "WITH torch.compile()" if use_compile else "WITHOUT torch.compile()"
        print(f"\n>>> Running Benchmarks {compile_str}\n")

        for b_size in args.batch_sizes:
            # Instantiate fresh model for clean memory layout
            model = DecoderTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff).to(device)

            if use_compile:
                try:
                    print(f"Compiling model with torch.compile()...", flush=True)
                    model = torch.compile(model)
                except Exception as comp_err:
                    print(f"   [Skipped] torch.compile() failed on this system: {comp_err}")
                    results.append({
                        "mode": compile_str,
                        "batch_size": b_size,
                        "status": f"COMPILE_FAILED ({comp_err})",
                    })
                    break

            res = run_benchmark_for_config(
                model=model,
                batch_size=b_size,
                seq_len=args.seq_len,
                vocab_size=vocab_size,
                warmup_steps=args.warmup,
                timed_steps=args.steps,
                device=device,
                use_compile=use_compile,
            )
            res["mode"] = compile_str
            results.append(res)

            if res["status"] == "SUCCESS":
                print(f"   -> Result: {res['tokens_per_sec']:.2f} tokens/sec | {res['ms_per_step']:.2f} ms/step | Peak VRAM: {res['max_mem_mb']:.2f} MB")
            elif res["status"] == "OOM":
                print(f"   -> Result: Out of Memory (OOM) at batch_size={b_size}")
            else:
                print(f"   -> Result: {res['status']}")

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Final Summary Table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Mode':<25} | {'Batch Size':<10} | {'Tokens / Sec':<15} | {'ms / Step':<12} | {'Peak VRAM (MB)':<15}")
    print("-" * 80)

    for r in results:
        mode = r["mode"]
        bs = r["batch_size"]
        if r["status"] == "SUCCESS":
            tps = f"{r['tokens_per_sec']:.2f}"
            ms = f"{r['ms_per_step']:.2f}"
            mem = f"{r['max_mem_mb']:.2f}"
        else:
            tps = r["status"]
            ms = "N/A"
            mem = "N/A"
        print(f"{mode:<25} | {bs:<10} | {tps:<15} | {ms:<12} | {mem:<15}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
