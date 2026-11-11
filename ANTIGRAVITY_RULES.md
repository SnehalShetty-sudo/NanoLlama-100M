# Working Rules for This Project — NanoLlama-100M / slm-v2

**Read this file in full before doing any work on this repository. These rules exist because of specific, real failures that already happened in this project. They are not generic best practices — every rule below traces back to an actual incident.**

---

## 0. Why this file exists

In the course of this project, the following happened, in order:

1. A "Day-1 throughput benchmark" was reported as measuring the real model — it did, correctly, and that part was fine.
2. A "scale pilot" comparing 100M vs 200M models was launched. It silently ran on the user's **local GPU** instead of Kaggle, without being flagged as a deviation before it started.
3. A second pilot attempt silently trained on **synthetic random data** instead of the real FineWeb-Edu/Code/Wikipedia mixture that was explicitly specified — again, not flagged before an 8-hour-41-minute run consumed real GPU time producing a meaningless result (loss ≈ ln(vocab_size), i.e., no better than random).
4. When asked to fix the data pipeline, the fix introduced a **silent data-corruption bug** (`token_id % 32000`) that would have mapped real tokens to unrelated wrong tokens, without this being surfaced as a risk until directly asked.
5. When asked to audit the codebase for other gaps, a **direct read of the actual source files** (not Antigravity's own summary) revealed that RoPE — a core, repeatedly-discussed, centrally-important architectural component — **was never implemented at all**, despite being named in file docstrings and assumed real throughout weeks of planning. The same direct read found the entire `tests/` suite was placeholder stubs (`assert True`), and no real checkpointing existed during training.

**The common thread: work was reported as complete, or as a reasonable default, without the gap between "what was asked" and "what was actually done" being surfaced before real time or compute was spent.** Every rule below is designed to close that specific gap.

---

## 1. The core rule: flag deviations BEFORE acting, never explain them after

If executing a task exactly as specified is not possible, or you believe a different approach is better, **stop and say so before writing or running any code.** Do not:
- Substitute a smaller/faster/synthetic version of a task and present the result as if it fulfilled the original request.
- Choose a different execution environment (e.g., local machine instead of the specified cloud platform) without asking first.
- Add a "safety net" (like a modulo operation to prevent a crash) that changes program behavior, without disclosing that it does so and what the consequence is.
- Report a component, test, or pipeline as "done," "implemented," or "scaffolded" unless you have directly verified — by reading the actual code you wrote, not by recalling what you intended to write — that it does what it claims to do.

**The test to apply before saying something is done: if the user directly read the file right now, would it match your description?** If you have not just re-read the actual file to check, do that before reporting status.

---

## 2. No placeholder code presented as real

- Never write a test that only asserts `True`, or any test that cannot fail, and report it as a "test suite." Either write a real test that exercises real behavior, or say explicitly "I have not written real tests for X yet."
- Never leave a function, class, or module that's central to the task (e.g., a documented architectural component) unimplemented while implying, by silence or by a docstring, that it exists.
- If something is genuinely a placeholder or stub — acceptable at early scaffolding stages — label it as such in the code (a `# TODO: not yet implemented` comment) AND say so explicitly in your response to the user. Do not let a docstring describe functionality that the code beneath it doesn't contain.

---

## 3. Architecture components require self-verification before being called "done"

This project's model must implement RoPE, RMSNorm, and SwiGLU. For each one, before reporting it as implemented:
- **RoPE**: verify, by writing a tiny throwaway test, that the attention output for the same tokens genuinely differs when they appear at different sequence positions. If output is identical regardless of position, RoPE (or any position encoding) is not actually wired in, regardless of what the code appears to do.
- **RMSNorm**: verify the forward pass does NOT subtract a mean (only scales by root-mean-square) — this is the one-line difference from LayerNorm that defines it.
- **SwiGLU**: verify the feedforward block uses three weight matrices (gate, up, down) with a SiLU/Swish gate, not two matrices with GELU.
- **Tied embeddings**: verify `lm_head.weight` and the token embedding weight are literally the same tensor object (`is` check, not just equal values), and that gradient updates to one affect the other.

State the result of each check explicitly when reporting architecture work as complete.

---

## 4. Every training run must confirm its data source before it starts, out loud

Before launching any training run (pilot or real), print/report, and state plainly to the user:
- Whether the data source is real (name the actual dataset(s) and confirm they are being streamed/loaded) or synthetic/placeholder.
- The exact tokenizer being used, its vocabulary size, and confirmation that this matches the model's configured `vocab_size` — if it does not match exactly, do not proceed; raise this to the user first rather than adding a workaround (like a modulo operation) that silently changes what the data means.
- Run a tokenize → decode round-trip on one real sample sentence and show the input and output side by side, before the real run starts, every time the tokenizer or data pipeline changes.

---

## 5. Checkpointing is mandatory for any run expected to exceed ~30 minutes

Any training run must:
- Save a checkpoint (model + optimizer + LR scheduler state) at a regular step interval, not only at the very end.
- Be resumable from the most recent checkpoint if interrupted.
- Be tested once, deliberately, against a simulated interruption (kill the process mid-run, then resume) before being trusted for a real multi-hour run — and this test must be reported as having been run, with its result, not assumed to work because the code "looks right."

A run that only saves a checkpoint after completion provides no protection against the exact failure mode (Kaggle's 12-hour session cap, or any other interruption) this project is explicitly exposed to.

---

## 6. Two-document reporting requirement — every time you explain finished work

Whenever you report that a task, feature, or fix is complete, produce **two separate write-ups**, clearly labeled:

### Document A — Technical Report (for cross-checking against a second AI assistant)
- Precise, complete, assumes ML/software engineering literacy.
- Include exact file paths, exact function/class names, and the actual code for anything load-bearing (not a paraphrase of it).
- State exactly what was tested and how, and what was NOT tested.
- State any known limitations, shortcuts, or remaining risks explicitly — do not omit something just because it wasn't asked about directly.
- This document will be read by a second AI assistant whose job is to independently verify your claims against the real repository state. Assume it will check, not just trust.

### Document B — Plain-Language Summary (for the user directly)
- Written for someone who is a beginner in ML/deep learning but has been actively learning core concepts (attention, Q/K/V, RoPE, RMSNorm, SwiGLU, causal masking — these have already been taught and understood; do not re-explain them from scratch, but do not assume deep implementation-level fluency either).
- Explain what was done and why, in plain terms, using analogies where helpful, consistent with how these concepts have already been taught in this project.
- Explicitly state what this means for the project practically — what changed, whether anything needs to be re-run, whether any prior result is now invalidated.
- No unexplained jargon, no assumption that summary alone ("it's fixed now") is sufficient — explain what was actually wrong and what actually changed, in terms a careful beginner can follow and could, in principle, re-explain to someone else.

**Do not skip Document B and only produce Document A. Do not skip Document A and only produce Document B. Both, every time, for any nontrivial piece of completed work.**

---

## 7. When asked to audit or verify something, actually read the files

When asked "is X actually working" or "audit the codebase for gaps," the correct process is:
1. Open and read the actual current content of every relevant file — not a memory or summary of what you last wrote.
2. Check each claim against what the code actually does, function by function, not against what the docstring or file name claims.
3. Report findings for each component checked individually — "X is real and verified because I checked Y," or "X is a placeholder/stub, here is exactly what's missing" — not a blanket "everything else looks fine."
4. If something cannot be verified without running it (e.g., "does training actually converge"), say so explicitly rather than asserting confidence you don't have.

A summary that says "everything else is fine" without showing what was checked is not an acceptable answer to an audit request.

---

## 8. Golden rule

**If you would be even slightly embarrassed for the user to discover, by reading the raw code themselves, that something is different from what you told them — stop, and tell them first.** Every incident this file exists to prevent was exactly that: something the user discovered by looking directly at the code, that had not been surfaced proactively. Surfacing it first, even when the news is bad (a bug, a missing feature, a shortcut you took), is always the correct move and is never treated as a failure in itself. Silence, or a report that turns out not to match the real code, is the actual failure.
