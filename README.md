# NanoLlama-100M (slm-v2)

NanoLlama-100M is a Small Language Model (~100M parameters) architecture and training codebase built for model design, custom BPE tokenizer training, data preprocessing, training loops, evaluation, and benchmarking.

## Project Structure

```text
slm-v2/
├── configs/          # Dataclass and YAML configurations for model architecture and training
├── data/             # Data ingestion, downloading, and preprocessing pipelines
├── model/            # Core model architecture (Attention, RoPE, RMSNorm, SwiGLU, Transformer Block)
├── training/         # Training loop, optimization, and checkpoint management
├── tokenizer/        # BPE tokenizer training scripts and wrapper classes
├── eval/             # Evaluation suite and fixed prompt evaluation sets
├── tests/            # pytest suite mirroring the module structure above
├── scripts/          # Utility scripts (benchmarking, export, profiling)
├── notebooks/        # Kaggle/Colab notebooks (kept separate from source module)
├── README.md         # Project documentation
├── requirements.txt   # Project dependencies
└── .gitignore        # Git ignore directives
```

## Setup Instructions

### 1. Virtual Environment

Create and activate a Python 3.10 virtual environment:

```bash
# Windows
py -3.10 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Tests

```bash
pytest
```
