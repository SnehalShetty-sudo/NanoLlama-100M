import os
import sys
import time
import torch
import torch.nn.functional as F
import streamlit as st

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from scripts.benchmark_throughput import DecoderTransformer
from tokenizers import Tokenizer

# Page configuration
st.set_page_config(
    page_title="NanoLlama-200M Interactive Demo",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e3440;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4e8cff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading so it stays in memory
@st.cache_resource(show_spinner="Loading NanoLlama-200M weights into memory...")
def load_model(checkpoint_path, model_size='200M'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_size == '200M':
        d_model, n_layers, n_heads, d_ff = 1024, 14, 16, 2752
    elif model_size == '10M':
        d_model, n_layers, n_heads, d_ff = 256, 4, 4, 680
    elif model_size == '100M':
        d_model, n_layers, n_heads, d_ff = 768, 11, 12, 2048
    else:
        d_model, n_layers, n_heads, d_ff = 64, 1, 4, 256

    model = DecoderTransformer(
        vocab_size=32000,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    
    tokenizer_path = os.path.join(base_dir, "tokenizer", "slm_v2_tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    metadata = {
        'step': ckpt.get('step', 61034),
        'loss': ckpt.get('loss', 2.6084),
        'device': str(device),
        'total_params': sum(p.numel() for p in model.parameters()) / 1e6
    }
    return model, tokenizer, metadata, device

# Generator for token-by-token streaming
def stream_generate(model, tokenizer, prompt, device, max_new_tokens=60, temp=0.8, top_k=40, top_p=0.9, rep_penalty=1.15):
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # We maintain decoded text to yield new tokens
    generated_ids = []
    
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
            
            # Top-P (nucleus)
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
            
            token_id = next_token.item()
            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Decode only the newest piece
            chunk = tokenizer.decode([token_id])
            yield chunk

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SnehalShetty-sudo/NanoLlama-100M/main/README.md", width=0) if False else None
    st.title("🦙 NanoLlama Control")
    
    # Available checkpoints
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    available_ckpts = []
    if os.path.exists(checkpoints_dir):
        available_ckpts = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        
    selected_ckpt = st.selectbox(
        "Checkpoint",
        options=available_ckpts if available_ckpts else ["pilot_200M_step61035.pt"],
        index=available_ckpts.index("pilot_200M_step61035.pt") if "pilot_200M_step61035.pt" in available_ckpts else 0
    )
    
    # Model size inference
    model_size = '200M' if '200M' in selected_ckpt else ('10M' if '10M' in selected_ckpt else '100M')
    
    st.subheader("⚙️ Sampling Hyperparameters")
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.75, step=0.05,
                            help="Lower = more deterministic, Higher = more creative.")
    top_k = st.slider("Top-K Filtering", min_value=1, max_value=100, value=40, step=5,
                      help="Limits pool to top K most probable tokens.")
    top_p = st.slider("Top-P (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=0.90, step=0.05,
                      help="Cumulative probability threshold.")
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=1.5, value=1.15, step=0.05,
                                   help="Penalizes repetitive sequences.")
    max_tokens = st.slider("Max New Tokens", min_value=10, max_value=256, value=64, step=10,
                           help="Maximum number of tokens to generate.")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Prompts")
    quick_prompts = [
        "In machine learning, neural networks",
        "The capital of France is",
        "Once upon a time in a distant kingdom,",
        "The theory of relativity states that",
        "def calculate_factorial(n):"
    ]
    for qp in quick_prompts:
        if st.button(qp, key=f"btn_{qp[:15]}", use_container_width=True):
            st.session_state['current_prompt'] = qp

# Load model
ckpt_full_path = os.path.join(checkpoints_dir, selected_ckpt)
model, tokenizer, meta, device = load_model(ckpt_full_path, model_size)

# --- MAIN UI ---
st.markdown('<div class="main-title">🦙 NanoLlama-200M Inference Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Decoder-Only Transformer trained from scratch on 250M real dataset tokens (FineWeb-Edu, Cosmopedia-v2, Wikipedia)</div>', unsafe_allow_html=True)

# Metrics bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["total_params"]:.1f}M</div><div class="metric-label">Parameters</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["step"]:,}</div><div class="metric-label">Training Steps Completed</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["loss"]:.3f}</div><div class="metric-label">Final Loss</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["device"].upper()}</div><div class="metric-label">Compute Device</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chat History state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display conversation
for item in st.session_state['chat_history']:
    with st.chat_message("user"):
        st.write(item['prompt'])
    with st.chat_message("assistant", avatar="🦙"):
        st.write(item['response'])
        if 'time' in item:
            st.caption(f"⚡ {item['time']:.2f}s | {item['speed']:.1f} tok/s")

# User input
default_prompt = st.session_state.get('current_prompt', '')
user_prompt = st.chat_input("Enter your prompt here...")

# If quick prompt was clicked, use it
if default_prompt and not user_prompt:
    user_prompt = default_prompt
    st.session_state['current_prompt'] = ''

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
        
    with st.chat_message("assistant", avatar="🦙"):
        start_time = time.time()
        
        # Stream the output live!
        response_placeholder = st.empty()
        full_response = ""
        token_count = 0
        
        for chunk in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user_prompt,
            device=device,
            max_new_tokens=max_tokens,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=repetition_penalty
        ):
            full_response += chunk
            token_count += 1
            response_placeholder.markdown(full_response + "▌")
            
        elapsed = time.time() - start_time
        speed = token_count / max(elapsed, 1e-5)
        response_placeholder.markdown(full_response)
        st.caption(f"⚡ Generated {token_count} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)")
        
        # Save to chat history
        st.session_state['chat_history'].append({
            'prompt': user_prompt,
            'response': full_response,
            'time': elapsed,
            'speed': speed
        })

# Expandable Architecture Details
with st.expander("🔍 View NanoLlama-200M Model Architecture"):
    st.markdown("""
    | Architecture Parameter | Value | Notes |
    | :--- | :--- | :--- |
    | **Vocabulary Size** | 32,000 | Custom Byte-Level BPE |
    | **Hidden Dimension ($d_{model}$)** | 1,024 | Tied Input/Output Embeddings |
    | **Layers ($n_{layers}$)** | 14 | Transformer Decoder Blocks |
    | **Attention Heads ($n_{heads}$)** | 16 | Multi-Head Self-Attention |
    | **Head Dimension ($d_{head}$)** | 64 | $1024 / 16$ |
    | **FeedForward Dimension ($d_{ff}$)** | 2,752 | SwiGLU / GeLU Activation |
    | **Position Embeddings** | RoPE | Rotary Position Embeddings |
    | **Total Parameters** | **209.88 Million** | Exact count |
    | **Training Mixture** | 70% FineWeb-Edu, 20% Cosmopedia, 10% Wiki | Resilient Streaming |
    """)
