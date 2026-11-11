import os
import random
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Global variables for special tokens
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]

def get_stream_iterator(num_docs=10000, seed=1337):
    from datasets import load_dataset
    print("Connecting to Hugging Face dataset streams (70% FineWeb-Edu, 20% Cosmopedia, 10% Wikipedia)...", flush=True)
    fineweb_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    code_ds = load_dataset("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True)
    wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    fineweb_iter = iter(fineweb_ds)
    code_iter = iter(code_ds)
    wiki_iter = iter(wiki_ds)

    rng = random.Random(seed)
    docs_collected = 0
    print(f"Streaming {num_docs} documents...", flush=True)
    while docs_collected < num_docs:
        prob = rng.random()
        try:
            if prob < 0.70:
                text = next(fineweb_iter).get("text", "")
            elif prob < 0.90:
                text = next(code_iter).get("text", "")
            else:
                text = next(wiki_iter).get("text", "")
        except StopIteration:
            fineweb_iter = iter(fineweb_ds)
            code_iter = iter(code_ds)
            wiki_iter = iter(wiki_ds)
            continue
            
        if not text or not text.strip():
            continue
            
        docs_collected += 1
        if docs_collected % 1000 == 0:
            print(f"Collected {docs_collected}/{num_docs} documents", flush=True)
            
        yield text

def get_file_iterator(path="tokenizer/tiny.txt"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Stream from HF datasets instead of local file")
    parser.add_argument("--num_docs", type=int, default=10000, help="Number of documents to stream")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size")
    args = parser.parse_args()

    print("Initializing ByteLevel BPE tokenizer...", flush=True)
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    if args.stream:
        iterator = get_stream_iterator(num_docs=args.num_docs)
    else:
        if os.path.exists("tokenizer/tiny.txt"):
            iterator = get_file_iterator("tokenizer/tiny.txt")
        else:
            iterator = get_stream_iterator(num_docs=args.num_docs)
            
    print("Training tokenizer...", flush=True)
    tokenizer.train_from_iterator(iterator, trainer)
    
    save_path = "tokenizer"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "slm_v2_tokenizer.json")
    tokenizer.save(file_path)
    print(f"Tokenizer trained and saved to {file_path} (vocab size: {tokenizer.get_vocab_size()})", flush=True)

if __name__ == "__main__":
    main()
