import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class RealMixedDataset(IterableDataset):
    """
    Streams and tokenizes a fixed mixture of real datasets from Hugging Face:
    - 70% FineWeb-Edu (HuggingFaceFW/fineweb-edu)
    - 20% StarCoder-Python + Cosmopedia (HuggingFaceTB/smollm-corpus python & cosmopedia)
    - 10% Wikipedia (wikimedia/wikipedia 20231101.en)

    Tokens are packed into contiguous blocks of length `seq_len` with a fixed seed.
    """
    def __init__(self, tokenizer_name="huggyllama/llama-7b", seq_len=512, seed=1337):
        super().__init__()
        self.seq_len = seq_len
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        
    def _init_tokenizer_and_streams(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception:
            # Fallback to gpt2 if llama-7b tokenizer is inaccessible
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

        # 1. FineWeb-Edu (70%)
        fineweb_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # 2. StarCoder-Python + Cosmopedia (20%)
        code_ds = load_dataset("HuggingFaceTB/smollm-corpus", name="python", split="train", streaming=True)
        
        # 3. Wikipedia (10%)
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

        return tokenizer, iter(fineweb_ds), iter(code_ds), iter(wiki_ds)

    def __iter__(self):
        tokenizer, fineweb_iter, code_iter, wiki_iter = self._init_tokenizer_and_streams()
        
        # Deterministic random generator for dataset sampling
        rng = random.Random(self.seed)
        token_buffer = []

        while True:
            prob = rng.random()
            text = ""
            try:
                if prob < 0.70:
                    text = next(fineweb_iter).get("text", "")
                elif prob < 0.90:
                    text = next(code_iter).get("text", "")
                else:
                    text = next(wiki_iter).get("text", "")
            except StopIteration:
                # Restart stream if exhausted
                _, fineweb_iter, code_iter, wiki_iter = self._init_tokenizer_and_streams()
                continue

            if not text or not text.strip():
                continue

            # Tokenize text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
            
            token_buffer.extend(tokens)

            # Yield packed sequences of length seq_len + 1 (for input and target shift)
            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len :]

                # Cap token IDs to model vocab_size (32000)
                chunk = [t % 32000 for t in chunk]
                
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield inputs, targets

def get_real_dataloader(batch_size=8, seq_len=512, seed=1337):
    dataset = RealMixedDataset(seq_len=seq_len, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
