import os
from tokenizers import Tokenizer

def test():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_dir, "tokenizer", "slm_v2_tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found at", tokenizer_path)
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    sentence = "The quick brown fox jumps over the lazy dog. def hello(): print('world')"
    
    encoded = tokenizer.encode(sentence).ids
    decoded = tokenizer.decode(encoded)
    
    print("--- Round Trip Test ---")
    print(f"Input:   {sentence}")
    print(f"Decoded: {decoded}")
    print(f"Tokens:  {encoded}")

if __name__ == "__main__":
    test()
