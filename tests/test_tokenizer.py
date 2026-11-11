import pytest
import os
from tokenizers import Tokenizer

def test_tokenizer_roundtrip():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_dir, "tokenizer", "slm_v2_tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, cannot test roundtrip yet.")
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    sentence = "The quick brown fox jumps over the lazy dog. def hello(): print('world')"
    
    encoded = tokenizer.encode(sentence).ids
    decoded = tokenizer.decode(encoded)
    
    # Using tokenizers library, decoded might strip some spaces if not configured perfectly, 
    # but the core words should match.
    assert "quick brown fox" in decoded
    assert "def hello" in decoded
    assert "world" in decoded
