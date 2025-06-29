from typing import List
import tiktoken

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """Chunk text based on token count using tiktoken (OpenAI tokenizer)."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]