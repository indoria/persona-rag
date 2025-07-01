from typing import List
import tiktoken
import nltk

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    Chunk text based on token count using tiktoken (OpenAI tokenizer).
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def chunker(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunking with sentence boundary awareness and optional token overlap.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        if current_tokens + len(tokens) > max_tokens:
            # finalize current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # start new chunk with overlap
            overlap_tokens = tokenizer.encode(" ".join(current_chunk[-overlap:])) if overlap > 0 else []
            current_chunk = [sentence]
            current_tokens = len(tokens)
        else:
            current_chunk.append(sentence)
            current_tokens += len(tokens)

    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
