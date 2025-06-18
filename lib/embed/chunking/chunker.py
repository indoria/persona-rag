import re
from abc import ABC, abstractmethod
from typing import List, Optional

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("NLTK, sentence-transformers, numpy, or scikit-learn not found. SemanticChunker and SentenceBasedChunker will be limited or unavailable.")
    _HAS_SENTENCE_TRANSFORMERS = False
    # Mock functions if libraries are not available
    def sent_tokenize(text):
        return re.split(r'(?<=[.!?])\s+', text)

    class MockSentenceTransformer:
        def encode(self, sentences):
            # Return dummy embeddings for demonstration
            return np.random.rand(len(sentences), 384) # Example dimension

    SentenceTransformer = MockSentenceTransformer
    cosine_similarity = lambda x, y: np.array([[0.5]]) # Dummy similarity
    np = type('module', (object,), {'array': lambda x: x, 'mean': lambda x, axis: x, 'zeros': lambda x: x})() # Mock numpy


class ChunkingStrategy(ABC):
    """
    Abstract Base Class for all chunking strategies.
    Defines the interface for chunking text.
    """
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Splits the input text into a list of chunks.

        Args:
            text (str): The input journalism piece content.

        Returns:
            List[str]: A list of text chunks.
        """
        pass

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """
        Applies overlap to a list of chunks. This is a helper method
        that can be used by various chunking strategies.
        """
        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks =
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Take the last 'overlap' characters from the previous chunk
                # and prepend them to the current chunk.
                # Ensure we don't go out of bounds for the previous chunk.
                prev_chunk_end = chunks[i-1][-overlap:]
                overlapped_chunks.append(prev_chunk_end + chunk)
        return overlapped_chunks


class FixedSizeChunker(ChunkingStrategy):
    """
    Implements fixed-size chunking with optional overlap.
    This is the simplest strategy, dividing text into segments of a predefined size.
    Good for initial experimentation or uniformly structured content.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            print("Warning: chunk_overlap is greater than or equal to chunk_size. This may lead to highly redundant chunks.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        chunks =
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            start_idx += self.chunk_size - self.chunk_overlap
            if start_idx < 0: # Handle cases where overlap is larger than chunk_size
                start_idx = 0
        return chunks

class SentenceBasedChunker(ChunkingStrategy):
    """
    Implements sentence-based chunking.
    Splits text based on natural sentence boundaries, ensuring semantic coherence
    at the sentence level. Can group multiple sentences per chunk.
    """
    def __init__(self, sentences_per_chunk: int = 3, chunk_overlap_sentences: int = 1):
        if sentences_per_chunk <= 0:
            raise ValueError("sentences_per_chunk must be positive.")
        if chunk_overlap_sentences < 0:
            raise ValueError("chunk_overlap_sentences cannot be negative.")
        if chunk_overlap_sentences >= sentences_per_chunk:
            print("Warning: chunk_overlap_sentences is greater than or equal to sentences_per_chunk. This may lead to highly redundant chunks.")
        self.sentences_per_chunk = sentences_per_chunk
        self.chunk_overlap_sentences = chunk_overlap_sentences

    def chunk(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks =
        i = 0
        while i < len(sentences):
            current_chunk_sentences = sentences[i : i + self.sentences_per_chunk]
            chunks.append(" ".join(current_chunk_sentences))
            i += self.sentences_per_chunk - self.chunk_overlap_sentences
            if self.sentences_per_chunk - self.chunk_overlap_sentences <= 0:
                # Prevent infinite loop if overlap is too large
                i += 1
            if i < 0: # Ensure index doesn't go negative
                i = 0
        return chunks

class ParagraphBasedChunker(ChunkingStrategy):
    """
    Implements paragraph-based chunking.
    Splits text based on natural paragraph breaks (double newlines).
    Each chunk aims to be a coherent paragraph.
    """
    def __init__(self, max_paragraph_length: Optional[int] = None, chunk_overlap_chars: int = 0):
        self.max_paragraph_length = max_paragraph_length
        self.chunk_overlap_chars = chunk_overlap_chars

    def chunk(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks =
        for paragraph in paragraphs:
            if self.max_paragraph_length and len(paragraph) > self.max_paragraph_length:
                # If a paragraph is too long, fall back to fixed-size chunking within it
                temp_chunker = FixedSizeChunker(self.max_paragraph_length, self.chunk_overlap_chars)
                chunks.extend(temp_chunker.chunk(paragraph))
            else:
                chunks.append(paragraph)
        return chunks

class RecursiveCharacterChunker(ChunkingStrategy):
    """
    Implements recursive character text splitting.
    Iteratively breaks down text using a hierarchical list of separators.
    Aims to preserve structural and semantic meaning.
    """
    def __init__(self, separators: Optional[List[str]] = None, chunk_size: int = 500, chunk_overlap: int = 50):
        self.separators = separators if separators is not None else ["\n\n", "\n", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_recursively(self, text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= self.chunk_size:
            # Base case: no more separators or chunk is small enough
            return [text] if text else

        current_separator = separators
        next_separators = separators[1:]

        parts = text.split(current_separator)
        chunks =
        current_chunk_content =
        current_chunk_length = 0

        for part in parts:
            sub_chunks = self._split_recursively(part, next_separators)
            for sub_chunk in sub_chunks:
                if current_chunk_length + len(sub_chunk) + len(current_separator) <= self.chunk_size:
                    current_chunk_content.append(sub_chunk)
                    current_chunk_length += len(sub_chunk) + len(current_separator)
                else:
                    if current_chunk_content:
                        chunks.append(current_separator.join(current_chunk_content).strip())
                    current_chunk_content = [sub_chunk]
                    current_chunk_length = len(sub_chunk)
        if current_chunk_content:
            chunks.append(current_separator.join(current_chunk_content).strip())

        # Apply overlap after initial recursive splitting
        final_chunks =
        for i, chunk in enumerate(chunks):
            if i == 0:
                final_chunks.append(chunk)
            else:
                # Find the overlap from the previous chunk
                overlap_text = chunks[i-1][-(self.chunk_overlap):]
                final_chunks.append(overlap_text + chunk)
        return final_chunks

    def chunk(self, text: str) -> List[str]:
        return self._split_recursively(text, self.separators)


class DocumentStructureChunker(ChunkingStrategy):
    """
    Implements document-structure-based chunking, tailored for journalism pieces.
    Assumes journalism pieces might have titles, subheadings, and paragraphs.
    It attempts to chunk based on these structural elements.
    """
    def __init__(self, heading_patterns: Optional[List[str]] = None,
                 max_section_length: Optional[int] = 1000, chunk_overlap_chars: int = 100):
        # Default patterns for common journalism structures (e.g., Markdown-like headings)
        # You might need to customize these based on your actual journalism file formats.
        self.heading_patterns = heading_patterns if heading_patterns is not None else [
            r"^\s*#\s*(.*)",  # Markdown H1
            r"^\s*##\s*(.*)", # Markdown H2
            r"^\s*[A-Z][A-Z0-9\s]*:\s*$", # ALL CAPS followed by colon (e.g., "INTRODUCTION:")
            r"^\s*[A-Z][a-z]+\s[A-Z][a-z]+.*?\n\n", # Potential bolded/capitalized section titles
        ]
        self.max_section_length = max_section_length
        self.chunk_overlap_chars = chunk_overlap_chars

    def chunk(self, text: str) -> List[str]:
        lines = text.split('\n')
        chunks =
        current_section_lines =
        
        for line in lines:
            is_heading = False
            for pattern in self.heading_patterns:
                if re.match(pattern, line):
                    is_heading = True
                    break
            
            if is_heading and current_section_lines:
                # End current section and start a new one
                section_content = "\n".join(current_section_lines).strip()
                if section_content:
                    if self.max_section_length and len(section_content) > self.max_section_length:
                        # If section is too long, sub-chunk it using recursive chunking
                        sub_chunker = RecursiveCharacterChunker(
                            separators=["\n\n", "\n", " "],
                            chunk_size=self.max_section_length,
                            chunk_overlap=self.chunk_overlap_chars
                        )
                        chunks.extend(sub_chunker.chunk(section_content))
                    else:
                        chunks.append(section_content)
                current_section_lines = [line] # Start new section with the heading
            else:
                current_section_lines.append(line)
        
        # Add the last section
        if current_section_lines:
            section_content = "\n".join(current_section_lines).strip()
            if section_content:
                if self.max_section_length and len(section_content) > self.max_section_length:
                    sub_chunker = RecursiveCharacterChunker(
                        separators=["\n\n", "\n", " "],
                        chunk_size=self.max_section_length,
                        chunk_overlap=self.chunk_overlap_chars
                    )
                    chunks.extend(sub_chunker.chunk(section_content))
                else:
                    chunks.append(section_content)
        
        # Apply overlap across the main sections if desired (optional, as sub-chunking handles overlap)
        # For simplicity, we'll rely on sub-chunking's overlap or assume sections are distinct.
        return [chunk for chunk in chunks if chunk] # Filter out empty chunks


class SemanticChunker(ChunkingStrategy):
    """
    Implements semantic chunking.
    Splits text based on semantic similarity between sentences, aiming to group
    semantically coherent ideas. Requires an embedding model.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.7, min_sentences_per_chunk: int = 2,
                 chunk_overlap_sentences: int = 1):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("SemanticChunker requires 'nltk', 'sentence-transformers', 'numpy', 'scikit-learn'. Please install them.")
        self.model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.chunk_overlap_sentences = chunk_overlap_sentences

    def chunk(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        if not sentences:
            return
        
        # Generate embeddings for each sentence
        sentence_embeddings = self.model.encode(sentences)
        
        # Calculate cosine similarity between adjacent sentences
        similarities =
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                sentence_embeddings[i].reshape(1, -1),
                sentence_embeddings[i+1].reshape(1, -1)
            )
            similarities.append(sim)
        
        # Identify break points where similarity drops below threshold
        break_points =  # Start of the first chunk
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                # Ensure minimum sentences per chunk before breaking
                if (i + 1) - break_points[-1] >= self.min_sentences_per_chunk:
                    break_points.append(i + 1)
        break_points.append(len(sentences)) # End of the last chunk

        chunks =
        for i in range(len(break_points) - 1):
            start_idx = break_points[i]
            end_idx = break_points[i+1]
            
            # Apply overlap for semantic chunks
            if i > 0 and self.chunk_overlap_sentences > 0:
                start_idx = max(0, start_idx - self.chunk_overlap_sentences)
            
            chunk_sentences = sentences[start_idx:end_idx]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
        
        return chunks
