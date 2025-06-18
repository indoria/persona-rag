from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class EmbeddingType(Enum):
    """Supported embedding types"""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_BERT = "sentence_bert"
    UNIVERSAL_SENTENCE_ENCODER = "universal_sentence_encoder"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    MPNET = "mpnet"
    E5 = "e5"
    BGE = "bge"
    INSTRUCTOR = "instructor"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_length: Optional[int] = None
    batch_size: int = 32
    device: str = "auto"
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"
    trust_remote_code: bool = False
    cache_folder: Optional[str] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

@dataclass
class EmbeddingResult:
    """Result container for embeddings"""
    embeddings: np.ndarray
    model_name: str
    embedding_dimension: int
    texts: List[str]
    processing_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextEmbedder(ABC):
    """Abstract base class for text embedding models"""
    
    def __init__(self, embedding_type: Union[str, EmbeddingType], config: Optional[EmbeddingConfig] = None):
        """
        Initialize the text embedder
        
        Args:
            embedding_type: Type of embedding model to use
            config: Configuration for the embedding model
        """
        if isinstance(embedding_type, str):
            try:
                self.embedding_type = EmbeddingType(embedding_type)
            except ValueError:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
        else:
            self.embedding_type = embedding_type
            
        self.config = config or EmbeddingConfig(model_name="default")
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model"""
        pass
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        pass
    
    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, EmbeddingResult]:
        """
        Embed text(s) and return vector embeddings
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Vector embeddings as numpy array or EmbeddingResult object
        """
        import time
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Validate inputs
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a string or list of strings")
        
        # Process in batches
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        processing_time = time.time() - start_time
        
        # Return as EmbeddingResult for detailed information
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.config.model_name,
            embedding_dimension=embeddings.shape[1],
            texts=texts,
            processing_time=processing_time,
            metadata={
                "embedding_type": self.embedding_type.value,
                "batch_size": batch_size,
                "normalized": self.config.normalize_embeddings
            }
        )
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text and return just the vector
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as numpy array
        """
        result = self.embed([text])
        return result.embeddings[0]
    
    def similarity(self, text1: str, text2: str, metric: str = "cosine") -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score
        """
        embeddings = self.embed([text1, text2])
        vec1, vec2 = embeddings.embeddings[0], embeddings.embeddings[1]
        
        if metric == "cosine":
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif metric == "euclidean":
            return 1 / (1 + np.linalg.norm(vec1 - vec2))
        elif metric == "dot":
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if not self.is_loaded:
            self.load_model()
        # Test with a dummy text
        test_embedding = self.embed_single("test")
        return test_embedding.shape[0]
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "embedding_type": self.embedding_type.value,
            "model_name": self.config.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.is_loaded else None,
            "device": self.config.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "normalized": self.config.normalize_embeddings
        }
    
    def __repr__(self) -> str:
        return f"TextEmbedder(type={self.embedding_type.value}, model={self.config.model_name})"