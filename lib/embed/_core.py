"""Abstract base class for text embedders."""

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np
import time
import logging

from .config import EmbeddingConfig
from .types import EmbeddingResult, EmbeddingType
from .exceptions import EmbeddingError, ModelNotLoadedError

logger = logging.getLogger(__name__)


class TextEmbedder(ABC):
    """Abstract base class for text embedding models."""
    
    def __init__(self, embedding_type: EmbeddingType, config: EmbeddingConfig):
        """
        Initialize the text embedder.
        
        Args:
            embedding_type: Type of embedding model to use
            config: Configuration for the embedding model
        """
        self.embedding_type = embedding_type
        self.config = config
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
        pass
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    def embed(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """
        Embed text(s) and return vector embeddings.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            EmbeddingResult object containing embeddings and metadata
            
        Raises:
            EmbeddingError: If embedding fails
            ModelNotLoadedError: If model is not loaded
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.is_loaded:
                self.load_model()
            
            # Handle single text input
            if isinstance(texts, str):
                texts = [texts]
            
            # Validate inputs
            self._validate_inputs(texts)
            
            # Process in batches
            all_embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_embeddings = self._embed_batch(batch)
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
                    raise EmbeddingError(f"Batch embedding failed: {e}")
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.config.normalize_embeddings:
                embeddings = self._normalize_embeddings(embeddings)
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.config.model_name,
                embedding_dimension=embeddings.shape[1],
                texts=texts,
                processing_time=processing_time,
                metadata={
                    "embedding_type": self.embedding_type.value,
                    "batch_size": batch_size,
                    "normalized": self.config.normalize_embeddings,
                    "device": self.config.device
                }
            )
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            if isinstance(e, (EmbeddingError, ModelNotLoadedError)):
                raise
            raise EmbeddingError(f"Unexpected error during embedding: {e}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text and return just the vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as numpy array
        """
        result = self.embed([text])
        return result.embeddings[0]
    
    def similarity(self, text1: str, text2: str, metric: str = "cosine") -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score
            
        Raises:
            ValueError: If metric is not supported
        """
        embeddings = self.embed([text1, text2])
        vec1, vec2 = embeddings.embeddings[0], embeddings.embeddings[1]
        
        if metric == "cosine":
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif metric == "euclidean":
            return float(1 / (1 + np.linalg.norm(vec1 - vec2)))
        elif metric == "dot":
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if not self.is_loaded:
            self.load_model()
        # Test with a dummy text
        test_embedding = self.embed_single("test")
        return test_embedding.shape[0]
    
    def _validate_inputs(self, texts: List[str]) -> None:
        """Validate input texts."""
        if not texts:
            raise ValueError("Input texts list is empty")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings")
        
        if any(len(text.strip()) == 0 for text in texts):
            logger.warning("Some input texts are empty or whitespace-only")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "embedding_type": self.embedding_type.value,
            "model_name": self.config.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.is_loaded else None,
            "device": self.config.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "normalized": self.config.normalize_embeddings,
            "is_loaded": self.is_loaded
        }
    
    def __repr__(self) -> str:
        return f"TextEmbedder(type={self.embedding_type.value}, model={self.config.model_name})"