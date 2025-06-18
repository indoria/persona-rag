import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from text_embedder_interface import TextEmbedder, EmbeddingType, EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformersEmbedder(TextEmbedder):
    """Sentence Transformers embedding implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.SENTENCE_TRANSFORMERS, config)
        
    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(
                model_name,
                device=self.config.device,
                cache_folder=self.config.cache_folder,
                trust_remote_code=self.config.trust_remote_code
            )
            self.is_loaded = True
            logger.info(f"Loaded SentenceTransformers model: {model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers package not found. Install with: pip install sentence-transformers")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

class OpenAIEmbedder(TextEmbedder):
    """OpenAI embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.OPENAI, config)
        
    def load_model(self):
        try:
            import openai
            
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")
            
            self.client = openai.OpenAI(api_key=self.config.api_key)
            self.model_name = self.config.model_name or "text-embedding-3-small"
            self.is_loaded = True
            logger.info(f"Initialized OpenAI embeddings with model: {self.model_name}")
            
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

class AzureOpenAIEmbedder(TextEmbedder):
    """Azure OpenAI embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.AZURE_OPENAI, config)
        
    def load_model(self):
        try:
            import openai
            
            if not self.config.api_key or not self.config.api_base:
                raise ValueError("Azure OpenAI API key and base URL are required")
            
            self.client = openai.AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.api_base,
                api_version="2024-02-01"
            )
            self.model_name = self.config.model_name or "text-embedding-ada-002"
            self.is_loaded = True
            logger.info(f"Initialized Azure OpenAI embeddings with model: {self.model_name}")
            
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

class CohereEmbedder(TextEmbedder):
    """Cohere embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.COHERE, config)
        
    def load_model(self):
        try:
            import cohere
            
            if not self.config.api_key:
                raise ValueError("Cohere API key is required")
            
            self.client = cohere.Client(self.config.api_key)
            self.model_name = self.config.model_name or "embed-english-v3.0"
            self.is_loaded = True
            logger.info(f"Initialized Cohere embeddings with model: {self.model_name}")
            
        except ImportError:
            raise ImportError("cohere package not found. Install with: pip install cohere")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document"
        )
        embeddings = np.array(response.embeddings)
        return embeddings

class HuggingFaceEmbedder(TextEmbedder):
    """HuggingFace transformers embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.HUGGINGFACE, config)
        
    def load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = self.config.model_name or "sentence-transformers/all-MiniLM-L6-v2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            self.model.to(self.device)
            self.is_loaded = True
            logger.info(f"Loaded HuggingFace model: {model_name}")
            
        except ImportError:
            raise ImportError("transformers package not found. Install with: pip install transformers torch")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        import torch
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length or 512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Apply pooling strategy
            if self.config.pooling_strategy == "mean":
                attention_mask = inputs['attention_mask']
                embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif self.config.pooling_strategy == "cls":
                embeddings = embeddings[:, 0]  # CLS token
            elif self.config.pooling_strategy == "max":
                embeddings = torch.max(embeddings, dim=1)[0]
            
        return embeddings.cpu().numpy()

class UniversalSentenceEncoderEmbedder(TextEmbedder):
    """Universal Sentence Encoder implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.UNIVERSAL_SENTENCE_ENCODER, config)
        
    def load_model(self):
        try:
            import tensorflow_hub as hub
            
            model_url = self.config.model_name or "https://tfhub.dev/google/universal-sentence-encoder/4"
            self.model = hub.load(model_url)
            self.is_loaded = True
            logger.info(f"Loaded Universal Sentence Encoder: {model_url}")
            
        except ImportError:
            raise ImportError("tensorflow-hub package not found. Install with: pip install tensorflow-hub")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model(texts)
        return embeddings.numpy()

class Word2VecEmbedder(TextEmbedder):
    """Word2Vec embeddings implementation with sentence averaging"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.WORD2VEC, config)
        
    def load_model(self):
        try:
            from gensim.models import Word2Vec
            import pickle
            
            model_path = self.config.model_name
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = Word2Vec.load(model_path)
            
            self.vector_size = self.model.wv.vector_size
            self.is_loaded = True
            logger.info(f"Loaded Word2Vec model: {model_path}")
            
        except ImportError:
            raise ImportError("gensim package not found. Install with: pip install gensim")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(self.vector_size)
            
            embeddings.append(sentence_vector)
        
        return np.array(embeddings)

class FastTextEmbedder(TextEmbedder):
    """FastText embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(EmbeddingType.FASTTEXT, config)
        
    def load_model(self):
        try:
            import fasttext
            
            model_path = self.config.model_name
            self.model = fasttext.load_model(model_path)
            self.vector_size = self.model.get_dimension()
            self.is_loaded = True
            logger.info(f"Loaded FastText model: {model_path}")
            
        except ImportError:
            raise ImportError("fasttext package not found. Install with: pip install fasttext")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # FastText can handle out-of-vocabulary words
            sentence_vector = self.model.get_sentence_vector(text)
            embeddings.append(sentence_vector)
        
        return np.array(embeddings)

# Factory function for creating embedders
def create_embedder(embedding_type: Union[str, EmbeddingType], config: EmbeddingConfig) -> TextEmbedder:
    """
    Factory function to create appropriate embedder based on type
    
    Args:
        embedding_type: Type of embedding to create
        config: Configuration for the embedder
        
    Returns:
        Configured TextEmbedder instance
    """
    if isinstance(embedding_type, str):
        embedding_type = EmbeddingType(embedding_type)
    
    embedder_classes = {
        EmbeddingType.SENTENCE_TRANSFORMERS: SentenceTransformersEmbedder,
        EmbeddingType.OPENAI: OpenAIEmbedder,
        EmbeddingType.AZURE_OPENAI: AzureOpenAIEmbedder,
        EmbeddingType.COHERE: CohereEmbedder,
        EmbeddingType.HUGGINGFACE: HuggingFaceEmbedder,
        EmbeddingType.UNIVERSAL_SENTENCE_ENCODER: UniversalSentenceEncoderEmbedder,
        EmbeddingType.WORD2VEC: Word2VecEmbedder,
        EmbeddingType.FASTTEXT: FastTextEmbedder,
        # Aliases for common models
        EmbeddingType.SENTENCE_BERT: SentenceTransformersEmbedder,
        EmbeddingType.BERT: HuggingFaceEmbedder,
        EmbeddingType.ROBERTA: HuggingFaceEmbedder,
        EmbeddingType.DISTILBERT: HuggingFaceEmbedder,
        EmbeddingType.MPNET: SentenceTransformersEmbedder,
        EmbeddingType.E5: SentenceTransformersEmbedder,
        EmbeddingType.BGE: SentenceTransformersEmbedder,
        EmbeddingType.INSTRUCTOR: SentenceTransformersEmbedder,
    }
    
    if embedding_type not in embedder_classes:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    return embedder_classes[embedding_type](config)

# Predefined configurations for popular models
POPULAR_MODELS = {
    # Sentence Transformers models
    "all-MiniLM-L6-v2": EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        normalize_embeddings=True
    ),
    "all-mpnet-base-v2": EmbeddingConfig(
        model_name="all-mpnet-base-v2",
        batch_size=16,
        normalize_embeddings=True
    ),
    "e5-large-v2": EmbeddingConfig(
        model_name="intfloat/e5-large-v2",
        batch_size=8,
        normalize_embeddings=True
    ),
    "bge-large-en-v1.5": EmbeddingConfig(
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=8,
        normalize_embeddings=True
    ),
    "instructor-xl": EmbeddingConfig(
        model_name="hkunlp/instructor-xl",
        batch_size=4,
        normalize_embeddings=True
    ),
    
    # OpenAI models
    "text-embedding-3-small": EmbeddingConfig(
        model_name="text-embedding-3-small",
        batch_size=100,
        normalize_embeddings=False
    ),
    "text-embedding-3-large": EmbeddingConfig(
        model_name="text-embedding-3-large",
        batch_size=50,
        normalize_embeddings=False
    ),
    
    # Cohere models
    "embed-english-v3.0": EmbeddingConfig(
        model_name="embed-english-v3.0",
        batch_size=50,
        normalize_embeddings=False
    ),
}

def get_embedder(model_name: str, **kwargs) -> TextEmbedder:
    """
    Convenience function to get a popular embedder with predefined config
    
    Args:
        model_name: Name of the popular model
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured TextEmbedder instance
    """
    if model_name not in POPULAR_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(POPULAR_MODELS.keys())}")
    
    config = POPULAR_MODELS[model_name]
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Determine embedder type based on model name
    if model_name.startswith("text-embedding"):
        embedding_type = EmbeddingType.OPENAI
    elif model_name.startswith("embed-"):
        embedding_type = EmbeddingType.COHERE
    else:
        embedding_type = EmbeddingType.SENTENCE_TRANSFORMERS
    
    return create_embedder(embedding_type, config)

# Example usage and testing
if __name__ == "__main__":
    # Example 1: Using Sentence Transformers
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        normalize_embeddings=True
    )
    
    embedder = create_embedder(EmbeddingType.SENTENCE_TRANSFORMERS, config)
    
    # Test embedding
    text = "This is a sample text for embedding."
    result = embedder.embed(text)
    print(f"Embedding shape: {result.embeddings.shape}")
    print(f"Model info: {embedder.model_info}")
    
    # Example 2: Using popular model shortcut
    # embedder = get_embedder("all-MiniLM-L6-v2")
    # result = embedder.embed(["Hello world", "How are you?"])
    # print(f"Batch embedding shape: {result.embeddings.shape}")
    
    # Example 3: Similarity calculation
    # similarity = embedder.similarity("Hello", "Hi there")
    # print(f"Similarity: {similarity}")