# TextEmbedder System - Requirements and Setup Guide

## Overview

This TextEmbedder system provides a unified interface for various enterprise-grade text embedding models including Sentence Transformers, OpenAI, Azure OpenAI, Cohere, HuggingFace Transformers, and more.

# TextEmbedder Package Structure

## Recommended Directory Hierarchy

```
text-embedder/
├── README.md
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── docs/
│   ├── README.md
│   ├── api_reference.md
│   ├── tutorials/
│   │   ├── getting_started.md
│   │   ├── advanced_usage.md
│   │   └── deployment.md
│   └── examples/
│       ├── basic_usage.py
│       ├── semantic_search.py
│       └── clustering.py
├── src/
│   └── text_embedder/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── config.py
│       │   ├── types.py
│       │   └── exceptions.py
│       ├── embedders/
│       │   ├── __init__.py
│       │   ├── sentence_transformers.py
│       │   ├── openai_embedder.py
│       │   ├── azure_openai.py
│       │   ├── cohere_embedder.py
│       │   ├── huggingface.py
│       │   ├── universal_encoder.py
│       │   └── traditional.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── factory.py
│       │   ├── popular_models.py
│       │   ├── validation.py
│       │   └── metrics.py
│       └── cli/
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_config.py
│   │   ├── test_sentence_transformers.py
│   │   ├── test_openai.py
│   │   ├── test_factory.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_embedder_integration.py
│   │   └── test_api_integration.py
│   └── performance/
│       ├── __init__.py
│       ├── benchmark.py
│       └── memory_profiling.py
├── examples/
│   ├── __init__.py
│   ├── basic_usage.py
│   ├── batch_processing.py
│   ├── semantic_search.py
│   ├── clustering.py
│   ├── similarity_analysis.py
│   └── model_comparison.py
├── scripts/
│   ├── install_models.py
│   ├── benchmark_models.py
│   └── validate_setup.py
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements-docker.txt
```

## File Breakdown

### Root Level Files

**setup.py**
```python
from setuptools import setup, find_packages

setup(
    name="text-embedder",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "api": ["openai>=1.0.0", "cohere>=4.0.0"],
        "full": ["tensorflow>=2.8.0", "gensim>=4.2.0", "fasttext>=0.9.2"],
        "dev": ["pytest>=7.0.0", "black", "flake8", "mypy"],
    }
)
```

**pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "text-embedder"
version = "1.0.0"
description = "Enterprise-grade text embedding interface"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
```

### Core Package Structure

**src/text_embedder/__init__.py**
```python
"""Text Embedder - Enterprise-grade text embedding interface."""

from .core.base import TextEmbedder
from .core.config import EmbeddingConfig
from .core.types import EmbeddingType, EmbeddingResult
from .utils.factory import create_embedder
from .utils.popular_models import get_embedder, POPULAR_MODELS

__version__ = "1.0.0"
__all__ = [
    "TextEmbedder",
    "EmbeddingConfig", 
    "EmbeddingType",
    "EmbeddingResult",
    "create_embedder",
    "get_embedder",
    "POPULAR_MODELS"
]
```

### Separation of Concerns

**src/text_embedder/core/base.py** - Abstract base class
**src/text_embedder/core/config.py** - Configuration classes
**src/text_embedder/core/types.py** - Type definitions and enums
**src/text_embedder/core/exceptions.py** - Custom exceptions

**src/text_embedder/embedders/** - Individual embedder implementations
**src/text_embedder/utils/** - Utility functions and factories

### Testing Structure

**tests/conftest.py**
```python
import pytest
from text_embedder.core.config import EmbeddingConfig

@pytest.fixture
def basic_config():
    return EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        normalize_embeddings=True
    )

@pytest.fixture
def sample_texts():
    return [
        "This is a test sentence.",
        "Another test sentence for embedding.",
        "Machine learning is fascinating."
    ]
```

## Benefits of This Structure

### 1. **Modularity**
- Each embedder type is in its own file
- Core functionality separated from implementations
- Easy to add new embedders without modifying existing code

### 2. **Testability**
- Unit tests for individual components
- Integration tests for full workflows
- Performance benchmarks separate from functional tests

### 3. **Maintainability**
- Clear separation of concerns
- Easy to locate and modify specific functionality
- Consistent naming conventions

### 4. **Extensibility**
- Plugin-like architecture for new embedders
- Factory pattern for easy instantiation
- Configuration-driven approach

### 5. **Professional Standards**
- Proper Python packaging structure
- Documentation and examples included
- CI/CD ready with GitHub Actions
- Docker support for deployment

## Installation Commands

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-org/text-embedder.git
cd text-embedder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,full]"

# Run tests
pytest tests/

# Format code
black src/ tests/
```

### Production Installation
```bash
# Basic installation
pip install text-embedder

# With API support
pip install "text-embedder[api]"

# Full installation
pip install "text-embedder[full]"
```

## Import Patterns

### Simple Usage
```python
from text_embedder import get_embedder

embedder = get_embedder("all-MiniLM-L6-v2")
result = embedder.embed("Hello world")
```

### Advanced Usage
```python
from text_embedder import create_embedder, EmbeddingConfig, EmbeddingType

config = EmbeddingConfig(
    model_name="all-mpnet-base-v2",
    batch_size=32,
    normalize_embeddings=True
)

embedder = create_embedder(EmbeddingType.SENTENCE_TRANSFORMERS, config)
result = embedder.embed(["text1", "text2"])
```

### Specific Embedder
```python
from text_embedder.embedders.openai_embedder import OpenAIEmbedder
from text_embedder.core.config import EmbeddingConfig

config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    api_key="your-key"
)
embedder = OpenAIEmbedder(config)
```


## Core Requirements

### Python Dependencies

```txt
# Core dependencies
numpy>=1.21.0
python>=3.8

# Sentence Transformers (recommended)
sentence-transformers>=2.2.0
torch>=1.9.0
transformers>=4.21.0

# Optional: HuggingFace Transformers (for custom models)
transformers>=4.21.0
torch>=1.9.0
tokenizers>=0.13.0

# Optional: OpenAI API
openai>=1.0.0

# Optional: Cohere API
cohere>=4.0.0

# Optional: TensorFlow Hub (for Universal Sentence Encoder)
tensorflow>=2.8.0
tensorflow-hub>=0.12.0

# Optional: Traditional embeddings
gensim>=4.2.0  # For Word2Vec
fasttext>=0.9.2  # For FastText

# Optional: Advanced features
scikit-learn>=1.1.0  # For clustering examples
```

## Installation Options

### Option 1: Minimal Installation (Sentence Transformers only)

```bash
pip install torch sentence-transformers numpy
```

### Option 2: Full Local Installation

```bash
pip install torch sentence-transformers transformers tokenizers numpy scikit-learn gensim
```

### Option 3: With API Services

```bash
pip install torch sentence-transformers numpy openai cohere
```

### Option 4: Everything (Full Enterprise)

```bash
pip install torch sentence-transformers transformers tokenizers numpy scikit-learn gensim openai cohere tensorflow tensorflow-hub fasttext
```

## Supported Embedding Types

### 1. Sentence Transformers (Recommended)
- **Models**: all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2, bge-large-en-v1.5
- **Pros**: Easy to use, good performance, works offline
- **Cons**: Requires local compute resources

### 2. OpenAI Embeddings
- **Models**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Pros**: High quality, fast API, no local compute needed
- **Cons**: Requires API key, costs money, needs internet

### 3. Azure OpenAI
- **Models**: text-embedding-ada-002, text-embedding-3-small
- **Pros**: Enterprise-grade, integrated with Azure services
- **Cons**: Requires Azure subscription and API key

### 4. Cohere
- **Models**: embed-english-v3.0, embed-multilingual-v3.0
- **Pros**: Good multilingual support, enterprise features
- **Cons**: Requires API key, costs money

### 5. HuggingFace Transformers
- **Models**: Any BERT-like model from HuggingFace Hub
- **Pros**: Huge model selection, customizable
- **Cons**: Requires more setup, slower than sentence-transformers

### 6. Universal Sentence Encoder
- **Models**: Google's USE models via TensorFlow Hub
- **Pros**: Good general-purpose embeddings
- **Cons**: Requires TensorFlow, large model size

### 7. Traditional Embeddings
- **Word2Vec**: Good for word-level semantics
- **FastText**: Handles out-of-vocabulary words well
- **Pros**: Fast, lightweight, interpretable
- **Cons**: Less sophisticated than transformer models

## Quick Start Examples

### Basic Usage
```python
from text_embedder_interface import EmbeddingConfig
from embedding_implementations import create_embedder

# Create embedder
config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
embedder = create_embedder("sentence_transformers", config)

# Embed text
result = embedder.embed("Hello, world!")
vector_embeddings = result.embeddings
print(f"Embedding shape: {vector_embeddings.shape}")
```

### Using Popular Models
```python
from embedding_implementations import get_embedder

# Get a popular model with predefined configuration
embedder = get_embedder("all-MiniLM-L6-v2")
result = embedder.embed(["Hello", "Hi there", "Goodbye"])
print(f"Batch embeddings shape: {result.embeddings.shape}")
```

### API-based Embeddings
```python
from text_embedder_interface import EmbeddingConfig
from embedding_implementations import create_embedder

# OpenAI
config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    api_key="your-api-key-here"
)
embedder = create_embedder("openai", config)

# Cohere
config = EmbeddingConfig(
    model_name="embed-english-v3.0",
    api_key="your-api-key-here"
)
embedder = create_embedder("cohere", config)
```

## Model Recommendations

### For Production Use
1. **OpenAI text-embedding-3-small** - Best balance of quality and cost
2. **Sentence Transformers all-mpnet-base-v2** - Best open-source option
3. **BGE-large-en-v1.5** - State-of-the-art open-source model

### For Development/Testing
1. **all-MiniLM-L6-v2** - Fast, lightweight, good quality
2. **all-distilroberta-v1** - Good performance, smaller size

### For Specialized Tasks
1. **E5-large-v2** - Excellent for retrieval tasks
2. **Instructor-XL** - Good for instruction-following tasks
3. **Cohere embed-english-v3.0** - Good for enterprise applications

## Performance Considerations

### GPU Acceleration
- Most models support GPU acceleration
- Set `device="cuda"` in EmbeddingConfig for GPU usage
- Significant speedup for large batches

### Batch Processing
- Use appropriate batch sizes based on your hardware
- Larger batches = better GPU utilization
- Start with batch_size=32 and adjust based on memory

### Memory Usage
- Larger models require more memory
- Consider model quantization for memory-constrained environments
- Use appropriate max_length settings

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'sentence_transformers'**
   ```bash
   pip install sentence-transformers
   ```

2. **CUDA out of memory**
   - Reduce batch_size in EmbeddingConfig
   - Use smaller model variants
   - Set device="cpu" to use CPU instead

3. **API key errors**
   - Ensure API keys are set correctly
   - Check API key permissions and quotas

4. **Slow performance**
   - Use GPU acceleration when available
   - Increase batch_size for better throughput
   - Consider using smaller, faster models

### Environment Variables
```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Cohere
export COHERE_API_KEY="your-key-here"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

## Best Practices

1. **Model Selection**
   - Start with sentence-transformers for prototyping
   - Use API-based models for production if budget allows
   - Benchmark different models on your specific task

2. **Configuration**
   - Set appropriate batch sizes based on your hardware
   - Use normalization for similarity tasks
   - Cache models when possible

3. **Error Handling**
   - Always handle API failures gracefully
   - Implement retry logic for API calls
   - Validate inputs before processing

4. **Performance**
   - Use GPU acceleration when available
   - Batch process multiple texts when possible
   - Consider model quantization for deployment

## License and Usage

This system is designed to work with various embedding providers. Please ensure you comply with the terms of service for any API-based services you use:

- **OpenAI**: Check OpenAI's usage policies
- **Cohere**: Check Cohere's terms of service
- **HuggingFace**: Most models are open-source, check individual model licenses
- **Sentence Transformers**: Generally Apache 2.0 license

## Support and Contributing

For issues or feature requests:
1. Check the troubleshooting section above
2. Review the model documentation
3. Test with minimal examples first
4. Provide detailed error messages when reporting issues