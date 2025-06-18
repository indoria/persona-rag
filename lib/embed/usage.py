"""
Comprehensive examples of using the TextEmbedder system
"""

import numpy as np
from text_embedder_interface import EmbeddingConfig, EmbeddingType
from embedding_implementations import create_embedder, get_embedder, POPULAR_MODELS

def example_basic_usage():
    """Basic usage example matching the requested API"""
    print("=== Basic Usage Example ===")
    
    # Create embedder with the requested API
    embedder = create_embedder(
        'sentence_transformers',
        EmbeddingConfig(model_name='all-MiniLM-L6-v2')
    )
    
    # Embed text
    text = "This is a sample text for embedding"
    result = embedder.embed(text)
    vector_embeddings = result.embeddings
    
    print(f"Text: {text}")
    print(f"Embedding shape: {vector_embeddings.shape}")
    print(f"First 5 dimensions: {vector_embeddings[0][:5]}")
    print()

def example_batch_processing():
    """Example of batch processing multiple texts"""
    print("=== Batch Processing Example ===")
    
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        normalize_embeddings=True
    )
    
    embedder = create_embedder(EmbeddingType.SENTENCE_TRANSFORMERS, config)
    
    texts = [
        "Machine learning is transforming industries",
        "Natural language processing enables computers to understand text",
        "Deep learning models require large amounts of data",
        "Artificial intelligence will reshape the future",
        "Python is a popular programming language for data science"
    ]
    
    result = embedder.embed(texts)
    
    print(f"Embedded {len(texts)} texts")
    print(f"Embeddings shape: {result.embeddings.shape}")
    print(f"Processing time: {result.processing_time:.3f} seconds")
    print(f"Model info: {result.model_name}")
    print()

def example_similarity_calculation():
    """Example of calculating text similarity"""
    print("=== Similarity Calculation Example ===")
    
    embedder = get_embedder("all-MiniLM-L6-v2")
    
    # Compare similar texts
    text1 = "The cat is sleeping on the couch"
    text2 = "A cat is resting on the sofa"
    text3 = "Dogs are running in the park"
    
    sim1_2 = embedder.similarity(text1, text2)
    sim1_3 = embedder.similarity(text1, text3)
    sim2_3 = embedder.similarity(text2, text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print()
    print(f"Similarity (Text 1 vs Text 2): {sim1_2:.4f}")
    print(f"Similarity (Text 1 vs Text 3): {sim1_3:.4f}")
    print(f"Similarity (Text 2 vs Text 3): {sim2_3:.4f}")
    print()

def example_different_models():
    """Example using different embedding models"""
    print("=== Different Models Example ===")
    
    text = "Artificial intelligence is revolutionizing technology"
    
    # Test different models (comment out models you don't have access to)
    models_to_test = [
        ("Sentence Transformers", "sentence_transformers", "all-MiniLM-L6-v2"),
        # ("OpenAI", "openai", "text-embedding-3-small"),  # Requires API key
        # ("Cohere", "cohere", "embed-english-v3.0"),      # Requires API key
        ("HuggingFace", "huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    
    for model_name, model_type, model_path in models_to_test:
        try:
            config = EmbeddingConfig(
                model_name=model_path,
                # api_key="your-api-key-here" if needed
            )
            
            embedder = create_embedder(model_type, config)
            result = embedder.embed(text)
            
            print(f"{model_name}:")
            print(f"  Model: {model_path}")
            print(f"  Embedding dimension: {result.embedding_dimension}")
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  First 3 dimensions: {result.embeddings[0][:3]}")
            print()
            
        except Exception as e:
            print(f"{model_name}: Error - {str(e)}")
            print()

def example_api_based_embeddings():
    """Example using API-based embedding services"""
    print("=== API-based Embeddings Example ===")
    
    # OpenAI Example (requires API key)
    try:
        openai_config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="your-openai-api-key",  # Replace with actual key
            batch_size=100
        )
        
        # Uncomment to test with actual API key
        # openai_embedder = create_embedder(EmbeddingType.OPENAI, openai_config)
        # result = openai_embedder.embed("Hello, world!")
        # print(f"OpenAI embedding dimension: {result.embedding_dimension}")
        
        print("OpenAI embedder configured (API key required for actual usage)")
        
    except Exception as e:
        print(f"OpenAI setup error: {e}")
    
    # Azure OpenAI Example
    try:
        azure_config = EmbeddingConfig(
            model_name="text-embedding-ada-002",
            api_key="your-azure-api-key",
            api_base="https://your-resource.openai.azure.com/",
            batch_size=50
        )
        
        print("Azure OpenAI embedder configured (API key required for actual usage)")
        
    except Exception as e:
        print(f"Azure OpenAI setup error: {e}")
    
    # Cohere Example
    try:
        cohere_config = EmbeddingConfig(
            model_name="embed-english-v3.0",
            api_key="your-cohere-api-key",
            batch_size=50
        )
        
        print("Cohere embedder configured (API key required for actual usage)")
        
    except Exception as e:
        print(f"Cohere setup error: {e}")
    
    print()

def example_advanced_configuration():
    """Example of advanced configuration options"""
    print("=== Advanced Configuration Example ===")
    
    # Advanced configuration with custom settings
    advanced_config = EmbeddingConfig(
        model_name="all-mpnet-base-v2",
        batch_size=8,
        device="auto",  # Will use GPU if available
        normalize_embeddings=True,
        pooling_strategy="mean",
        max_length=512,
        trust_remote_code=False,
        cache_folder="./model_cache",
        additional_params={
            "precision": "float32",
            "show_progress": True
        }
    )
    
    embedder = create_embedder(EmbeddingType.SENTENCE_TRANSFORMERS, advanced_config)
    
    # Long text example
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    "cognitive" functions that humans associate with the human mind, such as 
    "learning" and "problem solving".
    """
    
    result = embedder.embed(long_text.strip())
    
    print(f"Advanced configuration test:")
    print(f"  Model: {advanced_config.model_name}")
    print(f"  Device: {advanced_config.device}")
    print(f"  Max length: {advanced_config.max_length}")
    print(f"  Batch size: {advanced_config.batch_size}")
    print(f"  Normalized: {advanced_config.normalize_embeddings}")
    print(f"  Embedding shape: {result.embeddings.shape}")
    print(f"  Processing time: {result.processing_time:.3f}s")
    print()

def example_semantic_search():
    """Example of semantic search using embeddings"""
    print("=== Semantic Search Example ===")
    
    embedder = get_embedder("all-MiniLM-L6-v2")
    
    # Document corpus
    documents = [
        "Python is a high-level programming language",
        "Machine learning algorithms can predict future trends",
        "Natural language processing helps computers understand text",
        "Data science involves extracting insights from data",
        "Deep learning uses neural networks with multiple layers",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties",
        "Big data analytics processes large volumes of information"
    ]
    
    # Embed all documents
    doc_embeddings = embedder.embed(documents)
    
    # Query
    query = "What is machine learning?"
    query_embedding = embedder.embed_single(query)
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings.embeddings):
        similarity = np.dot(query_embedding, doc_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append((i, similarity, documents[i]))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Query: {query}")
    print("\nTop 3 most similar documents:")
    for i, (doc_idx, sim_score, doc_text) in enumerate(similarities[:3]):
        print(f"{i+1}. (Score: {sim_score:.4f}) {doc_text}")
    print()

def example_clustering():
    """Example of text clustering using embeddings"""
    print("=== Text Clustering Example ===")
    
    embedder = get_embedder("all-MiniLM-L6-v2")
    
    # Sample texts from different categories
    texts = [
        # Technology
        "Artificial intelligence is transforming industries",
        "Machine learning algorithms improve with more data",
        "Deep learning models require powerful GPUs",
        
        # Sports
        "The basketball team won the championship",
        "Football season starts in September",
        "Tennis players train for hours every day",
        
        # Food
        "Italian pasta is delicious with tomato sauce",
        "Sushi is a traditional Japanese dish",
        "French cuisine is known for its sophistication"
    ]
    
    # Get embeddings
    result = embedder.embed(texts)
    embeddings = result.embeddings
    
    # Simple clustering using k-means
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Perform clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Display results
        print("Text clustering results:")
        for i in range(n_clusters):
            print(f"\nCluster {i+1}:")
            cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
            for text in cluster_texts:
                print(f"  - {text}")
        
        # Calculate intra-cluster similarity
        print(f"\nCluster analysis:")
        for i in range(n_clusters):
            cluster_indices = [j for j in range(len(texts)) if clusters[j] == i]
            if len(cluster_indices) > 1:
                cluster_embeddings = embeddings[cluster_indices]
                similarities = cosine_similarity(cluster_embeddings)
                avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                print(f"Cluster {i+1} average similarity: {avg_similarity:.4f}")
        
    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        
        # Manual clustering using similarity threshold
        print("Using simple similarity-based grouping:")
        used_indices = set()
        cluster_id = 0
        
        for i, text in enumerate(texts):
            if i in used_indices:
                continue
                
            cluster_id += 1
            cluster = [text]
            used_indices.add(i)
            
            # Find similar texts
            for j, other_text in enumerate(texts):
                if j in used_indices or i == j:
                    continue
                
                similarity = embedder.similarity(text, other_text)
                if similarity > 0.7:  # Similarity threshold
                    cluster.append(other_text)
                    used_indices.add(j)
            
            print(f"\nGroup {cluster_id}:")
            for cluster_text in cluster:
                print(f"  - {cluster_text}")
    
    print()

def example_model_comparison():
    """Compare different models on the same task"""
    print("=== Model Comparison Example ===")
    
    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "The weather is sunny today",
    ]
    
    # Models to compare (using only sentence transformers for this example)
    model_configs = [
        ("MiniLM-L6", "all-MiniLM-L6-v2"),
        ("MPNet-Base", "all-mpnet-base-v2"),
        # Add more models as needed
    ]
    
    print("Comparing models on similarity task:")
    print(f"Sentence 1: {sentences[0]}")
    print(f"Sentence 2: {sentences[1]}")
    print(f"Sentence 3: {sentences[2]}")
    print()
    
    for model_name, model_path in model_configs:
        try:
            config = EmbeddingConfig(model_name=model_path)
            embedder = create_embedder(EmbeddingType.SENTENCE_TRANSFORMERS, config)
            
            # Calculate similarities
            sim_1_2 = embedder.similarity(sentences[0], sentences[1])
            sim_1_3 = embedder.similarity(sentences[0], sentences[2])
            
            print(f"{model_name}:")
            print(f"  Similarity (1 vs 2): {sim_1_2:.4f}")
            print(f"  Similarity (1 vs 3): {sim_1_3:.4f}")
            print(f"  Embedding dimension: {embedder.get_embedding_dimension()}")
            print()
            
        except Exception as e:
            print(f"{model_name}: Error - {str(e)}")
            print()

def example_popular_models_showcase():
    """Showcase all available popular models"""
    print("=== Popular Models Showcase ===")
    
    print("Available popular models:")
    for model_name, config in POPULAR_MODELS.items():
        print(f"  - {model_name}: {config.model_name}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    Normalized: {config.normalize_embeddings}")
    print()

def main():
    """Run all examples"""
    print("TextEmbedder System Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_similarity_calculation()
        example_different_models()
        example_api_based_embeddings()
        example_advanced_configuration()
        example_semantic_search()
        example_clustering()
        example_model_comparison()
        example_popular_models_showcase()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()