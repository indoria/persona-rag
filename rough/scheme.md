# AI Journalist Personas Implementation Guide

## Overview
This guide creates AI personas of Barkha Dutt and Palki Sharma Upadhyay using RAG (Retrieval Augmented Generation) with OpenAI GPT-4, LangChain, Qdrant vector database, and Flask.

## Architecture Components
- **OpenAI GPT-4**: Language model for generating responses
- **text-embedding-3-large**: Creating embeddings for documents
- **Qdrant**: Vector database for storing and retrieving embeddings
- **LangChain**: Framework for chaining LLM operations
- **Flask**: Web API framework

## Prerequisites

### Required Packages
```bash
pip install openai langchain qdrant-client flask python-dotenv requests beautifulsoup4 newspaper3k
```

### Environment Setup
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key  # if using cloud
```

## Step 1: Data Collection and Preparation

### 1.1 Create Data Collection Script
```python
# data_collector.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import json
import time

class JournalistDataCollector:
    def __init__(self):
        self.barkha_sources = [
            "interviews", "articles", "TV shows", "books", "social media"
        ]
        self.palki_sources = [
            "Gravitas episodes", "WION articles", "interviews", "reports"
        ]
    
    def collect_articles(self, journalist_name, sources):
        """Collect articles and transcripts for each journalist"""
        data = []
        # Implementation would involve web scraping from legitimate sources
        # Note: Ensure compliance with robots.txt and terms of service
        return data
    
    def save_data(self, data, filename):
        with open(f"data/{filename}.json", "w") as f:
            json.dump(data, f, indent=2)
```

### 1.2 Create Persona Profiles
```python
# persona_profiles.py
BARKHA_DUTT_PROFILE = {
    "name": "Barkha Dutt",
    "background": "Senior journalist and author, former NDTV anchor",
    "expertise": ["Politics", "Social issues", "International affairs", "Gender issues"],
    "style": "Direct, investigative, empathetic, questioning authority",
    "notable_works": ["This Unquiet Land", "Kafkaesque coverage"],
    "interviewing_style": "Probing, persistent, emotional intelligence",
    "political_stance": "Liberal, progressive, questions power structures"
}

PALKI_SHARMA_PROFILE = {
    "name": "Palki Sharma Upadhyay", 
    "background": "Anchor and Executive Editor at WION",
    "expertise": ["International news", "Geopolitics", "Global affairs", "Indian foreign policy"],
    "style": "Analytical, straightforward, fact-based, confident",
    "notable_works": ["Gravitas show", "International reporting"],
    "interviewing_style": "Direct questions, fact-checking, global perspective",
    "political_stance": "Nationalist perspective, pro-India stance in global context"
}
```

## Step 2: Set Up Qdrant Vector Database

### 2.1 Install and Run Qdrant
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```

### 2.2 Initialize Qdrant Collections
```python
# vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
import numpy as np

class VectorStore:
    def __init__(self, url="http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_names = ["barkha_dutt", "palki_sharma"]
        
    def create_collections(self):
        for collection_name in self.collection_names:
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=3072,  # text-embedding-3-large dimension
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created collection: {collection_name}")
            except Exception as e:
                print(f"Collection {collection_name} might already exist: {e}")
    
    def get_embedding(self, text):
        """Generate embedding using OpenAI text-embedding-3-large"""
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def add_documents(self, collection_name, documents):
        """Add documents with embeddings to collection"""
        points = []
        for i, doc in enumerate(documents):
            embedding = self.get_embedding(doc["content"])
            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "date": doc.get("date", "")
                }
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Added {len(points)} documents to {collection_name}")
    
    def search_similar(self, collection_name, query, limit=5):
        """Search for similar documents"""
        query_embedding = self.get_embedding(query)
        
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {
                "content": hit.payload["content"],
                "score": hit.score,
                "source": hit.payload.get("source", ""),
                "title": hit.payload.get("title", "")
            }
            for hit in search_result
        ]
```

## Step 3: Create LangChain Integration

### 3.1 Custom Retriever
```python
# retriever.py
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List

class QdrantRetriever(BaseRetriever):
    def __init__(self, vector_store, collection_name, k=5):
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.vector_store.search_similar(
            self.collection_name, 
            query, 
            limit=self.k
        )
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    "source": result["source"],
                    "title": result["title"],
                    "score": result["score"]
                }
            )
            documents.append(doc)
        
        return documents
```

### 3.2 Persona Chain
```python
# persona_chain.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class JournalistPersonaChain:
    def __init__(self, retriever, persona_profile):
        self.retriever = retriever
        self.persona_profile = persona_profile
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.chain = self._create_chain()
    
    def _create_chain(self):
        system_template = """You are {name}, a renowned journalist with the following profile:
        
Background: {background}
Expertise: {expertise}
Style: {style}
Notable Works: {notable_works}
Interviewing Style: {interviewing_style}
Perspective: {political_stance}

Based on the following context from your past work and interviews, respond as {name} would:

Context:
{context}

Guidelines:
1. Maintain the authentic voice and perspective of {name}
2. Reference your actual experiences and viewpoints when relevant
3. Stay true to your journalistic style and approach
4. Use your expertise areas to provide informed responses
5. Maintain your characteristic interviewing and communication style

Question: {question}

Respond as {name} would, drawing from your experience and the provided context."""

        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template("{question}")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "name": lambda _: self.persona_profile["name"],
                "background": lambda _: self.persona_profile["background"],
                "expertise": lambda _: ", ".join(self.persona_profile["expertise"]),
                "style": lambda _: self.persona_profile["style"],
                "notable_works": lambda _: ", ".join(self.persona_profile["notable_works"]),
                "interviewing_style": lambda _: self.persona_profile["interviewing_style"],
                "political_stance": lambda _: self.persona_profile["political_stance"]
            }
            | chat_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question):
        return self.chain.invoke(question)
```

## Step 4: Flask API Implementation

### 4.1 Main Application
```python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

from vector_store import VectorStore
from retriever import QdrantRetriever
from persona_chain import JournalistPersonaChain
from persona_profiles import BARKHA_DUTT_PROFILE, PALKI_SHARMA_PROFILE

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize components
vector_store = VectorStore()
vector_store.create_collections()

# Create retrievers and chains
barkha_retriever = QdrantRetriever(vector_store, "barkha_dutt", k=5)
palki_retriever = QdrantRetriever(vector_store, "palki_sharma", k=5)

barkha_chain = JournalistPersonaChain(barkha_retriever, BARKHA_DUTT_PROFILE)
palki_chain = JournalistPersonaChain(palki_retriever, PALKI_SHARMA_PROFILE)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        journalist = data.get('journalist', 'barkha').lower()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if journalist == 'barkha':
            response = barkha_chain.ask(question)
        elif journalist == 'palki':
            response = palki_chain.ask(question)
        else:
            return jsonify({"error": "Invalid journalist. Choose 'barkha' or 'palki'"}), 400
        
        return jsonify({
            "journalist": journalist,
            "question": question,
            "response": response
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/journalists', methods=['GET'])
def get_journalists():
    return jsonify({
        "journalists": [
            {
                "id": "barkha",
                "name": "Barkha Dutt",
                "profile": BARKHA_DUTT_PROFILE
            },
            {
                "id": "palki", 
                "name": "Palki Sharma Upadhyay",
                "profile": PALKI_SHARMA_PROFILE
            }
        ]
    })

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Endpoint to upload training data for personas"""
    try:
        data = request.json
        journalist = data.get('journalist')
        documents = data.get('documents', [])
        
        if journalist not in ['barkha_dutt', 'palki_sharma']:
            return jsonify({"error": "Invalid journalist"}), 400
        
        vector_store.add_documents(journalist, documents)
        
        return jsonify({
            "message": f"Successfully uploaded {len(documents)} documents for {journalist}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### 4.2 Data Loading Script
```python
# load_data.py
import json
from vector_store import VectorStore

def load_sample_data():
    """Load sample data for both journalists"""
    vector_store = VectorStore()
    
    # Sample data for Barkha Dutt
    barkha_docs = [
        {
            "content": "As a journalist, I believe in questioning power and holding those in authority accountable...",
            "source": "Interview",
            "title": "On Journalism Ethics",
            "date": "2023"
        },
        # Add more documents...
    ]
    
    # Sample data for Palki Sharma
    palki_docs = [
        {
            "content": "India's foreign policy has evolved significantly in recent years, with a focus on strategic autonomy...",
            "source": "Gravitas",  
            "title": "India's Global Position",
            "date": "2023"
        },
        # Add more documents...
    ]
    
    vector_store.add_documents("barkha_dutt", barkha_docs)
    vector_store.add_documents("palki_sharma", palki_docs)
    
    print("Sample data loaded successfully!")

if __name__ == "__main__":
    load_sample_data()
```

## Step 5: Testing and Deployment

### 5.1 Test Script
```python
# test_personas.py
import requests

def test_api():
    base_url = "http://localhost:5000/api"
    
    # Test Barkha Dutt persona
    barkha_question = {
        "journalist": "barkha",
        "question": "What are your thoughts on media accountability in India?"
    }
    
    response = requests.post(f"{base_url}/chat", json=barkha_question)
    print("Barkha Response:", response.json())
    
    # Test Palki Sharma persona
    palki_question = {
        "journalist": "palki", 
        "question": "How do you view India's position in global geopolitics?"
    }
    
    response = requests.post(f"{base_url}/chat", json=palki_question)
    print("Palki Response:", response.json())

if __name__ == "__main__":
    test_api()
```

### 5.2 Frontend Integration (Optional)
```html
<!-- simple_frontend.html -->
<!DOCTYPE html>
<html>
<head>
    <title>AI Journalist Personas</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chat-container { max-width: 800px; margin: 0 auto; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; }
        .ai { background-color: #f5f5f5; }
        select, input, button { padding: 10px; margin: 5px; }
        #question { width: 70%; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>AI Journalist Personas</h1>
        
        <div>
            <select id="journalist">
                <option value="barkha">Barkha Dutt</option>
                <option value="palki">Palki Sharma Upadhyay</option>
            </select>
            <input type="text" id="question" placeholder="Ask a question...">
            <button onclick="askQuestion()">Ask</button>
        </div>
        
        <div id="chat-messages"></div>
    </div>

    <script>
        async function askQuestion() {
            const journalist = document.getElementById('journalist').value;
            const question = document.getElementById('question').value;
            
            if (!question.trim()) return;
            
            // Add user message
            addMessage(question, 'user');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({journalist, question})
                });
                
                const data = await response.json();
                addMessage(data.response, 'ai', journalist);
                
            } catch (error) {
                addMessage('Error: ' + error.message, 'ai');
            }
            
            document.getElementById('question').value = '';
        }
        
        function addMessage(text, type, journalist = '') {
            const messages = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = `<strong>${journalist || 'You'}:</strong> ${text}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
```

## Step 6: Production Considerations

### 6.1 Environment Variables
```bash
# .env.production
OPENAI_API_KEY=your_production_key
QDRANT_URL=your_production_qdrant_url
QDRANT_API_KEY=your_production_qdrant_key
FLASK_ENV=production
```

### 6.2 Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### 6.3 Rate Limiting and Security
```python
# Add to app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    # existing code...
```

## Data Collection Guidelines

1. **Ethical Considerations**: Ensure all data collection respects copyright and fair use
2. **Source Verification**: Use only publicly available, legitimate sources
3. **Data Quality**: Focus on high-quality, representative content
4. **Regular Updates**: Periodically update the knowledge base with new content
5. **Bias Awareness**: Be mindful of potential biases in training data

## Usage Examples

```python
# Example usage
question_for_barkha = "What's your take on the role of women in Indian journalism?"
question_for_palki = "How should India respond to changing global alliances?"

# The system will retrieve relevant context and generate responses 
# that maintain each journalist's authentic voice and perspective
```

This implementation creates sophisticated AI personas that can engage in conversations while maintaining the distinctive voices, perspectives, and expertise of both journalists.