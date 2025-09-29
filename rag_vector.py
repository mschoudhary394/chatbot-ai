# rag_vector.py
"""
RAG implementation using a vector database (FAISS) and open-source embedding model (e.g., SentenceTransformers).
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize embedding model (you can change the model name)
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Example knowledge base
DOCUMENTS = [
    "Python is a high-level, interpreted programming language.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Flask is a lightweight WSGI web application framework in Python.",
    # Add more documents as needed
]

# Generate embeddings for the knowledge base
embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_similar_context(query, top_k=1):
    """Retrieve the most similar document(s) from the knowledge base for the given query."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    return [DOCUMENTS[i] for i in I[0]]

# Example usage for RAG
if __name__ == "__main__":
    user_query = "What is RAG?"
    context = retrieve_similar_context(user_query, top_k=2)
    print("Retrieved context:", context)
    # You can now augment the prompt with this context and send it to Gemini or any LLM
