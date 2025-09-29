# rag.py
"""
A simple Retrieval-Augmented Generation (RAG) implementation for Gemini API integration.
This module provides a function to retrieve relevant context from a knowledge base and augment the prompt before sending it to Gemini.
"""


import os
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

# Example: a simple in-memory knowledge base (replace with your own data source)
KNOWLEDGE_BASE = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted programming language known for its readability and versatility."
    },
    {
        "question": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with generative models to provide more accurate and context-aware responses."
    },
    # Add more Q&A pairs as needed
]

# Initialize SentenceTransformer model
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Precompute embeddings for the knowledge base questions
KB_QUESTIONS = [item["question"] for item in KNOWLEDGE_BASE]
KB_EMBEDDINGS = embedder.encode(KB_QUESTIONS, convert_to_numpy=True)

def retrieve_context(prompt, top_k=1):
    """
    Retrieve the most semantically similar context(s) from the knowledge base for the given prompt using embeddings.
    """
    prompt_emb = embedder.encode([prompt], convert_to_numpy=True)
    # Compute cosine similarity
    similarities = np.dot(KB_EMBEDDINGS, prompt_emb.T).squeeze() / (
        np.linalg.norm(KB_EMBEDDINGS, axis=1) * np.linalg.norm(prompt_emb)
    )
    # Get top_k most similar
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [KNOWLEDGE_BASE[i]["answer"] for i in top_indices if similarities[i] > 0.5]  # threshold for relevance

def rag_generate(prompt, model_name="gemini-2.5-flash"):
    """
    Augment the prompt with retrieved context and generate a response using Gemini.
    """
    context = retrieve_context(prompt)
    if context:
        augmented_prompt = f"Context: {context}\n\nUser: {prompt}"
    else:
        augmented_prompt = prompt
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(augmented_prompt)
    return {
        "augmented_prompt": augmented_prompt,
        "response": response.text
    }
