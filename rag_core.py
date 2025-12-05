import os
from typing import List

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
logging.basicConfig(level=logging.INFO)


# Load embedding model once
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Create Chroma client and collection
from chromadb import PersistentClient

# Create Chroma client (new API)
client = PersistentClient(path="chroma_store")

# Create or load the collection
collection = client.get_or_create_collection(name="pdf_knowledge_base")


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)

def split_text_into_chunks(text: str,
                           chunk_size: int = 500,
                           overlap: int = 50) -> List[str]:
    """
    chunk_size: target characters per chunk
    overlap: how many characters overlap between chunks to keep context
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # go back a bit for overlap

    return chunks

def add_pdf_to_vector_db(file_path: str, source_name: str):
    text = extract_text_from_pdf(file_path)
    chunks = split_text_into_chunks(text)

    embeddings = embedder.encode(chunks).tolist()  # list of vectors

    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def retrieve_relevant_chunks(query: str, top_k: int = 4) -> List[str]:
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # results["documents"] is a list of lists
    return results["documents"][0]

import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # Correct name

def call_llm(prompt: str) -> str:
    
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that answers based only on the given context."},
        {"role": "user", "content": prompt}
    ]
}

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()

    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"]
    else:
        return f"LLM API Error: {response_json}"

    

def answer_question(query: str):
    """Return both the answer and the retrieved chunks."""
    chunks = retrieve_relevant_chunks(query, top_k=4)

    context = "\n\n".join(chunks)

    prompt = f"""
You are an AI Knowledge Assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}
Answer:
"""

    answer = call_llm(prompt)
    return answer, chunks

