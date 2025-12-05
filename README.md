# 📚 Mini RAG-based AI Knowledge Assistant

A simple Retrieval-Augmented Generation (RAG) app where you can upload PDFs and ask questions based on their content.  
The system uses:

- **SentenceTransformers** to create text embeddings
- **ChromaDB** as a vector database
- **Groq Llama 3.1** models for answer generation
- **Streamlit** for the web UI

---

## ✨ Features

- Upload one or more PDF files
- Index documents into a vector database (Chroma)
- Ask natural language questions about the PDFs
- Retrieval-Augmented Generation (RAG) pipeline
- Shows **retrieved context chunks** used by the model
- Conversation-style **chat history**
- Button to **clear** the vector database

---

## 🧱 Project Structure

```text
mini_rag_assistant/
│
├── app.py          # Streamlit UI
├── rag_core.py     # Core RAG logic (PDF → chunks → embeddings → retrieval → LLM)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── docs/       # PDFs are stored here (not committed to Git)
└── chroma_store/   # Local ChromaDB files (ignored in Git)

Tech Stack: 
Python 3.10+
Streamlit
SentenceTransformers
ChromaDB
Groq LLM API
Llama 3.1 8B Instant (llama-3.1-8b-instant)
