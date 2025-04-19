# 📄 Document-Based Question Answering System

This project implements a document-based Question Answering (QA) system using LLMs (Large Language Models) and vector similarity search. It allows uploading documents (PDF, DOCX, TXT), indexing them, and querying their content via a REST API.

---

Features

- Upload and embed documents (PDF, DOCX, TXT)
- Query document contents using natural language
- Citation-based responses (page/document references)
- Modular support for LLMs (OpenAI, Gemini, Ollama, Gorilla)
- Uses FAISS for fast vector similarity search
- Works with  HuggingFace embeddings

---

Setup Instructions

1. Clone the Repository
2. pip install -r requirements.txt
3. OPENAI_API_KEY=your_openai_api_key
4. python app.py



