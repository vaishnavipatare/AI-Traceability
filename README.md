# 📑 AI Traceability Tool

This project is an **AI-powered document traceability system** that allows you to extract structured information from PDFs, store them in a searchable FAISS vector database, and query them with natural language. The tool uses **Sentence Transformers** for embeddings, **FAISS** for vector search, and a **local LLM (via llama.cpp)** for summarization.

---

## ✨ Features
- 📄 **PDF Parsing** – Extracts sections & subsections from technical/academic PDFs using PyMuPDF.  
- 🧩 **Chunking** – Breaks down documents into meaningful chunks (sections/subsections).  
- 🔍 **Vector Search with FAISS** – Finds the most relevant passages for a given query.  
- 🤖 **LLM Summarization** – Uses a local LLM (`llama.cpp`) to summarize retrieved passages.  
- 📂 **Multi-PDF Support** – Works on a folder of PDFs, not just single files.  
- ⚡ **Efficient Retrieval** – Embeddings generated using `all-MiniLM-L6-v2` from Sentence Transformers.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) – PDF parsing  
- [FAISS](https://github.c)
