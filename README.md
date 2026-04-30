# Fantasy Football RAG System

A multi-document Retrieval-Augmented Generation (RAG) system that lets you query fantasy football analytics reports using natural language.

Built with Python, LangChain, ChromaDB, and Llama 3 running locally via Ollama.

## What it does

- Loads multiple PDF documents from a local folder
- Splits them into chunks and converts them to vector embeddings
- Stores embeddings in a local ChromaDB vector database
- Accepts natural language questions and retrieves the most relevant context
- Uses Llama 3 (running locally) to generate answers grounded in your documents

## Tech Stack

- **LangChain** — orchestration and RAG pipeline
- **ChromaDB** — local vector database
- **Ollama + Llama 3** — local LLM, no data leaves your machine
- **HuggingFace Embeddings** — sentence-transformers/all-MiniLM-L6-v2
- **PyPDF** — PDF loading

## Setup

### 1. Install Ollama and pull Llama 3
Download Ollama from [ollama.com](https://ollama.com) then run:
