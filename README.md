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
Download Ollama from ollama.com then run:
ollama pull llama3
### 2. Install dependencies
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama pypdf sentence-transformers
### 3. Add your PDFs
Place your PDF files in a folder called data/ in the project root.
### 4. Run
python rag_system.py
The first run builds the vector database from your PDFs. Subsequent runs load the saved database instantly.
## Example Questions
- "Who are the best value running backs this season?"
- "What does the data say about wide receivers in PPR leagues?"
- "Which quarterbacks have the most consistent weekly scoring?"
- "Summarize the injury risk analysis for top tight ends"
## Notes
- The data/ and chroma_db/ folders are excluded from this repo
- All processing happens locally — no API keys required