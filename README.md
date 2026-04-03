# 🎬 YouTube / File RAG Streamlit App

## Setup

1. Add your secrets:
   - `.streamlit/secrets.toml`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run app:
   ```bash
   streamlit run app.py
   ```

## Features
- YouTube transcript ingestion
- File ingestion (TXT)
- Sentence chunking with NLTK
- BERT token-based chunking (≤512)
- OpenAI embeddings
- Supabase / PostgreSQL (pgvector)
- Context-aware Q&A