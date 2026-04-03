import os
import re
import streamlit as st
import nltk

# ---------------------------
# 🔧 Robust NLTK Setup
# ---------------------------
def setup_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=nltk_data_dir)

@st.cache_resource
def init_nltk():
    setup_nltk()

init_nltk()

from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from openai import OpenAI
import psycopg2

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DB_CONFIG = {
    "dbname": st.secrets["database"]["dbname"],
    "user": st.secrets["database"]["user"],
    "password": st.secrets["database"]["password"],
    "host": st.secrets["database"]["host"],
    "port": st.secrets["database"]["port"],
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

# ---------------------------
# 📥 Parsers
# ---------------------------
def parse_txt(content: str):
    return [{"text": content.strip(), "start": 0}]

# ---------------------------
# ✂️ Chunking
# ---------------------------
def chunk_text(transcript):
    sentences = []
    for entry in transcript:
        for s in sent_tokenize(entry["text"]):
            s = s.strip()
            if s:
                sentences.append({"sentence": s, "timestamp": entry["start"]})

    chunks = []
    current_chunk = []
    current_tokens = 0

    for item in sentences:
        tok = count_tokens(item["sentence"])

        if current_tokens + tok > 512:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(item)
        current_tokens += tok

    if current_chunk:
        chunks.append(current_chunk)

    return [
        {
            "content": " ".join(c["sentence"] for c in chunk),
            "start_time": chunk[0]["timestamp"],
        }
        for chunk in chunks
    ]

# ---------------------------
# 🗄️ Store embeddings
# ---------------------------
def store_embeddings(chunks):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS youtube_rag (
            id SERIAL PRIMARY KEY,
            content TEXT,
            start_time FLOAT,
            embedding VECTOR(1536)
        );
    """)

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["content"]
        ).data[0].embedding

        cur.execute(
            "INSERT INTO youtube_rag (content, start_time, embedding) VALUES (%s, %s, %s)",
            (chunk["content"], chunk["start_time"], embedding)
        )

    conn.commit()
    cur.close()
    conn.close()

# ---------------------------
# 🔍 Retrieval
# ---------------------------
def retrieve(query, top_k=5):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        "SELECT content, start_time FROM youtube_rag ORDER BY embedding <-> %s LIMIT %s",
        (query_embedding, top_k)
    )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ---------------------------
# 🤖 Answer
# ---------------------------
def answer_query(query):
    results = retrieve(query)

    if not results:
        return "No relevant context found."

    context = ""
    for content, start_time in results:
        context += f"[Time: {start_time}]\n{content}\n\n"

    prompt = f"""
Answer ONLY using the context below.
If not found, say: "This question is not covered in the provided content."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

# ---------------------------
# 🚀 Ingest
# ---------------------------
def ingest_data(url=None, file=None):
    transcript = []

    if file is not None:
        content = file.read().decode("utf-8", errors="ignore")
        transcript = parse_txt(content)

    elif url:
        from youtube_transcript_api import YouTubeTranscriptApi

        def extract_video_id(url):
            if "v=" in url:
                return url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[-1].split("?")[0]
            return None

        vid = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(vid)

    chunks = chunk_text(transcript)
    store_embeddings(chunks)