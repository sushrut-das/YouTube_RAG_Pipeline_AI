import os, nltk, psycopg2
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from openai import OpenAI

load_dotenv()

# NLTK setup
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_text(text):
    sentences = sent_tokenize(text)
    chunks, cur, tokens = [], [], 0

    for s in sentences:
        t = count_tokens(s)
        if tokens + t > 512:
            chunks.append(" ".join(cur))
            cur, tokens = [], 0
        cur.append(s)
        tokens += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def store(chunks):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE TABLE IF NOT EXISTS rag (content TEXT, embedding VECTOR(1536));")

    for c in chunks:
        emb = client.embeddings.create(model="text-embedding-3-small", input=c).data[0].embedding
        cur.execute("INSERT INTO rag VALUES (%s,%s)", (c, emb))
    conn.commit()
    conn.close()

def ingest_data(url=None, file=None):
    text = ""
    if file:
        text = file.read().decode()
    elif url:
        from youtube_transcript_api import YouTubeTranscriptApi
        vid = url.split("v=")[-1]
        data = YouTubeTranscriptApi.get_transcript(vid)
        text = " ".join([d["text"] for d in data])

    chunks = chunk_text(text)
    store(chunks)

def retrieve(q):
    emb = client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT content FROM rag ORDER BY embedding <-> %s LIMIT 3", (emb,))
    res = cur.fetchall()
    conn.close()
    return " ".join([r[0] for r in res])

def answer_query(q):
    ctx = retrieve(q)
    prompt = f"Answer from context only:\n{ctx}\nQ:{q}"
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content
