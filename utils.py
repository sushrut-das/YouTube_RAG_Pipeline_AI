
import nltk
nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from openai import OpenAI
import psycopg2

client = OpenAI()

DB_CONFIG = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "your_host",
    "port": "5432",
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_text(transcript):
    sentences = []
    for entry in transcript:
        for s in sent_tokenize(entry["text"]):
            sentences.append({"sentence": s, "timestamp": entry["start"]})

    chunks = []
    current_chunk = []
    current_tokens = 0

    for item in sentences:
        tok = count_tokens(item["sentence"])
        if current_tokens + tok > 512:
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

def store_embeddings(chunks):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

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

def answer_query(query):
    results = retrieve(query)

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

def ingest_data(url=None, file=None):
    # Simplified placeholder
    transcript = [{"text": "Sample transcript", "start": 0}]
    chunks = chunk_text(transcript)
    store_embeddings(chunks)
