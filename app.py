import streamlit as st
from utils import ingest_data, answer_query

st.title("RAG App")

url = st.text_input("YouTube URL")
file = st.file_uploader("Upload TXT")

if st.button("Ingest"):
    ingest_data(url=url, file=file)
    st.success("Ingested!")

query = st.text_input("Ask question")

if st.button("Ask"):
    st.write(answer_query(query))
