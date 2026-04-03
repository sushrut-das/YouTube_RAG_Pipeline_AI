import streamlit as st
from utils import ingest_data, answer_query

st.set_page_config(page_title="YouTube RAG App", layout="wide")

st.title("🎬 YouTube / File RAG System")

st.sidebar.header("Ingest Data")

option = st.sidebar.radio("Choose Input Type", ["YouTube URL", "Upload File"])

if option == "YouTube URL":
    url = st.sidebar.text_input("Enter YouTube URL")
    if st.sidebar.button("Ingest"):
        if not url:
            st.sidebar.error("Please enter a URL")
        else:
            with st.spinner("Ingesting from YouTube..."):
                ingest_data(url=url)
            st.sidebar.success("Data ingested successfully!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload SRT / VTT / TXT / PDF")
    if st.sidebar.button("Ingest"):
        if not uploaded_file:
            st.sidebar.error("Please upload a file")
        else:
            with st.spinner("Ingesting file..."):
                ingest_data(file=uploaded_file)
            st.sidebar.success("Data ingested successfully!")

st.header("💬 Ask Questions")

query = st.text_input("Enter your question")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response = answer_query(query)
    st.write(response)