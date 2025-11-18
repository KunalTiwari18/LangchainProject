import streamlit as st
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="LexiSummarize – Smart PDF Summarizer", page_icon="📘", layout="wide")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def recursive_summarize(text, max_len):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=max_len, min_length=60, do_sample=False)
        summaries.append(result[0]["summary_text"])
    combined = " ".join(summaries)
    if len(combined) > 2500:
        return recursive_summarize(combined, max_len)
    return combined

st.sidebar.header("⚙ Settings")
max_len = st.sidebar.slider("Summary Length", 80, 300, 200)
st.sidebar.markdown("---")
st.sidebar.info("LexiSummarize supports large PDFs using recursive summarization.")

st.markdown("<h1 style='text-align:center;color:#2C3E50;'>📘 LexiSummarize</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#7F8C8D;'>Smart PDF Summarizer & Voice Reader</p>", unsafe_allow_html=True)

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    st.info(f"PDF Loaded ✔ Total Chunks: {len(chunks)}")

    progress = st.progress(0)
    partial_summaries = []
    for i, d in enumerate(chunks):
        out = summarizer(d.page_content, max_length=max_len, min_length=60, do_sample=False)
        partial_summaries.append(out[0]["summary_text"])
        progress.progress((i + 1) / len(chunks))

    combined = " ".join(partial_summaries)

    final_summary = recursive_summarize(combined, max_len)

    st.success("Summary Generated Successfully ✔")
    st.subheader("📌 Final Summary")
    st.write(final_summary)

    st.download_button("⬇ Download Summary (TXT)", final_summary.encode(), "LexiSummary.txt", "text/plain")

    st.subheader("🔊 Summary Voice Reader")
    if st.button("🔊 Speak Summary"):
        tts = gTTS(final_summary, lang="en")
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        st.audio(audio, format="audio/mp3")