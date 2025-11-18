import streamlit as st
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
from io import BytesIO

st.set_page_config(
    page_title="LexiSummarize – Smart PDF Summarizer",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #2C3E50;
        padding-bottom: 5px;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #7F8C8D;
    }
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 25px;
        color: #34495E;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2c81ba;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📘 LexiSummarize</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Smart PDF Summarizer & Voice Reader</div>", unsafe_allow_html=True)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.sidebar.header("⚙ Settings")
chunk_size = st.sidebar.slider("Chunk Size", 500, 3000, 1500)
summary_length = st.sidebar.slider("Final Summary Length", 80, 300, 200)
st.sidebar.markdown("---")
st.sidebar.info("LexiSummarize uses the BART Large CNN model. No API key required.")

st.markdown("<div class='section-title'>📤 Upload PDF</div>", unsafe_allow_html=True)
pdf_file = st.file_uploader("", type=["pdf"])

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    st.info(f"PDF Loaded ✔ Total Chunks: {len(docs)}")

    chunk_summaries = []
    progress = st.progress(0)

    for i, d in enumerate(docs):
        part = summarizer(
            d.page_content,
            max_length=summary_length,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
        chunk_summaries.append(part)
        progress.progress((i + 1) / len(docs))

    combined_text = " ".join(chunk_summaries)

    final_summary = summarizer(
        combined_text,
        max_length=summary_length,
        min_length=80,
        do_sample=False
    )[0]["summary_text"]

    st.success("Summary Generated Successfully ✔")

    st.markdown("<div class='section-title'>📌 Final Summary</div>", unsafe_allow_html=True)
    st.write(final_summary)

    st.download_button(
        label="⬇ Download Summary (TXT)",
        data=final_summary.encode("utf-8"),
        file_name="LexiSummary.txt",
        mime="text/plain"
    )

    st.markdown("<div class='section-title'>🔊 Summary Voice Reader</div>", unsafe_allow_html=True)

    if st.button("🔊 Speak Summary"):
        tts = gTTS(final_summary, lang="en")
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        st.audio(audio, format="audio/mp3")
