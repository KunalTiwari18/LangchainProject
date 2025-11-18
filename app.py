import streamlit as st
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
from io import BytesIO
import base64

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Pro PDF Summarizer",
    page_icon="📘",
    layout="wide"
)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
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

st.markdown("<div class='main-title'>📘 Professional PDF Summarizer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Upload a PDF → Summarize → Download → Listen</div>", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙ Settings")
chunk_size = st.sidebar.slider("Text Chunk Size", 500, 3000, 1500)
summary_length = st.sidebar.slider("Summary Length (Max Tokens)", 50, 300, 180)

st.sidebar.markdown("---")
st.sidebar.info("This app uses *BART Large CNN* model (open-source, no API keys 🟢).")


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()


# ---------------------------
# FILE UPLOAD
# ---------------------------
st.markdown("<div class='section-title'>📤 Upload Your PDF</div>", unsafe_allow_html=True)
pdf_file = st.file_uploader("", type=["pdf"])

# ---------------------------
# PROCESS PDF
# ---------------------------
if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    full_text = " ".join([d.page_content for d in docs])

    st.markdown("<div class='section-title'>📝 Generating Summary...</div>", unsafe_allow_html=True)
    progress = st.progress(0)

    # Update progress
    progress.progress(30)

    summary = summarizer(
        full_text,
        max_length=summary_length,
        min_length=60,
        do_sample=False
    )[0]["summary_text"]

    progress.progress(100)
    st.success("Summary generated successfully!")

    # ---------------------------
    # SHOW SUMMARY
    # ---------------------------
    st.markdown("<div class='section-title'>📌 Summary</div>", unsafe_allow_html=True)
    st.write(summary)

    # ---------------------------
    # DOWNLOAD SUMMARY
    # ---------------------------
    summary_bytes = summary.encode("utf-8")

    st.download_button(
        label="⬇ Download Summary as TXT",
        data=summary_bytes,
        file_name="summary.txt",
        mime="text/plain",
    )

    # ---------------------------
    # TEXT TO SPEECH
    # ---------------------------
    st.markdown("<div class='section-title'>🔊 Listen to Summary</div>", unsafe_allow_html=True)

    if st.button("🔊 Generate Voice"):
        tts = gTTS(summary, lang="en")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        st.audio(audio_bytes, format="audio/mp3")