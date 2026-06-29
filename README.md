# YT-RAG-project.
import streamlit as st
import os
import re
import tempfile
import json
import zipfile
import shutil
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="YT-RAG: Multi-Video Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ SESSION STATE INIT ============
#Session State: Remembers values even when the script reruns.
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'video_metadata_list' not in st.session_state:
    st.session_state.video_metadata_list = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

# ============ HELPER FUNCTIONS ============
def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_metadata(video_url):
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'channel': info.get('uploader', 'Unknown'),
                'duration': info.get('duration_string', 'Unknown'),
                'url': video_url,
                'thumbnail': info.get('thumbnail', ''),
            }
    except:
        return {'title': 'Could not fetch', 'channel': 'Unknown', 'duration': 'Unknown', 'url': video_url, 'thumbnail': ''}

def fetch_transcript(video_id):
    for languages in [['en'], ['en', 'hi', 'es', 'fr', 'de']]:
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
            return " ".join([s.text for s in fetched]), len(fetched)
        except:
            continue
    raise Exception("No transcript available")

def preprocess_transcript(raw_text):
    text = re.sub(r'\[.*?\]', '', raw_text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def chunk_text(clean_text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = [Document(page_content=clean_text, metadata={"source": "transcript"})]
    chunks = text_splitter.split_documents(documents)
    return chunks

@st.cache_resource
def load_embedding_model():
    with st.spinner("🔄 Loading AI model..."):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=2048
    )

def process_multiple_urls(urls, progress_bar, status_text):
    all_chunks = []
    video_metadata = []
    embedding_model = load_embedding_model()
    
    for idx, url in enumerate(urls):
        if not url.strip():
            continue
        progress_bar.progress(idx / len(urls), f"Processing {idx+1}/{len(urls)}...")
        status_text.text(f"📹 Processing: {url[:50]}...")
        
        video_id = extract_video_id(url)
        if not video_id:
            st.error(f"❌ Invalid URL: {url}")
            continue
        
        metadata = get_video_metadata(url)
        metadata['video_id'] = video_id
        video_metadata.append(metadata)
        
        try:
            raw_transcript, _ = fetch_transcript(video_id)
            clean_text = preprocess_transcript(raw_transcript)
            chunks = chunk_text(clean_text)
            for chunk in chunks:
                chunk.metadata['video_title'] = metadata['title']
                chunk.metadata['channel'] = metadata['channel']
                chunk.metadata['video_url'] = url
            all_chunks.extend(chunks)
            st.success(f"✅ {metadata['title'][:50]}... ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"❌ Failed: {metadata['title'][:30]} - {str(e)[:80]}")
    
    progress_bar.progress(1.0, "Building vector database...")
    status_text.text("🔨 Building FAISS vector database...")
    
    if all_chunks:
        return FAISS.from_documents(all_chunks, embedding_model), video_metadata, all_chunks
    return None, video_metadata, []

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # Concise prompt (from app2 - faster, less tokens)
    prompt_template = """You are an expert video content analyst. Your job is to answer questions about a YouTube video
based ONLY on the transcript excerpts provided below.

IMPORTANT INSTRUCTIONS:
- Answer using ONLY the information from the transcript context below.
- If the answer is not found in the context, clearly say: "I couldn't find that information in the video transcript."
- Be concise, clear, and accurate.
- If asked to summarize, provide a structured summary with key points.
- Do NOT make up or infer information beyond what is explicitly in the transcript.

--- TRANSCRIPT CONTEXT ---
{context}
--------------------------

QUESTION: {question}

RULES: Answer in ENGLISH (HINDI if requested). 2-3 sentences max. bullet points if needed. If not in context, say "Not available in videos."

ANSWER:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

# ============ CUSTOM CSS (from app1 - professional look) ============
st.markdown("""
<style>
    /* Professional styling from app1 */
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .video-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    /* Success message styling */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============ TITLE SECTION (Centered, from app1) ============
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 style="text-align: center; color: #FF4B4B; white-space: nowrap;">🎓 YT-RAG Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Ask questions across multiple YouTube videos simultaneously</p>', unsafe_allow_html=True)

# ============ SIDEBAR (Improved from both) ============
with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=60)
    st.markdown("## 📋 Instructions")
    st.markdown("""
    1. Enter YouTube URLs (one per line)
    2. Click **Process Videos**
    3. Ask questions about ANY of the videos
    4. Get AI-powered answers with sources
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    - ✅ Multi-video support
    - ✅ Semantic search
    - ✅ Source attribution
    - ✅ Chat history
    - ✅ Export/Import Knowledge Base
    """)
    st.markdown("---")
    st.markdown("### 💾 Knowledge Base")
    
    # Import Knowledge Base (Improved - shows progress)
    uploaded_kb = st.file_uploader("📥 Import KB (.zip)", type=['zip'], help="Upload a previously exported knowledge base")
    if uploaded_kb is not None:
        with st.spinner("⏳ Importing knowledge base... Please wait..."):
            progress_msg = st.empty()
            
            progress_msg.info("📦 Extracting files...")
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_kb, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            progress_msg.info("🔢 Loading vector database...")
            st.session_state.vectorstore = FAISS.load_local(
                extract_dir, 
                load_embedding_model(), 
                allow_dangerous_deserialization=True
            )
            
            progress_msg.info("📋 Loading video metadata...")
            with open(os.path.join(extract_dir, "metadata.json"), "r") as f:
                st.session_state.video_metadata_list = json.load(f)
            
            progress_msg.info("🔧 Building RAG chain...")
            st.session_state.rag_chain = build_rag_chain(st.session_state.vectorstore, load_llm())
            
            shutil.rmtree(extract_dir)
            progress_msg.empty()
            st.success(f"✅ Imported {len(st.session_state.video_metadata_list)} videos!")
    
    st.markdown("---")
    
    # System Status (from app1 - better visual)
    if st.session_state.vectorstore:
        st.success(f"✅ Active: {len(st.session_state.video_metadata_list)} videos")
    else:
        st.warning("⚠️ No knowledge base loaded")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ============ MAIN CONTENT ============
col_left, col_right = st.columns(2)

# ============ LEFT COLUMN - Video Input (Styled from app1) ============
with col_left:
    st.markdown("### 📹 1. Add YouTube Videos")
    
    if 'urls_text' not in st.session_state:
        st.session_state.urls_text = ""
    
    urls_text = st.text_area(
        "Paste YouTube URLs (one per line):",
        value=st.session_state.urls_text,
        height=120,
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID_1\nhttps://youtu.be/VIDEO_ID_2\nhttps://www.youtube.com/watch?v=VIDEO_ID_3"
    )
    st.session_state.urls_text = urls_text
    
    # Button row
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📋 Load Examples", use_container_width=True):
            st.session_state.urls_text = """https://www.youtube.com/watch?v=ie4oGI85SAE&t=25s
https://www.youtube.com/watch?v=aircAruvnKk
https://www.youtube.com/watch?v=aGu0fbkHhek"""
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear URLs", use_container_width=True):
            st.session_state.urls_text = ""
            st.rerun()
    
    # Process button with full width (from app1)
    if st.button("🚀 PROCESS VIDEOS", type="primary", use_container_width=True):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        if urls:
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner(f"Processing {len(urls)} videos..."):
                vectorstore, metadata, chunks = process_multiple_urls(urls, progress_bar, status_text)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.video_metadata_list = metadata
                    st.session_state.rag_chain = build_rag_chain(vectorstore, load_llm())
                    st.success(f"✅ Processed {len(metadata)} videos, {len(chunks)} chunks!")
                    st.balloons()
            progress_bar.empty()
            status_text.empty()
    
    # Display processed videos with styled cards (from app1)
    if st.session_state.video_metadata_list:
        st.markdown(f"### ✅ Processed Videos ({len(st.session_state.video_metadata_list)})")
        for vid in st.session_state.video_metadata_list:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 10px; border-radius: 5px; margin: 5px 0; border: 1px solid #555;">
                🎥 <b style="color: #FFFFFF;">{vid['title'][:60]}</b><br>
                📺 Channel: <span style="color: #CCCCCC;">{vid['channel']}</span> | ⏱️ Duration: <span style="color: #CCCCCC;">{vid['duration']}</span>
            </div>
            """, unsafe_allow_html=True)

# ============ RIGHT COLUMN - Q&A ============
with col_right:
    st.markdown("### 💬 2. Ask Questions")
    
    question = st.text_input(
        "Your question:", 
        value=st.session_state.question_input,
        placeholder="e.g., What are the main topics discussed across these videos?"
    )
    st.session_state.question_input = question
    
    # Button row
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col_q2:
        clear_question = st.button("❌ Clear", use_container_width=True)
    with col_q3:
        clear_history = st.button("🗑️ Clear History", use_container_width=True)
    
    if clear_question:
        st.session_state.question_input = ""
        st.rerun()
    if clear_history:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and st.session_state.question_input:
        if not st.session_state.rag_chain:
            st.error("❌ Please process videos first!")
        else:
            with st.spinner("🔍 Searching videos and generating answer..."):
                try:
                    result = st.session_state.rag_chain.invoke({"query": st.session_state.question_input})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                    
                    st.session_state.chat_history.append({
                        "question": st.session_state.question_input,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Display answer (styled)
                    st.markdown("#### 🤖 Answer")
                    st.markdown(f'<div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; color: #ffffff;">{answer}</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if source_docs:
                        with st.expander(f"📚 Sources ({len(source_docs)} chunks retrieved)"):
                            for i, doc in enumerate(source_docs[:3], 1):
                                video_title = doc.metadata.get('video_title', 'Unknown')
                                content_preview = doc.page_content[:300]
                                st.markdown(f"""
                                **Source {i}** - 🎥 *{video_title[:50]}*  
                                `{content_preview}...`
                                ---
                                """)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Chat history display
    if st.session_state.chat_history:
        st.markdown(f"### 📜 Chat History ({len(st.session_state.chat_history)} messages)")
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"][:500] + "..." if len(chat["answer"]) > 500 else chat["answer"])
                st.caption(f"Answered at {chat['timestamp']}")

# ============ FOOTER - Management Buttons ============
st.markdown("---")
st.markdown("### 📦 Knowledge Base Management")
col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

with col_exp1:
    if st.button("📊 Show Statistics", use_container_width=True):
        if st.session_state.video_metadata_list:
            stats_df = pd.DataFrame(st.session_state.video_metadata_list)
            st.dataframe(stats_df[['title', 'channel', 'duration']])
        else:
            st.warning("No videos processed yet")

with col_exp2:
    if st.button("💾 Export KB", use_container_width=True) and st.session_state.vectorstore:
        export_dir = tempfile.mkdtemp()
        st.session_state.vectorstore.save_local(export_dir)
        with open(os.path.join(export_dir, "metadata.json"), "w") as f:
            json.dump(st.session_state.video_metadata_list, f, indent=2)
        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(export_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), export_dir))
        with open(zip_path, "rb") as f:
            st.download_button("📥 Download", f.read(), f"yt_rag_kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")
        shutil.rmtree(export_dir)
        os.remove(zip_path)
        st.success("✅ Knowledge base exported!")

with col_exp3:
    if st.button("🧹 Reset All", use_container_width=True):
        for key in ['vectorstore', 'all_chunks', 'video_metadata_list', 'rag_chain', 'chat_history', 'question_input', 'urls_text']:
            if key in st.session_state:
                if key in ['all_chunks', 'video_metadata_list', 'chat_history']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()

with col_exp4:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.markdown("---")
st.caption("🎓 YT-RAG | Multi-Video Knowledge Assistant | Final Year CSE Project")
