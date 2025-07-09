# YouTube LLM Assistant with TTS, Quiz, Search, Download, Multilingual UI, Chat Memory, and Meme Generation

import os
import streamlit as st
import requests
import whisper
import yt_dlp
import uuid
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from pinecone import Pinecone, ServerlessSpec
from gtts import gTTS
import tempfile
import base64
import warnings
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === ENV SETUP ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# === CONFIG FROM ENV ===
class Config:
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    INDEX_NAME = os.getenv("INDEX_NAME", "youtube-assistant")
    IMGFLIP_USERNAME = os.getenv("IMGFLIP_USERNAME")
    IMGFLIP_PASSWORD = os.getenv("IMGFLIP_PASSWORD")
    
    @classmethod
    def validate(cls):
        required_vars = ["HUGGINGFACE_API_KEY", "PINECONE_API_KEY"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            st.error(f"Missing required environment variables: {', '.join(missing)}")
            st.info("Please set these variables in your .env file")
            st.stop()

config = Config()
config.validate()

# === MODELS ===
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
    whisper_model = whisper.load_model("base")
    return embedder, whisper_model

embedder, whisper_model = load_models()

# === PINECONE SETUP ===
@st.cache_resource
def setup_pinecone():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=config.INDEX_NAME, 
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region=config.PINECONE_ENV)
        )
    return pc.Index(config.INDEX_NAME)

index = setup_pinecone()

# === LLM FUNCTIONS ===
def mistral_response(prompt):
    """Generate response using Mistral model via HuggingFace API"""
    headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt, 
        "parameters": {
            "max_new_tokens": 512, 
            "temperature": 0.7, 
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{config.MISTRAL_MODEL}", 
            headers=headers, 
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "Error generating response")
        return str(result)
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

# === TRANSCRIPT FUNCTIONS ===
def get_transcript(video_id):
    """Get transcript with fallback methods"""
    try:
        # Try English subtitles first
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([item['text'] for item in transcript])
        return text, 'en', 'subtitles', text
    except:
        try:
            # Try auto-generated subtitles
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcript_list:
                if t.is_generated:
                    fetched = t.fetch()
                    lang = t.language_code
                    text = " ".join([item['text'] for item in fetched])
                    translated = GoogleTranslator(source=lang, target='en').translate(text)
                    return translated, lang, 'translated-subtitles', text
        except:
            # Final fallback: Whisper transcription
            audio_text, lang = transcribe_audio(video_id)
            translated = GoogleTranslator(source=lang, target='en').translate(audio_text)
            return translated, lang, 'whisper-translated', audio_text

def transcribe_audio(video_id):
    """Transcribe audio using Whisper"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    uid = uuid.uuid4().hex[:6]
    audio_path = f"audio_{uid}.mp3"
    
    ydl_opts = {
        'format': 'bestaudio', 
        'outtmpl': f"temp_{uid}.%(ext)s", 
        'quiet': True, 
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }]
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        os.rename(f"temp_{uid}.mp3", audio_path)
        result = whisper_model.transcribe(audio_path, task="transcribe")
        
        return result["text"], result.get("language", "en")
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

# === UTILITY FUNCTIONS ===
def chunk_text(text):
    """Split text into chunks for vector storage"""
    return RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100
    ).split_text(text)

def store_chunks(chunks, namespace):
    """Store text chunks in Pinecone"""
    vectors = [
        {
            "id": f"{namespace}-{i}", 
            "values": embedder.encode(chunk).tolist(), 
            "metadata": {"text": chunk}
        } 
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors, namespace=namespace)

def search_similar(query, namespace):
    """Search for similar chunks in Pinecone"""
    query_vec = embedder.encode(query).tolist()
    results = index.query(
        vector=query_vec, 
        top_k=5, 
        include_metadata=True, 
        namespace=namespace
    )
    return [match["metadata"]["text"] for match in results["matches"]]

def is_video_processed(namespace):
    """Check if video is already processed"""
    try:
        res = index.query(vector=[0.0]*384, top_k=1, namespace=namespace)
        return len(res.get("matches", [])) > 0
    except:
        return False

def translate(text, source, target):
    """Translate text between languages"""
    if source == target:
        return text
    return GoogleTranslator(source=source, target=target).translate(text)

def tts_playback(text, lang='en'):
    """Generate TTS audio for text"""
    if not text.strip():
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            gTTS(text=text, lang=lang).save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode()
            os.unlink(fp.name)
            return f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    except Exception as e:
        st.warning(f"TTS failed: {str(e)}")
        return ""

def generate_quiz(summary):
    """Generate quiz questions from summary"""
    prompt = f"Create 3 multiple choice quiz questions with answers from the following summary:\n{summary}"
    return mistral_response(prompt)

def generate_meme_caption(summary):
    """Generate meme caption from summary"""
    prompt = f"Create a funny, relatable meme caption about this video summary (keep it short and punchy):\n{summary}"
    return mistral_response(prompt)

def create_imgflip_meme(caption):
    """Create meme using ImgFlip API"""
    if not config.IMGFLIP_USERNAME or not config.IMGFLIP_PASSWORD:
        return None, "ImgFlip credentials not configured"
    
    imgflip_api_url = "https://api.imgflip.com/caption_image"
    meme_templates = ["181913649", "112126428", "102156234"]  # Drake, Distracted Boyfriend, Mocking Spongebob
    
    caption_parts = caption.split(".")
    text0 = caption_parts[0] if len(caption_parts) > 0 else caption[:50]
    text1 = caption_parts[-1] if len(caption_parts) > 1 else caption[50:100]
    
    meme_payload = {
        'template_id': meme_templates[0],
        'username': config.IMGFLIP_USERNAME,
        'password': config.IMGFLIP_PASSWORD,
        'text0': text0,
        'text1': text1
    }
    
    try:
        response = requests.post(imgflip_api_url, data=meme_payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            return result['data']['url'], None
        else:
            return None, result.get("error_message", "Unknown error")
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"

def generate_ai_meme(caption):
    """Generate AI meme using HuggingFace"""
    dalle_prompt = f"Create a funny meme image that reflects this caption: '{caption}'"
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/openskyml/dalle-3",
            headers={"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"},
            json={"inputs": dalle_prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
    except Exception as e:
        st.warning(f"AI meme generation failed: {str(e)}")
        return None

# === STREAMLIT UI ===
def main():
    st.set_page_config(page_title="YouTube LLM Assistant", layout="wide")
    st.title("🧠 YouTube LLM Assistant Pro")
    st.markdown("*Powered by Mistral, Whisper, and Pinecone*")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input("📺 Paste YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        force_reprocess = st.checkbox("♻️ Reprocess", help="Force reprocessing even if video was already processed")
    
    if not video_url:
        st.info("👆 Enter a YouTube URL to get started")
        return
    
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        
        # Processing section
        with st.spinner("🔍 Processing video..."):
            if not force_reprocess and is_video_processed(video_id):
                st.success("✅ Video already processed. Using cached data.")
            else:
                # Get transcript
                result = get_transcript(video_id)
                if len(result) == 4:
                    transcript_en, lang, source, raw_transcript = result
                else:
                    transcript_en, lang, source = result
                    raw_transcript = transcript_en
                
                # Process and store
                chunks = chunk_text(transcript_en)
                store_chunks(chunks, video_id)
                
                # Generate summary
                summary_en = mistral_response(f"Summarize this video transcript in 2-3 sentences:\n{transcript_en[:2000]}")
                summary_native = translate(summary_en, 'en', lang) if lang != 'en' else summary_en
                
                # Display results
                st.subheader("📝 Summary")
                st.markdown(f"**English:** {summary_en}")
                if lang != 'en':
                    st.markdown(f"**{lang.upper()}:** {summary_native}")
                
                # TTS for summary
                if summary_en:
                    st.markdown(tts_playback(summary_en), unsafe_allow_html=True)
                
                # Transcript
                with st.expander("📜 View Full Transcript"):
                    st.text_area("Transcript", raw_transcript, height=300, key="transcript_display")
                
                # Quiz generation
                st.subheader("🧠 Quiz Questions")
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(summary_en)
                    st.text_area("Quiz Questions", quiz, height=200, key="quiz_display")
                
                # Meme generation
                st.subheader("🎭 Meme Generation")
                with st.spinner("Creating meme..."):
                    meme_caption = generate_meme_caption(summary_en)
                    st.text_area("Meme Caption", meme_caption, height=100, key="meme_caption")
                    
                    # Traditional meme
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📸 Traditional Meme**")
                        meme_url, error = create_imgflip_meme(meme_caption)
                        if meme_url:
                            st.image(meme_url, caption=meme_caption)
                            st.markdown(f"[🔗 View Full Size]({meme_url})")
                        else:
                            st.warning(f"Failed to generate meme: {error}")
                    
                    with col2:
                        st.markdown("**🤖 AI-Generated Meme**")
                        ai_meme = generate_ai_meme(meme_caption)
                        if ai_meme:
                            st.image(ai_meme, caption=meme_caption)
                        else:
                            st.warning("AI meme generation failed")
                
                # Search functionality
                st.subheader("🔍 Search Transcript")
                search_query = st.text_input("Search for specific content", placeholder="Enter keywords...")
                if search_query:
                    matching_chunks = [chunk for chunk in chunks if search_query.lower() in chunk.lower()]
                    if matching_chunks:
                        st.markdown(f"Found {len(matching_chunks)} matches:")
                        for i, match in enumerate(matching_chunks[:5]):  # Show top 5
                            st.markdown(f"**Match {i+1}:**")
                            st.markdown(f"> {match}")
                    else:
                        st.info("No matches found")
                
                # Download section
                st.subheader("⬇️ Downloads")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("📝 Download Summary", summary_en, file_name="summary.txt", mime="text/plain")
                with col2:
                    st.download_button("📜 Download Transcript", raw_transcript, file_name="transcript.txt", mime="text/plain")
                with col3:
                    st.download_button("🧠 Download Quiz", quiz, file_name="quiz.txt", mime="text/plain")
        
        # Chat section
        st.subheader("💬 Chat with the Video")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("**Previous Questions:**")
            for i, (question, answer) in enumerate(st.session_state.chat_history[-3:]):  # Show last 3
                if answer:
                    with st.expander(f"Q{i+1}: {question[:50]}..." if len(question) > 50 else f"Q{i+1}: {question}"):
                        st.markdown(f"**Answer:** {answer}")
        
        # New question input
        question = st.text_input("❓ Ask a question about the video", placeholder="What is the main topic discussed?")
        
        if st.button("🚀 Get Answer", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    # Detect language and translate if needed
                    question_lang = detect(question)
                    question_en = translate(question, question_lang, 'en')
                    
                    # Search for relevant chunks
                    relevant_chunks = search_similar(question_en, video_id)
                    context = "\n".join(relevant_chunks)
                    
                    # Generate answer
                    prompt = f"Based on this context from a video transcript, answer the question:\n\nContext:\n{context}\n\nQuestion: {question_en}\n\nAnswer:"
                    answer_en = mistral_response(prompt)
                    
                    # Translate back if needed
                    answer_native = translate(answer_en, 'en', question_lang) if question_lang != 'en' else answer_en
                    
                    # Store in history
                    st.session_state.chat_history.append((question, answer_en))
                    
                    # Display answer
                    st.markdown("### 🤖 Answer")
                    st.markdown(f"**English:** {answer_en}")
                    if question_lang != 'en':
                        st.markdown(f"**{question_lang.upper()}:** {answer_native}")
                    
                    # TTS for answer
                    if answer_en:
                        st.markdown(tts_playback(answer_en), unsafe_allow_html=True)
            else:
                st.warning("Please enter a question")
    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        st.info("Please check if the video URL is valid and the video has captions or audio")

if __name__ == "__main__":
    main()