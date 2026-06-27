# main.py - Talk to YouTube | Powered by Groq

import os
import re
import json
import html
import base64
import tempfile
import warnings
import requests
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from gtts import gTTS

warnings.filterwarnings("ignore")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THREADS_DIR = Path("threads")
THREADS_DIR.mkdir(exist_ok=True)

CHAT_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"


# === GROQ CLIENT ===

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


# === VIDEO HELPERS ===

def extract_video_id(url: str) -> str | None:
    for pattern in [
        r"(?:v=|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]:
        m = re.search(pattern, url.strip())
        if m:
            return m.group(1)
    return None


def get_video_info(video_id: str) -> dict:
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=5,
        )
        data = r.json()
        return {
            "title": data.get("title", "Unknown Title"),
            "author": data.get("author_name", "Unknown Channel"),
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        }
    except Exception:
        return {
            "title": "Unknown Title",
            "author": "Unknown Channel",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        }


def get_transcript(video_id: str) -> str:
    # Try English subtitles first
    try:
        items = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(i["text"] for i in items)
    except Exception:
        pass

    # Try any available transcript, translate if needed
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in tlist:
            items = t.fetch()
            text = " ".join(i["text"] for i in items)
            if t.language_code != "en":
                # Only translate the first 5000 chars to avoid API limits
                text = GoogleTranslator(source=t.language_code, target="en").translate(text[:5000])
            return text
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")


# === GROQ LLM ===

def groq_summarize(client: Groq, transcript: str) -> str:
    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following YouTube video transcript in 3-5 clear, informative sentences. "
                    "Capture the main topic, key points, and any notable conclusions."
                ),
            },
            {"role": "user", "content": transcript[:30000]},
        ],
        max_tokens=512,
        temperature=0.5,
    )
    return resp.choices[0].message.content


def groq_chat(client: Groq, messages: list[dict], transcript: str) -> str:
    system_content = (
        "You are an AI assistant that answers questions about a YouTube video "
        "based on its transcript. Be helpful, accurate, and concise. "
        "If something is not in the transcript, say so clearly.\n\n"
        f"--- VIDEO TRANSCRIPT ---\n{transcript[:100000]}\n--- END TRANSCRIPT ---"
    )
    groq_messages = [{"role": "system", "content": system_content}]
    groq_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=groq_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content


# === THREAD STORAGE ===

def thread_path(video_id: str) -> Path:
    return THREADS_DIR / f"{video_id}.json"


def load_thread(video_id: str) -> dict | None:
    p = thread_path(video_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_thread(thread: dict) -> None:
    p = thread_path(thread["video_id"])
    p.write_text(json.dumps(thread, indent=2, ensure_ascii=False), encoding="utf-8")


def list_threads() -> list[dict]:
    threads = []
    for f in THREADS_DIR.glob("*.json"):
        try:
            threads.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(threads, key=lambda x: x.get("last_updated", ""), reverse=True)


def delete_thread(video_id: str) -> None:
    p = thread_path(video_id)
    if p.exists():
        p.unlink()


# === TTS ===

def tts_audio_tag(text: str, lang: str = "en") -> str:
    if not text.strip():
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            gTTS(text=text[:500], lang=lang).save(fp.name)
            audio_b64 = base64.b64encode(open(fp.name, "rb").read()).decode()
        os.unlink(fp.name)
        return (
            f'<audio controls style="width:100%;margin-top:8px">'
            f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">'
            f"</audio>"
        )
    except Exception:
        return ""


# === CSS ===

YOUTUBE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', 'Arial', sans-serif;
}
.stApp {
    background-color: #0f0f0f;
    color: #ffffff;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background-color: #212121 !important;
    border-right: 1px solid #303030;
}
section[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* --- Inputs --- */
.stTextInput input, .stChatInput textarea {
    background-color: #121212 !important;
    color: #ffffff !important;
    border: 1.5px solid #303030 !important;
    border-radius: 24px !important;
    padding: 10px 18px !important;
    font-size: 15px !important;
}
.stTextInput input:focus {
    border-color: #1c62b9 !important;
    box-shadow: 0 0 0 2px rgba(28,98,185,0.25) !important;
}
[data-testid="stChatInput"] {
    background-color: #1a1a1a !important;
    border-top: 1px solid #303030 !important;
    border-radius: 0 !important;
    padding: 12px 16px !important;
}

/* --- Buttons --- */
.stButton > button {
    background-color: #cc0000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #aa0000 !important;
}

/* --- Chat messages --- */
[data-testid="stChatMessage"] {
    background-color: #1e1e1e !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
    border: 1px solid #2a2a2a !important;
}
[data-testid="stChatMessage"][data-message-author-role="user"] {
    background-color: #1a2744 !important;
    border-color: #1c3a6e !important;
}
[data-testid="stChatMessage"][data-message-author-role="assistant"] {
    background-color: #1e1e1e !important;
}

/* --- Video info card --- */
.yt-video-card {
    background: #1e1e1e;
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
    border: 1px solid #303030;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}
.yt-video-card img {
    border-radius: 8px;
    width: 240px;
    min-width: 240px;
    height: 135px;
    object-fit: cover;
}
.yt-video-card-info h2 {
    margin: 0 0 6px 0;
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    line-height: 1.3;
}
.yt-video-card-info .channel {
    font-size: 13px;
    color: #aaaaaa;
    margin-bottom: 8px;
}
.yt-badge {
    display: inline-block;
    background: #cc0000;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.yt-meta {
    font-size: 12px;
    color: #666;
    margin-top: 10px;
}

/* --- Summary box --- */
.yt-summary {
    background: #1a1a1a;
    border-left: 3px solid #cc0000;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    color: #cccccc;
    font-size: 14px;
    line-height: 1.8;
    margin: 12px 0 20px 0;
}

/* --- Section headers --- */
.yt-section-header {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin: 20px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #303030;
}

/* --- Thread card (sidebar) --- */
.thread-card {
    background: #2d2d2d;
    border-radius: 10px;
    padding: 10px;
    margin: 6px 0;
    border: 1.5px solid transparent;
}
.thread-card.active {
    border-color: #cc0000;
    background: #3a1a1a;
}
.thread-card-thumb {
    border-radius: 6px;
    width: 100%;
    height: 72px;
    object-fit: cover;
    margin-bottom: 6px;
    display: block;
}
.thread-title {
    font-size: 12px;
    color: #eee;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 180px;
}
.thread-meta {
    font-size: 10px;
    color: #888;
    margin-top: 2px;
}

/* --- Logo --- */
.yt-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #303030;
}
.yt-logo-icon {
    background: #cc0000;
    color: white;
    width: 36px;
    height: 26px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 900;
    flex-shrink: 0;
}
.yt-logo-text {
    font-size: 18px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}

/* --- Page hero --- */
.yt-hero {
    text-align: center;
    margin: 8px 0 28px 0;
}
.yt-hero h1 {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 6px;
}
.yt-hero p {
    font-size: 14px;
    color: #aaa;
}

/* --- Empty state --- */
.yt-empty {
    text-align: center;
    margin-top: 80px;
}
.yt-empty-icon {
    font-size: 64px;
    opacity: 0.3;
}
.yt-empty-text {
    color: #555;
    font-size: 15px;
    margin-top: 16px;
}

/* --- Spinners / misc --- */
.stSpinner > div { border-top-color: #cc0000 !important; }
.stAlert { border-radius: 10px !important; }
div[data-testid="stExpander"] {
    border: 1px solid #303030 !important;
    border-radius: 10px !important;
    background: #1a1a1a !important;
}
</style>
"""


# === MAIN ===

def main():
    st.set_page_config(
        page_title="Talk to YouTube",
        page_icon="▶️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(YOUTUBE_CSS, unsafe_allow_html=True)

    client = get_groq_client()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="yt-logo">'
            '<div class="yt-logo-icon">▶</div>'
            '<div class="yt-logo-text">Talk to YouTube</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        threads = list_threads()
        if threads:
            st.markdown(
                '<div class="yt-section-header">Your Chats</div>',
                unsafe_allow_html=True,
            )
            for t in threads:
                vid_id = t["video_id"]
                is_active = st.session_state.get("current_video_id") == vid_id
                card_cls = "thread-card active" if is_active else "thread-card"
                msg_count = len([m for m in t.get("messages", []) if m["role"] == "user"])
                title_safe = html.escape(t.get("title", "Unknown")[:45])
                author_safe = html.escape(t.get("author", "")[:28])

                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<img class="thread-card-thumb" src="{t.get("thumbnail","")}" '
                    f'onerror="this.style.display=\'none\'">'
                    f'<div class="thread-title">{title_safe}</div>'
                    f'<div class="thread-meta">{author_safe} &bull; {msg_count} questions</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button("Open", key=f"open_{vid_id}", use_container_width=True):
                        st.session_state.current_video_id = vid_id
                        st.session_state.input_url = f"https://www.youtube.com/watch?v={vid_id}"
                        st.rerun()
                with c2:
                    if st.button("🗑", key=f"del_{vid_id}"):
                        delete_thread(vid_id)
                        if st.session_state.get("current_video_id") == vid_id:
                            st.session_state.pop("current_video_id", None)
                            st.session_state.pop("input_url", None)
                        st.rerun()
        else:
            st.markdown(
                '<div style="color:#666;font-size:13px;padding:8px 0">'
                "No chats yet. Paste a YouTube URL to start."
                "</div>",
                unsafe_allow_html=True,
            )

        if not GROQ_API_KEY:
            st.error("⚠️ GROQ_API_KEY missing from .env")

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="yt-hero">'
        "<h1>Talk to any YouTube video</h1>"
        "<p>Paste a URL, get an AI summary, then chat about the content</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    url_col, btn_col = st.columns([6, 1])
    with url_col:
        video_url = st.text_input(
            "url",
            label_visibility="collapsed",
            placeholder="https://www.youtube.com/watch?v=...",
            value=st.session_state.get("input_url", ""),
            key="url_input",
        )
    with btn_col:
        load_btn = st.button("Load ▶", use_container_width=True)

    if load_btn and video_url:
        vid_id = extract_video_id(video_url)
        if vid_id:
            st.session_state.current_video_id = vid_id
            st.session_state.input_url = video_url
        else:
            st.error("Invalid YouTube URL. Paste a link like: https://youtube.com/watch?v=...")

    current_video_id = st.session_state.get("current_video_id")

    if not current_video_id:
        st.markdown(
            '<div class="yt-empty">'
            '<div class="yt-empty-icon">▶️</div>'
            '<div class="yt-empty-text">Enter a YouTube URL above to get started</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if not client:
        st.error("GROQ_API_KEY not set. Add it to your .env file and restart.")
        return

    # ── Load or create thread ─────────────────────────────────────────────────
    thread = load_thread(current_video_id)

    if thread is None:
        with st.spinner("Fetching transcript..."):
            info = get_video_info(current_video_id)
            try:
                transcript = get_transcript(current_video_id)
            except RuntimeError as e:
                st.error(str(e))
                st.info("The video may not have captions enabled. Try a different video.")
                return

        with st.spinner("Summarizing with Groq..."):
            summary = groq_summarize(client, transcript)

        thread = {
            "video_id": current_video_id,
            "url": f"https://www.youtube.com/watch?v={current_video_id}",
            "title": info["title"],
            "author": info["author"],
            "thumbnail": info["thumbnail"],
            "transcript": transcript,
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "messages": [],
        }
        save_thread(thread)
        st.rerun()

    # ── Video info card ───────────────────────────────────────────────────────
    word_count = len(thread.get("transcript", "").split())
    title_safe = html.escape(thread["title"])
    author_safe = html.escape(thread["author"])

    st.markdown(
        f'<div class="yt-video-card">'
        f'<img src="{thread["thumbnail"]}" '
        f'onerror="this.src=\'https://img.youtube.com/vi/{current_video_id}/hqdefault.jpg\'">'
        f'<div class="yt-video-card-info">'
        f"<h2>{title_safe}</h2>"
        f'<div class="channel">{author_safe}</div>'
        f'<span class="yt-badge">▶ YouTube</span>'
        f'<div class="yt-meta">{word_count:,} words in transcript</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="yt-section-header">AI Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="yt-summary">{html.escape(thread["summary"])}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        with st.expander("View Full Transcript"):
            st.text_area(
                "transcript",
                thread.get("transcript", ""),
                height=240,
                label_visibility="collapsed",
            )
    with col_b:
        with st.expander("Download"):
            st.download_button(
                "Download Transcript",
                thread.get("transcript", ""),
                file_name=f"transcript_{current_video_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                "Download Summary",
                thread.get("summary", ""),
                file_name=f"summary_{current_video_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            if st.button("🔄 Reprocess Video", use_container_width=True):
                delete_thread(current_video_id)
                st.session_state.pop("current_video_id", None)
                st.session_state.pop("input_url", None)
                st.rerun()

    # ── Chat interface ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="yt-section-header">Chat with this video</div>',
        unsafe_allow_html=True,
    )

    messages = thread.get("messages", [])

    # Render history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Clear button (only show if there are messages)
    if messages:
        if st.button("Clear chat history"):
            thread["messages"] = []
            thread["last_updated"] = datetime.now().isoformat()
            save_thread(thread)
            st.rerun()

    # New message input
    if user_input := st.chat_input("Ask anything about this video..."):
        messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(""):
                response = groq_chat(client, messages, thread["transcript"])
            st.markdown(response)

        messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })
        thread["messages"] = messages
        thread["last_updated"] = datetime.now().isoformat()
        save_thread(thread)


if __name__ == "__main__":
    main()
