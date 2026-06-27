# Talk to YouTube

Chat with any YouTube video using AI. Paste a URL, get an instant summary, then ask anything about the content — powered by Groq's fast LLM inference.

## Features

- **YouTube-style dark UI** — familiar dark theme with red accents
- **Per-video chat threads** — each video has its own persistent chat history stored locally as JSON
- **Instant AI summary** — transcript summarized automatically on load using Groq
- **Sidebar thread browser** — switch between past video chats, delete old threads
- **Transcript fallback** — tries English captions first, then auto-generated, then translates from any language
- **Download** — export transcript and summary as text files

## Stack

| Component | Technology |
|-----------|-----------|
| LLM (chat) | [Groq](https://groq.com) — `llama-3.3-70b-versatile` (128k context) |
| LLM (summary) | Groq — `llama-3.1-8b-instant` (fast) |
| Transcripts | `youtube-transcript-api` + `deep-translator` fallback |
| Chat persistence | JSON files per video ID in `threads/` directory |
| UI | Streamlit with custom YouTube dark theme CSS |

## Setup

```bash
git clone https://github.com/ganeshhgupta/talk_to_youtube
cd talk_to_youtube
pip install -r requirements.txt
```

Create a `.env` file (copy from `.env.example`):

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

## Run

```bash
streamlit run main.py
```

## How it works

```
User pastes URL
    └── Extract video ID
    └── Fetch transcript (YouTube captions → translate if needed)
    └── Summarize with Groq (llama-3.1-8b-instant)
    └── Save thread as threads/{video_id}.json
    └── Chat with Groq (llama-3.3-70b-versatile, 128k context)
         └── Full transcript passed as system context — no vector DB needed
         └── Message history appended to thread JSON on each exchange
```

## Thread storage format

Each video gets a file at `threads/{video_id}.json`:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "Video Title",
  "author": "Channel Name",
  "transcript": "Full transcript text...",
  "summary": "AI-generated summary...",
  "created_at": "2026-06-26T10:00:00",
  "last_updated": "2026-06-26T10:05:00",
  "messages": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "timestamp": "..." }
  ]
}
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key from console.groq.com |

## Notes

- Videos must have captions (manual or auto-generated). Videos with no captions cannot be processed.
- Transcript context is capped at 100,000 characters per chat message — covers virtually all YouTube videos.
- No external vector database or GPU required.
