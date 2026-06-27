# Talk to YouTube

Chat with any YouTube video using AI. Paste a URL, get an instant AI summary, then ask anything about the content — powered by Groq's fast LLM inference.

## Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 14 (App Router) + TypeScript |
| Styling | Tailwind CSS — YouTube dark theme |
| LLM (chat) | Groq — `llama-3.3-70b-versatile` (128k context, streaming) |
| LLM (summary) | Groq — `llama-3.1-8b-instant` |
| Transcripts | `youtube-transcript` npm package |
| Thread storage | JSON files per video ID in `threads/` |

## Features

- **YouTube-style dark UI** — dark theme, red accents, sidebar thread browser
- **Streaming AI responses** — real-time typewriter effect via Groq streaming
- **Per-video chat threads** — each video has its own persistent chat history
- **Sidebar thread browser** — switch between past video chats instantly
- **Transcript → context** — full transcript passed as LLM context (no vector DB)
- **Download** — export transcript and summary as text files

## Setup

```bash
git clone https://github.com/ganeshhgupta/talk_to_youtube
cd talk_to_youtube
npm install
```

Create `.env.local` (copy from `.env.local.example`):

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

## Run

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project structure

```
talk_to_youtube/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                   # Main client page
│   ├── globals.css
│   └── api/
│       ├── load/route.ts          # POST: fetch transcript + summarize
│       ├── threads/route.ts       # GET: list all threads
│       ├── threads/[id]/route.ts  # GET / DELETE / PATCH a thread
│       └── chat/route.ts          # POST: stream Groq response
├── components/
│   ├── Sidebar.tsx                # Thread list
│   ├── URLBar.tsx                 # URL input
│   ├── VideoCard.tsx              # Video info + summary
│   └── ChatInterface.tsx          # Streaming chat UI
├── lib/
│   ├── threads.ts                 # Thread CRUD (JSON files)
│   └── transcript.ts              # YouTube transcript + video info
├── types/index.ts
└── threads/                       # Created at runtime — one .json per video
```

## How it works

```
User pastes URL
    └── POST /api/load
        ├── Extract video ID
        ├── Fetch transcript (youtube-transcript)
        ├── Summarize with Groq (llama-3.1-8b-instant)
        └── Save as threads/{video_id}.json

User sends a message
    └── POST /api/chat
        ├── Load transcript from thread
        ├── Stream response from Groq (llama-3.3-70b-versatile)
        ├── Client shows typewriter effect in real-time
        └── Persist updated messages after stream ends
```

## Thread storage format

`threads/{video_id}.json`:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "Video Title",
  "author": "Channel Name",
  "transcript": "...",
  "summary": "AI-generated summary...",
  "created_at": "2026-06-26T10:00:00",
  "last_updated": "2026-06-26T10:05:00",
  "messages": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "timestamp": "..." }
  ]
}
```

## Notes

- Videos must have captions (manual or auto-generated).
- Transcript context is capped at 100,000 characters per chat message.
- No external database or GPU required — runs fully locally.
