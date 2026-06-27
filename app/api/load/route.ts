import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';
import { extractVideoId, fetchTranscript, fetchVideoInfo } from '@/lib/transcript';
import { getThread, saveThread } from '@/lib/threads';

// Collect a streaming Groq response — more resilient to network blips than one-shot fetch
async function streamingComplete(
  groq: Groq,
  messages: Groq.Chat.ChatCompletionMessageParam[],
  model: string,
  maxTokens: number
): Promise<string> {
  const stream = await groq.chat.completions.create({
    model,
    messages,
    max_tokens: maxTokens,
    temperature: 0.5,
    stream: true,
  });
  let text = '';
  for await (const chunk of stream) {
    text += chunk.choices[0]?.delta?.content ?? '';
  }
  return text;
}

// Retry wrapper for transient network errors
async function withRetry<T>(fn: () => Promise<T>, attempts = 3, delayMs = 1200): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      const msg = String(e).toLowerCase();
      const isTransient =
        msg.includes('premature close') ||
        msg.includes('fetch failed') ||
        msg.includes('econnreset') ||
        msg.includes('socket hang up') ||
        msg.includes('network');
      if (!isTransient) throw e;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, delayMs * (i + 1)));
    }
  }
  throw lastErr;
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: 'GROQ_API_KEY is not set. Create a .env.local file with your Groq API key.' },
        { status: 500 }
      );
    }

    const { url } = (await req.json()) as { url: string };

    const videoId = extractVideoId(url);
    if (!videoId) {
      return NextResponse.json({ error: 'Invalid YouTube URL' }, { status: 400 });
    }

    // Return cached thread if it exists
    const existing = await getThread(videoId);
    if (existing) return NextResponse.json(existing);

    // Fetch transcript
    let transcript: string;
    try {
      transcript = await fetchTranscript(videoId);
    } catch (e) {
      return NextResponse.json(
        { error: `Could not fetch transcript. The video may not have captions enabled. (${e})` },
        { status: 422 }
      );
    }

    const [info] = await Promise.all([fetchVideoInfo(videoId)]);

    const groq = new Groq({ apiKey });

    // Summarize — use streaming + retry for reliability
    const summary = await withRetry(() =>
      streamingComplete(
        groq,
        [
          {
            role: 'system',
            content:
              'Summarize this YouTube video transcript in 3-5 clear, informative sentences. Capture the main topic, key points, and any notable conclusions.',
          },
          // Send only the first 12k chars to keep payload small and fast
          { role: 'user', content: transcript.slice(0, 12000) },
        ],
        'llama-3.1-8b-instant',
        400
      )
    );

    const thread = {
      video_id: videoId,
      url: `https://www.youtube.com/watch?v=${videoId}`,
      title: info.title,
      author: info.author,
      thumbnail: info.thumbnail,
      transcript,
      summary,
      created_at: new Date().toISOString(),
      last_updated: new Date().toISOString(),
      messages: [],
    };

    await saveThread(thread);
    return NextResponse.json(thread);
  } catch (e) {
    console.error('[/api/load]', e);
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
