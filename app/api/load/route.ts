import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';
import { extractVideoId, fetchTranscript, fetchVideoInfo } from '@/lib/transcript';
import { getThread, saveThread } from '@/lib/threads';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function POST(req: NextRequest) {
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

  const info = await fetchVideoInfo(videoId);

  // Summarize
  const summaryRes = await groq.chat.completions.create({
    model: 'llama-3.1-8b-instant',
    messages: [
      {
        role: 'system',
        content:
          'Summarize this YouTube video transcript in 3-5 clear, informative sentences. Capture the main topic, key points, and any notable conclusions.',
      },
      { role: 'user', content: transcript.slice(0, 30000) },
    ],
    max_tokens: 512,
    temperature: 0.5,
  });

  const thread = {
    video_id: videoId,
    url: `https://www.youtube.com/watch?v=${videoId}`,
    title: info.title,
    author: info.author,
    thumbnail: info.thumbnail,
    transcript,
    summary: summaryRes.choices[0].message.content ?? '',
    created_at: new Date().toISOString(),
    last_updated: new Date().toISOString(),
    messages: [],
  };

  await saveThread(thread);
  return NextResponse.json(thread);
}
