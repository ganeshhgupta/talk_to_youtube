import { NextRequest } from 'next/server';
import Groq from 'groq-sdk';
import { getThread, saveThread } from '@/lib/threads';
import type { Message } from '@/types';

export async function POST(req: NextRequest) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return new Response(
      JSON.stringify({ error: 'GROQ_API_KEY is not set.' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }

  try {
    const { videoId, messages } = (await req.json()) as {
      videoId: string;
      messages: Message[];
    };

    const thread = await getThread(videoId);
    if (!thread) {
      return new Response(JSON.stringify({ error: 'Thread not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const groq = new Groq({ apiKey });

    const systemContent =
      `You are an AI assistant that answers questions about a YouTube video based on its transcript. ` +
      `Be helpful, accurate, and concise. If something is not in the transcript, say so clearly.\n\n` +
      `--- VIDEO TRANSCRIPT ---\n${thread.transcript.slice(0, 100000)}\n--- END TRANSCRIPT ---`;

    const groqMessages: Groq.Chat.ChatCompletionMessageParam[] = [
      { role: 'system', content: systemContent },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ];

    let fullResponse = '';

    const stream = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: groqMessages,
      max_tokens: 1024,
      temperature: 0.7,
      stream: true,
    });

    const encoder = new TextEncoder();

    const readable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const text = chunk.choices[0]?.delta?.content ?? '';
            fullResponse += text;
            if (text) controller.enqueue(encoder.encode(text));
          }

          // Persist updated messages after stream completes
          const updatedMessages: Message[] = [
            ...messages,
            { role: 'assistant', content: fullResponse, timestamp: new Date().toISOString() },
          ];
          thread.messages = updatedMessages;
          thread.last_updated = new Date().toISOString();
          await saveThread(thread);
        } finally {
          controller.close();
        }
      },
    });

    return new Response(readable, {
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
  } catch (e) {
    console.error('[/api/chat]', e);
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
