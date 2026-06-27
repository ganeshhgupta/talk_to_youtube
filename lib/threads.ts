import { promises as fs } from 'fs';
import path from 'path';
import type { Thread, Message } from '@/types';

const THREADS_DIR = path.join(process.cwd(), 'threads');

async function ensureDir() {
  await fs.mkdir(THREADS_DIR, { recursive: true });
}

export async function listThreads(): Promise<Thread[]> {
  await ensureDir();
  const files = await fs.readdir(THREADS_DIR);
  const threads = await Promise.all(
    files
      .filter((f) => f.endsWith('.json'))
      .map(async (f) => {
        const raw = await fs.readFile(path.join(THREADS_DIR, f), 'utf-8');
        return JSON.parse(raw) as Thread;
      })
  );
  return threads.sort(
    (a, b) => new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime()
  );
}

export async function getThread(videoId: string): Promise<Thread | null> {
  await ensureDir();
  const filePath = path.join(THREADS_DIR, `${videoId}.json`);
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw) as Thread;
  } catch {
    return null;
  }
}

export async function saveThread(thread: Thread): Promise<void> {
  await ensureDir();
  const filePath = path.join(THREADS_DIR, `${thread.video_id}.json`);
  await fs.writeFile(filePath, JSON.stringify(thread, null, 2), 'utf-8');
}

export async function deleteThread(videoId: string): Promise<void> {
  const filePath = path.join(THREADS_DIR, `${videoId}.json`);
  try {
    await fs.unlink(filePath);
  } catch {
    // already gone
  }
}

export async function updateMessages(videoId: string, messages: Message[]): Promise<Thread | null> {
  const thread = await getThread(videoId);
  if (!thread) return null;
  thread.messages = messages;
  thread.last_updated = new Date().toISOString();
  await saveThread(thread);
  return thread;
}
