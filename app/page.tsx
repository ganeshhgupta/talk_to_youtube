'use client';

import { useState, useEffect, useCallback } from 'react';
import type { Thread } from '@/types';
import Sidebar from '@/components/Sidebar';
import URLBar from '@/components/URLBar';
import VideoCard from '@/components/VideoCard';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [current, setCurrent] = useState<Thread | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshThreads = useCallback(async () => {
    const res = await fetch('/api/threads');
    const data: Thread[] = await res.json();
    setThreads(data);
  }, []);

  useEffect(() => {
    refreshThreads();
  }, [refreshThreads]);

  const handleLoad = async (url: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try {
          const data = await res.json();
          msg = data.error ?? msg;
        } catch { /* response was not JSON */ }
        throw new Error(msg);
      }
      const thread: Thread = await res.json();
      setCurrent(thread);
      await refreshThreads();
    } catch (e) {
      setError(String(e).replace('Error: ', ''));
    } finally {
      setIsLoading(false);
    }
  };

  const handleReprocess = async () => {
    if (!current) return;
    await fetch(`/api/threads/${current.video_id}`, { method: 'DELETE' });
    const url = current.url;
    setCurrent(null);
    await refreshThreads();
    await handleLoad(url);
  };

  const handleDelete = async (videoId: string) => {
    await fetch(`/api/threads/${videoId}`, { method: 'DELETE' });
    if (current?.video_id === videoId) setCurrent(null);
    await refreshThreads();
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#0f0f0f]">
      <Sidebar
        threads={threads}
        currentVideoId={current?.video_id}
        onSelect={setCurrent}
        onDelete={handleDelete}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <URLBar onLoad={handleLoad} isLoading={isLoading} />

        <div className="flex-1 overflow-y-auto">
          {/* Error */}
          {error && (
            <div className="mx-6 mt-5 p-4 bg-red-950/50 border border-red-900/50 rounded-xl text-red-300 text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Loading skeleton */}
          {isLoading && (
            <div className="max-w-3xl mx-auto p-6 space-y-4 animate-pulse">
              <div className="flex gap-4 bg-[#1e1e1e] rounded-2xl p-4">
                <div className="w-44 h-[99px] bg-[#2a2a2a] rounded-xl shrink-0" />
                <div className="flex-1 space-y-3 py-1">
                  <div className="h-4 bg-[#2a2a2a] rounded w-3/4" />
                  <div className="h-3 bg-[#2a2a2a] rounded w-1/4" />
                  <div className="h-3 bg-[#222] rounded w-1/3 mt-4" />
                </div>
              </div>
              <div className="h-[72px] bg-[#1a1a1a] rounded-xl border-l-2 border-[#cc0000]/30" />
              <p className="text-center text-[#444] text-xs pt-2">
                Fetching transcript &amp; summarizing with Groq...
              </p>
            </div>
          )}

          {/* Video + Chat */}
          {!isLoading && current && (
            <div className="max-w-3xl mx-auto p-6 space-y-6">
              <VideoCard thread={current} onReprocess={handleReprocess} />
              <div className="border-t border-[#1e1e1e] pt-6">
                <ChatInterface thread={current} onUpdate={setCurrent} />
              </div>
            </div>
          )}

          {/* Empty state */}
          {!isLoading && !current && !error && (
            <div className="flex flex-col items-center justify-center h-full text-center pb-16">
              <div
                className="w-20 h-14 rounded-2xl flex items-center justify-center text-4xl mb-6"
                style={{ background: 'linear-gradient(135deg, #cc0000, #990000)' }}
              >
                ▶
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Talk to any YouTube video</h2>
              <p className="text-[#555] text-sm max-w-xs leading-relaxed">
                Paste a YouTube URL above to get an AI summary and start chatting about the content.
              </p>
              {threads.length > 0 && (
                <p className="text-[#444] text-xs mt-4">
                  Or pick a past video from the sidebar →
                </p>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
