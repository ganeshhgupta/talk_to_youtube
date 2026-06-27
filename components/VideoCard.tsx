'use client';

import { useState } from 'react';
import { ChevronDown, Download, RefreshCw, ExternalLink } from 'lucide-react';
import type { Thread } from '@/types';

interface Props {
  thread: Thread;
  onReprocess: () => void;
}

export default function VideoCard({ thread, onReprocess }: Props) {
  const [showTranscript, setShowTranscript] = useState(false);
  const wordCount = thread.transcript.split(/\s+/).length;

  const downloadText = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  };

  return (
    <div className="space-y-4">
      {/* Video info card */}
      <div className="flex gap-4 bg-[#1e1e1e] rounded-2xl p-4 border border-[#2a2a2a]">
        <a href={thread.url} target="_blank" rel="noopener noreferrer" className="shrink-0">
          <img
            src={thread.thumbnail}
            alt={thread.title}
            className="w-44 h-[99px] rounded-xl object-cover hover:brightness-110 transition-all"
            onError={(e) => {
              (e.target as HTMLImageElement).src = `https://img.youtube.com/vi/${thread.video_id}/hqdefault.jpg`;
            }}
          />
        </a>

        <div className="flex flex-col justify-between min-w-0 py-0.5">
          <div>
            <a
              href={thread.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-1.5"
            >
              <span className="text-white font-semibold text-base leading-snug hover:text-[#bbb] transition-colors line-clamp-2">
                {thread.title}
              </span>
              <ExternalLink size={12} className="text-[#555] mt-1 shrink-0 group-hover:text-[#aaa] transition-colors" />
            </a>
            <p className="text-[#999] text-sm mt-1">{thread.author}</p>
          </div>

          <div className="flex items-center gap-2 flex-wrap mt-2">
            <span className="bg-[#cc0000] text-white text-[10px] font-bold px-2.5 py-0.5 rounded-full">
              ▶ YouTube
            </span>
            <span className="text-[#555] text-xs">{wordCount.toLocaleString()} words</span>

            {/* Downloads */}
            <button
              onClick={() => downloadText(thread.transcript, `transcript_${thread.video_id}.txt`)}
              className="flex items-center gap-1 text-[#666] hover:text-[#aaa] text-xs border border-[#303030] rounded-full px-2.5 py-0.5 transition-colors"
            >
              <Download size={10} /> Transcript
            </button>
            <button
              onClick={() => downloadText(thread.summary, `summary_${thread.video_id}.txt`)}
              className="flex items-center gap-1 text-[#666] hover:text-[#aaa] text-xs border border-[#303030] rounded-full px-2.5 py-0.5 transition-colors"
            >
              <Download size={10} /> Summary
            </button>
            <button
              onClick={onReprocess}
              className="flex items-center gap-1 text-[#666] hover:text-[#aaa] text-xs border border-[#303030] rounded-full px-2.5 py-0.5 transition-colors"
            >
              <RefreshCw size={10} /> Reprocess
            </button>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div>
        <p className="text-[11px] font-semibold text-[#666] uppercase tracking-widest mb-2">
          AI Summary
        </p>
        <div className="bg-[#1a1a1a] border-l-2 border-[#cc0000] rounded-r-xl px-4 py-3 text-[#ccc] text-sm leading-relaxed">
          {thread.summary}
        </div>
      </div>

      {/* Transcript toggle */}
      <button
        onClick={() => setShowTranscript((v) => !v)}
        className="flex items-center gap-1.5 text-[#666] hover:text-[#aaa] text-xs transition-colors"
      >
        <ChevronDown
          size={13}
          className={`transition-transform duration-200 ${showTranscript ? 'rotate-180' : ''}`}
        />
        {showTranscript ? 'Hide' : 'Show'} full transcript ({wordCount.toLocaleString()} words)
      </button>

      {showTranscript && (
        <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl p-4 max-h-56 overflow-y-auto">
          <p className="text-[#888] text-xs leading-relaxed whitespace-pre-wrap">{thread.transcript}</p>
        </div>
      )}
    </div>
  );
}
