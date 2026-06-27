'use client';

import { Trash2, MessageSquare } from 'lucide-react';
import type { Thread } from '@/types';

interface Props {
  threads: Thread[];
  currentVideoId?: string;
  onSelect: (t: Thread) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({ threads, currentVideoId, onSelect, onDelete }: Props) {
  return (
    <aside className="w-60 shrink-0 bg-[#212121] border-r border-[#303030] flex flex-col h-full overflow-hidden">
      {/* Logo */}
      <div className="px-4 py-4 border-b border-[#303030] shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-5 bg-[#cc0000] rounded flex items-center justify-center text-white text-[10px] font-black leading-none">
            ▶
          </div>
          <span className="font-bold text-white text-[15px] tracking-tight">Talk to YouTube</span>
        </div>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto py-3 px-2 space-y-1.5">
        {threads.length === 0 ? (
          <div className="text-center pt-12 px-4">
            <MessageSquare size={28} className="mx-auto text-[#444] mb-3" />
            <p className="text-[#555] text-xs leading-relaxed">
              No chats yet. Paste a YouTube URL to get started.
            </p>
          </div>
        ) : (
          threads.map((t) => {
            const isActive = t.video_id === currentVideoId;
            const msgCount = t.messages.filter((m) => m.role === 'user').length;

            return (
              <div
                key={t.video_id}
                onClick={() => onSelect(t)}
                className={`group relative rounded-xl overflow-hidden cursor-pointer transition-all duration-150 ${
                  isActive
                    ? 'ring-1 ring-[#cc0000] bg-[#2a1515]'
                    : 'bg-[#2d2d2d] hover:bg-[#333]'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative">
                  <img
                    src={t.thumbnail}
                    alt={t.title}
                    className="w-full h-[70px] object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = `https://img.youtube.com/vi/${t.video_id}/hqdefault.jpg`;
                    }}
                  />
                  {isActive && (
                    <div className="absolute inset-0 bg-[#cc0000]/10" />
                  )}
                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(t.video_id);
                    }}
                    className="absolute top-1 right-1 p-1 bg-black/60 backdrop-blur-sm rounded text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-[#cc0000]/80"
                    title="Delete chat"
                  >
                    <Trash2 size={10} />
                  </button>
                </div>

                {/* Info */}
                <div className="px-2.5 py-2">
                  <p className="text-white text-[11px] font-medium leading-tight line-clamp-2 mb-0.5">
                    {t.title}
                  </p>
                  <p className="text-[#777] text-[10px] truncate">
                    {t.author}
                    {msgCount > 0 && (
                      <span className="text-[#555]"> · {msgCount} msg{msgCount !== 1 ? 's' : ''}</span>
                    )}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
