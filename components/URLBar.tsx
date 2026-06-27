'use client';

import { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

interface Props {
  onLoad: (url: string) => void;
  isLoading: boolean;
}

export default function URLBar({ onLoad, isLoading }: Props) {
  const [url, setUrl] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (trimmed) onLoad(trimmed);
  };

  return (
    <header className="shrink-0 border-b border-[#303030] bg-[#0f0f0f]/95 backdrop-blur px-6 py-3.5">
      <form onSubmit={handleSubmit} className="flex gap-3 max-w-2xl">
        <div className="flex-1 flex items-center bg-[#121212] border border-[#303030] rounded-full px-4 gap-2.5 focus-within:border-[#555] transition-colors">
          <Search size={15} className="text-[#555] shrink-0" />
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste a YouTube URL..."
            disabled={isLoading}
            className="flex-1 bg-transparent text-white text-sm py-2.5 outline-none placeholder:text-[#444] disabled:opacity-50"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !url.trim()}
          className="flex items-center gap-2 bg-[#cc0000] hover:bg-[#aa0000] disabled:bg-[#2a2a2a] disabled:text-[#555] text-white px-5 py-2 rounded-full text-sm font-medium transition-colors shrink-0"
        >
          {isLoading ? <Loader2 size={14} className="animate-spin" /> : null}
          {isLoading ? 'Loading...' : 'Load'}
        </button>
      </form>
    </header>
  );
}
