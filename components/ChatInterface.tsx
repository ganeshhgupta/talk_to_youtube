'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Trash2, Bot, User } from 'lucide-react';
import type { Thread, Message } from '@/types';

interface Props {
  thread: Thread;
  onUpdate: (t: Thread) => void;
}

export default function ChatInterface({ thread, onUpdate }: Props) {
  const [messages, setMessages] = useState<Message[]>(thread.messages);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync when thread changes (e.g. sidebar switch)
  useEffect(() => {
    setMessages(thread.messages);
    setStreamingContent('');
  }, [thread.video_id]);

  // Scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Auto-resize textarea
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMsg: Message = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setIsStreaming(true);
    setStreamingContent('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoId: thread.video_id, messages: nextMessages }),
      });

      if (!res.ok || !res.body) throw new Error('Chat request failed');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value, { stream: true });
        setStreamingContent(accumulated);
      }

      const assistantMsg: Message = {
        role: 'assistant',
        content: accumulated,
        timestamp: new Date().toISOString(),
      };

      const finalMessages = [...nextMessages, assistantMsg];
      setMessages(finalMessages);
      setStreamingContent('');

      // Refresh thread from server (server persisted after streaming)
      const updated = await fetch(`/api/threads/${thread.video_id}`).then((r) => r.json());
      onUpdate(updated);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Something went wrong. Please try again.', timestamp: new Date().toISOString() },
      ]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = async () => {
    await fetch(`/api/threads/${thread.video_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [] }),
    });
    setMessages([]);
    onUpdate({ ...thread, messages: [], last_updated: new Date().toISOString() });
  };

  const allMessages = isStreaming
    ? [...messages, { role: 'assistant' as const, content: streamingContent, timestamp: '' }]
    : messages;

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <p className="text-[11px] font-semibold text-[#666] uppercase tracking-widest">
          Chat with this video
        </p>
        {messages.length > 0 && (
          <button
            onClick={handleClear}
            className="flex items-center gap-1 text-[#555] hover:text-[#aaa] text-xs transition-colors"
          >
            <Trash2 size={11} /> Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="space-y-3 mb-4">
        {allMessages.length === 0 && (
          <div className="text-center py-10">
            <Bot size={28} className="mx-auto text-[#444] mb-3" />
            <p className="text-[#555] text-sm">Ask anything about this video</p>
            <p className="text-[#3a3a3a] text-xs mt-1">Powered by Groq · llama-3.3-70b</p>
          </div>
        )}

        {allMessages.map((msg, i) => {
          const isUser = msg.role === 'user';
          const isLastAndStreaming = isStreaming && i === allMessages.length - 1 && !isUser;

          return (
            <div key={i} className={`flex gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
              {/* Avatar */}
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                  isUser ? 'bg-[#1c62b9]' : 'bg-[#2a2a2a] border border-[#383838]'
                }`}
              >
                {isUser ? <User size={13} /> : <Bot size={13} className="text-[#cc0000]" />}
              </div>

              {/* Bubble */}
              <div
                className={`max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  isUser
                    ? 'bg-[#1c62b9] text-white rounded-tr-md'
                    : 'bg-[#1e1e1e] text-[#e0e0e0] border border-[#2a2a2a] rounded-tl-md'
                } ${isLastAndStreaming && !streamingContent ? 'animate-pulse' : ''}`}
              >
                {isLastAndStreaming && !streamingContent ? (
                  <span className="flex items-center gap-2 text-[#666]">
                    <Loader2 size={12} className="animate-spin" /> Thinking...
                  </span>
                ) : (
                  <span className={isLastAndStreaming ? 'cursor-blink' : ''}>
                    {msg.content}
                  </span>
                )}
              </div>
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="flex gap-2.5 items-end bg-[#0f0f0f] pt-1">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            resizeTextarea();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about this video..."
          disabled={isStreaming}
          rows={1}
          className="flex-1 bg-[#1e1e1e] border border-[#303030] focus:border-[#444] rounded-2xl px-4 py-3 text-sm text-white placeholder:text-[#444] outline-none resize-none disabled:opacity-50 transition-colors leading-relaxed"
          style={{ minHeight: '48px', maxHeight: '120px' }}
        />
        <button
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
          className="w-11 h-11 rounded-full bg-[#cc0000] hover:bg-[#aa0000] disabled:bg-[#2a2a2a] disabled:text-[#444] text-white flex items-center justify-center shrink-0 transition-colors"
          title="Send (Enter)"
        >
          {isStreaming ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
        </button>
      </div>
      <p className="text-[#333] text-[10px] mt-1.5 text-center">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
