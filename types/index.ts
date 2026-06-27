export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface Thread {
  video_id: string;
  url: string;
  title: string;
  author: string;
  thumbnail: string;
  transcript: string;
  summary: string;
  created_at: string;
  last_updated: string;
  messages: Message[];
}
