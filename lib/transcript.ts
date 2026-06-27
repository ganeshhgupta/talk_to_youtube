import { YoutubeTranscript } from 'youtube-transcript';

export function extractVideoId(url: string): string | null {
  const patterns = [
    /(?:v=|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})/,
    /^([a-zA-Z0-9_-]{11})$/,
  ];
  for (const pattern of patterns) {
    const m = url.trim().match(pattern);
    if (m) return m[1];
  }
  return null;
}

export async function fetchTranscript(videoId: string): Promise<string> {
  const items = await YoutubeTranscript.fetchTranscript(videoId);
  return items.map((i) => i.text).join(' ');
}

export async function fetchVideoInfo(
  videoId: string
): Promise<{ title: string; author: string; thumbnail: string }> {
  try {
    const res = await fetch(
      `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`
    );
    const data = (await res.json()) as { title?: string; author_name?: string };
    return {
      title: data.title ?? 'Unknown Title',
      author: data.author_name ?? 'Unknown Channel',
      thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
    };
  } catch {
    return {
      title: 'Unknown Title',
      author: 'Unknown Channel',
      thumbnail: `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`,
    };
  }
}
