import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Talk to YouTube',
  description: 'Chat with any YouTube video using AI — powered by Groq',
  icons: { icon: 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>▶️</text></svg>' },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-[#0f0f0f] text-white antialiased">{children}</body>
    </html>
  );
}
