import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NextPick — 2-Stage Real-Time Recommender",
  description: "Two-Tower + Qdrant HNSW + Redis + LightGBM Pipeline Inspector",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
