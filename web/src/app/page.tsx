"use client";

import { useCallback, useEffect, useState } from "react";

interface RecommendedItem {
  article_id: string;
  prod_name: string;
  product_type_name?: string;
  product_group_name: string;
  department_name: string;
  price: number;
  image_url?: string;
  stage1_rank: number;
  stage2_rank: number;
  rank_delta: number;
  stage1_cosine_score: number;
  stage2_logit: number;
}

interface LatencyBreakdown {
  redis: number;
  onnx: number;
  qdrant: number;
  lightgbm: number;
  total: number;
}

interface RecommendResponse {
  session_id: string;
  mode: string;
  history_len: number;
  items: RecommendedItem[];
  latency_ms: LatencyBreakdown;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SEED_CATALOG: RecommendedItem[] = [
  {
    article_id: "0706016001",
    prod_name: "Jade HW Skinny Denim",
    product_group_name: "Garment Lower body",
    department_name: "Trouser",
    price: 0.033,
    stage1_rank: 1,
    stage2_rank: 1,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0372860001",
    prod_name: "7p Basic Low Socks",
    product_group_name: "Socks & Tights",
    department_name: "Socks",
    price: 0.013,
    stage1_rank: 2,
    stage2_rank: 2,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0610776002",
    prod_name: "Tilly Oversized Tee",
    product_group_name: "Garment Upper body",
    department_name: "Jersey Basic",
    price: 0.016,
    stage1_rank: 3,
    stage2_rank: 3,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0759871002",
    prod_name: "Tilda Hooded Sweatshirt",
    product_group_name: "Garment Upper body",
    department_name: "Jersey",
    price: 0.042,
    stage1_rank: 4,
    stage2_rank: 4,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
];

export default function HomePage() {
  const [sessionId, setSessionId] = useState("demo_user_1");
  const [productGroup, setProductGroup] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [data, setData] = useState<RecommendResponse | null>(null);
  const [clickedTrail, setClickedTrail] = useState<RecommendedItem[]>([]);
  const [apiOnline, setApiOnline] = useState(true);
  const [loading, setLoading] = useState(false);

  const fetchRecommendations = useCallback(
    async (sid: string, grp: string, priceCap: string) => {
      setLoading(true);
      try {
        const params = new URLSearchParams({ session_id: sid, top_k: "10" });
        if (grp) params.set("product_group", grp);
        if (priceCap) params.set("max_price", priceCap);
        const res = await fetch(`${API_BASE}/recommend?${params.toString()}`);
        if (!res.ok) throw new Error("API error");
        const json: RecommendResponse = await res.json();
        setData(json);
        setApiOnline(true);
      } catch {
        setApiOnline(false);
        setData({
          session_id: sid,
          mode: "offline_preview",
          history_len: clickedTrail.length,
          items: SEED_CATALOG,
          latency_ms: { redis: 0, onnx: 0, qdrant: 0, lightgbm: 0, total: 0 },
        });
      } finally {
        setLoading(false);
      }
    },
    [clickedTrail.length]
  );

  useEffect(() => {
    fetchRecommendations(sessionId, productGroup, maxPrice);
  }, [sessionId, productGroup, maxPrice, fetchRecommendations]);

  const handleProductClick = async (item: RecommendedItem) => {
    setClickedTrail((prev) => [item, ...prev].slice(0, 20));
    try {
      await fetch(`${API_BASE}/event`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          article_id: item.article_id,
          price: item.price,
          department_name: item.department_name,
          product_group_name: item.product_group_name,
        }),
      });
    } catch {
      // Offline preview fallback handled by fetchRecommendations
    }
    await fetchRecommendations(sessionId, productGroup, maxPrice);
  };

  const resetSession = () => {
    const nextId = `demo_${Math.random().toString(36).slice(2, 8)}`;
    setClickedTrail([]);
    setSessionId(nextId);
  };

  const items = data?.items ?? SEED_CATALOG;
  const lat = data?.latency_ms ?? { redis: 0, onnx: 0, qdrant: 0, lightgbm: 0, total: 0 };

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5 mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            NextPick <span className="text-blue-400 text-base font-medium">2-Stage Recommender</span>
          </h1>
          <p className="text-sm text-slate-400">
            Click any product to update the Redis 20-item window and watch Stage-1 → Stage-2 re-ranking live.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={productGroup}
            onChange={(e) => setProductGroup(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200"
          >
            <option value="">All Categories</option>
            <option value="Garment Upper body">Upper Body</option>
            <option value="Garment Lower body">Lower Body</option>
            <option value="Socks & Tights">Socks &amp; Tights</option>
          </select>
          <input
            type="number"
            step="0.01"
            placeholder="Max price"
            value={maxPrice}
            onChange={(e) => setMaxPrice(e.target.value)}
            className="w-28 bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200"
          />
          <button
            onClick={resetSession}
            className="bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm font-medium text-white"
          >
            New Session
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Panel: Storefront Recommendations */}
        <section className="lg:col-span-7">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Recommended For You ({items.length})
            </h2>
            {loading && <span className="text-xs text-blue-400">Refreshing...</span>}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {items.map((item) => (
              <button
                key={item.article_id}
                onClick={() => handleProductClick(item)}
                className="text-left bg-slate-900/90 hover:bg-slate-800/90 border border-slate-800 hover:border-blue-500/60 rounded-xl p-4 transition"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    #{item.stage2_rank}
                  </span>
                  {item.rank_delta !== 0 && (
                    <span
                      className={`text-xs font-mono px-2 py-0.5 rounded ${
                        item.rank_delta > 0
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-700/50"
                          : "bg-amber-950 text-amber-300 border border-amber-700/50"
                      }`}
                    >
                      S1 #{item.stage1_rank} → S2 #{item.stage2_rank} ({item.rank_delta > 0 ? `+${item.rank_delta}` : item.rank_delta})
                    </span>
                  )}
                </div>
                <div className="font-semibold text-white truncate">{item.prod_name || item.article_id}</div>
                <div className="text-xs text-slate-400 mt-1">
                  {item.department_name} · {item.product_group_name}
                </div>
                <div className="mt-3 flex items-center justify-between text-xs">
                  <span className="font-mono text-blue-300">Price: {item.price.toFixed(3)}</span>
                  <span className="text-slate-500 font-mono">ID {item.article_id}</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        {/* Right Panel: Pipeline Inspector */}
        <aside className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Pipeline Inspector</h2>
              <span
                className={`text-xs px-2 py-0.5 rounded font-mono ${
                  apiOnline ? "bg-emerald-950 text-emerald-300" : "bg-amber-950 text-amber-300"
                }`}
              >
                {data?.mode ?? "connecting"}
              </span>
            </div>

            <div className="text-xs text-slate-400 font-mono mb-3">
              session={sessionId} · redis_window={data?.history_len ?? clickedTrail.length}/20
            </div>

            <div className="grid grid-cols-5 gap-2 text-center bg-slate-950 border border-slate-800 rounded-lg p-2.5">
              <div>
                <div className="text-[11px] text-slate-500">Redis</div>
                <div className="text-xs font-mono text-slate-200">{lat.redis.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">ONNX</div>
                <div className="text-xs font-mono text-slate-200">{lat.onnx.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">Qdrant</div>
                <div className="text-xs font-mono text-slate-200">{lat.qdrant.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">LGBM</div>
                <div className="text-xs font-mono text-slate-200">{lat.lightgbm.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-blue-400">Total</div>
                <div className="text-xs font-mono font-bold text-blue-300">{lat.total.toFixed(1)}ms</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Stage-1 Cosine vs Stage-2 Logit
            </h3>
            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
              {items.slice(0, 8).map((item) => (
                <div
                  key={`insp-${item.article_id}`}
                  className="flex items-center justify-between text-xs bg-slate-950/70 rounded px-2.5 py-1.5 font-mono"
                >
                  <span className="truncate max-w-[160px] text-slate-200">
                    #{item.stage2_rank} {item.prod_name || item.article_id}
                  </span>
                  <span className="text-slate-400">
                    cos={item.stage1_cosine_score.toFixed(3)} | lgb={item.stage2_logit.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Live Session Trail ({clickedTrail.length}/20)
            </h3>
            {clickedTrail.length === 0 ? (
              <p className="text-xs text-slate-500">Click any product card on the left to start a live session.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {clickedTrail.map((c, idx) => (
                  <span
                    key={`${c.article_id}-${idx}`}
                    className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono"
                  >
                    {c.prod_name || c.article_id}
                  </span>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
