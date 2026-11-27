"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface RecommendedItem {
  article_id: string;
  prod_name: string;
  product_type_name?: string;
  product_group_name: string;
  index_group_name?: string;
  department_name: string;
  colour_group_name?: string;
  price: number;
  image_url?: string;
  stage1_rank: number;
  stage2_rank: number;
  rank_delta: number;
  stage1_cosine_score: number;
  stage2_logit: number;
}

const SWATCH_COLORS: Record<string, string> = {
  Black: "from-slate-800 to-slate-950",
  White: "from-slate-200 to-slate-400 text-slate-900",
  Blue: "from-blue-700 to-indigo-950",
  "Dark Blue": "from-blue-900 to-slate-950",
  Beige: "from-amber-200/80 to-stone-400 text-slate-900",
  Grey: "from-zinc-500 to-zinc-800",
  Red: "from-rose-700 to-red-950",
  Green: "from-emerald-700 to-teal-950",
  Pink: "from-pink-500 to-rose-900",
};

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
  ranker_mode?: string;
  is_synthetic_demo?: boolean;
  history_len: number;
  items: RecommendedItem[];
  latency_ms: LatencyBreakdown;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const IMAGE_CDN_BASE =
  process.env.NEXT_PUBLIC_IMAGE_CDN ??
  "https://raw.githubusercontent.com/AmoghRG/nextpick-images/main";
const INR_SCALE = 50000;

function formatInr(normPrice: number): string {
  return `~₹${Math.round(normPrice * INR_SCALE).toLocaleString("en-IN")}`;
}

function getImageUrl(articleId: string): string {
  const padded = articleId.padStart(10, "0");
  return `${IMAGE_CDN_BASE}/${padded.slice(0, 3)}/${padded}.jpg`;
}

function ProductThumb({ item }: { item: RecommendedItem }) {
  const [imgFailed, setImgFailed] = useState(false);

  if (imgFailed) {
    return (
      <div
        className={`w-20 h-28 rounded-lg bg-gradient-to-br ${
          SWATCH_COLORS[item.colour_group_name ?? ""] ?? "from-slate-600 to-slate-800"
        } text-white overflow-hidden shrink-0 flex flex-col items-center justify-center border border-slate-200 px-1.5 text-center shadow-inner`}
      >
        <span className="text-[10px] font-bold uppercase tracking-wider leading-tight">
          {(item.product_type_name || item.department_name || "SKU").slice(0, 8)}
        </span>
        <span className="text-[9px] opacity-85 mt-1 truncate max-w-full">
          {item.colour_group_name || "Fashion"}
        </span>
      </div>
    );
  }

  return (
    <div className="w-20 h-28 rounded-lg bg-slate-100 overflow-hidden shrink-0 border border-slate-200 shadow-inner">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={getImageUrl(item.article_id)}
        alt={item.prod_name || item.article_id}
        loading="lazy"
        onError={() => setImgFailed(true)}
        className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-200"
      />
    </div>
  );
}

function inferSegmentBadge(item: RecommendedItem): string | null {
  const idxGrp = (item.index_group_name ?? "").toLowerCase();
  const text = `${item.department_name} ${item.prod_name ?? ""}`.toLowerCase();
  const hasWomen = /\b(ladies|women|woman|mama|mom|hw|petite|bra|skirt|dress|blouse|tights)\b/.test(text);
  const hasMen = /\b(mens|men|boxer|chino)\b/.test(text);

  if (idxGrp === "ladieswear") return "Women";
  if (idxGrp === "menswear") return "Men";
  if (idxGrp === "baby/children") return "Kids";
  if (idxGrp === "divided") {
    if (hasWomen) return "Women · Divided";
    if (hasMen) return "Men · Divided";
    return "Unisex · Divided";
  }
  if (idxGrp === "sport") {
    if (hasWomen) return "Women · Sport";
    if (hasMen) return "Men · Sport";
    return "Unisex · Sport";
  }
  if (hasWomen) return "Women";
  if (hasMen) return "Men";
  if (/\b(baby|kids|boy|boys|girl|girls|junior)\b/.test(text)) return "Kids";
  return null;
}

const SEED_CATALOG: RecommendedItem[] = [
  {
    article_id: "0751471001",
    prod_name: "Pluto RW slacks (1)",
    product_group_name: "Garment Lower body",
    index_group_name: "Ladieswear",
    department_name: "Trouser",
    price: 0.033,
    stage1_rank: 1,
    stage2_rank: 1,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0706016001",
    prod_name: "Jade HW Skinny Denim TRS",
    product_group_name: "Garment Lower body",
    index_group_name: "Divided",
    department_name: "Trousers",
    price: 0.033,
    stage1_rank: 2,
    stage2_rank: 2,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0685814001",
    prod_name: "RICHIE HOOD",
    product_group_name: "Garment Upper body",
    index_group_name: "Menswear",
    department_name: "Heavy Basic Jersey",
    price: 0.033,
    stage1_rank: 3,
    stage2_rank: 3,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
  {
    article_id: "0916468003",
    prod_name: "Bailey",
    product_group_name: "Garment Upper body",
    index_group_name: "Ladieswear",
    department_name: "Knitwear",
    price: 0.033,
    stage1_rank: 4,
    stage2_rank: 4,
    rank_delta: 0,
    stage1_cosine_score: 0.0,
    stage2_logit: 0.0,
  },
];

export default function HomePage() {
  const [sessionId, setSessionId] = useState("");
  const [productGroup, setProductGroup] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [data, setData] = useState<RecommendResponse | null>(null);
  const [clickedTrail, setClickedTrail] = useState<RecommendedItem[]>([]);
  const [apiOnline, setApiOnline] = useState(true);
  const [eventError, setEventError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showGuide, setShowGuide] = useState(true);
  const reqSeqRef = useRef(0);

  const fetchRecommendations = useCallback(
    async (sid: string, grp: string, priceCap: string) => {
      const seq = ++reqSeqRef.current;
      setLoading(true);
      try {
        const params = new URLSearchParams({ session_id: sid, top_k: "10" });
        if (grp) params.set("product_group", grp);
        const parsedInr = Number(priceCap);
        if (priceCap && Number.isFinite(parsedInr) && parsedInr > 0) {
          params.set("max_price", (parsedInr / INR_SCALE).toFixed(6));
        }
        const res = await fetch(`${API_BASE}/recommend?${params.toString()}`);
        if (!res.ok) throw new Error(`API error (${res.status})`);
        const json: RecommendResponse = await res.json();
        if (seq !== reqSeqRef.current) return;
        setData(json);
        setApiOnline(true);
      } catch {
        if (seq !== reqSeqRef.current) return;
        setApiOnline(false);
        setData({
          session_id: sid,
          mode: "offline_preview",
          history_len: clickedTrail.length,
          items: SEED_CATALOG,
          latency_ms: { redis: 0, onnx: 0, qdrant: 0, lightgbm: 0, total: 0 },
        });
      } finally {
        if (seq === reqSeqRef.current) setLoading(false);
      }
    },
    [clickedTrail.length]
  );

  useEffect(() => {
    setSessionId(`demo_${Math.random().toString(36).slice(2, 8)}`);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    fetchRecommendations(sessionId, productGroup, maxPrice);
  }, [sessionId, productGroup, maxPrice, fetchRecommendations]);

  const handleProductClick = async (item: RecommendedItem) => {
    setEventError(null);
    setClickedTrail((prev) => [item, ...prev].slice(0, 20));
    try {
      const evRes = await fetch(`${API_BASE}/event`, {
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
      if (!evRes.ok) {
        throw new Error(`POST /event failed with HTTP ${evRes.status}`);
      }
    } catch (err) {
      setEventError(err instanceof Error ? err.message : "Failed to record click event");
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
    <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-5 mb-5">
        <div>
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">NextPick</h1>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
              2-Stage Real-Time Recommender
            </span>
          </div>
          <p className="text-sm text-slate-600 mt-1">
            Live fashion storefront powered by{" "}
            <span className="font-medium text-slate-800">Redis &rarr; ONNX Two-Tower &rarr; Qdrant HNSW &rarr; LightGBM</span>{" "}
            across 34,469 H&amp;M SKUs.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            aria-label="Filter by category"
            value={productGroup}
            onChange={(e) => setProductGroup(e.target.value)}
            className="bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-sm text-slate-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">All Categories</option>
            <option value="Garment Upper body">Upper Body</option>
            <option value="Garment Lower body">Lower Body</option>
            <option value="Socks & Tights">Socks &amp; Tights</option>
          </select>
          <input
            type="number"
            step="100"
            min="0"
            placeholder="Max price (₹)"
            value={maxPrice}
            onChange={(e) => setMaxPrice(e.target.value)}
            className="w-32 bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-sm text-slate-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="button"
            onClick={() => setShowGuide((v) => !v)}
            className="bg-white hover:bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition"
          >
            {showGuide ? "Hide Overview" : "How it Works"}
          </button>
          <button
            type="button"
            onClick={resetSession}
            className="bg-slate-900 hover:bg-slate-800 rounded-lg px-3.5 py-1.5 text-sm font-medium text-white shadow-sm transition"
          >
            New Session
          </button>
        </div>
      </header>

      {showGuide && (
        <section className="mb-6 rounded-2xl bg-gradient-to-br from-white to-slate-50 border border-slate-200 p-5 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
            <div className="max-w-2xl">
              <div className="text-[11px] font-bold uppercase tracking-wider text-indigo-600 mb-1">
                Real-Time Fashion Retrieval &amp; Ranking Demo
              </div>
              <h2 className="text-lg font-semibold text-slate-900">
                Watch AI re-rank 34,469 H&amp;M products in milliseconds as you browse.
              </h2>
              <p className="text-xs text-slate-600 mt-1">
                Click any product card below to simulate a shopper view. Even <strong className="text-slate-800">1 click</strong> activates
                two-stage personalization — pick <strong className="text-slate-800">1–3 items</strong> (up to 20 remembered in Redis) to see
                how recommendations adapt to your style.
              </p>
            </div>
            <div className="flex items-center gap-2 text-[11px] font-mono text-slate-600 bg-white border border-slate-200 rounded-lg px-3 py-2">
              <span className="font-semibold text-slate-900">Est. ₹</span>
              <span>= H&amp;M normalized &times; 50,000</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-600">
            <div className="bg-white rounded-xl p-3.5 border border-slate-200/80">
              <div className="font-semibold text-slate-900 mb-1">1. Pick 1–3 Clothes Below</div>
              Starts in <span className="font-mono text-emerald-700">cold_start_trending</span> (<span className="font-mono">0/20</span>).
              Clicking 1 item writes to Upstash Redis and triggers <span className="font-mono text-indigo-700">two_stage_personalized</span>.
            </div>
            <div className="bg-white rounded-xl p-3.5 border border-slate-200/80">
              <div className="font-semibold text-slate-900 mb-1">2. Stage-1 &rarr; Stage-2 Re-ranking</div>
              ONNX Two-Tower + Qdrant HNSW retrieves Top-100 candidates, then LightGBM LambdaRank re-ranks them (<span className="font-mono">S1 &rarr; S2</span>).
            </div>
            <div className="bg-white rounded-xl p-3.5 border border-slate-200/80">
              <div className="font-semibold text-slate-900 mb-1">3. Filter, Inspect &amp; Reset</div>
              Filter by category or ₹ budget anytime, inspect live latencies on the right, or click{" "}
              <strong className="text-slate-900">New Session</strong> to reset to <span className="font-mono">0/20</span>.
            </div>
          </div>
        </section>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {eventError && (
          <div className="lg:col-span-12 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-lg px-4 py-2 font-mono">
            {eventError}
          </div>
        )}
        {/* Left Panel: Storefront Recommendations */}
        <section className="lg:col-span-7">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2.5">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Recommended For You ({data ? items.length : "..."})
              </h2>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                {(data?.history_len ?? clickedTrail.length) === 0
                  ? "0/20 clicks · Pick 1–3 items to personalize"
                  : `${data?.history_len ?? clickedTrail.length}/20 clicks · Personalized active`}
              </span>
            </div>
            {loading && (
              <span className="text-xs font-medium text-indigo-600">
                {data ? "Updating recommendations..." : "Connecting to ML API..."}
              </span>
            )}
          </div>

          {!data && loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div
                  key={`skel-${idx}`}
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex gap-3.5 items-start animate-pulse"
                >
                  <div className="w-20 h-28 rounded-lg bg-slate-200 shrink-0" />
                  <div className="flex-1 space-y-2.5 py-1">
                    <div className="h-4 bg-slate-200 rounded w-20" />
                    <div className="h-4 bg-slate-200 rounded w-3/4" />
                    <div className="h-3 bg-slate-100 rounded w-1/2" />
                    <div className="h-4 bg-slate-100 rounded w-1/3 mt-3" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {items.map((item) => {
              const segment = inferSegmentBadge(item);
              const isClicked = clickedTrail.some((c) => c.article_id === item.article_id);
              return (
              <button
                key={item.article_id}
                onClick={() => handleProductClick(item)}
                className={`group text-left bg-white hover:bg-indigo-50/30 border rounded-xl p-4 shadow-sm hover:shadow-md transition flex gap-3.5 items-start ${
                  isClicked ? "border-indigo-300 ring-1 ring-indigo-100" : "border-slate-200 hover:border-indigo-400"
                }`}
              >
                <ProductThumb item={item} />

                <div className="min-w-0 flex-1">
                  <div className="flex items-start justify-between gap-2 mb-1.5 flex-wrap">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                        #{item.stage2_rank}
                      </span>
                      {segment && (
                        <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                          {segment}
                        </span>
                      )}
                      {isClicked && (
                        <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                          Clicked ✓
                        </span>
                      )}
                    </div>
                    {item.rank_delta !== 0 && (
                      <span
                        className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded ${
                          item.rank_delta > 0
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                            : "bg-amber-50 text-amber-700 border border-amber-200"
                        }`}
                      >
                        S1 #{item.stage1_rank} &rarr; S2 #{item.stage2_rank} ({item.rank_delta > 0 ? `+${item.rank_delta}` : item.rank_delta})
                      </span>
                    )}
                  </div>
                  <div className="font-semibold text-slate-900 group-hover:text-indigo-700 truncate">
                    {item.prod_name || item.article_id}
                  </div>
                  <div className="text-xs text-slate-500 mt-1 truncate">
                    {item.department_name} &middot; {item.product_group_name}
                  </div>
                  <div className="mt-2.5 flex items-center justify-between text-xs gap-2">
                    <span className="font-mono font-semibold text-indigo-700">
                      {formatInr(item.price)}{" "}
                      <span className="text-[10px] font-normal text-slate-400">({item.price.toFixed(3)})</span>
                    </span>
                    <span className="text-slate-400 font-mono truncate">ID {item.article_id}</span>
                  </div>
                </div>
              </button>
              );
            })}
          </div>
          )}
        </section>

        {/* Right Panel: Pipeline Inspector */}
        <aside className="lg:col-span-5 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">Pipeline Inspector</h2>
              <span
                className={`text-xs px-2.5 py-0.5 rounded-full font-mono font-medium ${
                  apiOnline && data?.ranker_mode === "trained_lightgbm"
                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    : "bg-amber-50 text-amber-700 border border-amber-200"
                }`}
              >
                {data?.mode ?? "connecting"} &middot; {data?.ranker_mode ?? (apiOnline ? "unknown" : "offline_preview")}
              </span>
            </div>

            {(!apiOnline || data?.is_synthetic_demo || (data?.ranker_mode && data.ranker_mode !== "trained_lightgbm")) && (
              <div className="mb-3 rounded-lg bg-amber-50 border border-amber-200 px-3 py-1.5 text-[11px] font-mono text-amber-800">
                {!apiOnline
                  ? "Offline preview mode — showing static seed items (API unreachable)."
                  : data?.is_synthetic_demo
                  ? "SYNTHETIC DEMO — RANDOM EMBEDDINGS / UNTRAINED MODEL (heuristic fallback)."
                  : "Heuristic fallback active (trained LightGBM model not loaded)."}
              </div>
            )}

            <div className="text-xs text-slate-500 font-mono mb-3">
              session={sessionId} &middot; redis_window={data?.history_len ?? clickedTrail.length}/20
            </div>

            <div className="grid grid-cols-5 gap-2 text-center bg-slate-50 border border-slate-200 rounded-lg p-2.5">
              <div>
                <div className="text-[11px] text-slate-500">Redis</div>
                <div className="text-xs font-mono font-semibold text-slate-800">{lat.redis.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">ONNX</div>
                <div className="text-xs font-mono font-semibold text-slate-800">{lat.onnx.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">Qdrant</div>
                <div className="text-xs font-mono font-semibold text-slate-800">{lat.qdrant.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">LGBM</div>
                <div className="text-xs font-mono font-semibold text-slate-800">{lat.lightgbm.toFixed(1)}ms</div>
              </div>
              <div>
                <div className="text-[11px] font-semibold text-indigo-600">Total</div>
                <div className="text-xs font-mono font-bold text-indigo-700">{lat.total.toFixed(1)}ms</div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
              Stage-1 Cosine vs Stage-2 Logit
            </h3>
            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
              {items.slice(0, 8).map((item) => (
                <div
                  key={`insp-${item.article_id}`}
                  className="flex items-center justify-between text-xs bg-slate-50 border border-slate-100 rounded px-2.5 py-1.5 font-mono"
                >
                  <span className="truncate max-w-[160px] text-slate-800 font-medium">
                    #{item.stage2_rank} {item.prod_name || item.article_id}
                  </span>
                  <span className="text-slate-500">
                    cos={item.stage1_cosine_score.toFixed(3)} | lgb={item.stage2_logit.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
              Live Session Trail ({clickedTrail.length}/20)
            </h3>
            {clickedTrail.length === 0 ? (
              <p className="text-xs text-slate-500">Click any product card on the left to start a live session.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {clickedTrail.map((c, idx) => (
                  <span
                    key={`${c.article_id}-${idx}`}
                    className="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded font-mono"
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
