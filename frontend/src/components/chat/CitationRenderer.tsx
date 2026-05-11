/**
 * CitationRenderer — replaces `(Source: xxx)` inline patterns in Markdown
 * with interactive numbered superscript badges.
 *
 * Features:
 * - Numbered references [1], [2], … replace "(Source: fundamentals)" etc.
 * - Clicking a badge shows a popover: source label + always-visible open link
 * - Web search results link to the actual publisher URL (parsed domain shown)
 * - Internal tools (calculator) appear only in the References section at the bottom
 * - References section is always rendered below the report for cross-validation
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, useRef, useEffect, type ReactNode } from "react";
import type { WebSource } from "../../lib/api";

// ---------------------------------------------------------------------------
// Source metadata
// ---------------------------------------------------------------------------

interface SourceMeta {
  label: string;
  icon: string;
  urlFor?: (ticker?: string) => string;  // constructs an external link
  isInternal: boolean;  // if true: hide inline badge, show in References only
}

const TOOL_META: Record<string, SourceMeta> = {
  fundamentals: {
    label: "Yahoo Finance — Fundamentals",
    icon: "📊",
    urlFor: (t) => `https://finance.yahoo.com/quote/${t ?? ""}`,
    isInternal: false,
  },
  price_history: {
    label: "Yahoo Finance — Price History",
    icon: "📈",
    urlFor: (t) => `https://finance.yahoo.com/quote/${t ?? ""}`,
    isInternal: false,
  },
  balance_sheet: {
    label: "Yahoo Finance — Balance Sheet",
    icon: "📋",
    urlFor: (t) => `https://finance.yahoo.com/quote/${t ?? ""}/balance-sheet/`,
    isInternal: false,
  },
  benchmark_lookup: {
    label: "Sector Benchmarks (2024 averages)",
    icon: "📐",
    urlFor: () => "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/pedata.html",
    isInternal: false,
  },
  calculator: {
    label: "Derived Calculation",
    icon: "🧮",
    isInternal: true,  // hidden from inline badges
  },
  web_search: {
    label: "Web Search",
    icon: "🌐",
    isInternal: false,
  },
};

function getDomainLabel(url: string): string {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    const known: Record<string, string> = {
      "finance.yahoo.com": "Yahoo Finance",
      "reuters.com": "Reuters",
      "bloomberg.com": "Bloomberg",
      "marketwatch.com": "MarketWatch",
      "cnbc.com": "CNBC",
      "wsj.com": "Wall Street Journal",
      "ft.com": "Financial Times",
      "seekingalpha.com": "Seeking Alpha",
      "fool.com": "Motley Fool",
      "investopedia.com": "Investopedia",
      "businessinsider.com": "Business Insider",
      "thestreet.com": "The Street",
    };
    return known[host] ?? host;
  } catch {
    return "Source";
  }
}

// ---------------------------------------------------------------------------
// Badge + Popover
// ---------------------------------------------------------------------------

interface BadgeProps {
  n: number;
  sourceTool: string;
  ticker?: string;
  webSources?: WebSource[];
}

function CitationBadge({ n, sourceTool, ticker, webSources }: BadgeProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const meta = TOOL_META[sourceTool] ?? {
    label: sourceTool,
    icon: "📌",
    isInternal: false,
  };

  // For web_search, find the first matching web source
  const webSource = sourceTool === "web_search" && webSources?.length
    ? webSources[0]
    : undefined;

  const linkUrl = webSource?.url
    ?? meta.urlFor?.(ticker)
    ?? null;

  const linkLabel = webSource
    ? getDomainLabel(webSource.url)
    : meta.label;

  return (
    <span ref={ref} className="relative inline">
      <button
        onClick={() => setOpen((v) => !v)}
        className="citation-badge"
        aria-label={`Citation ${n}: ${meta.label}`}
      >
        {n}
      </button>

      {open && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl p-3 text-left animate-fade-in">
          <span className="flex items-start gap-2">
            <span className="text-base flex-shrink-0 mt-0.5">{meta.icon}</span>
            <span className="flex flex-col gap-1 min-w-0">
              <span className="text-[11px] font-medium text-zinc-200 leading-tight">{linkLabel}</span>
              {webSource?.title && (
                <span className="text-[10px] text-zinc-500 leading-tight line-clamp-2">
                  {webSource.title}
                </span>
              )}
              {linkUrl && (
                <a
                  href={linkUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-[10px] text-violet-400 hover:text-violet-300 mt-0.5"
                  onClick={(e) => e.stopPropagation()}
                >
                  Open source
                  <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              )}
            </span>
          </span>
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main renderer
// ---------------------------------------------------------------------------

interface CitationRendererProps {
  content: string;
  ticker?: string;
  webSources?: WebSource[];
}

// Matches: (Source: fundamentals) or (source: web_search) — case-insensitive
const SOURCE_PATTERN = /\(source:\s*([a-z_]+)\)/gi;

function processContent(
  content: string,
  ticker: string | undefined,
  webSources: WebSource[] | undefined,
): { processed: string; refs: Array<{ n: number; tool: string; meta: SourceMeta; url: string | null; webSource?: WebSource }> } {
  const refs: Array<{ n: number; tool: string; meta: SourceMeta; url: string | null; webSource?: WebSource }> = [];
  const toolToN = new Map<string, number>();
  let counter = 0;

  const processed = content.replace(SOURCE_PATTERN, (_match, tool) => {
    const toolKey = tool.toLowerCase();
    const meta = TOOL_META[toolKey] ?? { label: toolKey, icon: "📌", isInternal: false };

    if (meta.isInternal) return "";  // remove internal citations from body

    if (!toolToN.has(toolKey)) {
      counter += 1;
      toolToN.set(toolKey, counter);
      const webSource = toolKey === "web_search" ? webSources?.[refs.filter(r => r.tool === "web_search").length] : undefined;
      const url = webSource?.url ?? meta.urlFor?.(ticker) ?? null;
      refs.push({ n: counter, tool: toolKey, meta, url, webSource });
    }
    const n = toolToN.get(toolKey)!;
    return `[CITATION:${n}:${toolKey}]`;
  });

  return { processed, refs };
}

export function CitationRenderer({ content, ticker, webSources }: CitationRendererProps) {
  const { processed, refs } = processContent(content, ticker, webSources);

  const BADGE_RE = /\[CITATION:(\d+):([a-z_]+)\]/;

  const renderParts = (text: string): ReactNode[] =>
    text.split(/(\[CITATION:\d+:[a-z_]+\])/).map((part, i) => {
      const m = BADGE_RE.exec(part);
      if (m) {
        const n = parseInt(m[1], 10);
        const tool = m[2];
        const refEntry = refs.find((r) => r.n === n && r.tool === tool);
        const webSource = refEntry?.webSource ?? webSources?.find((ws) => ws.ticker === ticker);
        return (
          <CitationBadge
            key={`badge-${n}-${i}`}
            n={n}
            sourceTool={tool}
            ticker={ticker}
            webSources={webSource ? [webSource] : webSources}
          />
        );
      }
      return part;
    });

  // Custom paragraph renderer that injects badges inline
  const components = {
    p: ({ children }: { children?: ReactNode }) => {
      const childStr = String(children ?? "");
      if (!childStr.includes("[CITATION:")) {
        return <p>{children}</p>;
      }
      return <p>{renderParts(childStr)}</p>;
    },
    li: ({ children }: { children?: ReactNode }) => {
      const childStr = String(children ?? "");
      if (!childStr.includes("[CITATION:")) {
        return <li>{children}</li>;
      }
      return <li>{renderParts(childStr)}</li>;
    },
  };

  return (
    <>
      <div className="prose prose-sm prose-invert max-w-none text-zinc-200 leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components as never}>
          {processed}
        </ReactMarkdown>
      </div>

      {/* References section — always visible for cross-validation */}
      {refs.length > 0 && (
        <div className="mt-4 pt-3 border-t border-zinc-800">
          <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-2 font-medium">References</p>
          <ol className="space-y-1.5 list-none m-0 p-0">
            {refs.map((ref) => (
              <li key={ref.n} className="flex items-start gap-2 text-[11px]">
                <span className="citation-badge flex-shrink-0 mt-0.5">{ref.n}</span>
                <span className="text-zinc-500">
                  <span className="text-zinc-400 font-medium mr-1">
                    {ref.webSource ? getDomainLabel(ref.webSource.url) : ref.meta.label}
                  </span>
                  {ref.url && (
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-violet-500 hover:text-violet-400 break-all"
                    >
                      {ref.webSource
                        ? new URL(ref.url).hostname.replace("www.", "")
                        : ref.url.replace("https://", "")}
                    </a>
                  )}
                  {!ref.url && (
                    <span className="text-zinc-700">Internal</span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </>
  );
}
