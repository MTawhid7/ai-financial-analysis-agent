import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PlotlyChart } from "../PlotlyChart";
import { ExportMenu } from "./ExportMenu";
import { ProvenancePanel } from "./ProvenancePanel";
import { CitationRenderer } from "./CitationRenderer";
import type { StepEvent, ChartDescriptor } from "../../hooks/useStreamingChat";

interface UserBubbleProps {
  content: string;
}

interface AssistantBubbleProps {
  content: string;
  isStreaming?: boolean;
  stepEvents?: StepEvent[];
  charts?: ChartDescriptor[];
  reportId?: string | null;
}

export function UserBubble({ content }: UserBubbleProps) {
  return (
    <div className="flex justify-end animate-fade-in-up">
      <div className="max-w-[78%] rounded-2xl rounded-tr-sm bg-violet-600 px-4 py-3 text-sm text-white shadow-sm whitespace-pre-wrap break-words leading-relaxed">
        {content}
      </div>
    </div>
  );
}

export function AssistantBubble({
  content,
  isStreaming,
  stepEvents,
  charts,
  reportId,
}: AssistantBubbleProps) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="flex gap-3 group animate-fade-in-up">
      {/* Agent avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-800 flex items-center justify-center mt-0.5">
        <svg className="w-4 h-4 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {/* Live tool-step indicators (shown during and after streaming) */}
        {(isStreaming || (stepEvents && stepEvents.length > 0)) && (
          <div className="space-y-0.5 font-mono">
            {stepEvents?.map((evt, i) => (
              <div key={i} className="flex items-center gap-1.5 text-[11px] text-zinc-600">
                <span className={evt.ok !== false ? "text-emerald-600" : "text-red-500"}>
                  {evt.ok !== false ? "✓" : "✗"}
                </span>
                <span>
                  [{evt.step}] {evt.agent} → {evt.tool}
                  {evt.cache_hit ? " · cached" : ""}
                </span>
              </div>
            ))}
            {isStreaming && (
              <div className="flex items-center gap-1.5 text-[11px] text-zinc-700">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" />
                <span>Working…</span>
              </div>
            )}
          </div>
        )}

        {/* Report content — with citation renderer for (Source: xxx) patterns */}
        {content && (
          content.includes("(Source:") || content.includes("(source:")
            ? <CitationRenderer content={content} />
            : (
              <div className="prose prose-sm prose-invert max-w-none text-zinc-200 leading-relaxed">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              </div>
            )
        )}

        {/* Interactive Plotly charts */}
        {!isStreaming && charts && charts.length > 0 && (
          <div className="space-y-2">
            {charts.map((chart, i) => (
              <PlotlyChart key={i} figure={chart.figure} title={chart.title} />
            ))}
          </div>
        )}

        {/* Export menu (PDF / Word / Excel) */}
        {!isStreaming && reportId && <ExportMenu reportId={reportId} />}

        {/* Action bar — View Sources only (feedback removed) */}
        {!isStreaming && content && reportId && (
          <div className="flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
            <button
              onClick={() => setShowSources((v) => !v)}
              className="flex items-center gap-1 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {showSources ? "Hide sources" : "View sources"}
            </button>
          </div>
        )}

        {/* Provenance panel */}
        {showSources && reportId && <ProvenancePanel reportId={reportId} />}
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-800 flex items-center justify-center">
        <svg className="w-4 h-4 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
      </div>
      <div className="flex items-center gap-1 py-2">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
