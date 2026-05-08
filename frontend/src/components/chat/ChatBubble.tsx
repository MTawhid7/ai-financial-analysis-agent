import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PlotlyChart } from "../PlotlyChart";
import { ExportMenu } from "./ExportMenu";
import { ProvenancePanel } from "./ProvenancePanel";
import { submitFeedback } from "../../lib/api";
import type { StepEvent, ChartDescriptor } from "../../hooks/useStreamingChat";

interface UserBubbleProps {
  content: string;
}

interface AssistantBubbleProps {
  content: string;
  messageIndex?: number;
  conversationId?: string;
  isStreaming?: boolean;
  stepEvents?: StepEvent[];
  charts?: ChartDescriptor[];
  reportId?: string | null;
  existingRating?: 1 | -1 | null;
}

export function UserBubble({ content }: UserBubbleProps) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[78%] rounded-2xl rounded-tr-sm bg-violet-600 px-4 py-2.5 text-sm text-white shadow-sm whitespace-pre-wrap break-words leading-relaxed">
        {content}
      </div>
    </div>
  );
}

export function AssistantBubble({
  content,
  messageIndex,
  conversationId,
  isStreaming,
  stepEvents,
  charts,
  reportId,
  existingRating,
}: AssistantBubbleProps) {
  const [rating, setRating] = useState<1 | -1 | null>(existingRating ?? null);
  const [showSources, setShowSources] = useState(false);

  const handleRating = async (value: 1 | -1) => {
    if (rating !== null || isStreaming) return;
    if (conversationId === undefined || messageIndex === undefined) return;
    setRating(value);
    await submitFeedback(conversationId, messageIndex, value).catch(() => {});
  };

  const showActions = !isStreaming && content;
  const showFeedback = showActions && conversationId !== undefined && messageIndex !== undefined;

  return (
    <div className="flex gap-3 group">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-sm mt-0.5">
        📊
      </div>
      <div className="flex-1 min-w-0 space-y-2">
        {/* Live tool-step indicators */}
        {(isStreaming || (stepEvents && stepEvents.length > 0)) && (
          <div className="space-y-1 font-mono">
            {stepEvents?.map((evt, i) => (
              <div key={i} className="flex items-center gap-1.5 text-xs text-zinc-500">
                <span className={evt.ok !== false ? "text-emerald-500" : "text-red-500"}>
                  {evt.ok !== false ? "✓" : "✗"}
                </span>
                <span>[{evt.step}] {evt.agent} → {evt.tool}{evt.cache_hit ? " (cached)" : ""}</span>
              </div>
            ))}
            {isStreaming && (
              <div className="flex items-center gap-1.5 text-xs text-zinc-600">
                <span className="animate-pulse">●</span>
                <span>Processing…</span>
              </div>
            )}
          </div>
        )}

        {/* Rendered markdown report */}
        {content && (
          <div className="prose prose-sm prose-invert max-w-none text-zinc-200">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        )}

        {/* Charts */}
        {!isStreaming && charts && charts.length > 0 && (
          <div className="space-y-1 mt-2">
            {charts.map((chart, i) => (
              <PlotlyChart key={i} figure={chart.figure} title={chart.title} />
            ))}
          </div>
        )}

        {/* Export menu */}
        {!isStreaming && reportId && <ExportMenu reportId={reportId} />}

        {/* Action bar — feedback + provenance */}
        {showActions && (
          <div className="flex items-center gap-3 pt-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {showFeedback && (
              <div className="flex items-center gap-1">
                <button
                  onClick={() => handleRating(1)}
                  disabled={rating !== null}
                  title="Helpful"
                  className={`p-1 rounded text-xs transition-colors ${
                    rating === 1
                      ? "text-emerald-400"
                      : "text-zinc-600 hover:text-emerald-400 disabled:cursor-not-allowed"
                  }`}
                >
                  👍
                </button>
                <button
                  onClick={() => handleRating(-1)}
                  disabled={rating !== null}
                  title="Not helpful"
                  className={`p-1 rounded text-xs transition-colors ${
                    rating === -1
                      ? "text-red-400"
                      : "text-zinc-600 hover:text-red-400 disabled:cursor-not-allowed"
                  }`}
                >
                  👎
                </button>
              </div>
            )}

            {/* View Sources button (only for analysis reports) */}
            {reportId && (
              <button
                onClick={() => setShowSources((v) => !v)}
                className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors flex items-center gap-1"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                {showSources ? "Hide Sources" : "View Sources"}
              </button>
            )}
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
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-sm">
        📊
      </div>
      <div className="flex items-center gap-1 py-2">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
