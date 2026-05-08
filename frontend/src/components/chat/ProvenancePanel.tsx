import { useQuery } from "@tanstack/react-query";
import { getReportSources } from "../../lib/api";

interface Props {
  reportId: string;
}

const TOOL_LABELS: Record<string, string> = {
  calculator: "Calculator",
  benchmark_lookup: "Sector Benchmarks",
  yahoo_finance: "Yahoo Finance",
  web_search: "Web Search",
  report_writer: "Report Writer",
};

export function ProvenancePanel({ reportId }: Props) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["sources", reportId],
    queryFn: () => getReportSources(reportId),
  });

  if (isLoading) {
    return (
      <div className="mt-2 text-xs text-zinc-600 animate-pulse">Loading sources…</div>
    );
  }

  if (isError || !data) {
    return (
      <div className="mt-2 text-xs text-red-500">Could not load sources.</div>
    );
  }

  const hasAny = Object.values(data.analysis).some(
    (metrics) => Object.keys(metrics as object).length > 0
  );

  if (!hasAny) {
    return (
      <div className="mt-2 text-xs text-zinc-600">No source citations available for this report.</div>
    );
  }

  return (
    <div className="mt-3 rounded-lg border border-zinc-700 bg-zinc-900 text-xs overflow-hidden">
      <div className="px-3 py-2 bg-zinc-800 text-zinc-300 font-medium flex items-center gap-1.5">
        <svg className="w-3.5 h-3.5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Analysis Sources — {data.tickers}
      </div>

      <div className="divide-y divide-zinc-800">
        {Object.entries(data.analysis).map(([ticker, metrics]) => (
          <div key={ticker} className="p-3 space-y-1.5">
            <p className="font-semibold text-zinc-300 mb-2">{ticker}</p>
            {Object.entries(metrics as Record<string, { value: unknown; source_tool: string; observation_step: number }>)
              .map(([metric, info]) => (
                <div key={metric} className="flex items-start justify-between gap-4">
                  <span className="text-zinc-500 flex-shrink-0 w-48">
                    {_formatMetricName(metric)}
                  </span>
                  <span className="tabular-nums text-zinc-300 font-mono flex-shrink-0">
                    {_formatValue(metric, info.value)}
                  </span>
                  <span className="text-zinc-600 text-right">
                    {TOOL_LABELS[info.source_tool] ?? info.source_tool}
                    {info.observation_step != null && (
                      <span className="ml-1 text-zinc-700">· step {info.observation_step}</span>
                    )}
                  </span>
                </div>
              ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function _formatMetricName(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/pct$/, "%")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function _formatValue(metric: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (metric.includes("pct")) return `${value.toFixed(1)}%`;
    if (metric.includes("pe") || metric.includes("ratio")) return `${value.toFixed(1)}x`;
    if (metric.includes("cagr")) return `${value.toFixed(1)}%`;
    return value.toFixed(2);
  }
  return String(value);
}
