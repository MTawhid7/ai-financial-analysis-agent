import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { clearAllMemory, getPreferences, getSummaries } from "../../lib/api";

// Convert raw preference keys to natural-language sentences
function _preferenceToSentence(key: string, value: string): string {
  const k = key.toLowerCase().replace(/_/g, " ");
  const v = value.toLowerCase();
  const map: Record<string, string> = {
    "investment style": `prefers ${v} investment style`,
    "summary length": `wants ${v} summaries`,
    "investor type": `is a ${v} investor`,
    "focus": `focuses on ${v}`,
    "risk tolerance": `has ${v} risk tolerance`,
  };
  return map[k] ?? `set ${k} to ${v}`;
}

export function MemoryPanel() {
  const [open, setOpen] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const qc = useQueryClient();

  const { data: prefs = {} } = useQuery({
    queryKey: ["preferences"],
    queryFn: getPreferences,
    enabled: open,
  });

  const { data: summaries = [] } = useQuery({
    queryKey: ["summaries"],
    queryFn: () => getSummaries(8),
    enabled: open,
  });

  const clearMut = useMutation({
    mutationFn: clearAllMemory,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["preferences"] });
      qc.invalidateQueries({ queryKey: ["summaries"] });
      setConfirmClear(false);
    },
  });

  const prefEntries = Object.entries(prefs);

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full px-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors rounded"
      >
        <span className="flex items-center gap-1.5 font-medium">
          <span>🧠</span>
          Memory
          {(prefEntries.length > 0 || summaries.length > 0) && (
            <span className="ml-1 rounded-full bg-zinc-700 text-zinc-300 px-1.5 py-0.5 text-[10px]">
              {prefEntries.length + summaries.length}
            </span>
          )}
        </span>
        <svg
          className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-1 space-y-3 px-2 pb-2">
          {/* Preferences — shown as natural language */}
          {prefEntries.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1.5">
                What I know about you
              </p>
              <ul className="space-y-1">
                {prefEntries.map(([k, v]) => (
                  <li key={k} className="text-xs text-zinc-400 flex gap-1.5">
                    <span className="text-zinc-600 flex-shrink-0">•</span>
                    <span>You {_preferenceToSentence(k, v)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Past analyses */}
          {summaries.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1.5">
                Past analyses ({summaries.length})
              </p>
              <div className="space-y-2">
                {summaries.map((s) => (
                  <div key={s.id} className="bg-zinc-800 rounded p-2 space-y-0.5">
                    <p className="text-[10px] font-medium text-zinc-400 font-mono">
                      [{s.tickers}]
                    </p>
                    <p className="text-[11px] text-zinc-400 leading-relaxed line-clamp-2">
                      {s.summary_text}
                    </p>
                    <p className="text-[10px] text-zinc-600">
                      {new Date(s.created_at * 1000).toLocaleDateString()}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {prefEntries.length === 0 && summaries.length === 0 && (
            <p className="text-xs text-zinc-600">
              No memory yet. Run an analysis or say{" "}
              <em className="not-italic text-zinc-500">"I prefer conservative picks"</em>.
            </p>
          )}

          {/* Clear all — with confirmation */}
          {!confirmClear ? (
            <button
              onClick={() => setConfirmClear(true)}
              disabled={prefEntries.length === 0 && summaries.length === 0}
              className="text-xs text-zinc-600 hover:text-red-400 transition-colors disabled:opacity-30"
            >
              Clear all memory…
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">Are you sure?</span>
              <button
                onClick={() => clearMut.mutate()}
                disabled={clearMut.isPending}
                className="text-xs text-red-400 hover:text-red-300 font-medium"
              >
                {clearMut.isPending ? "Clearing…" : "Yes, clear"}
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                className="text-xs text-zinc-600 hover:text-zinc-400"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
