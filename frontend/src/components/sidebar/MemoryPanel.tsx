import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { clearAllMemory, getPreferences, getSummaries } from "../../lib/api";

export function MemoryPanel() {
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();

  const { data: prefs = {} } = useQuery({
    queryKey: ["preferences"],
    queryFn: getPreferences,
    enabled: open,
  });

  const { data: summaries = [] } = useQuery({
    queryKey: ["summaries"],
    queryFn: () => getSummaries(5),
    enabled: open,
  });

  const clearMut = useMutation({
    mutationFn: clearAllMemory,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["preferences"] });
      qc.invalidateQueries({ queryKey: ["summaries"] });
    },
  });

  const prefEntries = Object.entries(prefs);

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full px-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <span>🧠</span> Memory
        </span>
        <svg
          className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-1 space-y-3 px-2">
          {/* Preferences */}
          {prefEntries.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Preferences</p>
              <div className="space-y-0.5">
                {prefEntries.map(([k, v]) => (
                  <div key={k} className="text-xs text-zinc-400">
                    <span className="text-zinc-500">{k}:</span> {v}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Past analyses */}
          {summaries.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">
                Past analyses ({summaries.length})
              </p>
              <div className="space-y-1">
                {summaries.map((s) => (
                  <div key={s.id} className="text-xs text-zinc-500 leading-relaxed">
                    <span className="text-zinc-400 font-medium">[{s.tickers}]</span>{" "}
                    {s.summary_text.slice(0, 80)}…
                  </div>
                ))}
              </div>
            </div>
          )}

          {prefEntries.length === 0 && summaries.length === 0 && (
            <p className="text-xs text-zinc-600">
              No memory stored yet.
            </p>
          )}

          <button
            onClick={() => clearMut.mutate()}
            disabled={clearMut.isPending}
            className="text-xs text-red-500 hover:text-red-400 transition-colors disabled:opacity-40"
          >
            {clearMut.isPending ? "Clearing…" : "Clear all memory"}
          </button>
        </div>
      )}
    </div>
  );
}
