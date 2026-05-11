import { useRef, useState, type DragEvent } from "react";
import { uploadFile } from "../../lib/api";

interface Props {
  onParsed: (summary: Record<string, unknown>, filename: string) => void;
  disabled?: boolean;
}

const ACCEPTED_EXTENSIONS = ".csv,.pdf,.xlsx,.xls,.docx,.txt,.md,.json";
const ACCEPTED_LIST = "CSV, PDF, Excel, Word, TXT, MD, JSON";

export function FileUploadZone({ onParsed, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    try {
      const summary = await uploadFile(file);
      onParsed(summary, file.name);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Surface a readable error without an alert (add as assistant message via parent)
      onParsed({ error: msg }, file.name);
    } finally {
      setUploading(false);
      // Reset input so the same file can be re-uploaded
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const isDisabled = disabled || uploading;

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!isDisabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { if (!isDisabled) onDrop(e); }}
      onClick={() => { if (!isDisabled) inputRef.current?.click(); }}
      title={`Supported formats: ${ACCEPTED_LIST}`}
      className={[
        "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs cursor-pointer",
        "transition-all duration-150 select-none",
        dragging
          ? "border-zinc-500 bg-zinc-800/40 text-zinc-300"
          : "border-zinc-700/60 text-zinc-600 hover:border-zinc-600 hover:text-zinc-400",
        isDisabled ? "opacity-40 cursor-not-allowed pointer-events-none" : "",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        disabled={isDisabled}
      />
      {uploading ? (
        <>
          <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>Uploading…</span>
        </>
      ) : (
        <>
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
          <span>Attach file</span>
        </>
      )}
    </div>
  );
}
