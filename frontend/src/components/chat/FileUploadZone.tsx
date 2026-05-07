import { useRef, useState, type DragEvent } from "react";
import { uploadFile } from "../../lib/api";

interface Props {
  onParsed: (summary: Record<string, unknown>, filename: string) => void;
  disabled?: boolean;
}

export function FileUploadZone({ onParsed, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ext || !["csv", "pdf"].includes(ext)) {
      alert("Only CSV and PDF files are supported.");
      return;
    }
    setUploading(true);
    try {
      const summary = await uploadFile(file);
      onParsed(summary, file.name);
    } catch (err) {
      alert(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`
        flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs cursor-pointer
        transition-colors select-none
        ${dragging
          ? "border-violet-500 bg-violet-950 text-violet-300"
          : "border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-300"
        }
        ${disabled || uploading ? "opacity-40 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.pdf"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        disabled={disabled || uploading}
      />
      {uploading ? (
        <>
          <span className="animate-spin">⟳</span>
          <span>Uploading…</span>
        </>
      ) : (
        <>
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
          <span>Attach CSV or PDF</span>
        </>
      )}
    </div>
  );
}
