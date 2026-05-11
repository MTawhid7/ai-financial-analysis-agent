import { type KeyboardEvent, useRef, useState } from "react";

const ACCEPTED_EXTENSIONS = ".csv,.pdf,.xlsx,.xls,.docx,.txt,.md,.json";

interface PendingAttachment {
  filename: string;
  description: string;
}

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  pendingAttachment?: PendingAttachment | null;
  onAttach?: (file: File) => void;
  onClearAttachment?: () => void;
}

export function MessageInput({
  onSend,
  disabled,
  pendingAttachment,
  onAttach,
  onClearAttachment,
}: MessageInputProps) {
  const [value, setValue] = useState("");
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if ((!trimmed && !pendingAttachment) || disabled) return;
    onSend(trimmed || "(File attachment — please analyse)");
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onAttach) return;
    setUploading(true);
    try {
      await onAttach(file);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const canSend = (!!value.trim() || !!pendingAttachment) && !disabled && !uploading;
  const isDisabled = disabled || uploading;

  return (
    <div className="px-4 pb-4 pt-2">
      <div
        className={[
          "relative flex flex-col rounded-2xl border bg-zinc-900",
          "transition-all duration-150",
          isDisabled
            ? "border-zinc-800 opacity-60"
            : "border-zinc-700/80 focus-within:border-zinc-600",
        ].join(" ")}
      >
        {/* Attachment badge */}
        {pendingAttachment && (
          <div className="flex items-center gap-2 px-4 pt-3 pb-1">
            <div className="flex items-center gap-1.5 bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1 text-xs text-zinc-300 max-w-full">
              <svg className="w-3 h-3 text-zinc-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
              <span className="truncate max-w-xs">{pendingAttachment.filename}</span>
              <button
                onClick={onClearAttachment}
                className="ml-1 text-zinc-500 hover:text-zinc-300 flex-shrink-0"
                aria-label="Remove attachment"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Input row */}
        <div className="flex items-end gap-2 px-3 py-3">
          {/* Paperclip button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isDisabled}
            title="Attach file (CSV, PDF, Excel, Word, TXT, MD, JSON)"
            className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800 transition-all"
          >
            {uploading ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            )}
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS}
            className="hidden"
            onChange={handleFileChange}
            disabled={isDisabled}
          />

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder="Ask about stocks, request analysis, compare companies…"
            disabled={isDisabled}
            rows={1}
            className="flex-1 resize-none bg-transparent text-[14px] text-zinc-200 placeholder-zinc-600 outline-none leading-relaxed"
            style={{ maxHeight: 180 }}
          />

          {/* Send button — white when active */}
          <button
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className={[
              "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150",
              canSend
                ? "bg-white text-zinc-900 hover:bg-zinc-100 shadow-sm"
                : "bg-zinc-800 text-zinc-600 cursor-not-allowed",
            ].join(" ")}
          >
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          </button>
        </div>
      </div>

      <p className="mt-1.5 text-center text-[11px] text-zinc-700">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
