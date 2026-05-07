import { type KeyboardEvent, useRef, useState } from "react";

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function MessageInput({ onSend, disabled }: MessageInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset height
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
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  return (
    <div className="border-t border-zinc-800 bg-zinc-950 p-4">
      <div className="relative flex items-end gap-2 rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 focus-within:border-violet-500 transition-colors">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask about stocks, request analysis, or recall past research…"
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-zinc-200 placeholder-zinc-500 outline-none disabled:opacity-50"
          style={{ maxHeight: 200 }}
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="flex-shrink-0 rounded-lg bg-violet-600 p-1.5 text-white transition-colors hover:bg-violet-500 disabled:opacity-30 disabled:cursor-not-allowed"
          aria-label="Send message"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </div>
      <p className="mt-1.5 text-center text-[11px] text-zinc-600">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
