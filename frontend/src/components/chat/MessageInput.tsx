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

  const canSend = !!value.trim() && !disabled;

  return (
    <div className="px-4 pb-4 pt-2">
      <div
        className={[
          "relative flex items-end gap-2 rounded-2xl border bg-zinc-900 px-4 py-3",
          "transition-all duration-150",
          disabled
            ? "border-zinc-800 opacity-60"
            : "border-zinc-700/80 focus-within:border-violet-500/60 focus-within:shadow-[0_0_0_3px_rgb(124,58,237,0.08)]",
        ].join(" ")}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask about stocks, request analysis, compare companies…"
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-transparent text-[14px] text-zinc-200 placeholder-zinc-600 outline-none leading-relaxed"
          style={{ maxHeight: 180 }}
        />

        <button
          onClick={submit}
          disabled={!canSend}
          aria-label="Send message"
          className={[
            "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150",
            canSend
              ? "bg-violet-600 hover:bg-violet-500 text-white shadow-sm"
              : "bg-zinc-800 text-zinc-600 cursor-not-allowed",
          ].join(" ")}
        >
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        </button>
      </div>

      <p className="mt-1.5 text-center text-[11px] text-zinc-700">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
