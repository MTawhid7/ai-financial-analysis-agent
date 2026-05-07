import { useCallback, useRef } from "react";
import { createEventSource, sendMessage } from "../lib/api";

export interface StepEvent {
  type: "step";
  step: number | string;
  agent: string;
  tool: string;
  cache_hit?: boolean;
  ok?: boolean;
}

export interface CompleteEvent {
  type: "complete";
  response: string;
}

export interface ErrorEvent {
  type: "error";
  detail: string;
}

type SSEEvent = StepEvent | CompleteEvent | ErrorEvent | { type: "ping" };

export function useStreamingChat() {
  const esRef = useRef<EventSource | null>(null);

  const abort = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const streamMessage = useCallback(
    async (
      conversationId: string,
      message: string,
      onStep: (e: StepEvent) => void,
      onComplete: (response: string) => void,
      onError: (detail: string) => void
    ) => {
      // Abort any in-flight stream
      abort();

      try {
        const { event_id } = await sendMessage(conversationId, message);

        const es = createEventSource(event_id);
        esRef.current = es;

        es.onmessage = (e: MessageEvent) => {
          let data: SSEEvent;
          try {
            data = JSON.parse(e.data);
          } catch {
            return;
          }

          if (data.type === "step") onStep(data);
          else if (data.type === "complete") {
            onComplete(data.response);
            es.close();
            esRef.current = null;
          } else if (data.type === "error") {
            onError(data.detail);
            es.close();
            esRef.current = null;
          }
          // "ping" events are silently ignored
        };

        es.onerror = () => {
          onError("Connection to the server was lost.");
          es.close();
          esRef.current = null;
        };
      } catch (err) {
        onError(err instanceof Error ? err.message : String(err));
      }
    },
    [abort]
  );

  return { streamMessage, abort };
}
