import { useRef, useCallback } from 'react';

export interface SSECallbacks {
  onTextToken?: (token: string) => void;
  onAudioChunk?: (chunkUrl: string, chunkIndex: number) => void;
  onVideoChunk?: (chunkUrl: string, chunkIndex: number) => void;
  onStageUpdate?: (stage: string, durationMs: number) => void;
  onSources?: (sources: { chunk_text: string; page: number; score: number }[]) => void;
  onDone?: (totalDurationMs: number) => void;
  onError?: (message: string) => void;
}

export function useSSE() {
  const abortRef = useRef<AbortController | null>(null);

  const ask = useCallback(async (question: string, avatarId: string, callbacks: SSECallbacks) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, avatar_id: avatarId }),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      callbacks.onError?.(err.detail || 'Request failed');
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ') && eventType) {
          try {
            const data = JSON.parse(line.slice(6));
            switch (eventType) {
              case 'text_token':
                callbacks.onTextToken?.(data.token);
                break;
              case 'audio_chunk':
                callbacks.onAudioChunk?.(data.chunk_url, data.chunk_index);
                break;
              case 'video_chunk':
                callbacks.onVideoChunk?.(data.chunk_url, data.chunk_index);
                break;
              case 'stage_update':
                callbacks.onStageUpdate?.(data.stage, data.duration_ms);
                break;
              case 'sources':
                callbacks.onSources?.(data.sources);
                break;
              case 'done':
                callbacks.onDone?.(data.total_duration_ms);
                break;
              case 'error':
                callbacks.onError?.(data.message || 'Stream error');
                break;
            }
          } catch {
            // skip malformed JSON
          }
          eventType = '';
        }
      }
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { ask, cancel };
}
