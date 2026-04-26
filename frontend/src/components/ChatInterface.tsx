import { useState, useRef, useEffect } from 'react';
import { useSSE } from '../hooks/useSSE';
import type { LipSyncMode } from '../App';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  latency?: number;
  audioUrl?: string;
  sentence?: string;
  audioDuration?: number;
  videoUrl?: string;       // for real mode replay
}

interface Props {
  avatarId: string | null;
  lipSyncMode: LipSyncMode;
  onAudioChunk?: (url: string, index: number, sentence: string, duration: number) => void;
  onVideoChunk?: (url: string) => void;
  onStageUpdate?: (stage: string) => void;
  onProcessingChange?: (processing: boolean) => void;
  onNewQuery?: () => void;
  onReplay?: (audioUrl: string, sentence: string, duration: number, videoUrl?: string) => void;
}

export default function ChatInterface({
  avatarId,
  lipSyncMode,
  onAudioChunk,
  onVideoChunk,
  onStageUpdate,
  onProcessingChange,
  onNewQuery,
  onReplay,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [processing, setProcessing] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);
  const { ask } = useSSE();

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [messages]);

  // Cleanup session on browser tab close
  useEffect(() => {
    const cleanup = () => {
      navigator.sendBeacon('/api/reset', '');
    };
    window.addEventListener('beforeunload', cleanup);
    return () => window.removeEventListener('beforeunload', cleanup);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || !avatarId || processing) return;

    setInput('');
    onNewQuery?.();
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setProcessing(true);
    onProcessingChange?.(true);

    const startTime = performance.now();
    let assistantText = '';
    let lastAudioUrl = '';
    let lastSentence = '';
    let lastDuration = 0;
    let lastVideoUrl = '';

    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    await ask(question, avatarId, {
      onTextToken(token) {
        assistantText += token;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: assistantText };
          return updated;
        });
      },
      onAudioChunk(url, index, sentence, duration) {
        lastAudioUrl = url;
        lastSentence = sentence;
        lastDuration = duration;
        onAudioChunk?.(url, index, sentence, duration);
      },
      onVideoChunk(url) {
        lastVideoUrl = url;
        onVideoChunk?.(url);
      },
      onStageUpdate(stage) {
        onStageUpdate?.(stage);
      },
      onDone() {
        const latency = (performance.now() - startTime) / 1000;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: assistantText,
            latency,
            audioUrl: lastAudioUrl,
            sentence: lastSentence || assistantText,
            audioDuration: lastDuration,
            videoUrl: lastVideoUrl,
          };
          return updated;
        });
        setProcessing(false);
        onProcessingChange?.(false);
      },
      onError(msg) {
        const latency = (performance.now() - startTime) / 1000;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: `Error: ${msg}`,
            latency,
          };
          return updated;
        });
        setProcessing(false);
        onProcessingChange?.(false);
      },
    }, lipSyncMode);
  }

  function handleReplay(msg: ChatMessage) {
    if (onReplay) {
      onReplay(msg.audioUrl || '', msg.sentence || msg.content, msg.audioDuration || 3, msg.videoUrl);
    }
  }

  return (
    <div className="chat-interface">
      <div className="chat-history" ref={historyRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">💬</div>
            <p>Ask a question about your uploaded document.</p>
            <p className="chat-empty-hint">The avatar will read the answer aloud with lip-sync.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg--${msg.role}`}>
            <span className="chat-role">{msg.role === 'user' ? 'You' : '🎓 AI Gurukul'}</span>
            <p>{msg.content}{msg.role === 'assistant' && processing && i === messages.length - 1 ? '▊' : ''}</p>
            {msg.role === 'assistant' && msg.latency != null && (
              <div className="chat-meta">
                <span className="chat-latency">⏱️ {msg.latency.toFixed(1)}s</span>
                {msg.audioUrl && (
                  <button className="chat-replay-btn" onClick={() => handleReplay(msg)}>
                    🔄 Replay
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={avatarId ? 'Type your question…' : 'Upload an avatar first'}
          disabled={!avatarId || processing}
          aria-label="Question input"
        />
        <button type="submit" disabled={!avatarId || processing || !input.trim()}>
          {processing ? '⏳' : 'Send'}
        </button>
      </form>
    </div>
  );
}
