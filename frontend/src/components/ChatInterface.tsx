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
  onQAComplete?: (question: string, answer: string) => void;
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
  onQAComplete,
  onReplay,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [processing, setProcessing] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);
  const { ask } = useSSE();

  // Session cache: normalized question → cached answer
  const cacheRef = useRef<Map<string, { answer: string; audioUrl: string; sentence: string; duration: number; videoUrl: string }>>(new Map());

  const normalizeQuestion = (q: string) => q.trim().toLowerCase().replace(/[?.!,]+$/g, '').replace(/\s+/g, ' ');

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

    // Check session cache
    const key = normalizeQuestion(question);
    const cached = cacheRef.current.get(key);
    if (cached) {
      const cachedContent = `🔄 You Already Asked This\n\n${cached.answer}`;
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: cachedContent,
          latency: 0,
          audioUrl: cached.audioUrl,
          sentence: cached.sentence || cached.answer,
          audioDuration: cached.duration,
          videoUrl: cached.videoUrl,
        },
      ]);
      // Replay the cached audio/video
      if (cached.audioUrl) {
        onAudioChunk?.(cached.audioUrl, 0, cached.sentence || cached.answer, cached.duration);
      }
      if (cached.videoUrl) {
        onVideoChunk?.(cached.videoUrl);
      }
      return;
    }

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
        if (assistantText) {
          // Cache the response
          cacheRef.current.set(key, {
            answer: assistantText,
            audioUrl: lastAudioUrl,
            sentence: lastSentence || assistantText,
            duration: lastDuration,
            videoUrl: lastVideoUrl,
          });
          onQAComplete?.(question, assistantText);
        }
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
