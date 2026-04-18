import { useState, useRef, useEffect } from 'react';
import { useSSE } from '../hooks/useSSE';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface Props {
  avatarId: string | null;
  onTextToken?: (token: string) => void;
  onAudioChunk?: (url: string, index: number) => void;
  onVideoChunk?: (url: string, index: number) => void;
  onStageUpdate?: (stage: string) => void;
  onProcessingChange?: (processing: boolean) => void;
}

export default function ChatInterface({
  avatarId,
  onTextToken,
  onAudioChunk,
  onVideoChunk,
  onStageUpdate,
  onProcessingChange,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [processing, setProcessing] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);
  const { ask } = useSSE();

  // Auto-scroll chat history
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || !avatarId || processing) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setProcessing(true);
    onProcessingChange?.(true);

    // Add empty assistant message to fill progressively
    let assistantText = '';
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    await ask(question, avatarId, {
      onTextToken(token) {
        assistantText += token;
        onTextToken?.(token);
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: assistantText };
          return updated;
        });
      },
      onAudioChunk(url, index) {
        onAudioChunk?.(url, index);
      },
      onVideoChunk(url, index) {
        onVideoChunk?.(url, index);
      },
      onStageUpdate(stage) {
        onStageUpdate?.(stage);
      },
      onDone() {
        setProcessing(false);
        onProcessingChange?.(false);
      },
      onError(msg) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: `Error: ${msg}` };
          return updated;
        });
        setProcessing(false);
        onProcessingChange?.(false);
      },
    });
  }

  return (
    <div className="chat-interface">
      <div className="chat-history" ref={historyRef}>
        {messages.length === 0 && (
          <p className="chat-empty">Ask a question about your uploaded document.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg--${msg.role}`}>
            <span className="chat-role">{msg.role === 'user' ? 'You' : 'Avatar'}</span>
            <p>{msg.content}</p>
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
          Send
        </button>
      </form>
    </div>
  );
}
