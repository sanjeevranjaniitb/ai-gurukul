import { useState, useCallback, useRef } from 'react';
import AvatarUpload from './components/AvatarUpload';
import PdfUpload from './components/PdfUpload';
import ChatInterface from './components/ChatInterface';
import AvatarPlayer from './components/AvatarPlayer';
import RealVideoPlayer from './components/RealVideoPlayer';
import ProcessingIndicator from './components/ProcessingIndicator';
import RollingQuizBanner from './components/RollingQuizBanner';
import DocQuizModal from './components/DocQuizModal';
import type { AvatarUploadResponse, PdfUploadResponse } from './api';
import './App.css';

export type LipSyncMode = 'animated' | 'real';

interface AudioChunkData {
  url: string;
  sentence: string;
  duration: number;
}

function App() {
  const [avatarId, setAvatarId] = useState<string | null>(null);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [visemes, setVisemes] = useState<Record<string, string>>({});
  const [docId, setDocId] = useState<string | null>(null);
  const [docName, setDocName] = useState<string | null>(null);
  const [audioQueue, setAudioQueue] = useState<AudioChunkData[]>([]);
  const [videoQueue, setVideoQueue] = useState<string[]>([]);
  const [stage, setStage] = useState('');
  const [processing, setProcessing] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [lipSyncMode, setLipSyncMode] = useState<LipSyncMode>('animated');

  // Quiz state
  const [qaHistory, setQaHistory] = useState<Array<{ question: string; answer: string }>>([]);
  const [showDocQuiz, setShowDocQuiz] = useState(false);
  const sessionQuizTrigger = useRef<(() => void) | null>(null);

  const handleAvatarUploaded = useCallback((data: AvatarUploadResponse) => {
    setAvatarId(data.avatar_id);
    setAvatarPreview(data.frame_url || data.preview_url);
    if (data.visemes) setVisemes(data.visemes);
  }, []);

  const handlePdfUploaded = useCallback((data: PdfUploadResponse) => {
    setDocId(data.document_id);
    setDocName(data.name);
  }, []);

  const handleAudioChunk = useCallback((url: string, _index: number, sentence: string, duration: number) => {
    setAudioQueue((prev) => [...prev, { url, sentence, duration }]);
  }, []);

  const handleVideoChunk = useCallback((url: string) => {
    setVideoQueue((prev) => [...prev, url]);
  }, []);

  const handleStageUpdate = useCallback((s: string) => {
    setStage(s);
  }, []);

  const handleProcessingChange = useCallback((p: boolean) => {
    setProcessing(p);
    if (!p) setStage('');
  }, []);

  const handleNewQuery = useCallback(() => {
    setAudioQueue([]);
    setVideoQueue([]);
    setSidebarOpen(false);
  }, []);

  const handleQAComplete = useCallback((question: string, answer: string) => {
    setQaHistory((prev) => [...prev, { question, answer }]);
  }, []);

  const handleSegmentEnd = useCallback(() => {}, []);

  // Replay: clear queue first, then set the replay item on next tick
  // so the AvatarPlayer detects the length change (0 → 1) and starts playback
  const handleReplay = useCallback((audioUrl: string, sentence: string, duration: number, videoUrl?: string) => {
    if (lipSyncMode === 'animated') {
      setAudioQueue([]);
      setTimeout(() => {
        setAudioQueue([{ url: audioUrl, sentence, duration }]);
      }, 50);
    } else if (lipSyncMode === 'real' && videoUrl) {
      setVideoQueue([]);
      setTimeout(() => {
        setVideoQueue([videoUrl]);
      }, 50);
    }
  }, [lipSyncMode]);

  const ready = !!avatarId && !!docName;

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-inner">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
          <h1>🎓 AI Gurukul</h1>
          <span className="header-tagline">Talk to your documents through a living avatar</span>

          {/* Quiz buttons — right side of header */}
          <div className="header-quiz-buttons">
            {docId && (
              <button
                className="header-quiz-btn header-quiz-btn--doc header-quiz-btn--flash"
                onClick={() => setShowDocQuiz(true)}
                aria-label="Test Document Knowledge"
              >
                📄 Test Document Knowledge
              </button>
            )}
            {qaHistory.length > 0 && (
              <button
                className="header-quiz-btn header-quiz-btn--session header-quiz-btn--flash"
                onClick={() => sessionQuizTrigger.current?.()}
                aria-label="Quiz Your Understanding"
              >
                🧠 Quiz Your Understanding
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="app-body">
        <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar--collapsed'}`}>
          {sidebarOpen && (
            <>
              <AvatarUpload onUploaded={handleAvatarUploaded} />
              <PdfUpload onUploaded={handlePdfUploaded} />
              {ready && (
                <div className="ready-badge">
                  <span className="ready-dot" />
                  Ready to chat
                </div>
              )}
            </>
          )}
        </aside>

        <main className="main-content">
          <div className="conversation-area">
            <div className="avatar-column">
              <div className="mode-selector">
                <label htmlFor="lip-sync-mode">Lip Sync Mode</label>
                <select
                  id="lip-sync-mode"
                  value={lipSyncMode}
                  onChange={(e) => setLipSyncMode(e.target.value as LipSyncMode)}
                  disabled={processing}
                >
                  <option value="animated">Animated Lip Sync</option>
                  <option value="real">Real Lip Sync</option>
                </select>
              </div>

              {lipSyncMode === 'animated' ? (
                <AvatarPlayer
                  avatarPreview={avatarPreview}
                  visemes={visemes}
                  audioQueue={audioQueue}
                  onSegmentEnd={handleSegmentEnd}
                />
              ) : (
                <RealVideoPlayer
                  avatarPreview={avatarPreview}
                  videoQueue={videoQueue}
                  onSegmentEnd={handleSegmentEnd}
                />
              )}

              <ProcessingIndicator stage={stage} visible={processing} />

              {lipSyncMode === 'real' && processing && (
                <div className="mode-hint">
                  Generating real lip-sync video — this takes longer but looks better
                </div>
              )}
            </div>

            <div className="chat-column">
              {/* Rolling quiz overlay (no inline button — triggered from header) */}
              <RollingQuizBanner qaHistory={qaHistory} triggerRef={sessionQuizTrigger} />

              <ChatInterface
                avatarId={avatarId}
                lipSyncMode={lipSyncMode}
                onAudioChunk={handleAudioChunk}
                onVideoChunk={handleVideoChunk}
                onStageUpdate={handleStageUpdate}
                onProcessingChange={handleProcessingChange}
                onNewQuery={handleNewQuery}
                onQAComplete={handleQAComplete}
                onReplay={handleReplay}
              />
            </div>
          </div>
        </main>
      </div>

      {/* Document quiz modal */}
      {showDocQuiz && docId && (
        <DocQuizModal
          documentId={docId}
          documentName={docName || 'Document'}
          onClose={() => setShowDocQuiz(false)}
        />
      )}
    </div>
  );
}

export default App;
