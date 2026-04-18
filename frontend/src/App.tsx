import { useState, useCallback } from 'react';
import AvatarUpload from './components/AvatarUpload';
import PdfUpload from './components/PdfUpload';
import ChatInterface from './components/ChatInterface';
import AvatarPlayer from './components/AvatarPlayer';
import ProcessingIndicator from './components/ProcessingIndicator';
import type { AvatarUploadResponse, PdfUploadResponse } from './api';
import './App.css';

function App() {
  const [avatarId, setAvatarId] = useState<string | null>(null);
  const [, setDocId] = useState<string | null>(null);
  const [videoQueue, setVideoQueue] = useState<string[]>([]);
  const [stage, setStage] = useState('');
  const [processing, setProcessing] = useState(false);

  const handleAvatarUploaded = useCallback((data: AvatarUploadResponse) => {
    setAvatarId(data.avatar_id);
  }, []);

  const handlePdfUploaded = useCallback((data: PdfUploadResponse) => {
    setDocId(data.document_id);
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

  const handleSegmentEnd = useCallback(() => {
    // segment finished playing — queue handles auto-advance
  }, []);

  return (
    <div className="app-layout">
      <header className="app-header">
        <h1>AI Gurukul</h1>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <AvatarUpload onUploaded={handleAvatarUploaded} />
          <PdfUpload onUploaded={handlePdfUploaded} />
        </aside>

        <main className="main-content">
          <div className="avatar-area">
            <AvatarPlayer videoQueue={videoQueue} onSegmentEnd={handleSegmentEnd} />
            <ProcessingIndicator stage={stage} visible={processing} />
          </div>
          <ChatInterface
            avatarId={avatarId}
            onVideoChunk={handleVideoChunk}
            onStageUpdate={handleStageUpdate}
            onProcessingChange={handleProcessingChange}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
