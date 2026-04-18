import { useState, useRef } from 'react';
import { uploadPdf, type PdfUploadResponse } from '../api';

const MAX_SIZE = 50 * 1024 * 1024; // 50 MB

interface Props {
  onUploaded: (data: PdfUploadResponse) => void;
}

export default function PdfUpload({ onUploaded }: Props) {
  const [doc, setDoc] = useState<PdfUploadResponse | null>(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setError('');
    if (file.size > MAX_SIZE) {
      setError('PDF must be 50 MB or smaller.');
      return;
    }
    setUploading(true);
    try {
      const data = await uploadPdf(file);
      setDoc(data);
      onUploaded(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="upload-section">
      <h3>PDF Document</h3>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        aria-label="Upload PDF document"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {uploading && <p className="info">Uploading…</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {doc && (
        <p className="success">
          {doc.name} — {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}, {doc.chunk_count} chunks
        </p>
      )}
    </div>
  );
}
