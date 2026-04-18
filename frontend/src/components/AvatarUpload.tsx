import { useState, useRef } from 'react';
import { uploadAvatar, type AvatarUploadResponse } from '../api';

const ACCEPTED = '.png,.jpg,.jpeg';
const MAX_SIZE = 10 * 1024 * 1024; // 10 MB
const MIN_RES = 256;

interface Props {
  onUploaded: (data: AvatarUploadResponse) => void;
}

function validateImage(file: File): Promise<string | null> {
  return new Promise((resolve) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !['png', 'jpg', 'jpeg'].includes(ext)) {
      return resolve('Only PNG, JPG, and JPEG files are accepted.');
    }
    if (file.size > MAX_SIZE) {
      return resolve('Image must be 10 MB or smaller.');
    }
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      if (img.width < MIN_RES || img.height < MIN_RES) {
        resolve(`Image must be at least ${MIN_RES}×${MIN_RES} pixels.`);
      } else {
        resolve(null);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      resolve('Could not read image file.');
    };
    img.src = URL.createObjectURL(file);
  });
}

export default function AvatarUpload({ onUploaded }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setError('');
    const validationError = await validateImage(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setUploading(true);
    try {
      const data = await uploadAvatar(file);
      setPreview(data.preview_url);
      onUploaded(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="upload-section">
      <h3>Avatar Image</h3>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        aria-label="Upload avatar image"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {uploading && <p className="info">Uploading…</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {preview && <img src={preview} alt="Avatar preview" className="avatar-preview" />}
    </div>
  );
}
