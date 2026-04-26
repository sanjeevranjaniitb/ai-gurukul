import { useState, useRef, useEffect } from 'react';
import { uploadAvatar, type AvatarUploadResponse } from '../api';

const ACCEPTED = '.png,.jpg,.jpeg';
const MAX_SIZE = 10 * 1024 * 1024;
const MIN_RES = 256;

interface CatalogItem {
  name: string;
  file: string;
}

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
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load catalog on mount
  useEffect(() => {
    fetch('/avatars/catalog.json')
      .then((r) => (r.ok ? r.json() : []))
      .then((items: CatalogItem[]) => {
        // Only keep items whose image actually exists
        const checks = items.map((item) =>
          fetch(item.file, { method: 'HEAD' }).then((r) => (r.ok ? item : null))
        );
        return Promise.all(checks);
      })
      .then((results) => setCatalog(results.filter(Boolean) as CatalogItem[]))
      .catch(() => setCatalog([]));
  }, []);

  async function handleFile(file: File) {
    setError('');
    const validationError = await validateImage(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    await doUpload(file);
  }

  async function handleCatalogPick(item: CatalogItem) {
    setError('');
    setUploading(true);
    try {
      const res = await fetch(item.file);
      const blob = await res.blob();
      const ext = item.file.split('.').pop() || 'jpg';
      const file = new File([blob], `${item.name}.${ext}`, { type: blob.type });
      await doUpload(file);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load catalog image');
      setUploading(false);
    }
  }

  async function doUpload(file: File) {
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

      {/* Default catalog */}
      {catalog.length > 0 && !preview && (
        <div className="avatar-catalog">
          <p className="catalog-label">Choose a default avatar:</p>
          <div className="catalog-grid">
            {catalog.map((item) => (
              <button
                key={item.file}
                className="catalog-item"
                onClick={() => handleCatalogPick(item)}
                disabled={uploading}
                title={item.name}
              >
                <img src={item.file} alt={item.name} />
              </button>
            ))}
          </div>
          <p className="catalog-or">— or upload your own —</p>
        </div>
      )}

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
