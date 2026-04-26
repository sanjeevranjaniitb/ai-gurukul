import { useRef, useEffect, useState } from 'react';

interface Props {
  avatarPreview: string | null;
  videoQueue: string[];
  onSegmentEnd: () => void;
}

/**
 * Real Lip Sync player — waits for the full Wav2Lip video to arrive,
 * then plays it once smoothly. No flickering, no partial segments.
 */
export default function RealVideoPlayer({ avatarPreview, videoQueue, onSegmentEnd }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasPlayed, setHasPlayed] = useState(false);
  const lastPlayedUrl = useRef('');

  // When a new video URL arrives and we haven't played it yet, play it
  useEffect(() => {
    if (videoQueue.length === 0) {
      // Queue cleared (new query) — reset state
      setIsPlaying(false);
      setHasPlayed(false);
      lastPlayedUrl.current = '';
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.removeAttribute('src');
        videoRef.current.load();
      }
      return;
    }

    // Get the latest video (in real mode there's typically just one)
    const latestUrl = videoQueue[videoQueue.length - 1];
    if (latestUrl === lastPlayedUrl.current) return; // Already playing/played this one

    const video = videoRef.current;
    if (!video) return;

    lastPlayedUrl.current = latestUrl;
    video.src = latestUrl;
    video.load();

    video.onended = () => {
      setIsPlaying(false);
      setHasPlayed(true);
      onSegmentEnd();
    };

    video.onerror = () => {
      setIsPlaying(false);
    };

    video.play()
      .then(() => setIsPlaying(true))
      .catch(() => {
        // Autoplay blocked — try muted
        video.muted = true;
        video.play()
          .then(() => setIsPlaying(true))
          .catch(() => setIsPlaying(false));
      });
  }, [videoQueue, onSegmentEnd]);

  const showPlaceholder = !avatarPreview && videoQueue.length === 0;
  const showPreview = !!avatarPreview && !isPlaying;

  return (
    <div className="avatar-player-container">
      <div className={`avatar-player ${showPlaceholder ? 'placeholder' : ''}`}>
        {showPlaceholder && (
          <div className="placeholder-content">
            <div className="placeholder-icon">👤</div>
            <p>Upload a photo to create your avatar</p>
          </div>
        )}

        {/* Static preview — visible when not playing video */}
        {showPreview && (
          <img src={avatarPreview} alt="Avatar" className="avatar-base-img" />
        )}

        {/* Video element — visible only when playing */}
        <video
          ref={videoRef}
          playsInline
          aria-label="Avatar video with real lip sync"
          className="avatar-video"
          style={{ display: isPlaying ? 'block' : 'none' }}
        />
      </div>

      {isPlaying && (
        <div className="avatar-speaking-indicator">
          <span className="speaking-dot" />
          Speaking (Real Lip Sync)…
        </div>
      )}

      {hasPlayed && !isPlaying && (
        <div className="avatar-replay-hint">
          ✓ Playback complete
        </div>
      )}
    </div>
  );
}
