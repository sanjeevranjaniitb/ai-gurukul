import { useRef, useEffect, useState, useCallback } from 'react';

// Character → viseme name mapping
const CHAR_TO_VISEME: Record<string, string> = {
  a: 'a', h: 'a', r: 'a', l: 'a',
  e: 'e', i: 'e', y: 'e', s: 'e', z: 'e', j: 'e',
  o: 'o', u: 'o', w: 'o', q: 'o',
  m: 'm', b: 'm', p: 'm', f: 'm', v: 'm',
  t: 'a', d: 'a', n: 'a', k: 'a', g: 'a', c: 'a', x: 'a',
  ' ': 'idle', '.': 'idle', ',': 'idle', '?': 'idle', '!': 'idle',
};

function getVisemeForChar(char: string): string {
  return CHAR_TO_VISEME[char.toLowerCase()] || 'a';
}

export interface AudioChunkData {
  url: string;
  sentence: string;
  duration: number;
}

interface Props {
  avatarPreview: string | null;
  visemes: Record<string, string>;
  audioQueue: AudioChunkData[];
  onSegmentEnd: () => void;
}

export default function AvatarPlayer({ avatarPreview, visemes, audioQueue, onSegmentEnd }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const animRef = useRef<number>(0);
  const idxRef = useRef(0);
  const playingRef = useRef(false);
  const loadedImagesRef = useRef<Record<string, HTMLImageElement>>({});
  const [isAnimating, setIsAnimating] = useState(false);
  const [canvasReady, setCanvasReady] = useState(false);

  // Load all viseme images into HTMLImageElement objects for canvas drawing
  useEffect(() => {
    const images: Record<string, HTMLImageElement> = {};
    const urls = { ...visemes };
    if (avatarPreview && !urls['idle']) {
      urls['idle'] = avatarPreview;
    }

    let loaded = 0;
    const total = Object.keys(urls).length;
    if (total === 0) return;

    Object.entries(urls).forEach(([name, url]) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        images[name] = img;
        loaded++;
        if (loaded >= total) {
          loadedImagesRef.current = images;
          setCanvasReady(true);
          // Draw idle frame
          drawFrame('idle');
        }
      };
      img.onerror = () => {
        loaded++;
        if (loaded >= total) {
          loadedImagesRef.current = images;
          setCanvasReady(true);
        }
      };
      img.src = url;
    });
  }, [visemes, avatarPreview]);

  // Draw a single viseme frame on the canvas
  const drawFrame = useCallback((visemeName: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = loadedImagesRef.current[visemeName] || loadedImagesRef.current['idle'];
    if (!img) return;

    // Clear and draw
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }, []);

  // Draw idle when avatar preview changes
  useEffect(() => {
    if (avatarPreview && !isAnimating) {
      // Draw the preview image on canvas
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = avatarPreview;
    }
  }, [avatarPreview, isAnimating]);

  // Play next audio segment with canvas animation
  useEffect(() => {
    if (playingRef.current) return;
    if (idxRef.current >= audioQueue.length) return;
    if (!canvasReady) return;

    const segment = audioQueue[idxRef.current];
    const audio = audioRef.current;
    if (!audio || !segment) return;

    playingRef.current = true;
    const segIdx = idxRef.current;
    idxRef.current += 1;

    audio.src = segment.url;
    audio.load();

    const text = segment.sentence;
    const dur = segment.duration || 3;
    const charSpeed = text.length > 0 ? text.length / dur : 14;

    const runAnimation = () => {
      setIsAnimating(true);
      const t0 = performance.now();

      const tick = () => {
        const elapsed = (performance.now() - t0) / 1000;
        const ci = Math.min(Math.floor(elapsed * charSpeed), text.length - 1);

        if (ci >= 0 && ci < text.length) {
          const visemeName = getVisemeForChar(text[ci]);
          drawFrame(visemeName);
        }

        if (elapsed < dur) {
          animRef.current = requestAnimationFrame(tick);
        }
      };
      animRef.current = requestAnimationFrame(tick);
    };

    const onEnded = () => {
      cancelAnimationFrame(animRef.current);
      drawFrame('idle');
      playingRef.current = false;
      setIsAnimating(idxRef.current < audioQueue.length);
      onSegmentEnd();
    };

    audio.onended = onEnded;
    audio.onerror = () => {
      playingRef.current = false;
      setIsAnimating(false);
      drawFrame('idle');
    };

    audio.play()
      .then(runAnimation)
      .catch(() => {
        audio.muted = true;
        audio.play()
          .then(runAnimation)
          .catch(() => {
            // Can't play audio — still animate visually
            runAnimation();
            setTimeout(onEnded, dur * 1000);
          });
      });
  }, [audioQueue.length, canvasReady, drawFrame, onSegmentEnd]);

  // Reset when queue is cleared
  useEffect(() => {
    if (audioQueue.length === 0) {
      idxRef.current = 0;
      playingRef.current = false;
      cancelAnimationFrame(animRef.current);
      setIsAnimating(false);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.onended = null;
        audioRef.current.removeAttribute('src');
      }
      drawFrame('idle');
    }
  }, [audioQueue.length, drawFrame]);

  // Cleanup
  useEffect(() => () => cancelAnimationFrame(animRef.current), []);

  const showPlaceholder = !avatarPreview && Object.keys(visemes).length === 0;

  return (
    <div className="avatar-player-container">
      <div className={`avatar-player ${showPlaceholder ? 'placeholder' : ''}`}>
        {showPlaceholder && (
          <div className="placeholder-content">
            <div className="placeholder-icon">👤</div>
            <p>Upload a photo to create your avatar</p>
          </div>
        )}

        <canvas
          ref={canvasRef}
          width={512}
          height={512}
          className="avatar-canvas"
          style={{ display: showPlaceholder ? 'none' : 'block' }}
        />
      </div>

      {isAnimating && (
        <div className="avatar-speaking-indicator">
          <span className="speaking-dot" />
          Speaking…
        </div>
      )}

      <audio ref={audioRef} preload="auto" />
    </div>
  );
}
