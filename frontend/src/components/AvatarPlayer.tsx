import { useRef, useEffect, useState, useCallback } from 'react';

// Character → viseme name mapping (20 visemes for near-phoneme-level animation)
const CHAR_TO_VISEME: Record<string, string> = {
  // Vowels
  a: 'ah', h: 'ah',
  e: 'eh', 
  i: 'ih',
  o: 'oh',
  u: 'oo', q: 'oo',
  // Bilabial
  m: 'mm', b: 'bv', p: 'mm',
  // Labiodental
  f: 'ff', v: 'ff',
  // Dental / alveolar
  t: 'td', d: 'td',
  n: 'nn',
  l: 'll',
  // Sibilant
  s: 'ss', z: 'ss',
  j: 'sh', c: 'sh', x: 'sh',
  // Velar / glottal
  k: 'kk', g: 'kk',
  // Approximants
  w: 'ww',
  r: 'rr',
  y: 'ee',
  // Silence / punctuation
  ' ': 'idle', '.': 'idle', ',': 'idle', '?': 'idle', '!': 'idle',
  '\n': 'idle', '-': 'idle', "'": 'idle',
};

function getVisemeForChar(char: string): string {
  return CHAR_TO_VISEME[char.toLowerCase()] || 'ae';
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
  const [segmentTrigger, setSegmentTrigger] = useState(0);

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

    // Draw directly without clearing — avoids flash of background color
    // between frames. drawImage at full canvas size covers all pixels.
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
    let animStopped = false;

    const startAnimation = () => {
      setIsAnimating(true);

      const tick = () => {
        if (animStopped) return;

        // Check if audio has ended or paused — stop immediately
        if (audio.ended || audio.paused) {
          drawFrame('idle');
          return;
        }

        const audioDur = audio.duration || segment.duration || 3;
        const elapsed = audio.currentTime || 0;
        const charSpeed = text.length > 0 ? text.length / audioDur : 14;
        const ci = Math.min(Math.floor(elapsed * charSpeed), text.length - 1);

        if (ci >= 0 && ci < text.length) {
          const visemeName = getVisemeForChar(text[ci]);
          drawFrame(visemeName);
        }

        animRef.current = requestAnimationFrame(tick);
      };
      animRef.current = requestAnimationFrame(tick);
    };

    const cleanup = () => {
      animStopped = true;
      cancelAnimationFrame(animRef.current);
      drawFrame('idle');
      playingRef.current = false;
      setIsAnimating(idxRef.current < audioQueue.length);
      onSegmentEnd();
      setSegmentTrigger((prev) => prev + 1);
    };

    audio.onended = cleanup;
    audio.onpause = () => {
      // Also stop on pause (covers edge cases)
      animStopped = true;
      cancelAnimationFrame(animRef.current);
      drawFrame('idle');
    };
    audio.onerror = () => {
      animStopped = true;
      cancelAnimationFrame(animRef.current);
      playingRef.current = false;
      setIsAnimating(false);
      drawFrame('idle');
    };

    audio.play()
      .then(startAnimation)
      .catch(() => {
        audio.muted = true;
        audio.play()
          .then(startAnimation)
          .catch(() => {
            // Can't play audio at all — animate with timer fallback
            setIsAnimating(true);
            const dur = segment.duration || 3;
            const t0 = performance.now();
            const fallbackTick = () => {
              if (animStopped) return;
              const elapsed = (performance.now() - t0) / 1000;
              const charSpeed = text.length > 0 ? text.length / dur : 14;
              const ci = Math.min(Math.floor(elapsed * charSpeed), text.length - 1);
              if (ci >= 0 && ci < text.length) {
                drawFrame(getVisemeForChar(text[ci]));
              }
              if (elapsed < dur) {
                animRef.current = requestAnimationFrame(fallbackTick);
              } else {
                cleanup();
              }
            };
            animRef.current = requestAnimationFrame(fallbackTick);
          });
      });
  }, [audioQueue.length, canvasReady, drawFrame, onSegmentEnd, segmentTrigger]);

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
