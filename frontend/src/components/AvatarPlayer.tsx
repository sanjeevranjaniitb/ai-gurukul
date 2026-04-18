import { useRef, useEffect, useCallback } from 'react';

interface Props {
  videoQueue: string[];
  onSegmentEnd: () => void;
}

export default function AvatarPlayer({ videoQueue, onSegmentEnd }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const playingIndex = useRef(-1);

  const playNext = useCallback(() => {
    const nextIdx = playingIndex.current + 1;
    if (nextIdx < videoQueue.length && videoRef.current) {
      playingIndex.current = nextIdx;
      videoRef.current.src = videoQueue[nextIdx];
      videoRef.current.play().catch(() => {});
    }
  }, [videoQueue]);

  // When new segments arrive, start playback if idle
  useEffect(() => {
    if (videoQueue.length > 0 && playingIndex.current < 0) {
      playNext();
    }
  }, [videoQueue, playNext]);

  function handleEnded() {
    onSegmentEnd();
    playNext();
  }

  if (videoQueue.length === 0) {
    return (
      <div className="avatar-player placeholder">
        <p>Avatar video will appear here</p>
      </div>
    );
  }

  return (
    <div className="avatar-player">
      <video
        ref={videoRef}
        onEnded={handleEnded}
        playsInline
        aria-label="Avatar video"
      />
    </div>
  );
}
