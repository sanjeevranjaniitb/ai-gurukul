import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import AvatarPlayer from './AvatarPlayer';

// Stub HTMLMediaElement.play since jsdom doesn't implement it
beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('AvatarPlayer', () => {
  it('renders placeholder when video queue is empty', () => {
    render(<AvatarPlayer videoQueue={[]} onSegmentEnd={vi.fn()} />);
    expect(
      screen.getByText('Avatar video will appear here')
    ).toBeInTheDocument();
    expect(screen.queryByRole('video')).not.toBeInTheDocument();
  });

  it('renders video element when queue has items', () => {
    render(
      <AvatarPlayer
        videoQueue={['/data/media/v1.mp4']}
        onSegmentEnd={vi.fn()}
      />
    );
    const video = document.querySelector('video');
    expect(video).toBeInTheDocument();
    expect(video).toHaveAttribute('aria-label', 'Avatar video');
  });

  it('starts playback when first item is added to queue', () => {
    render(
      <AvatarPlayer
        videoQueue={['/data/media/v1.mp4']}
        onSegmentEnd={vi.fn()}
      />
    );
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it('has playsInline attribute for mobile compatibility', () => {
    render(
      <AvatarPlayer
        videoQueue={['/data/media/v1.mp4']}
        onSegmentEnd={vi.fn()}
      />
    );
    const video = document.querySelector('video');
    expect(video).toHaveAttribute('playsinline');
  });
});
