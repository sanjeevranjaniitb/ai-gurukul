import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ChatInterface from './ChatInterface';

// Mock the useSSE hook
const mockAsk = vi.fn();
vi.mock('../hooks/useSSE', () => ({
  useSSE: () => ({ ask: mockAsk, cancel: vi.fn() }),
}));

describe('ChatInterface', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders question input and send button', () => {
    render(<ChatInterface avatarId="abc" />);
    expect(screen.getByLabelText('Question input')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('disables input when no avatarId is provided', () => {
    render(<ChatInterface avatarId={null} />);
    const input = screen.getByLabelText('Question input');
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute('placeholder', 'Upload an avatar first');
  });

  it('shows empty state message when no messages', () => {
    render(<ChatInterface avatarId="abc" />);
    expect(
      screen.getByText('Ask a question about your uploaded document.')
    ).toBeInTheDocument();
  });

  it('adds user message to chat history on submit', async () => {
    // Make ask resolve immediately
    mockAsk.mockImplementation(
      async (
        _q: string,
        _id: string,
        callbacks: { onDone?: () => void }
      ) => {
        callbacks.onDone?.();
      }
    );

    const user = userEvent.setup();
    render(<ChatInterface avatarId="abc" />);

    const input = screen.getByLabelText('Question input');
    await user.type(input, 'What is AI?');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(screen.getByText('What is AI?')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();
  });

  it('displays streamed assistant response tokens', async () => {
    mockAsk.mockImplementation(
      async (
        _q: string,
        _id: string,
        callbacks: { onTextToken?: (t: string) => void; onDone?: () => void }
      ) => {
        callbacks.onTextToken?.('Hello');
        callbacks.onTextToken?.(' world');
        callbacks.onDone?.();
      }
    );

    const user = userEvent.setup();
    render(<ChatInterface avatarId="abc" />);

    const input = screen.getByLabelText('Question input');
    await user.type(input, 'Hi');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(screen.getByText('Hello world')).toBeInTheDocument();
  });

  it('shows error message from SSE stream', async () => {
    mockAsk.mockImplementation(
      async (
        _q: string,
        _id: string,
        callbacks: { onError?: (msg: string) => void }
      ) => {
        callbacks.onError?.('Something went wrong');
      }
    );

    const user = userEvent.setup();
    render(<ChatInterface avatarId="abc" />);

    const input = screen.getByLabelText('Question input');
    await user.type(input, 'Test');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(
      screen.getByText('Error: Something went wrong')
    ).toBeInTheDocument();
  });

  it('calls onVideoChunk and onStageUpdate callbacks', async () => {
    const onVideoChunk = vi.fn();
    const onStageUpdate = vi.fn();

    mockAsk.mockImplementation(
      async (
        _q: string,
        _id: string,
        callbacks: {
          onVideoChunk?: (url: string, idx: number) => void;
          onStageUpdate?: (stage: string) => void;
          onDone?: () => void;
        }
      ) => {
        callbacks.onStageUpdate?.('generating');
        callbacks.onVideoChunk?.('/data/media/v1.mp4', 0);
        callbacks.onDone?.();
      }
    );

    const user = userEvent.setup();
    render(
      <ChatInterface
        avatarId="abc"
        onVideoChunk={onVideoChunk}
        onStageUpdate={onStageUpdate}
      />
    );

    const input = screen.getByLabelText('Question input');
    await user.type(input, 'Q');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(onStageUpdate).toHaveBeenCalledWith('generating');
    expect(onVideoChunk).toHaveBeenCalledWith('/data/media/v1.mp4', 0);
  });
});
