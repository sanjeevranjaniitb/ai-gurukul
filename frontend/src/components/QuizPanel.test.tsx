import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import QuizPanel from './QuizPanel';
import type { QuizGenerateResponse } from '../api';

// ---------------------------------------------------------------------------
// Mock the API module
// ---------------------------------------------------------------------------

const mockGenerateQuiz = vi.fn<() => Promise<QuizGenerateResponse>>();

vi.mock('../api', () => ({
  generateQuiz: (...args: unknown[]) => mockGenerateQuiz(...args),
}));

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const qaHistory = [
  { question: 'What is photosynthesis?', answer: 'The process by which plants convert sunlight into energy.' },
  { question: 'Where does it occur?', answer: 'In the chloroplasts of plant cells.' },
];

const mockQuizResponse: QuizGenerateResponse = {
  quiz_id: 'test-uuid',
  questions: [
    {
      id: 0,
      question: 'What organelle is responsible for photosynthesis?',
      options: [
        { index: 0, text: 'A) Mitochondria' },
        { index: 1, text: 'B) Nucleus' },
        { index: 2, text: 'C) Chloroplast' },
        { index: 3, text: 'D) Ribosome' },
      ],
      correct_answer: 2,
      explanation: 'Chloroplasts contain chlorophyll.',
    },
    {
      id: 1,
      question: 'Which is a product of photosynthesis?',
      options: [
        { index: 0, text: 'A) Carbon dioxide' },
        { index: 1, text: 'B) Glucose' },
        { index: 2, text: 'C) Nitrogen' },
        { index: 3, text: 'D) Water' },
      ],
      correct_answer: 1,
      explanation: 'Glucose is produced during photosynthesis.',
    },
  ],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderPanel(overrides: Partial<Parameters<typeof QuizPanel>[0]> = {}) {
  const onClose = vi.fn();
  const onQuizComplete = vi.fn();
  const utils = render(
    <QuizPanel
      qaHistory={qaHistory}
      onClose={onClose}
      onQuizComplete={onQuizComplete}
      {...overrides}
    />,
  );
  return { ...utils, onClose, onQuizComplete };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('QuizPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---- Loading state ----

  it('shows a loading indicator while waiting for the API response', () => {
    // Never resolve so we stay in loading
    mockGenerateQuiz.mockReturnValue(new Promise(() => {}));
    renderPanel();

    expect(screen.getByRole('status', { name: /loading quiz/i })).toBeInTheDocument();
    expect(screen.getByText(/generating your quiz/i)).toBeInTheDocument();
  });

  // ---- Questions render correctly ----

  it('renders questions and radio-button options after API response', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    renderPanel();

    // Wait for questions to appear
    await waitFor(() => {
      expect(screen.getByText(/What organelle is responsible for photosynthesis/)).toBeInTheDocument();
    });

    expect(screen.getByText(/Which is a product of photosynthesis/)).toBeInTheDocument();

    // All options rendered
    expect(screen.getByText('A) Mitochondria')).toBeInTheDocument();
    expect(screen.getByText('B) Nucleus')).toBeInTheDocument();
    expect(screen.getByText('C) Chloroplast')).toBeInTheDocument();
    expect(screen.getByText('D) Ribosome')).toBeInTheDocument();
    expect(screen.getByText('B) Glucose')).toBeInTheDocument();

    // Radio buttons present (2 questions × 4 options = 8)
    const radios = screen.getAllByRole('radio');
    expect(radios).toHaveLength(8);
  });

  // ---- Option selection & submit enable/disable ----

  it('disables submit until all questions are answered, then enables it', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    const submitBtn = screen.getByRole('button', { name: /submit quiz answers/i });
    expect(submitBtn).toBeDisabled();

    // Answer only the first question
    await user.click(screen.getByLabelText('C) Chloroplast'));
    expect(submitBtn).toBeDisabled();

    // Answer the second question — now submit should be enabled
    await user.click(screen.getByLabelText('B) Glucose'));
    expect(submitBtn).toBeEnabled();
  });

  // ---- Grading: all correct ----

  it('shows perfect score when all answers are correct', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    const { onQuizComplete } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    // Select correct answers: Q0 → index 2 (C) Chloroplast), Q1 → index 1 (B) Glucose)
    await user.click(screen.getByLabelText('C) Chloroplast'));
    await user.click(screen.getByLabelText('B) Glucose'));
    await user.click(screen.getByRole('button', { name: /submit quiz answers/i }));

    expect(screen.getByText('2/2')).toBeInTheDocument();
    expect(screen.getByText(/perfect score/i)).toBeInTheDocument();
    expect(onQuizComplete).toHaveBeenCalledWith(2, 2);
  });

  // ---- Grading: all wrong ----

  it('shows zero score when all answers are wrong', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    const { onQuizComplete } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    // Select wrong answers: Q0 → index 0 (Mitochondria), Q1 → index 0 (Carbon dioxide)
    await user.click(screen.getByLabelText('A) Mitochondria'));
    await user.click(screen.getByLabelText('A) Carbon dioxide'));
    await user.click(screen.getByRole('button', { name: /submit quiz answers/i }));

    expect(screen.getByText('0/2')).toBeInTheDocument();
    expect(screen.getByText(/keep studying/i)).toBeInTheDocument();
    expect(onQuizComplete).toHaveBeenCalledWith(0, 2);
  });

  // ---- Grading: mixed results with score display ----

  it('shows mixed score with correct/incorrect indicators and explanations', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    const { onQuizComplete } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    // Q0 correct (Chloroplast), Q1 wrong (Carbon dioxide)
    await user.click(screen.getByLabelText('C) Chloroplast'));
    await user.click(screen.getByLabelText('A) Carbon dioxide'));
    await user.click(screen.getByRole('button', { name: /submit quiz answers/i }));

    expect(screen.getByText('1/2')).toBeInTheDocument();
    // Correct/incorrect indicators
    expect(screen.getByText('✅ Correct')).toBeInTheDocument();
    expect(screen.getByText('❌ Incorrect')).toBeInTheDocument();
    // Explanations shown
    expect(screen.getByText('Chloroplasts contain chlorophyll.')).toBeInTheDocument();
    expect(screen.getByText('Glucose is produced during photosynthesis.')).toBeInTheDocument();
    expect(onQuizComplete).toHaveBeenCalledWith(1, 2);
  });

  // ---- Error state with retry ----

  it('shows error message with retry button on API failure', async () => {
    mockGenerateQuiz.mockRejectedValue(new Error('Service unavailable'));
    renderPanel();

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    expect(screen.getByText('Service unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('retries quiz generation when retry button is clicked', async () => {
    mockGenerateQuiz
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(mockQuizResponse);

    const user = userEvent.setup();
    renderPanel();

    // Wait for error
    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    // Click retry
    await user.click(screen.getByRole('button', { name: /retry/i }));

    // Should now show questions
    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    expect(mockGenerateQuiz).toHaveBeenCalledTimes(2);
  });

  // ---- onClose callback ----

  it('calls onClose when the close button is clicked', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const { onClose } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /close quiz$/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Escape key is pressed', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    const { onClose } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  // ---- onQuizComplete callback ----

  it('fires onQuizComplete with score and total after grading', async () => {
    mockGenerateQuiz.mockResolvedValue(mockQuizResponse);
    const user = userEvent.setup();
    const { onQuizComplete } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/What organelle/)).toBeInTheDocument();
    });

    await user.click(screen.getByLabelText('C) Chloroplast'));
    await user.click(screen.getByLabelText('B) Glucose'));
    await user.click(screen.getByRole('button', { name: /submit quiz answers/i }));

    expect(onQuizComplete).toHaveBeenCalledOnce();
    expect(onQuizComplete).toHaveBeenCalledWith(2, 2);
  });
});
