import { useState, useEffect, useCallback } from 'react';
import { generateQuiz } from '../api';
import type { QuizQuestion } from '../api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface QuestionResult {
  questionId: number;
  selectedAnswer: number;
  correctAnswer: number;
  isCorrect: boolean;
  explanation: string;
}

export interface GradeResult {
  score: number;
  total: number;
  results: QuestionResult[];
}

// ---------------------------------------------------------------------------
// Pure grading function (exported for property-based testing)
// ---------------------------------------------------------------------------

export function gradeQuiz(
  questions: QuizQuestion[],
  userAnswers: Map<number, number>,
): GradeResult {
  let score = 0;
  const results: QuestionResult[] = [];

  for (const q of questions) {
    const selected = userAnswers.get(q.id) ?? -1;
    const isCorrect = selected === q.correct_answer;
    if (isCorrect) score++;

    results.push({
      questionId: q.id,
      selectedAnswer: selected,
      correctAnswer: q.correct_answer,
      isCorrect,
      explanation: q.explanation,
    });
  }

  return { score, total: questions.length, results };
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface QuizPanelProps {
  qaHistory: Array<{ question: string; answer: string }>;
  onClose: () => void;
  onQuizComplete: (score: number, total: number) => void;
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: '1rem',
  },
  panel: {
    background: '#fff',
    borderRadius: '12px',
    maxWidth: '640px',
    width: '100%',
    maxHeight: '85vh',
    overflowY: 'auto',
    padding: '2rem',
    boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
    position: 'relative',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.5rem',
  },
  title: {
    margin: 0,
    fontSize: '1.4rem',
    color: '#1a1a2e',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    fontSize: '1.5rem',
    cursor: 'pointer',
    color: '#666',
    padding: '0.25rem 0.5rem',
    borderRadius: '4px',
  },
  loading: {
    textAlign: 'center' as const,
    padding: '3rem 1rem',
  },
  spinner: {
    display: 'inline-block',
    width: '36px',
    height: '36px',
    border: '3px solid #e0e0e0',
    borderTopColor: '#4a6cf7',
    borderRadius: '50%',
    animation: 'quizSpin 0.8s linear infinite',
  },
  error: {
    textAlign: 'center' as const,
    padding: '2rem 1rem',
    color: '#c0392b',
  },
  retryBtn: {
    marginTop: '1rem',
    padding: '0.5rem 1.5rem',
    background: '#4a6cf7',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '0.95rem',
  },
  questionCard: {
    marginBottom: '1.5rem',
    padding: '1rem',
    border: '1px solid #e0e0e0',
    borderRadius: '8px',
    background: '#fafafa',
  },
  questionText: {
    fontWeight: 600,
    marginBottom: '0.75rem',
    color: '#1a1a2e',
  },
  optionLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.4rem 0.5rem',
    borderRadius: '4px',
    cursor: 'pointer',
    marginBottom: '0.25rem',
  },
  optionLabelHover: {
    background: '#f0f0f0',
  },
  submitBtn: {
    display: 'block',
    width: '100%',
    padding: '0.75rem',
    background: '#4a6cf7',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
  submitBtnDisabled: {
    display: 'block',
    width: '100%',
    padding: '0.75rem',
    background: '#b0b0b0',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'not-allowed',
  },
  scoreSummary: {
    textAlign: 'center' as const,
    marginBottom: '1.5rem',
  },
  scoreValue: {
    fontSize: '2.5rem',
    fontWeight: 700,
    color: '#4a6cf7',
  },
  resultCard: {
    marginBottom: '1rem',
    padding: '1rem',
    borderRadius: '8px',
    border: '1px solid',
  },
  resultCorrect: {
    borderColor: '#27ae60',
    background: '#eafaf1',
  },
  resultIncorrect: {
    borderColor: '#c0392b',
    background: '#fdedec',
  },
  indicator: {
    fontWeight: 600,
    marginBottom: '0.25rem',
  },
  explanation: {
    fontSize: '0.9rem',
    color: '#555',
    marginTop: '0.5rem',
    fontStyle: 'italic',
  },
};

// Keyframe animation injected once
const SPIN_STYLE_ID = 'quiz-panel-spin';
function ensureSpinKeyframes() {
  if (typeof document === 'undefined') return;
  if (document.getElementById(SPIN_STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = SPIN_STYLE_ID;
  style.textContent = `@keyframes quizSpin { to { transform: rotate(360deg); } }`;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type Phase = 'loading' | 'quiz' | 'results' | 'error';

export default function QuizPanel({
  qaHistory,
  onClose,
  onQuizComplete,
}: QuizPanelProps) {
  const [phase, setPhase] = useState<Phase>('loading');
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [userAnswers, setUserAnswers] = useState<Map<number, number>>(new Map());
  const [gradeResult, setGradeResult] = useState<GradeResult | null>(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Inject spinner keyframes
  useEffect(() => {
    ensureSpinKeyframes();
  }, []);

  const fetchQuiz = useCallback(async () => {
    setPhase('loading');
    setErrorMessage('');
    try {
      const response = await generateQuiz(qaHistory);
      if (response.questions.length === 0) {
        setErrorMessage(
          "Couldn't generate questions — try asking more questions first.",
        );
        setPhase('error');
        return;
      }
      setQuestions(response.questions);
      setUserAnswers(new Map());
      setGradeResult(null);
      setPhase('quiz');
    } catch (err: unknown) {
      setErrorMessage(
        err instanceof Error ? err.message : 'Failed to generate quiz.',
      );
      setPhase('error');
    }
  }, [qaHistory]);

  // Fetch quiz on mount
  useEffect(() => {
    fetchQuiz();
  }, [fetchQuiz]);

  const handleSelect = (questionId: number, optionIndex: number) => {
    setUserAnswers((prev) => {
      const next = new Map(prev);
      next.set(questionId, optionIndex);
      return next;
    });
  };

  const allAnswered = questions.length > 0 && questions.every((q) => userAnswers.has(q.id));

  const handleSubmit = () => {
    if (!allAnswered) return;
    const result = gradeQuiz(questions, userAnswers);
    setGradeResult(result);
    setPhase('results');
    onQuizComplete(result.score, result.total);
  };

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div
      style={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-label="Knowledge Quiz"
    >
      <div style={styles.panel}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>🧠 Knowledge Quiz</h2>
          <button
            style={styles.closeBtn}
            onClick={onClose}
            aria-label="Close quiz"
          >
            ✕
          </button>
        </div>

        {/* Loading */}
        {phase === 'loading' && (
          <div style={styles.loading} role="status" aria-label="Loading quiz">
            <div style={styles.spinner} />
            <p>Generating your quiz…</p>
          </div>
        )}

        {/* Error */}
        {phase === 'error' && (
          <div style={styles.error} role="alert">
            <p>{errorMessage}</p>
            <button
              style={styles.retryBtn}
              onClick={fetchQuiz}
              aria-label="Retry quiz generation"
            >
              Retry
            </button>
          </div>
        )}

        {/* Quiz questions */}
        {phase === 'quiz' && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSubmit();
            }}
            aria-label="Quiz questions"
          >
            {questions.map((q, qi) => (
              <fieldset
                key={q.id}
                style={styles.questionCard}
                aria-label={`Question ${qi + 1}`}
              >
                <legend style={styles.questionText}>
                  {qi + 1}. {q.question}
                </legend>
                {q.options.map((opt) => (
                  <label
                    key={opt.index}
                    style={styles.optionLabel}
                  >
                    <input
                      type="radio"
                      name={`question-${q.id}`}
                      value={opt.index}
                      checked={userAnswers.get(q.id) === opt.index}
                      onChange={() => handleSelect(q.id, opt.index)}
                      aria-label={opt.text}
                    />
                    <span>{opt.text}</span>
                  </label>
                ))}
              </fieldset>
            ))}

            <button
              type="submit"
              disabled={!allAnswered}
              style={allAnswered ? styles.submitBtn : styles.submitBtnDisabled}
              aria-label="Submit quiz answers"
            >
              Submit
            </button>
          </form>
        )}

        {/* Results */}
        {phase === 'results' && gradeResult && (
          <div aria-label="Quiz results">
            <div style={styles.scoreSummary}>
              <div style={styles.scoreValue}>
                {gradeResult.score}/{gradeResult.total}
              </div>
              <p>
                {gradeResult.score === gradeResult.total
                  ? 'Perfect score! 🎉'
                  : gradeResult.score >= gradeResult.total / 2
                    ? 'Good job! 👍'
                    : 'Keep studying! 📚'}
              </p>
            </div>

            {gradeResult.results.map((r, i) => {
              const q = questions.find((qq) => qq.id === r.questionId);
              return (
                <div
                  key={r.questionId}
                  style={{
                    ...styles.resultCard,
                    ...(r.isCorrect
                      ? styles.resultCorrect
                      : styles.resultIncorrect),
                  }}
                  aria-label={`Question ${i + 1} result: ${r.isCorrect ? 'correct' : 'incorrect'}`}
                >
                  <div style={styles.indicator}>
                    {r.isCorrect ? '✅ Correct' : '❌ Incorrect'}
                  </div>
                  <div style={styles.questionText}>
                    {i + 1}. {q?.question}
                  </div>
                  {!r.isCorrect && q && (
                    <div style={{ fontSize: '0.9rem', color: '#333' }}>
                      Your answer:{' '}
                      {q.options.find((o) => o.index === r.selectedAnswer)?.text ??
                        '(none)'}
                      <br />
                      Correct answer:{' '}
                      {q.options.find((o) => o.index === r.correctAnswer)?.text}
                    </div>
                  )}
                  {r.explanation && (
                    <div style={styles.explanation}>{r.explanation}</div>
                  )}
                </div>
              );
            })}

            <button
              style={{ ...styles.submitBtn, marginTop: '1.5rem' }}
              onClick={onClose}
              aria-label="Close quiz results"
            >
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
