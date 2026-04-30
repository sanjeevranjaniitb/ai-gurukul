import { useState, useEffect, useCallback, useRef } from 'react';
import { generateQuiz } from '../api';
import type { QuizQuestion } from '../api';

// Re-export gradeQuiz for backward compatibility with property tests
export { gradeQuiz } from './QuizPanel';
export type { QuestionResult, GradeResult } from './QuizPanel';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface RollingQuizBannerProps {
  qaHistory: Array<{ question: string; answer: string }>;
  triggerRef?: React.MutableRefObject<(() => void) | null>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type BannerPhase = 'loading' | 'question' | 'feedback' | 'idle' | 'error';

export default function RollingQuizBanner({ qaHistory, triggerRef }: RollingQuizBannerProps) {
  const [phase, setPhase] = useState<BannerPhase>('idle');
  const [currentQuestion, setCurrentQuestion] = useState<QuizQuestion | null>(null);
  const [feedbackCorrect, setFeedbackCorrect] = useState<boolean | null>(null);
  const [feedbackExplanation, setFeedbackExplanation] = useState('');
  const [correctAnswerText, setCorrectAnswerText] = useState('');
  const [totalAsked, setTotalAsked] = useState(0);
  const [totalCorrect, setTotalCorrect] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const feedbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Expose trigger to parent via ref
  useEffect(() => {
    if (triggerRef) triggerRef.current = fetchOneQuestion;
    return () => { if (triggerRef) triggerRef.current = null; };
  });

  const fetchOneQuestion = useCallback(async () => {
    setPhase('loading');
    setErrorMsg('');
    try {
      const response = await generateQuiz(qaHistory, 1);
      if (response.questions.length === 0) {
        setPhase('idle');
        return;
      }
      setCurrentQuestion(response.questions[0]);
      setFeedbackCorrect(null);
      setPhase('question');
    } catch {
      setErrorMsg('Could not generate quiz question.');
      setPhase('error');
      // Auto-dismiss error after 4s
      setTimeout(() => setPhase('idle'), 4000);
    }
  }, [qaHistory]);

  const handleOptionClick = (optionIndex: number) => {
    if (!currentQuestion || phase !== 'question') return;

    const isCorrect = optionIndex === currentQuestion.correct_answer;
    setFeedbackCorrect(isCorrect);
    setFeedbackExplanation(currentQuestion.explanation || '');
    setCorrectAnswerText(
      currentQuestion.options.find((o) => o.index === currentQuestion.correct_answer)?.text ?? '',
    );
    setTotalAsked((prev) => prev + 1);
    if (isCorrect) setTotalCorrect((prev) => prev + 1);
    setPhase('feedback');

    // Auto-dismiss feedback after 3 seconds
    if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current);
    feedbackTimerRef.current = setTimeout(() => {
      setPhase('idle');
      setCurrentQuestion(null);
    }, 3500);
  };

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current);
    };
  }, []);

  // Don't render anything if nothing active
  if (phase === 'idle') return null;

  return (
    <div style={containerStyle} role="region" aria-label="Rolling quiz">
      {/* Error (inline) */}
      {phase === 'error' && (
        <div style={errorStyle} role="alert">
          {errorMsg}
        </div>
      )}

      {/* Popup overlay for loading, question, and feedback */}
      {(phase === 'loading' || phase === 'question' || phase === 'feedback') && (
        <div style={overlayStyle} role="dialog" aria-modal="true" aria-label="Quiz popup">
          <div style={popupStyle}>

            {/* Loading */}
            {phase === 'loading' && (
              <div style={loadingStyle}>
                <span style={spinnerSmallStyle} />
                <span>Generating quiz question…</span>
              </div>
            )}

            {/* Question */}
            {phase === 'question' && currentQuestion && (
              <>
                <div style={questionHeaderStyle}>
                  <span style={quizBadgeStyle}>🧠 Quick Quiz</span>
                  {totalAsked > 0 && (
                    <span style={scoreInlineStyle}>
                      Score: {totalCorrect}/{totalAsked}
                    </span>
                  )}
                </div>
                <p style={questionTextStyle}>{currentQuestion.question}</p>
                <div style={optionsGridStyle}>
                  {currentQuestion.options.map((opt) => (
                    <button
                      key={opt.index}
                      style={optionBtnStyle}
                      onClick={() => handleOptionClick(opt.index)}
                      aria-label={opt.text}
                    >
                      {opt.text}
                    </button>
                  ))}
                </div>
              </>
            )}

            {/* Feedback flash */}
            {phase === 'feedback' && (
              <div
                style={{
                  ...feedbackContainerStyle,
                  background: feedbackCorrect ? '#d4edda' : '#f8d7da',
                  borderColor: feedbackCorrect ? '#28a745' : '#dc3545',
                }}
                role="alert"
              >
                <div style={feedbackHeaderStyle}>
                  <span style={{ fontSize: '1.5rem' }}>
                    {feedbackCorrect ? '✅' : '❌'}
                  </span>
                  <span
                    style={{
                      fontWeight: 700,
                      color: feedbackCorrect ? '#155724' : '#721c24',
                      fontSize: '1.1rem',
                    }}
                  >
                    {feedbackCorrect ? 'Correct!' : 'Incorrect'}
                  </span>
                  <span style={scoreInlineStyle}>
                    Score: {totalCorrect}/{totalAsked}
                  </span>
                </div>
                {!feedbackCorrect && correctAnswerText && (
                  <p style={{ margin: '6px 0 0', fontSize: '0.9rem', color: '#721c24' }}>
                    Correct answer: {correctAnswerText}
                  </p>
                )}
                {feedbackExplanation && (
                  <p style={explanationStyle}>{feedbackExplanation}</p>
                )}
              </div>
            )}

          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const containerStyle: React.CSSProperties = {
  width: '100%',
  flexShrink: 0,
};

const idleBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-end',
  gap: '10px',
  padding: '8px 16px',
  margin: '0 0 6px',
};

const scoreChipStyle: React.CSSProperties = {
  padding: '5px 14px',
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  borderRadius: '16px',
  color: '#fff',
  fontWeight: 700,
  fontSize: '0.85rem',
  letterSpacing: '0.3px',
};

const takeQuizBtnStyle: React.CSSProperties = {
  padding: '8px 18px',
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: '#fff',
  border: 'none',
  borderRadius: '20px',
  fontWeight: 600,
  fontSize: '0.88rem',
  cursor: 'pointer',
  transition: 'transform 0.15s, box-shadow 0.15s',
  boxShadow: '0 2px 8px rgba(102, 126, 234, 0.35)',
  letterSpacing: '0.3px',
};

const loadingStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '10px',
  padding: '24px 16px',
  fontSize: '1rem',
  color: '#4a6cf7',
};

const spinnerSmallStyle: React.CSSProperties = {
  display: 'inline-block',
  width: '20px',
  height: '20px',
  border: '2px solid #c0c8f0',
  borderTopColor: '#4a6cf7',
  borderRadius: '50%',
  animation: 'quizSpin 0.8s linear infinite',
};

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1000,
  padding: '1rem',
  animation: 'quizFadeIn 0.25s ease-out',
};

const popupStyle: React.CSSProperties = {
  background: '#fff',
  borderRadius: '14px',
  maxWidth: '520px',
  width: '100%',
  padding: '24px',
  boxShadow: '0 12px 40px rgba(0,0,0,0.25)',
};

const questionHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: '8px',
};

const quizBadgeStyle: React.CSSProperties = {
  fontWeight: 700,
  fontSize: '0.85rem',
  color: '#4a6cf7',
};

const scoreInlineStyle: React.CSSProperties = {
  fontSize: '0.8rem',
  fontWeight: 600,
  color: '#666',
};

const questionTextStyle: React.CSSProperties = {
  margin: '0 0 14px',
  fontWeight: 600,
  fontSize: '1.05rem',
  color: '#1a1a2e',
  lineHeight: 1.4,
};

const optionsGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '8px',
};

const optionBtnStyle: React.CSSProperties = {
  padding: '10px 12px',
  background: '#f8f9ff',
  border: '1px solid #d0d7ff',
  borderRadius: '8px',
  cursor: 'pointer',
  fontSize: '0.9rem',
  textAlign: 'left',
  transition: 'background 0.15s, border-color 0.15s, transform 0.1s',
  color: '#333',
};

const feedbackContainerStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderRadius: '10px',
  border: '2px solid',
  margin: '0 0 8px',
  animation: 'quizFadeIn 0.3s ease-out',
};

const feedbackHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
};

const explanationStyle: React.CSSProperties = {
  margin: '6px 0 0',
  fontSize: '0.85rem',
  color: '#555',
  fontStyle: 'italic',
};

const errorStyle: React.CSSProperties = {
  padding: '8px 16px',
  background: '#fff3f3',
  border: '1px solid #ffcdd2',
  borderRadius: '8px',
  margin: '0 0 8px',
  fontSize: '0.85rem',
  color: '#c0392b',
};

// Inject keyframes once
if (typeof document !== 'undefined') {
  const id = 'rolling-quiz-keyframes';
  if (!document.getElementById(id)) {
    const style = document.createElement('style');
    style.id = id;
    style.textContent = `
      @keyframes quizSpin { to { transform: rotate(360deg); } }
      @keyframes quizFadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
    `;
    document.head.appendChild(style);
  }
}
