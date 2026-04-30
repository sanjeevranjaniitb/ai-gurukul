import { useState, useEffect, useCallback, useRef } from 'react';
import { generateDocQuiz } from '../api';
import type { QuizQuestion, QuizGenerateResponse } from '../api';

interface DocQuizModalProps {
  documentId: string;
  documentName: string;
  onClose: () => void;
  preloadedQuiz?: QuizGenerateResponse | null;
}

type AnswerState = { selected: number; isCorrect: boolean } | null;

export default function DocQuizModal({ documentId, documentName, onClose, preloadedQuiz }: DocQuizModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [answers, setAnswers] = useState<Map<number, AnswerState>>(new Map());
  const [score, setScore] = useState(0);
  const [totalAnswered, setTotalAnswered] = useState(0);
  const fetchedRef = useRef(false);

  const fetchQuiz = useCallback(async () => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    // Use preloaded data if available
    if (preloadedQuiz && preloadedQuiz.questions.length > 0) {
      setQuestions(preloadedQuiz.questions);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError('');
    try {
      const resp = await generateDocQuiz(documentId, 10);
      setQuestions(resp.questions);
      setAnswers(new Map());
      setScore(0);
      setTotalAnswered(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate quiz.');
    } finally {
      setLoading(false);
    }
  }, [documentId, preloadedQuiz]);

  useEffect(() => { fetchQuiz(); }, [fetchQuiz]);

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  const handleSelect = (questionId: number, optionIndex: number) => {
    if (answers.has(questionId)) return; // already answered
    const q = questions.find(qq => qq.id === questionId);
    if (!q) return;
    const isCorrect = optionIndex === q.correct_answer;
    setAnswers(prev => {
      const next = new Map(prev);
      next.set(questionId, { selected: optionIndex, isCorrect });
      return next;
    });
    setTotalAnswered(prev => prev + 1);
    if (isCorrect) setScore(prev => prev + 1);
  };

  const allDone = totalAnswered === questions.length && questions.length > 0;

  return (
    <div style={overlay} role="dialog" aria-modal="true" aria-label="Document Knowledge Quiz">
      <div style={modal}>
        {/* Header */}
        <div style={header}>
          <div>
            <h2 style={title}>📄 Document Knowledge Quiz</h2>
            <p style={subtitle}>{documentName}</p>
          </div>
          <div style={headerRight}>
            {totalAnswered > 0 && (
              <span style={scoreBadge}>{score}/{totalAnswered}</span>
            )}
            <button style={closeBtn} onClick={onClose} aria-label="Close">✕</button>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div style={centerMsg}>
            <div style={spinner} />
            <p>Generating 10 questions from your document…</p>
            <p style={hintText}>This may take a moment</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div style={centerMsg}>
            <p style={{ color: '#c0392b' }}>{error}</p>
            <button style={retryBtn} onClick={fetchQuiz}>Retry</button>
          </div>
        )}

        {/* Questions */}
        {!loading && !error && questions.length > 0 && (
          <div style={questionsContainer}>
            {questions.map((q, qi) => {
              const ans = answers.get(q.id);
              const answered = ans !== null && ans !== undefined;
              return (
                <div key={q.id} style={{
                  ...questionCard,
                  ...(answered ? (ans!.isCorrect ? correctCard : incorrectCard) : {}),
                }}>
                  <p style={qText}>{qi + 1}. {q.question}</p>
                  <div style={optGrid}>
                    {q.options.map(opt => {
                      const isSelected = answered && ans!.selected === opt.index;
                      const isCorrectOpt = answered && opt.index === q.correct_answer;
                      let btnStyle = { ...optBtn };
                      if (answered) {
                        if (isCorrectOpt) btnStyle = { ...optBtn, ...optCorrect };
                        else if (isSelected) btnStyle = { ...optBtn, ...optWrong };
                        else btnStyle = { ...optBtn, opacity: 0.5, cursor: 'default' };
                      }
                      return (
                        <button
                          key={opt.index}
                          style={btnStyle}
                          onClick={() => handleSelect(q.id, opt.index)}
                          disabled={answered}
                          aria-label={opt.text}
                        >
                          {opt.text}
                        </button>
                      );
                    })}
                  </div>
                  {answered && (
                    <div style={feedbackRow}>
                      <span>{ans!.isCorrect ? '✅ Correct' : '❌ Incorrect'}</span>
                      {q.explanation && <span style={explText}>{q.explanation}</span>}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Final score */}
            {allDone && (
              <div style={finalScore}>
                <h3 style={{ margin: '0 0 8px' }}>Quiz Complete!</h3>
                <div style={bigScore}>{score}/{questions.length}</div>
                <p style={{ margin: '8px 0 0', color: '#666' }}>
                  {score === questions.length ? 'Perfect! You know this document well 🎉' :
                   score >= questions.length * 0.7 ? 'Great understanding! 👍' :
                   score >= questions.length * 0.4 ? 'Good start — keep learning! 📚' :
                   'Time to study the document more carefully 💪'}
                </p>
                <button style={closeFinalBtn} onClick={onClose}>Done</button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Styles
const overlay: React.CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  zIndex: 1000, padding: '1rem',
};
const modal: React.CSSProperties = {
  background: '#fff', borderRadius: '14px', maxWidth: '700px', width: '100%',
  maxHeight: '90vh', overflowY: 'auto', padding: '24px',
  boxShadow: '0 12px 40px rgba(0,0,0,0.25)',
};
const header: React.CSSProperties = {
  display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
  marginBottom: '20px', borderBottom: '1px solid #eee', paddingBottom: '12px',
};
const headerRight: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: '12px',
};
const title: React.CSSProperties = { margin: 0, fontSize: '1.3rem', color: '#1a1a2e' };
const subtitle: React.CSSProperties = { margin: '4px 0 0', fontSize: '0.85rem', color: '#888' };
const scoreBadge: React.CSSProperties = {
  padding: '4px 14px', background: 'linear-gradient(135deg, #667eea, #764ba2)',
  borderRadius: '16px', color: '#fff', fontWeight: 700, fontSize: '0.95rem',
};
const closeBtn: React.CSSProperties = {
  background: 'none', border: 'none', fontSize: '1.4rem', cursor: 'pointer', color: '#666',
};
const centerMsg: React.CSSProperties = { textAlign: 'center', padding: '40px 16px' };
const spinner: React.CSSProperties = {
  display: 'inline-block', width: '32px', height: '32px',
  border: '3px solid #e0e0e0', borderTopColor: '#4a6cf7', borderRadius: '50%',
  animation: 'quizSpin 0.8s linear infinite', margin: '0 auto 12px',
};
const hintText: React.CSSProperties = { fontSize: '0.85rem', color: '#999' };
const retryBtn: React.CSSProperties = {
  marginTop: '12px', padding: '8px 20px', background: '#4a6cf7', color: '#fff',
  border: 'none', borderRadius: '8px', cursor: 'pointer', fontSize: '0.9rem',
};
const questionsContainer: React.CSSProperties = { display: 'flex', flexDirection: 'column', gap: '12px' };
const questionCard: React.CSSProperties = {
  padding: '14px', border: '1px solid #e8e8e8', borderRadius: '10px',
  background: '#fafbff', transition: 'border-color 0.3s, background 0.3s',
};
const correctCard: React.CSSProperties = { borderColor: '#28a745', background: '#f0faf3' };
const incorrectCard: React.CSSProperties = { borderColor: '#dc3545', background: '#fef5f5' };
const qText: React.CSSProperties = { margin: '0 0 10px', fontWeight: 600, fontSize: '0.95rem', color: '#1a1a2e' };
const optGrid: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' };
const optBtn: React.CSSProperties = {
  padding: '8px 10px', background: '#fff', border: '1px solid #d0d7ff',
  borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem', textAlign: 'left',
  transition: 'all 0.2s', color: '#333',
};
const optCorrect: React.CSSProperties = {
  background: '#d4edda', borderColor: '#28a745', color: '#155724', fontWeight: 600,
};
const optWrong: React.CSSProperties = {
  background: '#f8d7da', borderColor: '#dc3545', color: '#721c24',
};
const feedbackRow: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: '10px', marginTop: '8px', fontSize: '0.85rem',
};
const explText: React.CSSProperties = { color: '#666', fontStyle: 'italic' };
const finalScore: React.CSSProperties = {
  textAlign: 'center', padding: '20px', background: 'linear-gradient(135deg, #f8f9ff, #eef1ff)',
  borderRadius: '12px', border: '1px solid #d0d7ff',
};
const bigScore: React.CSSProperties = { fontSize: '2.5rem', fontWeight: 700, color: '#4a6cf7' };
const closeFinalBtn: React.CSSProperties = {
  marginTop: '16px', padding: '10px 28px',
  background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
  border: 'none', borderRadius: '20px', fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer',
};
