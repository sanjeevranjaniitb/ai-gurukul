import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { gradeQuiz } from './QuizPanel';
import type { QuizQuestion } from '../api';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/** Generate a single QuizQuestion with 2–4 options and a valid correct_answer. */
const quizOptionArb = (index: number) =>
  fc.string({ minLength: 1, maxLength: 50 }).map((text) => ({ index, text }));

const quizQuestionArb = (id: number): fc.Arbitrary<QuizQuestion> =>
  fc
    .integer({ min: 2, max: 4 })
    .chain((optionCount) =>
      fc.tuple(
        fc.string({ minLength: 1, maxLength: 100 }), // question text
        fc.tuple(
          ...Array.from({ length: optionCount }, (_, i) => quizOptionArb(i)),
        ),
        fc.integer({ min: 0, max: optionCount - 1 }), // correct_answer
        fc.string({ minLength: 0, maxLength: 100 }), // explanation
      ),
    )
    .map(([question, options, correct_answer, explanation]) => ({
      id,
      question,
      options,
      correct_answer,
      explanation,
    }));

/** Generate a non-empty array of QuizQuestions with unique sequential ids. */
const questionsArb: fc.Arbitrary<QuizQuestion[]> = fc
  .integer({ min: 1, max: 10 })
  .chain((count) =>
    fc.tuple(...Array.from({ length: count }, (_, i) => quizQuestionArb(i))),
  )
  .map((qs) => [...qs]);

/**
 * Generate a Map<number, number> of user answers for the given questions.
 * Each answer is a random option index (may or may not match correct_answer).
 */
function userAnswersArb(
  questions: QuizQuestion[],
): fc.Arbitrary<Map<number, number>> {
  if (questions.length === 0) return fc.constant(new Map());
  return fc
    .tuple(
      ...questions.map((q) =>
        fc.integer({ min: 0, max: q.options.length - 1 }),
      ),
    )
    .map((answers) => {
      const map = new Map<number, number>();
      questions.forEach((q, i) => map.set(q.id, answers[i]));
      return map;
    });
}

// ---------------------------------------------------------------------------
// Property Tests
// ---------------------------------------------------------------------------

describe('gradeQuiz property tests', () => {
  /**
   * **Validates: Requirements 3.5**
   *
   * Property 4: Grading score bound — ∀ grading result G:
   *   G.score ≤ G.total AND G.total === questions.length
   */
  it('Property 4: score is bounded by total and total equals questions.length', () => {
    fc.assert(
      fc.property(
        questionsArb.chain((qs) =>
          userAnswersArb(qs).map((answers) => ({ qs, answers })),
        ),
        ({ qs, answers }) => {
          const result = gradeQuiz(qs, answers);

          // score never exceeds total
          expect(result.score).toBeLessThanOrEqual(result.total);
          // score is non-negative
          expect(result.score).toBeGreaterThanOrEqual(0);
          // total matches the number of questions
          expect(result.total).toBe(qs.length);
          // results array has one entry per question
          expect(result.results).toHaveLength(qs.length);
        },
      ),
      { numRuns: 200 },
    );
  });

  /**
   * **Validates: Requirements 3.5**
   *
   * Property 8: Grading idempotency — Given the same questions and
   * userAnswers, gradeQuiz() always returns the same score and results.
   */
  it('Property 8: grading is idempotent for the same inputs', () => {
    fc.assert(
      fc.property(
        questionsArb.chain((qs) =>
          userAnswersArb(qs).map((answers) => ({ qs, answers })),
        ),
        ({ qs, answers }) => {
          const first = gradeQuiz(qs, answers);
          const second = gradeQuiz(qs, answers);

          // Same score
          expect(second.score).toBe(first.score);
          // Same total
          expect(second.total).toBe(first.total);
          // Same per-question results
          expect(second.results).toEqual(first.results);
        },
      ),
      { numRuns: 200 },
    );
  });
});
