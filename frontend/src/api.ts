export interface AvatarUploadResponse {
  avatar_id: string;
  preview_url: string;
  landmarks_ready: boolean;
  visemes?: Record<string, string>;
  frame_url?: string;
}

export interface PdfUploadResponse {
  document_id: string;
  name: string;
  page_count: number;
  chunk_count: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  memory_usage_mb: number;
}

export async function uploadAvatar(file: File): Promise<AvatarUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/upload/avatar', { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Avatar upload failed');
  }
  return res.json();
}

export async function uploadPdf(file: File): Promise<PdfUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/upload/pdf', { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'PDF upload failed');
  }
  return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ---------------------------------------------------------------------------
// Quiz types and API client
// ---------------------------------------------------------------------------

export interface QuizQuestion {
  id: number;
  question: string;
  options: Array<{ index: number; text: string }>;
  correct_answer: number;
  explanation: string;
}

export interface QuizGenerateResponse {
  quiz_id: string;
  questions: QuizQuestion[];
}

export async function generateQuiz(
  qaHistory: Array<{ question: string; answer: string }>,
  numQuestions?: number
): Promise<QuizGenerateResponse> {
  const res = await fetch('/api/quiz/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      qa_history: qaHistory,
      ...(numQuestions !== undefined && { num_questions: numQuestions }),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Quiz generation failed');
  }
  return res.json();
}


export async function generateDocQuiz(
  documentId: string,
  numQuestions?: number
): Promise<QuizGenerateResponse> {
  const res = await fetch('/api/quiz/generate-from-doc', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      document_id: documentId,
      ...(numQuestions !== undefined && { num_questions: numQuestions }),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Document quiz generation failed');
  }
  return res.json();
}
