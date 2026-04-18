import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import PdfUpload from './PdfUpload';

vi.mock('../api', () => ({
  uploadPdf: vi.fn(),
}));

import { uploadPdf } from '../api';

const mockedUploadPdf = vi.mocked(uploadPdf);

function createFile(name: string, sizeBytes: number, type: string): File {
  const buffer = new ArrayBuffer(sizeBytes);
  return new File([buffer], name, { type });
}

describe('PdfUpload', () => {
  const onUploaded = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders a file input accepting PDF', () => {
    render(<PdfUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload PDF document');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'file');
    expect(input).toHaveAttribute('accept', '.pdf');
  });

  it('shows error when PDF exceeds 50 MB', async () => {
    const user = userEvent.setup();
    render(<PdfUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload PDF document');

    const file = createFile('large.pdf', 51 * 1024 * 1024, 'application/pdf');
    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'PDF must be 50 MB or smaller.'
      );
    });
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it('calls uploadPdf and shows success info on valid file', async () => {
    const mockResponse = {
      document_id: 'doc-456',
      name: 'report.pdf',
      page_count: 12,
      chunk_count: 45,
    };
    mockedUploadPdf.mockResolvedValue(mockResponse);

    const user = userEvent.setup();
    render(<PdfUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload PDF document');

    const file = createFile('report.pdf', 2 * 1024 * 1024, 'application/pdf');
    await user.upload(input, file);

    await waitFor(() => {
      expect(mockedUploadPdf).toHaveBeenCalledWith(file);
    });

    await waitFor(() => {
      expect(onUploaded).toHaveBeenCalledWith(mockResponse);
    });

    expect(screen.getByText(/report\.pdf/)).toBeInTheDocument();
    expect(screen.getByText(/12 pages/)).toBeInTheDocument();
    expect(screen.getByText(/45 chunks/)).toBeInTheDocument();
  });

  it('shows upload error from API', async () => {
    mockedUploadPdf.mockRejectedValue(new Error('Corrupted PDF'));

    const user = userEvent.setup();
    render(<PdfUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload PDF document');

    const file = createFile('bad.pdf', 1024, 'application/pdf');
    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Corrupted PDF');
    });
    expect(onUploaded).not.toHaveBeenCalled();
  });
});
