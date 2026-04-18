import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import AvatarUpload from './AvatarUpload';

// Mock the api module
vi.mock('../api', () => ({
  uploadAvatar: vi.fn(),
}));

import { uploadAvatar } from '../api';

const mockedUploadAvatar = vi.mocked(uploadAvatar);

function createFile(name: string, sizeBytes: number, type: string): File {
  const buffer = new ArrayBuffer(sizeBytes);
  return new File([buffer], name, { type });
}

describe('AvatarUpload', () => {
  const onUploaded = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders a file input accepting PNG/JPG/JPEG', () => {
    render(<AvatarUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload avatar image');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'file');
    expect(input).toHaveAttribute('accept', '.png,.jpg,.jpeg');
  });

  it('shows error for invalid file format', async () => {
    // fireEvent.change bypasses the accept attribute filter,
    // allowing us to test the component's own validation logic
    const { fireEvent } = await import('@testing-library/react');
    render(<AvatarUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload avatar image');

    const file = createFile('avatar.gif', 1024, 'image/gif');
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Only PNG, JPG, and JPEG files are accepted.'
      );
    });
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it('shows error when file exceeds 10 MB', async () => {
    const user = userEvent.setup();
    render(<AvatarUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload avatar image');

    const file = createFile('avatar.png', 11 * 1024 * 1024, 'image/png');
    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Image must be 10 MB or smaller.'
      );
    });
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it('calls uploadAvatar and onUploaded on valid file', async () => {
    // Mock Image to simulate valid resolution
    const originalImage = globalThis.Image;
    const mockImage = class extends originalImage {
      constructor() {
        super();
        setTimeout(() => {
          Object.defineProperty(this, 'width', { value: 512 });
          Object.defineProperty(this, 'height', { value: 512 });
          this.onload?.(new Event('load'));
        }, 0);
      }
    };
    vi.stubGlobal('Image', mockImage);

    // Mock URL.createObjectURL
    const createObjectURLSpy = vi.fn(() => 'blob:mock-url');
    const revokeObjectURLSpy = vi.fn();
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: createObjectURLSpy,
      revokeObjectURL: revokeObjectURLSpy,
    });

    const mockResponse = {
      avatar_id: 'abc-123',
      preview_url: '/data/avatars/abc-123/original.png',
      landmarks_ready: true,
    };
    mockedUploadAvatar.mockResolvedValue(mockResponse);

    const user = userEvent.setup();
    render(<AvatarUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload avatar image');

    const file = createFile('avatar.png', 5 * 1024 * 1024, 'image/png');
    await user.upload(input, file);

    await waitFor(() => {
      expect(mockedUploadAvatar).toHaveBeenCalledWith(file);
    });

    await waitFor(() => {
      expect(onUploaded).toHaveBeenCalledWith(mockResponse);
    });

    vi.unstubAllGlobals();
  });

  it('shows upload error from API', async () => {
    const originalImage = globalThis.Image;
    const mockImage = class extends originalImage {
      constructor() {
        super();
        setTimeout(() => {
          Object.defineProperty(this, 'width', { value: 512 });
          Object.defineProperty(this, 'height', { value: 512 });
          this.onload?.(new Event('load'));
        }, 0);
      }
    };
    vi.stubGlobal('Image', mockImage);
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:mock-url'),
      revokeObjectURL: vi.fn(),
    });

    mockedUploadAvatar.mockRejectedValue(new Error('No face detected'));

    const user = userEvent.setup();
    render(<AvatarUpload onUploaded={onUploaded} />);
    const input = screen.getByLabelText('Upload avatar image');

    const file = createFile('avatar.jpg', 1024, 'image/jpeg');
    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('No face detected');
    });
    expect(onUploaded).not.toHaveBeenCalled();

    vi.unstubAllGlobals();
  });
});
