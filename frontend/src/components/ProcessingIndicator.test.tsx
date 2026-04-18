import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ProcessingIndicator from './ProcessingIndicator';

describe('ProcessingIndicator', () => {
  it('renders nothing when not visible', () => {
    const { container } = render(
      <ProcessingIndicator stage="generating" visible={false} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('displays mapped stage label for known stages', () => {
    render(<ProcessingIndicator stage="retrieving" visible={true} />);
    expect(
      screen.getByText('Retrieving relevant context…')
    ).toBeInTheDocument();
  });

  it('displays generating stage text', () => {
    render(<ProcessingIndicator stage="generating" visible={true} />);
    expect(screen.getByText('Generating answer…')).toBeInTheDocument();
  });

  it('displays synthesizing stage text', () => {
    render(<ProcessingIndicator stage="synthesizing" visible={true} />);
    expect(screen.getByText('Synthesizing speech…')).toBeInTheDocument();
  });

  it('displays animating stage text', () => {
    render(<ProcessingIndicator stage="animating" visible={true} />);
    expect(screen.getByText('Animating avatar…')).toBeInTheDocument();
  });

  it('falls back to raw stage string for unknown stages', () => {
    render(<ProcessingIndicator stage="custom-stage" visible={true} />);
    expect(screen.getByText('custom-stage')).toBeInTheDocument();
  });

  it('has role="status" for accessibility', () => {
    render(<ProcessingIndicator stage="generating" visible={true} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});
