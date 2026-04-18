interface Props {
  stage: string;
  visible: boolean;
}

const STAGE_LABELS: Record<string, string> = {
  retrieving: 'Retrieving relevant context…',
  generating: 'Generating answer…',
  synthesizing: 'Synthesizing speech…',
  animating: 'Animating avatar…',
};

export default function ProcessingIndicator({ stage, visible }: Props) {
  if (!visible) return null;

  const label = STAGE_LABELS[stage] || stage;

  return (
    <div className="processing-indicator" role="status" aria-live="polite">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}
