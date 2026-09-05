const EXAMPLES = [
  "Where is the CSS box model explained?",
  "What are semantic HTML tags and why use them?",
  "How does CSS Flexbox work for layout alignment?",
  "Where is DOM manipulation and event handling introduced?",
];

interface Props {
  onSelect: (question: string) => void;
  disabled?: boolean;
}

export function ExampleQuestions({ onSelect, disabled }: Props) {
  return (
    <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
      {EXAMPLES.map((q) => (
        <button
          key={q}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(q)}
          className="rounded-full border border-white/10 bg-slate-900/60 px-3.5 py-1.5 text-xs text-slate-300 transition hover:border-indigo-500/50 hover:bg-indigo-500/10 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {q}
        </button>
      ))}
    </div>
  );
}
