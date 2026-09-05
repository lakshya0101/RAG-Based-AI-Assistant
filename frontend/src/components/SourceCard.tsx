import type { Source } from "../lib/api";

interface Props {
  source: Source;
}

export function SourceCard({ source }: Props) {
  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/70 p-3.5 transition hover:border-indigo-500/40 hover:bg-slate-900/90 shadow-sm">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-indigo-500/20 border border-indigo-500/30 px-2 py-0.5 text-xs font-semibold text-indigo-300">
            Video {source.video_number}
          </span>
          <span className="rounded-md bg-slate-800 border border-slate-700/80 px-2 py-0.5 font-mono text-xs text-emerald-400">
            {source.start} – {source.end}
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          score: {source.score.toFixed(3)}
        </span>
      </div>
      <p className="text-sm font-medium text-slate-200">{source.title}</p>
      <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-slate-400 italic">
        "{source.excerpt}"
      </p>
    </div>
  );
}
