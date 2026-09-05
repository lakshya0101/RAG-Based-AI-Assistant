import { API_URL } from "../lib/api";

type Status = "checking" | "online" | "offline";

interface Props {
  status: Status;
  chunkCount?: number;
}

export function Header({ status, chunkCount }: Props) {
  const statusConfig = {
    checking: { dot: "bg-amber-400 animate-pulse", label: "Connecting..." },
    online: { dot: "bg-emerald-400", label: "Backend Ready" },
    offline: { dot: "bg-rose-400", label: "Backend Offline" },
  }[status];

  return (
    <header className="glass sticky top-0 z-10 border-b border-white/10 shadow-lg shadow-black/20">
      <div className="mx-auto flex max-w-4xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-indigo-600 to-violet-700 shadow-md shadow-indigo-500/25">
            <svg
              className="h-5 w-5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-white sm:text-lg">
              RAG AI Teaching Assistant
            </h1>
            <p className="text-xs text-slate-400">
              Video lecture knowledge retrieval & timestamp citation
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-3 py-1">
            <span className={`h-2 w-2 rounded-full ${statusConfig.dot}`} />
            <span className="text-xs font-medium text-slate-300">{statusConfig.label}</span>
          </div>
          {chunkCount !== undefined && chunkCount > 0 && (
            <span className="hidden text-xs text-indigo-300 sm:inline bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded-full">
              {chunkCount} chunks indexed
            </span>
          )}
        </div>
      </div>

      {status === "offline" && (
        <div className="border-t border-rose-500/20 bg-rose-500/10 px-4 py-2 text-center text-xs text-rose-200">
          Backend service unreachable at <code className="font-mono text-white/90">{API_URL}</code>. Ensure the FastAPI server is running.
        </div>
      )}
    </header>
  );
}
