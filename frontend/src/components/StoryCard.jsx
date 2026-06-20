import React, { useState } from 'react'

// A single generated story with the human-feedback controls that feed the
// continuous-improvement loop: thumbs up/down, or "edit & approve" which sends
// the improved text back as the preferred training target.
export default function StoryCard({ item, onFeedback }) {
  const [submitted, setSubmitted] = useState(null) // 'up' | 'down'
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(item.output)
  const [busy, setBusy] = useState(false)

  const send = async (rating, edited_text) => {
    setBusy(true)
    try {
      await onFeedback({ generation_id: item.generation_id, rating, edited_text })
      setSubmitted(rating)
      setEditing(false)
    } finally {
      setBusy(false)
    }
  }

  const empty = !item.output || item.output.trim() === ''

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 shadow-lg">
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-500">
        <span className="truncate pr-2">▶ {item.prompt}</span>
        <span className="shrink-0 rounded bg-zinc-800 px-2 py-0.5">v{item.model_version}</span>
      </div>

      {empty ? (
        <p className="italic text-zinc-500">
          (model returned no text — it is barely trained; run a longer pretrain)
        </p>
      ) : (
        <p className="whitespace-pre-wrap leading-relaxed text-zinc-100">{item.output}</p>
      )}

      {editing ? (
        <div className="mt-3">
          <textarea
            className="h-40 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm text-zinc-100 outline-none focus:border-red-500"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="mt-2 flex gap-2">
            <button
              disabled={busy}
              onClick={() => send('up', draft)}
              className="rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              Approve improved version →
            </button>
            <button
              onClick={() => setEditing(false)}
              className="rounded-lg bg-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-600"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : submitted ? (
        <div className="mt-3 text-sm text-zinc-400">
          {submitted === 'up' ? '✓ Saved as a preferred sample' : '✓ Marked as poor'} — thanks!
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            disabled={busy}
            onClick={() => send('up')}
            className="rounded-lg bg-emerald-600/90 px-3 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
          >
            👍 Good story
          </button>
          <button
            disabled={busy}
            onClick={() => send('down')}
            className="rounded-lg bg-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-600 disabled:opacity-50"
          >
            👎 Poor
          </button>
          <button
            disabled={busy || empty}
            onClick={() => { setDraft(item.output); setEditing(true) }}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800 disabled:opacity-40"
          >
            ✎ Edit &amp; approve
          </button>
        </div>
      )}
    </div>
  )
}
