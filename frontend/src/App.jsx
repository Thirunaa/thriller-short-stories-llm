import React, { useEffect, useRef, useState } from 'react'
import { api } from './api'
import StoryCard from './components/StoryCard'
import MonitorPanel from './components/MonitorPanel'

// Story openings — the model continues whatever you write.
const PRESETS = [
  'Once upon a time, in a quiet little town,',
  'The old house at the end of the street held a secret.',
  'A curious child discovered a hidden door in the garden.',
  'On a dark and stormy night, something stirred in the woods.',
]

export default function App() {
  const [prompt, setPrompt] = useState(PRESETS[0])
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [topK, setTopK] = useState(40)

  const [stories, setStories] = useState([])
  const [generating, setGenerating] = useState(false)
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  const refreshStatus = async () => {
    try {
      setStatus(await api.status())
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    refreshStatus()
    pollRef.current = setInterval(refreshStatus, 4000)
    return () => clearInterval(pollRef.current)
  }, [])

  const doGenerate = async () => {
    if (!prompt.trim()) return
    setGenerating(true)
    setError(null)
    try {
      const res = await api.generate({
        prompt,
        max_new_tokens: Number(maxTokens),
        temperature: Number(temperature),
        top_k: Number(topK),
      })
      setStories((s) => [res, ...s])
    } catch (e) {
      setError(e.message)
    } finally {
      setGenerating(false)
    }
  }

  const doFeedback = async (payload) => {
    await api.feedback(payload)
    refreshStatus()
  }

  const doTrigger = async () => {
    setBusy(true)
    try {
      await api.triggerTraining()
      setTimeout(refreshStatus, 800)
    } finally {
      setBusy(false)
    }
  }

  const doReload = async () => {
    setBusy(true)
    try {
      await api.reloadModel()
      refreshStatus()
    } finally {
      setBusy(false)
    }
  }

  const untrained = status?.model && !status.model.trained

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 bg-zinc-900/50 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              🗡️ Thriller Forge <span className="text-zinc-500">— MiniGPT Story Studio</span>
            </h1>
            <p className="text-xs text-zinc-500">
              A from-scratch JAX/Flax GPT that learns from your feedback in real time.
            </p>
          </div>
          <span className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-400">
            model v{status?.model?.version ?? '…'} · {status?.model?.source ?? '…'}
          </span>
        </div>
      </header>

      {untrained && (
        <div className="bg-amber-900/40 px-6 py-2 text-center text-sm text-amber-200">
          Model is untrained — run <code className="rounded bg-black/40 px-1">python prepare_data.py</code> then{' '}
          <code className="rounded bg-black/40 px-1">python train.py</code> for coherent stories.
        </div>
      )}

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[1fr_320px]">
        {/* Studio */}
        <div className="space-y-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <textarea
              className="h-24 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm outline-none focus:border-red-500"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Start your thriller with an opening line… the model continues it."
            />
            <div className="mt-2 flex flex-wrap gap-1">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => setPrompt(p)}
                  className="rounded-full border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800"
                >
                  {p.slice(0, 28)}…
                </button>
              ))}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-zinc-400">
              <label className="flex items-center gap-2">
                temp
                <input type="range" min="0" max="1.5" step="0.05" value={temperature}
                  onChange={(e) => setTemperature(e.target.value)} />
                <span className="w-8 text-zinc-200">{Number(temperature).toFixed(2)}</span>
              </label>
              <label className="flex items-center gap-2">
                tokens
                <input type="range" min="20" max="500" step="10" value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)} />
                <span className="w-8 text-zinc-200">{maxTokens}</span>
              </label>
              <label className="flex items-center gap-2">
                top-k
                <input type="number" min="0" max="200" value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  className="w-16 rounded border border-zinc-700 bg-zinc-950 px-2 py-1" />
              </label>
              <button
                onClick={doGenerate}
                disabled={generating}
                className="ml-auto rounded-lg bg-red-600 px-5 py-2 text-sm font-semibold hover:bg-red-500 disabled:opacity-50"
              >
                {generating ? 'Forging…' : 'Forge story'}
              </button>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-red-800 bg-red-950/50 p-3 text-sm text-red-300">
              {error}
            </div>
          )}

          {stories.length === 0 && !generating && (
            <div className="rounded-xl border border-dashed border-zinc-800 p-10 text-center text-zinc-600">
              No stories yet. Forge one above, then rate it to teach the model.
            </div>
          )}

          {stories.map((s) => (
            <StoryCard key={s.generation_id} item={s} onFeedback={doFeedback} />
          ))}
        </div>

        {/* Monitor */}
        <aside className="lg:sticky lg:top-6 lg:h-fit rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <MonitorPanel
            status={status}
            onTrigger={doTrigger}
            onReload={doReload}
            busy={busy}
          />
        </aside>
      </main>
    </div>
  )
}
