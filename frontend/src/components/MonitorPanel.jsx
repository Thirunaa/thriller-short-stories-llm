import React from 'react'

function Stat({ label, value, accent }) {
  return (
    <div className="rounded-lg bg-zinc-950/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`text-lg font-semibold ${accent || 'text-zinc-100'}`}>{value}</div>
    </div>
  )
}

// Live monitoring sidebar: model lineage, feedback funnel, and the continuous
// trainer's state. Drives the "improve now" + "reload" controls.
export default function MonitorPanel({ status, onTrigger, onReload, busy }) {
  if (!status) return <div className="text-sm text-zinc-500">Connecting to backend…</div>

  const m = status.model || {}
  const fb = status.feedback || {}
  const tr = status.trainer || {}
  const data = status.data || {}
  const last = tr.last_result || {}

  const progressTo = tr.min_new_samples || 1
  const pct = Math.min(100, Math.round((100 * (tr.pending_positive || 0)) / progressTo))

  return (
    <div className="space-y-4">
      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Served model
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Version" value={`v${m.version ?? 0}`} accent="text-red-400" />
          <Stat label="Source" value={m.source || '—'} />
          <Stat label="Params" value={(m.params || 0).toLocaleString()} />
          <Stat
            label="State"
            value={m.trained ? 'trained' : 'untrained'}
            accent={m.trained ? 'text-emerald-400' : 'text-amber-400'}
          />
        </div>
        {m.config && (
          <div className="mt-2 text-[11px] text-zinc-500">
            L{m.config.n_layer} · H{m.config.n_head} · D{m.config.n_embd} · ctx{m.config.block_size}
            {data.train_tokens ? ` · ${Number(data.train_tokens).toLocaleString()} train tokens` : ''}
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Human feedback
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Generations" value={fb.generations ?? 0} />
          <Stat label="👍 / 👎" value={`${fb.thumbs_up ?? 0} / ${fb.thumbs_down ?? 0}`} />
          <Stat label="Used in training" value={fb.used_in_training ?? 0} accent="text-emerald-400" />
          <Stat label="Pending good" value={fb.pending_positive ?? 0} accent="text-amber-400" />
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Continuous trainer
        </h3>
        <div className="rounded-lg bg-zinc-950/60 p-3">
          <div className="mb-1 flex justify-between text-xs text-zinc-400">
            <span>{tr.in_progress ? 'fine-tuning…' : tr.running ? 'watching feedback' : 'stopped'}</span>
            <span>{tr.pending_positive ?? 0}/{progressTo} to auto-trigger</span>
          </div>
          <div className="h-2 overflow-hidden rounded bg-zinc-800">
            <div
              className={`h-full ${tr.in_progress ? 'animate-pulse bg-red-500' : 'bg-emerald-500'}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <Stat label="Rounds done" value={tr.rounds_completed ?? 0} />
            <Stat label="Last status" value={last.status || 'idle'} />
          </div>
          {last.status === 'improved' && (
            <div className="mt-2 text-[11px] text-emerald-400">
              → v{last.new_version} from {last.samples} samples (loss {last.final_loss})
            </div>
          )}
          {last.status === 'no_base_model' && (
            <div className="mt-2 text-[11px] text-amber-400">Pretrain first: python train.py</div>
          )}
        </div>
      </section>

      <div className="flex gap-2">
        <button
          disabled={busy}
          onClick={onTrigger}
          className="flex-1 rounded-lg bg-red-600 px-3 py-2 text-sm font-medium hover:bg-red-500 disabled:opacity-50"
        >
          ⚡ Improve now
        </button>
        <button
          disabled={busy}
          onClick={onReload}
          className="rounded-lg border border-zinc-700 px-3 py-2 text-sm hover:bg-zinc-800 disabled:opacity-50"
        >
          ↻ Reload
        </button>
      </div>
    </div>
  )
}
