// Thin fetch wrapper around the FastAPI backend. Calls go through the Vite proxy
// (/api -> http://localhost:8000) in dev.

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export const api = {
  status: () => jsonFetch('/api/status'),
  generate: (payload) =>
    jsonFetch('/api/generate', { method: 'POST', body: JSON.stringify(payload) }),
  feedback: (payload) =>
    jsonFetch('/api/feedback', { method: 'POST', body: JSON.stringify(payload) }),
  triggerTraining: () => jsonFetch('/api/train/trigger', { method: 'POST' }),
  reloadModel: () => jsonFetch('/api/model/reload', { method: 'POST' }),
}
