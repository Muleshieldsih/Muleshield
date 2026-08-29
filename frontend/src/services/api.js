import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' }
})

/**
 * Normalise an axios failure into something the UI can show an operator.
 * A console that silently swaps in demo data when the backend is down is worse
 * than one that says it is degraded — the officer must know what they are
 * looking at.
 */
export function describeError(err) {
  if (!err) return 'Unknown error'
  if (err.code === 'ECONNABORTED') return 'Backend timed out'
  if (err.response) {
    const detail = err.response.data?.detail
    if (typeof detail === 'string') return detail
    return `Backend returned HTTP ${err.response.status}`
  }
  if (err.request) return 'Cannot reach backend on ' + BASE
  return err.message || 'Unknown error'
}

export const endpoints = {
  health: () => api.get('/health').then(r => r.data),
  listComplaints: (limit = 60, offset = 0) =>
    api.get('/api/v1/complaint/list', { params: { limit, offset } }).then(r => r.data),
  getComplaint: (id) => api.get(`/api/v1/complaint/${id}`).then(r => r.data),
  ingestComplaint: (payload) => api.post('/api/v1/complaint/ingest', payload).then(r => r.data),
  getGraph: (id) => api.get(`/api/v1/graph/${id}`).then(r => r.data),
  getEmbeddings: (id, topN = 5) =>
    api.get(`/api/v1/embeddings/${id}`, { params: { top_n: topN } }).then(r => r.data),
  predictCashout: (id) => api.get(`/api/v1/predict/cashout/${id}`).then(r => r.data),
  microFreeze: (payload) => api.post('/api/v1/bank/micro-freeze', payload).then(r => r.data),
}

export function wsUrl() {
  const base = BASE.replace(/^http/, 'ws')
  return `${base}/ws/feed`
}
