import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE,
  timeout: 8000,
  headers: { 'Content-Type': 'application/json' }
})

export const endpoints = {
  health: () => api.get('/health').then(r => r.data),
  listComplaints: () => api.get('/api/v1/complaint/list').then(r => r.data),
  getComplaint: (id) => api.get(`/api/v1/complaint/${id}`).then(r => r.data),
  ingestComplaint: (payload) => api.post('/api/v1/complaint/ingest', payload).then(r => r.data),
  getGraph: (id) => api.get(`/api/v1/graph/${id}`).then(r => r.data),
  getEmbeddings: (id) => api.get(`/api/v1/embeddings/${id}`).then(r => r.data).catch(() => ({ nodes: [] })),
  predictCashout: (id) => api.get(`/api/v1/predict/cashout/${id}`).then(r => r.data),
  microFreeze: (payload) => api.post('/api/v1/bank/micro-freeze', payload).then(r => r.data),
}

export function wsUrl() {
  const base = BASE.replace(/^http/, 'ws')
  return `${base}/ws/feed`
}
