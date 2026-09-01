import axios from 'axios'
import { getToken, clearToken, notifyExpired } from './auth'

/**
 * Where the API lives.
 *
 * Three cases, in order:
 *   1. VITE_API_BASE_URL is set     -> use it (the isolated smoke build does this)
 *   2. dev server                   -> localhost:8000, where uvicorn runs
 *   3. production build             -> same origin, empty string
 *
 * Case 3 matters for deployment: the container serves this bundle from the same
 * FastAPI process that answers the API, so requests must be relative. The old
 * fallback was a hardcoded localhost:8000, which meant a deployed build asked
 * the VIEWER'S machine for the API and failed for everyone.
 */
const BASE = import.meta.env.VITE_API_BASE_URL
  ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

export const api = axios.create({
  baseURL: BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' }
})

// Attach the bearer token to every request that has one.
api.interceptors.request.use(config => {
  const token = getToken()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// One place to handle "your session is over".
//
// The /auth/login exclusion matters: a rejected sign-in is a form error, not an
// expired session. Without it, typing the wrong password would clear the token
// and bounce the officer to the login screen they are already looking at, and
// the error message would be lost in the remount.
api.interceptors.response.use(
  response => response,
  error => {
    const url = error?.config?.url || ''
    if (error?.response?.status === 401 && !url.includes('/auth/login')) {
      clearToken()
      notifyExpired()
    }
    return Promise.reject(error)
  }
)

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
  // auth
  login: (username, password) =>
    api.post('/api/v1/auth/login', { username, password }).then(r => r.data),
  logout: () => api.post('/api/v1/auth/logout').then(r => r.data),
  me: () => api.get('/api/v1/auth/me').then(r => r.data),
  forgotPassword: username =>
    api.post('/api/v1/auth/forgot-password', { username }).then(r => r.data),
  resetPassword: (token, newPassword) =>
    api.post('/api/v1/auth/reset-password',
             { token, new_password: newPassword }).then(r => r.data),

  // administrator
  listUsers: () => api.get('/api/v1/auth/users').then(r => r.data),
  createUser: payload => api.post('/api/v1/auth/users', payload).then(r => r.data),
  unlockUser: id => api.post(`/api/v1/auth/users/${id}/unlock`).then(r => r.data),
  listResetRequests: (status = 'pending') =>
    api.get('/api/v1/auth/reset-requests', { params: { status } }).then(r => r.data),
  approveReset: id =>
    api.post(`/api/v1/auth/reset-requests/${id}/approve`).then(r => r.data),
  denyReset: id =>
    api.post(`/api/v1/auth/reset-requests/${id}/deny`).then(r => r.data),

  // cross-case intelligence
  listAtmIntel: (limit = 25) =>
    api.get('/api/v1/intel/atms', { params: { limit } }).then(r => r.data),

  // forward hotspot surface
  //
  // Params are passed through axios rather than hand-built with URLSearchParams
  // so that undefined keys drop out on their own and the bearer interceptor
  // above still applies -- this endpoint is Tier A and 401s without it.
  //
  // `fraud_type` is comma-separated; `state` filters the SOURCE complaints by
  // the victim's state, which is not what the console's drill-down does. The
  // breadcrumb scopes CELLS by cell.state, client-side, because the response is
  // already a complete national aggregate of ~222 cells.
  listHotspots: ({ windowStartMin, windowEndMin, fraudTypes, asOf, openMinutes } = {}) =>
    api.get('/api/v1/hotspots/cells', {
      params: {
        window_start_min: windowStartMin,
        window_end_min: windowEndMin,
        fraud_type: fraudTypes?.length ? fraudTypes.join(',') : undefined,
        as_of: asOf || undefined,
        open_minutes: openMinutes,
      },
    }).then(r => r.data),

  // Phase 4 (alerting) owns this route. It is declared here so /risk's
  // "Raise for review" needs no frontend change when the service lands; until
  // then the screen catches the 404 and says the queue is not enabled yet
  // rather than reporting a success it did not get.
  raiseAlert: payload => api.post('/api/v1/alerts', payload).then(r => r.data),

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
  updateCase: (id, payload) =>
    api.patch(`/api/v1/complaint/${id}`, payload).then(r => r.data),
  addNote: (id, payload) =>
    api.post(`/api/v1/complaint/${id}/note`, payload).then(r => r.data),
  listNotes: (id) => api.get(`/api/v1/complaint/${id}/notes`).then(r => r.data),
  listTransactions: (id) =>
    api.get(`/api/v1/complaint/${id}/transactions`).then(r => r.data),
  listAudit: (params = {}) =>
    api.get('/api/v1/audit', { params }).then(r => r.data),
}

export function wsUrl() {
  // Same-origin build: BASE is empty, so the socket host comes from the page.
  // Keeps wss:// on an https deployment, which a hardcoded ws:// would break.
  const origin = BASE || window.location.origin
  return `${origin.replace(/^http/, 'ws')}/ws/feed`
}
