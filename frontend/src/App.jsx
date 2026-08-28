import { useState, useCallback, useEffect, useMemo, lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { Topbar, Sidebar, MobileNav } from './components/Shell'
import TriageFeed from './pages/TriageFeed'
import useWebSocket from './hooks/useWebSocket'

// Leaflet and React Flow are the two heaviest dependencies in the bundle and
// neither is needed for the landing screen. Splitting them keeps the triage
// queue — the first thing shown at a demo — fast to paint.
const TacticalMap = lazy(() => import('./pages/TacticalMap'))
const ForensicGraph = lazy(() => import('./pages/ForensicGraph'))
const Interception = lazy(() => import('./pages/Interception'))

function RouteFallback() {
  return (
    <div className="p-10 grid place-items-center mono text-[12px] text-zinc-500">
      <span className="flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading module…
      </span>
    </div>
  )
}
import { endpoints } from './services/api'
import { readStoredComplaintId, storeComplaintId } from './hooks/useActiveComplaint'

const QUEUE_LIMIT = 60

function Layout() {
  const [complaints, setComplaints] = useState([])
  const [loadError, setLoadError] = useState('')
  const [selected, setSelected] = useState(readStoredComplaintId)
  const [lastEvent, setLastEvent] = useState(null)

  const location = useLocation()
  const navigate = useNavigate()

  // ── Initial queue load ─────────────────────────────────────────────────────
  // One fetch owned here and shared with every child, instead of each screen
  // pulling its own copy of the same list.
  useEffect(() => {
    let cancelled = false
    endpoints.listComplaints(QUEUE_LIMIT)
      .then(list => { if (!cancelled) { setComplaints(list || []); setLoadError('') } })
      .catch(err => { if (!cancelled) setLoadError(err?.message || 'Backend unreachable') })
    return () => { cancelled = true }
  }, [])

  // ── Live events ────────────────────────────────────────────────────────────
  const onWs = useCallback((msg) => {
    if (!msg?.event_type) return
    setLastEvent({ type: msg.event_type, at: Date.now(), complaintId: msg.complaint_id })

    if (msg.event_type === 'NEW_COMPLAINT' && msg.payload?.ticket_id) {
      setComplaints(prev => {
        if (prev.some(c => c.ticket_id === msg.payload.ticket_id)) return prev
        return [msg.payload, ...prev].slice(0, QUEUE_LIMIT)
      })
      setSelected(msg.payload.ticket_id)
      storeComplaintId(msg.payload.ticket_id)
    }
  }, [])

  const { connected } = useWebSocket(onWs)

  // ── Keep ?c= and the stored selection in step ──────────────────────────────
  useEffect(() => {
    const urlCid = new URLSearchParams(location.search).get('c')
    if (urlCid && urlCid !== selected) {
      setSelected(urlCid)
      storeComplaintId(urlCid)
    }
  }, [location.search, selected])

  useEffect(() => { storeComplaintId(selected) }, [selected])

  const handleSelect = useCallback((id) => {
    setSelected(id)
    storeComplaintId(id)
  }, [])

  const handleIngested = useCallback((record) => {
    if (!record?.ticket_id) return
    setComplaints(prev =>
      prev.some(c => c.ticket_id === record.ticket_id) ? prev : [record, ...prev].slice(0, QUEUE_LIMIT)
    )
    handleSelect(record.ticket_id)
  }, [handleSelect])

  const goTo = useCallback((path) => {
    navigate(path + (selected ? `?c=${encodeURIComponent(selected)}` : ''))
  }, [navigate, selected])

  const backendDown = !!loadError && complaints.length === 0

  const contextValue = useMemo(() => ({
    complaints, selected, onSelect: handleSelect, onIngested: handleIngested, backendDown,
  }), [complaints, selected, handleSelect, handleIngested, backendDown])

  return (
    <div className="h-screen flex flex-col bg-ink-bg overflow-hidden">
      <Topbar
        wsConnected={connected}
        complaintId={selected}
        backendDown={backendDown}
        lastEvent={lastEvent}
      />

      <div className="flex flex-1 min-h-0">
        <Sidebar
          complaintId={selected}
          complaints={complaints}
          onSelect={handleSelect}
          wsConnected={connected}
        />

        <main className="flex-1 min-w-0 bg-ink-bg overflow-y-auto">
          <MobileNav onNavigate={goTo} pathname={location.pathname} />

          {backendDown && (
            <div className="m-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 mono text-[11px] text-red-300">
              <span className="font-bold">BACKEND UNREACHABLE</span> — {loadError}. Start the API with
              <span className="text-white"> python -m uvicorn backend.main:app --port 8000</span>
            </div>
          )}

          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/" element={<TriageFeed {...contextValue} />} />
              <Route path="/map" element={<TacticalMap />} />
              <Route path="/graph" element={<ForensicGraph />} />
              <Route path="/intercept" element={<Interception />} />
            </Routes>
          </Suspense>
        </main>
      </div>

      <footer className="h-6 border-t border-ink-border bg-ink-panel flex items-center justify-between px-3 mono text-[10px] text-zinc-500 shrink-0">
        <span>SIH26184 · MHA / I4C · MuleShield AI v3.0 · GraphSAGE 64-d + XGBoost v2</span>
        <span className="hidden md:block">Real-Time Law Enforcement Interdiction</span>
      </footer>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  )
}
