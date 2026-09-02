import { useState, useCallback, useEffect, useMemo, useRef, lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { Topbar, Sidebar, MobileNav } from './components/Shell'
import TriageFeed from './pages/TriageFeed'
import useWebSocket from './hooks/useWebSocket'
import { ToastProvider } from './components/Toast'
import { AuthProvider, useAuth } from './context/AuthContext'
import Login from './pages/Login'

// Leaflet and React Flow are the two heaviest dependencies in the bundle and
// neither is needed for the landing screen. Splitting them keeps the triage
// queue — the first thing shown at a demo — fast to paint.
const TacticalMap = lazy(() => import('./pages/TacticalMap'))
const ForensicGraph = lazy(() => import('./pages/ForensicGraph'))
const Interception = lazy(() => import('./pages/Interception'))
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'))
const RiskHeatmap = lazy(() => import('./pages/RiskHeatmap'))
const AlertInbox = lazy(() => import('./pages/AlertInbox'))

function NotFound({ onHome }) {
  return (
    <div className="p-10 grid place-items-center text-center">
      <div>
        <div className="text-[13px] text-zinc-200 font-semibold">Screen not found</div>
        <div className="text-[11px] text-zinc-500 mt-1.5">
          <span className="mono text-zinc-400">{window.location.pathname}</span> is not a route in this console.
        </div>
        <button
          onClick={onHome}
          className="mt-4 px-3 py-1.5 rounded border border-ink-border bg-ink-panel
                     text-[12px] text-zinc-200 hover:border-zinc-600 hover:text-white transition-colors"
        >
          Back to cases
        </button>
      </div>
    </div>
  )
}

function RouteFallback() {
  return (
    <div className="p-10 grid place-items-center text-[12.5px] text-zinc-500">
      <span className="flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading…
      </span>
    </div>
  )
}
import { endpoints } from './services/api'
import { clearComplaintId, readStoredComplaintId, storeComplaintId } from './hooks/useActiveComplaint'

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
      .then(async list => {
        if (cancelled) return
        setComplaints(list || [])
        setLoadError('')

        // Drop a stored selection that no longer exists.
        //
        // A ticket id in localStorage outlives the data it points at — a
        // regenerated dataset, or one of the demo fixtures that were removed —
        // and every screen then reports "complaint not found" while the live
        // queue beside them is populated. Verified against the backend rather
        // than against this page of the list, since a valid complaint can sit
        // outside the first QUEUE_LIMIT rows.
        const urlCid = new URLSearchParams(window.location.search).get('c')
        const stored = urlCid || readStoredComplaintId()
        if (!stored) return
        if ((list || []).some(c => c.ticket_id === stored)) return
        try {
          await endpoints.getComplaint(stored)
        } catch (err) {
          if (err?.response?.status === 404 && !cancelled) {
            clearComplaintId()
            const fallback = list?.length ? list[0].ticket_id : ''
            setSelected(fallback)
            // The dead id must leave the URL as well, or the ?c= sync effect
            // below immediately restores it and the screens stay stuck.
            navigate(
              window.location.pathname + (fallback ? `?c=${encodeURIComponent(fallback)}` : ''),
              { replace: true },
            )
          }
        }
      })
      .catch(err => { if (!cancelled) setLoadError(err?.message || 'Backend unreachable') })
    return () => { cancelled = true }
  }, [navigate])

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
  //
  // One direction each way, and never both at once.
  //
  // This effect used to list `selected` as a dependency and overwrite it from
  // the URL whenever the two disagreed. Since selecting a row set state but
  // never touched the URL, every click re-ran this effect, found the stale ?c=
  // disagreeing with the new selection, and set it straight back -- pinning the
  // console to whichever complaint the URL named. The queue looked frozen: rows
  // took focus but the incident panel never moved.
  //
  // It is invisible without a ?c= in the address bar, which is why it survived:
  // the guard below short-circuits on a bare "/" and the bug only shows once a
  // complaint has been selected once.
  const lastUrlCid = useRef(null)

  useEffect(() => {
    const urlCid = new URLSearchParams(location.search).get('c')
    if (urlCid && urlCid !== lastUrlCid.current) {
      lastUrlCid.current = urlCid
      setSelected(urlCid)
      storeComplaintId(urlCid)
    }
  }, [location.search])

  useEffect(() => { storeComplaintId(selected) }, [selected])

  const handleSelect = useCallback((id) => {
    lastUrlCid.current = id || null
    setSelected(id || '')
    storeComplaintId(id)
    // The URL follows the selection rather than competing with it, so the
    // current complaint survives a refresh and a copied link opens on it.
    navigate(
      window.location.pathname + (id ? `?c=${encodeURIComponent(id)}` : ''),
      { replace: true },
    )
  }, [navigate])

  // A case whose status or assignee changed has to update in the queue too,
  // or the row an analyst just acted on still reads as untouched beside the
  // panel that says otherwise.
  const handleCaseUpdated = useCallback((record) => {
    if (!record?.ticket_id) return
    setComplaints(prev => prev.map(c => (
      c.ticket_id === record.ticket_id ? { ...c, ...record } : c
    )))
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
    complaints, selected, onSelect: handleSelect, onIngested: handleIngested,
    onCaseUpdated: handleCaseUpdated, backendDown,
  }), [complaints, selected, handleSelect, handleIngested, handleCaseUpdated, backendDown])

  return (
    <div className="h-screen flex flex-col bg-ink-bg overflow-hidden">
      <Topbar
        wsConnected={connected}
        complaintId={selected}
        backendDown={backendDown}
        lastEvent={lastEvent}
      />

      <div className="flex flex-1 min-h-0">
        {/* No wsConnected: the sidebar no longer carries a connection badge of
            its own. The Topbar owns that state. */}
        <Sidebar
          complaintId={selected}
          complaints={complaints}
          onSelect={handleSelect}
        />

        <main className="flex-1 min-w-0 bg-ink-bg overflow-y-auto">
          <MobileNav onNavigate={goTo} pathname={location.pathname} />

          {backendDown && (
            <div className="m-3 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-[12px] text-red-300">
              <span className="font-semibold">Cannot reach the service</span> — {loadError}. Start it with
              <span className="mono text-white"> python -m uvicorn backend.main:app --port 8000</span>
            </div>
          )}

          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/" element={<TriageFeed {...contextValue} />} />
              <Route path="/map" element={<TacticalMap />} />
              <Route path="/graph" element={<ForensicGraph />} />
              <Route path="/intercept" element={<Interception />} />
              <Route path="/model" element={<ModelPerformance />} />
              <Route path="/risk" element={<RiskHeatmap />} />
              <Route path="/alerts" element={<AlertInbox />} />
              {/* Without a catch-all, an unknown URL rendered the shell around an
                  empty <main> -- a blank console with no indication anything was
                  wrong. A mistyped link should say so and offer the way back. */}
              <Route path="*" element={<NotFound onHome={() => goTo('/')} />} />
            </Routes>
          </Suspense>
        </main>
      </div>

      <footer className="h-6 border-t border-ink-border bg-ink-panel flex items-center justify-between px-3 text-[10.5px] text-zinc-500 shrink-0">
        <span>MuleShield AI · SIH26184 · MHA / I4C</span>
        <span className="hidden md:block">v3.0</span>
      </footer>
    </div>
  )
}

/**
 * Signed out, Layout never mounts.
 *
 * That is the whole reason this sits above <Layout /> rather than being a guard
 * inside it: Layout fetches the case queue and opens a WebSocket the moment it
 * mounts, and an unauthenticated console would sit behind the login form
 * retrying a socket it can never open and toasting errors nobody can act on.
 * Not mounting it at all is simpler than gating every effect inside it.
 */
function Gate() {
  const { user, ready } = useAuth()

  // A stored token has not been checked yet. Rendering the login form here would
  // flash it in front of an officer who is perfectly signed in.
  if (!ready) {
    return (
      <div className="h-screen bg-ink-bg grid place-items-center">
        <Loader2 size={18} className="animate-spin text-zinc-600" />
      </div>
    )
  }

  if (!user) return <Login />

  // Keyed on the officer, so signing in as somebody else rebuilds the console
  // rather than leaving the previous session's selected case in place.
  return <Layout key={user.username} />
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <AuthProvider>
          <Gate />
        </AuthProvider>
      </ToastProvider>
    </BrowserRouter>
  )
}
