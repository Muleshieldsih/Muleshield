import { useState, useCallback, useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { Topbar, Sidebar } from './components/Shell'
import TriageFeed from './pages/TriageFeed'
import TacticalMap from './pages/TacticalMap'
import ForensicGraph from './pages/ForensicGraph'
import Interception from './pages/Interception'
import useWebSocket from './hooks/useWebSocket'

function Layout() {
  const [complaints, setComplaints] = useState([])
  const [selected, setSelected] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('muleshield:selected') || '' : ''))

  const onWs = useCallback((msg) => {
    if (msg?.event_type === 'NEW_COMPLAINT' && msg.payload) {
      setComplaints((prev) => [msg.payload, ...prev].slice(0, 50))
      setSelected(msg.payload.ticket_id)
      if (typeof window !== 'undefined') {
        localStorage.setItem('muleshield:selected', msg.payload.ticket_id)
      }
    }
  }, [])

  const { connected } = useWebSocket(onWs)
  const location = useLocation()
  const navigate = useNavigate()

  // Sync selected state with URL query parameter on navigation
  useEffect(() => {
    const sp = new URLSearchParams(location.search)
    const urlCid = sp.get('c')
    if (urlCid && urlCid !== selected) {
      setSelected(urlCid)
      localStorage.setItem('muleshield:selected', urlCid)
    }
  }, [location.search, selected])

  useEffect(() => {
    if (selected) {
      localStorage.setItem('muleshield:selected', selected)
    }
  }, [selected])

  return (
    <div className="min-h-screen flex flex-col bg-ink-bg">
      <Topbar wsConnected={connected} complaintId={selected} />
      <div className="flex flex-1 min-h-0">
        <Sidebar complaintId={selected} />
        <div className="flex-1 min-w-0 bg-ink-bg overflow-y-auto">
          {/* mobile nav */}
          <div className="md:hidden flex gap-1 p-2 border-b border-ink-border overflow-x-auto">
            {[
              ['/', 'Triage'],
              ['/map', 'Map'],
              ['/graph', 'Graph'],
              ['/intercept', 'Intercept']
            ].map(([to, label]) => (
              <button
                key={to}
                onClick={() => navigate(to + (selected ? `?c=${encodeURIComponent(selected)}` : ''))}
                className="px-3 py-1.5 rounded border border-ink-border bg-ink-panel mono text-[11px] whitespace-nowrap text-zinc-300 hover:text-white"
              >
                {label}
              </button>
            ))}
          </div>
          <Routes>
            <Route path="/" element={<TriageFeed complaints={complaints} onSelect={setSelected} />} />
            <Route path="/map" element={<TacticalMap />} />
            <Route path="/graph" element={<ForensicGraph />} />
            <Route path="/intercept" element={<Interception />} />
          </Routes>
        </div>
      </div>
      <div className="h-6 border-t border-ink-border bg-ink-panel flex items-center justify-between px-3 mono text-[10px] text-zinc-600 shrink-0">
        <span>SIH26184 · MHA/I4C · MuleShield AI v3.0 · GraphSAGE 64-d · XGB v2 · {new Date().getFullYear()}</span>
        <span className="hidden md:block">Real-Time Law Enforcement Interdiction</span>
      </div>
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
