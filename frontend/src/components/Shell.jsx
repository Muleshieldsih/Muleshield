import { NavLink, useNavigate } from 'react-router-dom'
import { Shield, Activity, GitBranch, MapPinned, Zap, Radio, Circle, WifiOff } from 'lucide-react'
import { useMemo } from 'react'
import { amountShort, MODEL_STATS } from '../utils/constants'
import { timeAgo, useNow } from '../hooks/useCountdown'

/** Compact city code for the ticker, derived from the city name itself. */
function cityCode(city) {
  const s = String(city || '').trim()
  if (!s) return '—'
  const words = s.split(/\s+/)
  if (words.length > 1) return (words[0][0] + words[1][0] + (words[1][1] || '')).toUpperCase()
  return s.slice(0, 3).toUpperCase()
}

/**
 * Live intake ticker.
 *
 * Renders the real complaint queue. It used to invent random ticket IDs on a
 * timer, which looked live but navigated to tickets the backend had never
 * heard of — every click produced a 404 on the graph and prediction screens.
 */
function SidebarStreamTicker({ complaints = [], onSelect, activeId, wsConnected }) {
  const now = useNow(1000)
  const navigate = useNavigate()

  const rows = useMemo(() => (complaints || []).slice(0, 24), [complaints])

  const handleClick = (id) => {
    onSelect?.(id)
    navigate(`/?c=${encodeURIComponent(id)}`)
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 border-t border-ink-border px-3 py-2">
      <div className="flex items-center justify-between pb-1.5 mono text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">
        <span>1930 Live Stream</span>
        <span
          className={`flex items-center gap-1 font-bold text-[9px] lowercase tracking-normal ${
            wsConnected ? 'text-aegis-green' : 'text-zinc-500'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              wsConnected ? 'bg-aegis-green animate-pulse-dot' : 'bg-zinc-600'
            }`}
          />
          {wsConnected ? 'live' : 'offline'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {rows.length === 0 ? (
          <div className="mono text-[10px] text-zinc-600 py-4 text-center leading-relaxed">
            No complaints in queue.
            <br />
            Ingest one from the Triage screen.
          </div>
        ) : (
          rows.map(c => {
            const isActive = c.ticket_id === activeId
            return (
              <button
                key={c.ticket_id}
                onClick={() => handleClick(c.ticket_id)}
                title={`${c.ticket_id} — ${c.victim_name}, ${c.city}`}
                className={`w-full text-left p-1.5 rounded border transition-colors duration-100 flex items-center justify-between mono text-[10px] ${
                  isActive
                    ? 'bg-ink-panel border-aegis-green/60'
                    : 'bg-ink-panel/70 border-ink-border/80 hover:border-aegis-green/40 hover:bg-ink-panel'
                }`}
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="px-1 py-0.5 rounded bg-ink-bg border border-ink-border text-[9px] font-bold text-zinc-300 shrink-0">
                    {cityCode(c.city)}
                  </span>
                  <div className="truncate">
                    <span className="font-semibold text-white">{c.victim_bank}</span>
                    <span className="text-zinc-500"> · {c.fraud_type}</span>
                  </div>
                </div>
                <div className="text-right shrink-0 pl-1">
                  <div className="font-bold text-aegis-green">{amountShort(c.stolen_amount)}</div>
                  <div className="text-[9px] text-zinc-500">{timeAgo(c.complaint_timestamp, now)}</div>
                </div>
              </button>
            )
          })
        )}
      </div>
    </div>
  )
}

const EVENT_LABEL = {
  NEW_COMPLAINT: 'complaint ingested',
  PREDICTION_READY: 'prediction ready',
  FREEZE_EXECUTED: 'freeze executed',
  CONNECTED: 'feed connected',
}

export function Topbar({ wsConnected, complaintId, backendDown, lastEvent }) {
  const now = useNow(1000)
  const recent = lastEvent && now - lastEvent.at < 12000 ? lastEvent : null

  return (
    <header className="h-[48px] flex items-center justify-between gap-3 px-4 border-b border-ink-border bg-ink-bg shrink-0 z-30">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded bg-aegis-green grid place-items-center shadow-[0_0_10px_rgba(124,240,0,0.4)] shrink-0">
          <Shield size={16} className="text-black" />
        </div>
        <div className="leading-tight min-w-0">
          <div className="text-[12px] font-bold tracking-[0.14em] text-white mono">MULESHIELD AI</div>
          <div className="text-[10px] mono text-zinc-400 truncate">
            1930 HELPLINE INTERDICTION CONSOLE · MHA / I4C
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {recent && (
          <span className="hidden lg:flex items-center gap-1.5 mono text-[10px] text-aegis-green bg-aegis-green/10 border border-aegis-green/30 rounded-full px-2.5 py-1 animate-pulse-dot">
            <Radio size={10} /> {EVENT_LABEL[recent.type] || recent.type.toLowerCase()}
          </span>
        )}

        <div className="hidden md:flex items-center gap-2 text-[11px] mono bg-ink-panel border border-ink-border rounded-full px-3 py-1">
          {backendDown ? (
            <span className="flex items-center gap-1.5 text-red-400 font-semibold">
              <WifiOff size={11} /> API OFFLINE
            </span>
          ) : (
            <span
              className={`flex items-center gap-1.5 ${
                wsConnected ? 'text-aegis-green font-semibold' : 'text-zinc-500'
              }`}
            >
              <Circle size={8} className={wsConnected ? 'fill-aegis-green animate-pulse-dot' : ''} />
              {wsConnected ? 'LIVE 1930 TELEMETRY' : 'RECONNECTING…'}
            </span>
          )}
        </div>

        {complaintId && (
          <div className="hidden sm:block text-[11px] mono text-zinc-300 border border-aegis-green/40 rounded-full px-3 py-1 bg-aegis-green/10">
            Active: <span className="font-bold text-white">{complaintId}</span>
          </div>
        )}
      </div>
    </header>
  )
}

const NAV_ITEMS = [
  ['/', <Activity size={15} key="i" />, 'TRIAGE QUEUE', 'Live 1930 feed'],
  ['/map', <MapPinned size={15} key="i" />, 'TACTICAL MAP', 'ATM GPS routing'],
  ['/graph', <GitBranch size={15} key="i" />, 'MONEY FLOW', 'Forensic graph'],
  ['/intercept', <Zap size={15} key="i" />, 'INTERCEPTION', '1-Click freeze'],
]

export function Sidebar({ complaintId, complaints, onSelect, wsConnected }) {
  const withCid = (path) =>
    complaintId ? `${path}?c=${encodeURIComponent(complaintId)}` : path

  return (
    <nav className="w-[210px] shrink-0 border-r border-ink-border bg-ink-bg hidden md:flex flex-col min-h-0">
      <div className="p-3 space-y-1.5 shrink-0">
        {NAV_ITEMS.map(([to, icon, label, sub]) => (
          <NavLink
            key={to}
            to={withCid(to)}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg border text-[12px] mono transition ${
                isActive
                  ? 'bg-ink-panel border-aegis-green/50 text-white font-bold shadow-[0_0_10px_rgba(124,240,0,0.1)]'
                  : 'border-transparent text-zinc-400 hover:text-white hover:bg-ink-panel/50'
              }`
            }
          >
            <span className="w-6 h-6 grid place-items-center rounded bg-ink-surface border border-ink-border shrink-0">
              {icon}
            </span>
            <span className="flex-1 leading-tight">
              {label}
              <br />
              <span className="text-[10px] text-zinc-500 font-normal">{sub}</span>
            </span>
          </NavLink>
        ))}
      </div>

      <SidebarStreamTicker
        complaints={complaints}
        onSelect={onSelect}
        activeId={complaintId}
        wsConnected={wsConnected}
      />

      <div className="p-3 border-t border-ink-border shrink-0">
        <div className="aegis-panel p-2.5 bg-ink-panel/40">
          <div className="text-[10px] mono tracking-[0.12em] text-zinc-400 uppercase font-semibold">
            Benchmarked Model SLA
          </div>
          <div className="text-[11px] mono text-aegis-green flex items-center gap-1.5 mt-1 font-bold">
            <Radio size={12} className="animate-pulse-dot" /> {MODEL_STATS.top3Accuracy} Top-3 Accuracy
          </div>
          <div className="text-[10px] mono text-zinc-500 mt-1">
            Countdown MAE: <span className="text-zinc-300">{MODEL_STATS.countdownMae}</span>
          </div>
          <div className="text-[10px] mono text-zinc-500">
            GNN F1: <span className="text-zinc-300">{MODEL_STATS.gnnF1}</span>
          </div>
        </div>
        <div className="text-[10px] mono text-zinc-600 mt-2 text-center">SIH26184 · MHA / I4C</div>
      </div>
    </nav>
  )
}

export function MobileNav({ onNavigate, pathname }) {
  return (
    <div className="md:hidden flex gap-1 p-2 border-b border-ink-border overflow-x-auto">
      {NAV_ITEMS.map(([to, , label]) => (
        <button
          key={to}
          onClick={() => onNavigate(to)}
          className={`px-3 py-1.5 rounded border mono text-[11px] whitespace-nowrap transition ${
            pathname === to
              ? 'border-aegis-green/50 bg-ink-panel text-white font-bold'
              : 'border-ink-border bg-ink-panel text-zinc-300 hover:text-white'
          }`}
        >
          {label.split(' ')[0]}
        </button>
      ))}
    </div>
  )
}

export function Panel({ title, live, right, children, className = '', bodyClass = '' }) {
  return (
    <section className={`aegis-panel ${className}`}>
      <div className="aegis-panel-header">
        <div className="flex items-center gap-2 text-[11px] mono tracking-[0.12em] text-zinc-400 min-w-0">
          <span className="truncate">{title}</span>
          {live && (
            <span className="flex items-center gap-1.5 text-aegis-green normal-case tracking-normal shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-aegis-green animate-pulse-dot" />
              Live
            </span>
          )}
        </div>
        <div className="text-[11px] mono text-zinc-500 shrink-0 pl-2">{right}</div>
      </div>
      <div className={bodyClass}>{children}</div>
    </section>
  )
}

/** Small labelled metric cell used across the tactical screens. */
export function Stat({ label, value, tone = 'default', sub }) {
  const tones = {
    default: 'text-white',
    good: 'text-aegis-green',
    warn: 'text-amber-400',
    bad: 'text-red-400',
    info: 'text-blue-400',
  }
  return (
    <div className="rounded-lg border border-ink-border bg-ink-panel px-2.5 py-2">
      <div className="mono text-[10px] text-zinc-500 uppercase tracking-wide">{label}</div>
      <div className={`mono text-[15px] font-bold mt-0.5 ${tones[tone] || tones.default}`}>{value}</div>
      {sub && <div className="mono text-[10px] text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  )
}
