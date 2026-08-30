import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { Shield, Activity, GitBranch, MapPinned, Zap, BarChart3, Radio, Circle, WifiOff } from 'lucide-react'
import { useMemo } from 'react'
import { amountShort, formatTicket } from '../utils/constants'
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
 * Complaint intake queue.
 *
 * Renders the real queue. It used to invent random ticket IDs on a timer, which
 * looked live but navigated to tickets the backend had never heard of — every
 * click produced a 404 on the graph and prediction screens.
 *
 * It was then labelled "1930 LIVE STREAM · live" while showing a list that never
 * moved, because the badge reported the WebSocket being *open*, not complaints
 * arriving. Offline there is no NCRP feed pushing new complaints, so the queue is
 * static by definition and the label was claiming something untrue.
 *
 * It no longer carries a connection badge of its own. That was the third one on
 * screen — two in the topbar, one here — all reading the same boolean. The
 * topbar owns connection state; this list is a switcher.
 *
 * The Sidebar renders it only away from "/", where the main panel already shows
 * the same complaints in a form that has room for severity and amounts.
 */
function SidebarStreamTicker({ complaints = [], onSelect, activeId }) {
  const now = useNow(1000)
  const navigate = useNavigate()
  const location = useLocation()

  const rows = useMemo(() => (complaints || []).slice(0, 24), [complaints])

  // Stay on the screen the operator is using. This used to navigate to "/"
  // unconditionally, so picking a complaint while working the map threw you back
  // to the triage queue and you had to navigate to the map a second time.
  const handleClick = (id) => {
    onSelect?.(id)
    navigate(`${location.pathname}?c=${encodeURIComponent(id)}`)
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 border-t border-ink-border px-3 py-2">
      <div className="flex items-center justify-between pb-1.5 text-[11px] text-zinc-500 font-medium">
        <span>Case queue{rows.length ? ` · ${rows.length}` : ''}</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {rows.length === 0 ? (
          <div className="text-[11px] text-zinc-500 py-4 text-center leading-relaxed">
            No complaints in queue.
            <br />
            Add one from the Cases screen.
            <br />
            <span className="text-zinc-600">No live feed in this build.</span>
          </div>
        ) : (
          rows.map(c => {
            const isActive = c.ticket_id === activeId
            return (
              <button
                key={c.ticket_id}
                onClick={() => handleClick(c.ticket_id)}
                title={`${formatTicket(c.ticket_id)} — ${c.victim_name}, ${c.city}`}
                className={`w-full text-left p-1.5 rounded border transition-colors duration-100 flex items-center justify-between text-[11px] ${
                  isActive
                    ? 'bg-ink-panel border-aegis-green/50'
                    : 'bg-ink-panel/60 border-ink-border/80 hover:border-ink-border2 hover:bg-ink-panel'
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
                  <div className="mono tnum font-semibold text-zinc-200">{amountShort(c.stolen_amount)}</div>
                  <div className="text-[10px] text-zinc-500">{timeAgo(c.complaint_timestamp, now)}</div>
                </div>
              </button>
            )
          })
        )}
      </div>
    </div>
  )
}

// CONNECTED is deliberately absent. The backend broadcasts it on socket open
// (backend/main.py), so for 12 seconds after every connect the header rendered
// "feed connected" beside the persistent "1930 FEED CONNECTED" pill -- same
// green, same shape, saying the same thing twice. This pill is for events that
// carry news; connection state is owned by the pill next to it.
const EVENT_LABEL = {
  NEW_COMPLAINT: 'complaint ingested',
  PREDICTION_READY: 'prediction ready',
  FREEZE_EXECUTED: 'freeze executed',
}

export function Topbar({ wsConnected, complaintId, backendDown, lastEvent }) {
  const now = useNow(1000)
  const isNews = lastEvent && EVENT_LABEL[lastEvent.type]
  const recent = isNews && now - lastEvent.at < 12000 ? lastEvent : null

  return (
    <header className="h-[48px] flex items-center justify-between gap-3 px-4 border-b border-ink-border bg-ink-bg shrink-0 z-30">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded bg-aegis-green grid place-items-center shrink-0">
          <Shield size={16} className="text-black" />
        </div>
        <div className="leading-tight min-w-0">
          <div className="text-[13px] font-semibold tracking-tight text-white">MuleShield AI</div>
          <div className="text-[11px] text-zinc-400 truncate">
            Fraud investigation console · 1930 / I4C
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {recent && (
          <span className="hidden lg:flex items-center gap-1.5 text-[11px] text-aegis-green bg-aegis-green/10 border border-aegis-green/30 rounded px-2.5 py-1">
            <Radio size={10} /> {EVENT_LABEL[recent.type]}
          </span>
        )}

        <div className="hidden md:flex items-center gap-2 text-[11px] bg-ink-panel border border-ink-border rounded px-3 py-1">
          {backendDown ? (
            <span className="flex items-center gap-1.5 text-red-400 font-semibold">
              <WifiOff size={11} /> Service offline
            </span>
          ) : (
            <span
              className={`flex items-center gap-1.5 ${
                wsConnected ? 'text-aegis-green font-semibold' : 'text-zinc-500'
              }`}
            >
              <Circle size={8} className={wsConnected ? 'fill-aegis-green' : ''} />
              {wsConnected ? 'Connected' : 'Reconnecting…'}
            </span>
          )}
        </div>

        {complaintId && (
          <div className="hidden sm:flex items-baseline gap-1.5 text-[11px] text-zinc-400 border border-ink-border rounded px-3 py-1 bg-ink-panel">
            Case
            <span className="mono tnum font-semibold text-zinc-100">{formatTicket(complaintId)}</span>
          </div>
        )}
      </div>
    </header>
  )
}

const NAV_ITEMS = [
  ['/', <Activity size={15} key="i" />, 'Cases', 'Queue and triage'],
  ['/graph', <GitBranch size={15} key="i" />, 'Transaction trail', 'Fund movement'],
  ['/map', <MapPinned size={15} key="i" />, 'Locations', 'Cash-out points'],
  ['/intercept', <Zap size={15} key="i" />, 'Intervention', 'Freeze and escalate'],
  ['/model', <BarChart3 size={15} key="i" />, 'Model performance', 'Detection accuracy'],
]

export function Sidebar({ complaintId, complaints, onSelect }) {
  const location = useLocation()
  const withCid = (path) =>
    complaintId ? `${path}?c=${encodeURIComponent(complaintId)}` : path

  // On the triage screen the main panel already lists these complaints with room
  // for severity, victim and amount. Repeating a truncated copy of the same rows
  // in a 210px column taught the operator nothing and cost the queue the width.
  const showTicker = location.pathname !== '/'

  return (
    <nav className="w-[210px] shrink-0 border-r border-ink-border bg-ink-bg hidden md:flex flex-col min-h-0">
      <div className="p-3 space-y-1.5 shrink-0">
        {NAV_ITEMS.map(([to, icon, label, sub]) => (
          <NavLink
            key={to}
            to={withCid(to)}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md border text-[12.5px] transition-colors ${
                isActive
                  ? 'bg-ink-panel border-ink-border text-white font-medium'
                  : 'border-transparent text-zinc-400 hover:text-zinc-100 hover:bg-ink-panel/50'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span className={`shrink-0 ${isActive ? 'text-aegis-green' : 'text-zinc-500'}`}>
                  {icon}
                </span>
                <span className="flex-1 leading-tight">
                  {label}
                  <br />
                  <span className="text-[10.5px] text-zinc-500 font-normal">{sub}</span>
                </span>
              </>
            )}
          </NavLink>
        ))}
      </div>

      {showTicker ? (
        <SidebarStreamTicker complaints={complaints} onSelect={onSelect} activeId={complaintId} />
      ) : (
        <div className="flex-1 min-h-0" />
      )}

      {/* The model metrics that used to live here are on /model.
          They describe the detector across a test set, not the case an analyst
          is reading, and standing beside one invited them to be read as its
          confidence. */}
      <div className="p-3 border-t border-ink-border shrink-0 text-[10.5px] text-zinc-600">
        SIH26184 · MHA / I4C
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

// Carried a `live` prop that rendered a pulsing "Live" tag. No caller ever set
// it, and it was the same unearned claim the intake badge was making, so it is
// gone rather than left available to switch on.
export function Panel({ title, right, children, className = '', bodyClass = '' }) {
  return (
    <section className={`aegis-panel ${className}`}>
      <div className="aegis-panel-header">
        <div className="flex items-center gap-2 text-[12px] font-medium text-zinc-300 min-w-0">
          <span className="truncate">{title}</span>
        </div>
        <div className="text-[11px] text-zinc-500 shrink-0 pl-2">{right}</div>
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
    <div className="rounded border border-ink-border bg-ink-panel px-2.5 py-2">
      <div className="text-[11px] text-zinc-500">{label}</div>
      <div className={`mono tnum text-[16px] font-semibold mt-0.5 ${tones[tone] || tones.default}`}>{value}</div>
      {sub && <div className="text-[11px] text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  )
}
