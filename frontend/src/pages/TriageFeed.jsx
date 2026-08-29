import { useEffect, useState, useMemo, useCallback, memo } from 'react'
import { useNavigate } from 'react-router-dom'
import { endpoints, describeError } from '../services/api'
import { amountFmt, amountShort, FRAUD_TYPES, BANKS, CITIES } from '../utils/constants'
import { timeAgo, useNow } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import {
  ShieldAlert, Plus, Search, MapPinned, GitBranch, Zap, ArrowUpRight,
  CheckCircle2, AlertTriangle, Loader2, Radio,
} from 'lucide-react'

const GOLDEN_HOUR_MIN = 60
const CRITICAL_MIN = 15
const WARNING_MIN = 35
const HIGH_VALUE = 150000
const MID_VALUE = 75000

/**
 * Severity of a complaint, from how long the money has been moving and how
 * much of it there is. Age is bucketed rather than read continuously so a row
 * only changes class when it crosses a real threshold.
 */
function severityOf(complaint, now) {
  const ts = new Date(complaint.complaint_timestamp).getTime()
  const mins = Number.isNaN(ts) ? Infinity : Math.max(0, (now - ts) / 60000)
  const amount = Number(complaint.stolen_amount) || 0

  if (mins < CRITICAL_MIN || amount >= HIGH_VALUE) {
    return { weight: 3, label: `CRITICAL · <${CRITICAL_MIN}m`, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', dot: 'bg-red-500', critical: true }
  }
  if (mins < WARNING_MIN || amount >= MID_VALUE) {
    return { weight: 2, label: `WARNING · <${WARNING_MIN}m`, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', dot: 'bg-amber-500', critical: false }
  }
  return { weight: 1, label: 'MONITORING', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30', dot: 'bg-emerald-500', critical: false }
}

/** Fraction of the golden hour already burned, for the row's urgency bar. */
function goldenHourUsed(complaint, now) {
  const ts = new Date(complaint.complaint_timestamp).getTime()
  if (Number.isNaN(ts)) return 1
  const mins = Math.max(0, (now - ts) / 60000)
  return Math.min(1, mins / GOLDEN_HOUR_MIN)
}

const TriageRow = memo(function TriageRow({ c, isSelected, onSelect, now }) {
  const sev = severityOf(c, now)
  const used = goldenHourUsed(c, now)

  return (
    <button
      onClick={() => onSelect(c.ticket_id)}
      className={`w-full text-left p-3 rounded-lg border transition-colors duration-150 relative overflow-hidden ${
        isSelected
          ? 'bg-ink-panel border-aegis-green ring-1 ring-aegis-green/30'
          : sev.critical
          ? 'bg-ink-surface/80 border-ink-border hover:bg-ink-panel hover:border-red-500/40'
          : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`w-2 h-2 rounded-full shrink-0 ${sev.dot}`} />
          <span className="mono text-[12px] font-bold text-white tracking-wide truncate">{c.ticket_id}</span>
          {c.is_live && (
            <span className="px-1.5 py-0.5 rounded text-[9px] mono border border-aegis-green/40 bg-aegis-green/10 text-aegis-green font-bold shrink-0">
              LIVE
            </span>
          )}
          <span className={`px-2 py-0.5 rounded text-[10px] mono border font-medium shrink-0 ${sev.bg} ${sev.color}`}>
            {sev.label}
          </span>
        </div>
        <span className={`mono text-[13px] font-bold shrink-0 ${sev.critical ? 'text-red-400' : 'text-aegis-green'}`}>
          {amountFmt(c.stolen_amount)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 mt-2 text-[11px] mono text-zinc-400">
        <div className="truncate">
          Victim: <span className="text-zinc-200 font-medium">{c.victim_name}</span>
        </div>
        <div className="text-right text-zinc-300 font-medium truncate">{c.city}, {c.state}</div>
        <div className="truncate">{c.victim_bank} · <span className="text-zinc-300">{c.fraud_type}</span></div>
        <div className="text-right text-zinc-500">{timeAgo(c.complaint_timestamp, now)}</div>
      </div>

      {/* Golden-hour burn-down */}
      <div className="mt-2 h-0.5 bg-ink-bg rounded overflow-hidden">
        <span
          className="block h-full transition-all duration-500"
          style={{
            width: `${used * 100}%`,
            background: used > 0.75 ? '#ff3b3b' : used > 0.45 ? '#ff8c42' : '#7cf000',
          }}
        />
      </div>
    </button>
  )
})

const IngestModal = memo(function IngestModal({ isOpen, onClose, onSuccess }) {
  const [form, setForm] = useState({
    victim_name: '', victim_bank: 'State Bank of India', victim_account: '',
    fraud_type: 'UPI Fraud', stolen_amount: '250000', city: 'Pune', state: 'Maharashtra',
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Reset between openings so a previous failure doesn't linger.
  useEffect(() => {
    if (isOpen) setError('')
  }, [isOpen])

  if (!isOpen) return null

  const set = (k) => (e) => setForm(prev => ({ ...prev, [k]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      const res = await endpoints.ingestComplaint({
        ...form,
        stolen_amount: Number(form.stolen_amount) || 50000,
        victim_account: form.victim_account || `ACC-${Math.floor(10000000 + Math.random() * 90000000)}`,
      })
      onSuccess(res)
      onClose()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const field = 'w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green transition-colors'

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 grid place-items-center p-4" onClick={onClose}>
      <div className="aegis-panel w-full max-w-lg p-5 bg-ink-bg border-ink-border2 shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between pb-3 border-b border-ink-border">
          <div className="mono text-[13px] font-bold text-white flex items-center gap-2">
            <Plus size={16} className="text-aegis-green" /> Ingest 1930 Cybercrime Complaint
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-white mono text-[12px] px-1">✕</button>
        </div>

        <form onSubmit={submit} className="space-y-3 mt-4 mono text-[11px]">
          <div>
            <label className="block text-zinc-400 mb-1">Victim Full Name</label>
            <input required type="text" placeholder="e.g. Ramesh Chandra" value={form.victim_name} onChange={set('victim_name')} className={field} />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-zinc-400 mb-1">Victim Bank</label>
              <select value={form.victim_bank} onChange={set('victim_bank')} className={field}>
                {BANKS.map(b => <option key={b} value={b}>{b}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-zinc-400 mb-1">Stolen Amount (₹)</label>
              <input required type="number" min="1000" step="1000" value={form.stolen_amount} onChange={set('stolen_amount')} className={field} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-zinc-400 mb-1">Fraud Category</label>
              <select value={form.fraud_type} onChange={set('fraud_type')} className={field}>
                {FRAUD_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-zinc-400 mb-1">City</label>
              <input required list="ms-cities" type="text" value={form.city} onChange={set('city')} className={field} />
              <datalist id="ms-cities">{CITIES.map(c => <option key={c} value={c} />)}</datalist>
            </div>
          </div>

          <div>
            <label className="block text-zinc-400 mb-1">State</label>
            <input required type="text" value={form.state} onChange={set('state')} className={field} />
          </div>

          <p className="text-[10px] text-zinc-500 leading-relaxed">
            The bank/NPCI feed supplies the transaction ledger in production. For this console the
            layering chain is drawn from real graph accounts in the victim's city, so the GNN
            embeddings, ATM directory and XGBoost inference all run unchanged.
          </p>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-300 px-3 py-2 rounded">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-3 border-t border-ink-border">
            <button type="button" onClick={onClose} className="px-4 py-2 rounded border border-ink-border text-zinc-400 hover:text-white">
              Cancel
            </button>
            <button type="submit" disabled={busy} className="px-4 py-2 rounded bg-aegis-green text-black font-bold hover:bg-emerald-400 disabled:opacity-50 flex items-center gap-2">
              {busy && <Loader2 size={13} className="animate-spin" />}
              {busy ? 'Broadcasting…' : 'Ingest & Trigger AI'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
})

export default function TriageFeed({ complaints = [], selected, onSelect, onIngested, backendDown }) {
  const [dismissed, setDismissed] = useState(() => new Set())
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [modalOpen, setModalOpen] = useState(false)
  const [health, setHealth] = useState(null)
  const navigate = useNavigate()

  // 15s is enough for relative timestamps and severity buckets. A 1s tick here
  // re-sorted the entire queue sixty times a minute for no visible benefit.
  const now = useNow(15000)

  useEffect(() => {
    endpoints.health().then(setHealth).catch(() => setHealth(null))
  }, [complaints.length])

  const visible = useMemo(() => {
    const rows = complaints.filter(c => c?.ticket_id && !dismissed.has(c.ticket_id))

    const scored = rows.map(c => ({ c, w: severityOf(c, now).weight }))
    scored.sort((a, b) => (b.w - a.w) || ((b.c.stolen_amount || 0) - (a.c.stolen_amount || 0)))

    const needle = q.trim().toLowerCase()
    return scored
      .map(s => s.c)
      .filter(c => {
        if (filter !== 'ALL' && c.fraud_type !== filter) return false
        if (!needle) return true
        return `${c.ticket_id} ${c.victim_name} ${c.city} ${c.state} ${c.fraud_type} ${c.victim_account}`
          .toLowerCase().includes(needle)
      })
  }, [complaints, dismissed, q, filter, now])

  const active = useMemo(
    () => visible.find(c => c.ticket_id === selected) || visible[0] || null,
    [visible, selected]
  )

  // Adopt the top of the queue when nothing is selected yet.
  useEffect(() => {
    if (!selected && visible.length) onSelect?.(visible[0].ticket_id)
  }, [selected, visible, onSelect])

  const stats = useMemo(() => ({
    critical: visible.filter(c => severityOf(c, now).critical).length,
    atRisk: visible.reduce((sum, c) => sum + (Number(c.stolen_amount) || 0), 0),
  }), [visible, now])

  const resolve = useCallback(() => {
    if (active) setDismissed(prev => new Set(prev).add(active.ticket_id))
  }, [active])

  const activeSev = active ? severityOf(active, now) : null

  const jump = (path) => {
    if (!active) return
    navigate(`${path}?c=${encodeURIComponent(active.ticket_id)}`)
  }

  return (
    <div className="p-4 space-y-4">
      {/* ── Command metrics ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="aegis-panel p-3 border-l-4 border-l-blue-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">Queue Depth</div>
          <div className="text-2xl font-bold text-white mt-1 mono">{visible.length}</div>
          <div className="text-[11px] text-zinc-500 mt-1">
            {health ? `${health.active_complaints.toLocaleString('en-IN')} on national feed` : 'Queue total unavailable'}
          </div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-red-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">Critical Severity</div>
          <div className="text-2xl font-bold text-red-400 mt-1 mono">{stats.critical}</div>
          <div className="text-[11px] text-red-400/80 mt-1">&lt;{CRITICAL_MIN}m window / high-loss</div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-emerald-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">Funds At Risk</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1 mono">{amountShort(stats.atRisk)}</div>
          <div className="text-[11px] text-emerald-400/80 mt-1">Across the visible queue</div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-purple-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">Graph Corpus</div>
          <div className="text-2xl font-bold text-purple-400 mt-1 mono">
            {health ? health.embeddings_loaded.toLocaleString('en-IN') : '—'}
          </div>
          <div className="text-[11px] text-zinc-500 mt-1">64-d GraphSAGE embeddings</div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* ── Queue ───────────────────────────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-7">
          <div className="aegis-panel p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <ShieldAlert size={16} className="text-aegis-green" />
                <span className="mono text-[12px] font-bold tracking-wider text-white">1930 TRIAGE QUEUE</span>
                <span className="text-[10px] mono text-red-400 font-semibold bg-red-500/10 border border-red-500/30 px-2 py-0.5 rounded flex items-center gap-1">
                  <AlertTriangle size={11} /> SEVERITY RANKED
                </span>
              </div>
              <button
                onClick={() => setModalOpen(true)}
                disabled={backendDown}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-aegis-green text-black mono text-[11px] font-bold hover:bg-emerald-400 disabled:opacity-40 transition"
              >
                <Plus size={14} /> Ingest Complaint
              </button>
            </div>

            <div className="flex flex-wrap gap-2 mb-3">
              <div className="flex-1 min-w-[200px] flex items-center gap-2 bg-ink-bg border border-ink-border rounded px-2.5 py-1.5">
                <Search size={13} className="text-zinc-500 shrink-0" />
                <input
                  type="text"
                  placeholder="Search ticket, victim, city…"
                  value={q}
                  onChange={e => setQ(e.target.value)}
                  className="bg-transparent outline-none text-[12px] mono w-full text-zinc-200 placeholder:text-zinc-600"
                />
              </div>
              <select
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="bg-ink-bg border border-ink-border rounded px-2.5 py-1.5 text-[11px] mono text-zinc-300 outline-none"
              >
                <option value="ALL">All Fraud Types</option>
                {FRAUD_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>

            <div className="space-y-2 max-h-[58vh] overflow-y-auto pr-1">
              {visible.length === 0 ? (
                <div className="p-8 text-center text-zinc-500 mono text-[12px] leading-relaxed">
                  {backendDown
                    ? 'Backend unreachable — no live queue to display.'
                    : complaints.length === 0
                    ? 'Queue is empty. Ingest a complaint to start the pipeline.'
                    : 'No complaints match the current filter.'}
                </div>
              ) : (
                visible.map(c => (
                  <TriageRow
                    key={c.ticket_id}
                    c={c}
                    isSelected={active?.ticket_id === c.ticket_id}
                    onSelect={onSelect}
                    now={now}
                  />
                ))
              )}
            </div>
          </div>
        </div>

        {/* ── Selected incident ───────────────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-5 space-y-3">
          {active ? (
            <div className="aegis-panel p-4 space-y-4">
              <div className="border-b border-ink-border pb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="mono text-[10px] uppercase tracking-wider text-zinc-400 font-semibold flex items-center gap-2 flex-wrap">
                    <span>Selected Incident</span>
                    {activeSev && (
                      <span className={`px-2 py-0.5 rounded text-[9px] mono border ${activeSev.bg} ${activeSev.color}`}>
                        {activeSev.label}
                      </span>
                    )}
                  </div>
                  <div className="text-lg font-bold text-white mono mt-0.5 truncate">{active.ticket_id}</div>
                  <div className="text-[12px] mono text-zinc-400 mt-1 truncate">
                    <span className="text-white font-medium">{active.victim_name}</span> · {active.victim_bank}
                  </div>
                  <div className="text-[11px] mono text-zinc-500 truncate">{active.victim_account}</div>
                </div>
                <button
                  onClick={resolve}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 mono text-[11px] font-bold transition shrink-0"
                  title="Remove from the active queue"
                >
                  <CheckCircle2 size={13} /> Resolve
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] mono">
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-400">Stolen Amount</div>
                  <div className="text-base font-bold text-red-400 mt-0.5">{amountFmt(active.stolen_amount)}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-400">Category</div>
                  <div className="text-sm font-semibold text-white mt-1">{active.fraud_type}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-400">Location</div>
                  <div className="text-sm text-zinc-200 mt-1">{active.city}, {active.state}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-400">Reported</div>
                  <div className="text-sm text-zinc-200 mt-1">{timeAgo(active.complaint_timestamp, now)}</div>
                </div>
              </div>

              <div className="pt-2 border-t border-ink-border space-y-2">
                <div className="mono text-[11px] font-bold text-zinc-300">TACTICAL ACTIONS</div>
                {[
                  ['/map', MapPinned, 'text-red-400', 'hover:border-red-500/60 hover:bg-red-500/10', 'Tactical GIS Map', 'Predicted ATM cluster & PCR dispatch'],
                  ['/graph', GitBranch, 'text-blue-400', 'hover:border-blue-500/60 hover:bg-blue-500/10', 'Forensic Money-Flow Graph', 'Multi-hop layering & GNN risk per node'],
                  ['/intercept', Zap, 'text-aegis-green', 'hover:border-aegis-green/60 hover:bg-aegis-green/10', '1-Click Emergency Freeze', 'Lock terminal accounts before cashout'],
                ].map(([path, Icon, iconColor, hover, title, sub]) => (
                  <button
                    key={path}
                    onClick={() => jump(path)}
                    className={`w-full flex items-center justify-between p-2.5 rounded bg-ink-bg border border-ink-border text-left transition-colors duration-150 group ${hover}`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <Icon size={16} className={`${iconColor} shrink-0`} />
                      <div className="min-w-0">
                        <div className="mono text-[11px] font-bold text-white truncate">{title}</div>
                        <div className="mono text-[10px] text-zinc-400 truncate">{sub}</div>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-zinc-500 group-hover:text-white shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="aegis-panel p-8 text-center text-zinc-500 mono text-[12px]">
              Select a complaint from the queue.
            </div>
          )}

          <Panel title="PIPELINE STATUS" right={health ? 'ONLINE' : 'OFFLINE'}>
            <div className="p-3 grid grid-cols-2 gap-2 mono text-[11px]">
              {[
                ['ATM Directory', health ? health.atm_directory_size.toLocaleString('en-IN') : '—'],
                ['GNN Embeddings', health ? health.embeddings_loaded.toLocaleString('en-IN') : '—'],
                ['WS Clients', health ? health.ws_connections : '—'],
                ['API Version', health ? health.version : '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between bg-ink-panel border border-ink-border rounded px-2.5 py-1.5">
                  <span className="text-zinc-400">{k}</span>
                  <span className="text-zinc-200 font-bold">{v}</span>
                </div>
              ))}
              <div className="col-span-2 flex items-center gap-1.5 text-[10px] text-zinc-500 pt-1">
                <Radio size={11} className={health ? 'text-aegis-green' : 'text-zinc-600'} />
                Figures read live from <span className="text-zinc-400">/health</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <IngestModal isOpen={modalOpen} onClose={() => setModalOpen(false)} onSuccess={onIngested} />
    </div>
  )
}
