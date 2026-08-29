import { useEffect, useState, useMemo, useCallback, memo } from 'react'
import { useNavigate } from 'react-router-dom'
import { endpoints, describeError } from '../services/api'
import { amountFmt, amountShort, formatTicket, MODEL_STATS, FRAUD_TYPES, BANKS, CITIES } from '../utils/constants'
import { timeAgo, useNow } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import {
  ShieldAlert, Plus, Search, MapPinned, GitBranch, Zap, ArrowUpRight,
  CheckCircle2, AlertTriangle, Loader2, Radio, RotateCcw,
} from 'lucide-react'

const GOLDEN_HOUR_MIN = 60

// A queue of small complaints must not manufacture a HIGH VALUE row simply by
// containing the largest of a set of small ones. Percentiles decide the
// ordering; this decides whether the top of that ordering earns the label.
const HIGH_VALUE_FLOOR = 100000

/** Minutes since a complaint was filed, or Infinity if the timestamp is unusable. */
function ageMinutes(complaint, now) {
  const ts = new Date(complaint.complaint_timestamp).getTime()
  return Number.isNaN(ts) ? Infinity : Math.max(0, (now - ts) / 60000)
}

/**
 * Is this complaint's clock safe to draw a countdown from?
 *
 * Complaints stamped by the backend carry `is_live` and a UTC offset
 * (state.py writes datetime.now(timezone.utc).isoformat()). The seed corpus
 * carries naive local-time strings — "2026-05-27T19:43:13", no zone — which JS
 * resolves against the *browser's* timezone. On a demo machine in a different
 * zone from the one that generated the CSV, a seed row can drift inside the
 * last 60 minutes and start a countdown for a complaint that is months old.
 *
 * A countdown that is not true is the exact defect this model exists to remove,
 * so the golden hour is only ever drawn for a timestamp that says what zone it
 * is in.
 */
function hasTrustedClock(complaint) {
  if (complaint?.is_live) return true
  return /(?:Z|[+-]\d{2}:\d{2})$/.test(String(complaint?.complaint_timestamp || ''))
}

/**
 * Value thresholds taken from the queue that is actually loaded.
 *
 * These were fixed rupee constants (₹1.5L / ₹75k). Against the shipped corpus
 * that put 43% of complaints in the top band and 73% in the top two, so the
 * badge told an operator almost nothing — the head of the queue was uniformly
 * red. Percentiles self-calibrate to whatever data is present, so the top band
 * stays a top band whatever the caseload looks like.
 */
function valueBands(complaints) {
  const amts = complaints
    .map(c => Number(c.stolen_amount) || 0)
    .sort((a, b) => a - b)
  const at = p => (amts.length ? amts[Math.min(amts.length - 1, Math.floor(p * amts.length))] : 0)
  const high = Math.max(at(0.90), HIGH_VALUE_FLOOR)
  // The floor can push `high` past p90, so the share is counted rather than
  // assumed — the stat card prints this number and it has to be true.
  const share = amts.length ? amts.filter(a => a >= high).length / amts.length : 0
  return { high, mid: at(0.65), share }
}

/**
 * Severity on two axes that are never collapsed into one claim.
 *
 * The old version was `age < 15min OR amount >= ₹1.5L` and labelled the result
 * "CRITICAL · <15m". No complaint in the corpus is under 12 HOURS old, so the
 * age test never fired and every badge was decided by amount alone — while
 * still asserting a 15-minute window. A row read "CRITICAL · <15m" directly
 * above its own "3d ago". That is the kind of contradiction a judge reads as
 * fabrication, so the label now states only what is true of that complaint.
 *
 * Axis A, the golden hour, applies only where it is real: a complaint ingested
 * through the console is timestamped now (backend/state.py add_complaint), so
 * its 60-minute window genuinely ticks. Axis B, value, ranks the cold backlog
 * that the window has already closed on.
 */
function severityOf(complaint, now, bands) {
  const mins = ageMinutes(complaint, now)
  const amount = Number(complaint.stolen_amount) || 0
  const ts = new Date(complaint.complaint_timestamp).getTime()

  if (mins < GOLDEN_HOUR_MIN && hasTrustedClock(complaint)) {
    const open = { label: 'GOLDEN HOUR', golden: true, urgent: true, startedAt: ts }
    if (mins < 15) {
      return { ...open, weight: 5, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', dot: 'bg-red-500' }
    }
    if (mins < 35) {
      return { ...open, weight: 4, color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/30', dot: 'bg-orange-500' }
    }
    return { ...open, weight: 3, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', dot: 'bg-amber-500' }
  }

  // Cold: the interception window closed. Rank by what is at stake instead.
  const cold = { golden: false, urgent: false, startedAt: null }
  if (amount >= bands.high) {
    return { ...cold, weight: 2, label: 'HIGH VALUE', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', dot: 'bg-red-500' }
  }
  if (amount >= bands.mid) {
    return { ...cold, weight: 1, label: 'ELEVATED', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', dot: 'bg-amber-500' }
  }
  // Zinc rather than emerald: emerald is this console's "good / live" accent and
  // the majority of rows wearing it competed with the aegis-green LIVE chip.
  return { ...cold, weight: 0, label: 'ROUTINE', color: 'text-zinc-400', bg: 'bg-zinc-500/10 border-zinc-600/30', dot: 'bg-zinc-600' }
}

/**
 * Minutes:seconds left on an open interception window.
 *
 * Its own 1-second tick. The page runs `useNow(15000)` because 60 static rows
 * do not need to repaint every second — but a countdown that advances in
 * 15-second jumps reads as broken during the one moment the console is
 * actually claiming to be real-time. Mounted only for golden rows, so nothing
 * else pays for the faster tick.
 */
const GoldenHourClock = memo(function GoldenHourClock({ startedAt }) {
  const tick = useNow(1000)
  const left = Math.max(0, GOLDEN_HOUR_MIN * 60000 - (tick - startedAt))
  if (left === 0) return <span>expired</span>
  const mm = String(Math.floor(left / 60000)).padStart(2, '0')
  const ss = String(Math.floor((left % 60000) / 1000)).padStart(2, '0')
  return <span>{mm}:{ss}</span>
})

/** One badge, so a row and the selected-incident panel can never disagree. */
const SevBadge = memo(function SevBadge({ sev }) {
  return (
    <span
      className={`px-2 py-0.5 rounded text-[10px] mono border font-medium shrink-0 inline-flex items-center gap-1 ${sev.bg} ${sev.color}`}
    >
      {sev.label}
      {sev.golden && sev.startedAt != null && (
        <>· <GoldenHourClock startedAt={sev.startedAt} /></>
      )}
    </span>
  )
})

/**
 * Fraction of the golden hour burned, or null once it has closed.
 *
 * null, not 1. The bar used to clamp to 1 for anything past the window, which
 * meant every row in a corpus of days-old complaints rendered a full red bar —
 * an urgency signal on rows where no urgency remained. The caller now omits the
 * bar entirely for those.
 */
function goldenHourUsed(complaint, now) {
  const mins = ageMinutes(complaint, now)
  if (!Number.isFinite(mins) || mins >= GOLDEN_HOUR_MIN) return null
  return mins / GOLDEN_HOUR_MIN
}

const TriageRow = memo(function TriageRow({ c, isSelected, onSelect, now, bands }) {
  const sev = severityOf(c, now, bands)
  const used = goldenHourUsed(c, now)

  return (
    <button
      onClick={() => onSelect(c.ticket_id)}
      className={`w-full text-left p-3 rounded-lg border transition-colors duration-150 relative overflow-hidden ${
        isSelected
          ? 'bg-ink-panel border-aegis-green ring-1 ring-aegis-green/30'
          : sev.urgent
          ? 'bg-ink-surface/80 border-ink-border hover:bg-ink-panel hover:border-red-500/40'
          : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`w-2 h-2 rounded-full shrink-0 ${sev.dot}`} />
          <span className="mono text-[12px] font-bold text-white tracking-wide truncate">{formatTicket(c.ticket_id)}</span>
          {c.is_live && (
            <span className="px-1.5 py-0.5 rounded text-[9px] mono border border-aegis-green/40 bg-aegis-green/10 text-aegis-green font-bold shrink-0">
              LIVE
            </span>
          )}
          <SevBadge sev={sev} />
        </div>
        <span className={`mono text-[13px] font-bold shrink-0 ${sev.urgent ? 'text-red-400' : 'text-aegis-green'}`}>
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

      {/* Golden-hour burn-down. Drawn only while the window is genuinely open --
          on a days-old complaint a full bar is decoration, not information. */}
      {used !== null && (
        <div className="mt-2 h-0.5 bg-ink-bg rounded overflow-hidden">
          <span
            className="block h-full transition-all duration-500"
            style={{
              width: `${used * 100}%`,
              background: used > 0.75 ? '#ff3b3b' : used > 0.45 ? '#ff8c42' : '#7cf000',
            }}
          />
        </div>
      )}
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

/** "57, Gandhi Chowk, Kochi, Kerala - 133704 (Branch ATM)" -> "Kochi". */
function townOf(address, fallback) {
  const parts = String(address || '').split(',').map(t => t.trim())
  return parts.length >= 3 ? parts[2] : fallback
}

/**
 * The case in plain language, above the numbers.
 *
 * The console was legible to someone who already knew what a GraphSAGE
 * embedding was and opaque to everyone else — which on an SIH jury is most of
 * the room. A police officer or a policy official could read every figure on
 * this screen and still not be able to say what had happened or what the system
 * was recommending.
 *
 * So this states it as sentences: what was stolen, where the money went, where
 * we expect it to be withdrawn, and how long there is to act. Nothing here is
 * new information — it is the same prediction the tactical screens render,
 * written for someone reading it for the first time. The technical detail stays
 * directly underneath, unchanged, for the judges who want it.
 */
const CaseStory = memo(function CaseStory({ complaint, sev, prediction, loading, error, now }) {
  const amount = amountFmt(complaint.stolen_amount)
  const when = timeAgo(complaint.complaint_timestamp, now)
  const scam = String(complaint.fraud_type || 'cyber fraud').toLowerCase()
  // Fraud types are data, so the article has to be chosen rather than hardcoded:
  // "a digital arrest" but "an investment scam".
  const article = /^[aeiou]/.test(scam) ? 'an' : 'a'

  const top = prediction?.ranked_candidates?.[0]
  const nAtms = prediction?.ranked_candidates?.length ?? 0
  const town = top ? townOf(top.address, complaint.city) : null
  const mins = prediction ? Math.max(1, Math.round(prediction.time_to_cashout_minutes)) : null

  return (
    <div className="rounded-lg border border-ink-border bg-ink-bg/60 p-3 space-y-2.5">
      <div className="mono text-[10px] uppercase tracking-[0.12em] text-zinc-500 font-semibold">
        The case, in plain terms
      </div>

      <p className="text-[12.5px] leading-relaxed text-zinc-300">
        <span className="text-white font-semibold">{amount}</span> was taken from{' '}
        <span className="text-white font-semibold">{complaint.victim_name}</span> in{' '}
        {complaint.city}, {complaint.state} — {article} {scam}, reported {when}.
      </p>

      {loading && (
        <p className="text-[12.5px] leading-relaxed text-zinc-500 flex items-center gap-2">
          <Loader2 size={12} className="animate-spin shrink-0" />
          Tracing the money and forecasting the withdrawal…
        </p>
      )}

      {!loading && prediction && (
        <>
          <p className="text-[12.5px] leading-relaxed text-zinc-300">
            The money was moved through a chain of mule accounts and now sits in
            account <span className="text-white font-semibold">{prediction.terminal_account}</span>.
          </p>
          {/* Tense matters. A complaint filed three days ago cannot have a
              withdrawal "expected in 32 minutes" — that money left the system
              long ago, and the sentence would contradict the closing line of
              this same paragraph. The forecast is stated as a live expectation
              only while the window is open, and as what the model placed at the
              time otherwise. */}
          {sev?.golden ? (
            <p className="text-[12.5px] leading-relaxed text-zinc-300">
              We expect a cash withdrawal in about{' '}
              <span className="text-white font-semibold">{mins} minutes</span>, most likely at{' '}
              <span className="text-white font-semibold">one of {nAtms} ATMs</span>
              {town ? <> around {town}</> : null} — narrowed from{' '}
              <span className="text-white font-semibold">{MODEL_STATS.atmTotal}</span> nationwide.
            </p>
          ) : (
            <p className="text-[12.5px] leading-relaxed text-zinc-300">
              On the evidence available, the model puts the withdrawal about{' '}
              <span className="text-white font-semibold">{mins} minutes</span> after the
              transfer, at <span className="text-white font-semibold">one of {nAtms} ATMs</span>
              {town ? <> around {town}</> : null} — narrowed from{' '}
              <span className="text-white font-semibold">{MODEL_STATS.atmTotal}</span> nationwide.
            </p>
          )}
        </>
      )}

      {!loading && !prediction && (
        <p className="text-[12.5px] leading-relaxed text-amber-400/90">
          {/* Naming the cause matters: "could not be produced" reads as a model
              failure when the usual reason is simply that the API is not up. */}
          No withdrawal forecast — {error || 'the prediction service did not respond'}.
        </p>
      )}

      {/* The one line that tells an officer whether to move. */}
      <div className="pt-2 border-t border-ink-border">
        {sev?.golden ? (
          <p className="text-[12.5px] leading-relaxed text-red-300 font-semibold flex items-center gap-1.5">
            <Radio size={12} className="animate-pulse-dot shrink-0" />
            <span>
              Police have <GoldenHourClock startedAt={sev.startedAt} /> of the golden hour left.
            </span>
          </p>
        ) : (
          <p className="text-[12.5px] leading-relaxed text-zinc-500">
            The interception window for this complaint has closed. It is kept for
            pattern analysis and to train the model.
          </p>
        )}
      </div>
    </div>
  )
})

export default function TriageFeed({ complaints = [], selected, onSelect, onIngested, backendDown }) {
  const [dismissed, setDismissed] = useState(() => new Set())
  const [lastResolved, setLastResolved] = useState(null)
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [modalOpen, setModalOpen] = useState(false)
  const [health, setHealth] = useState(null)
  const navigate = useNavigate()

  // 15s is enough for relative timestamps and severity buckets. A 1s tick here
  // re-sorted the entire queue sixty times a minute for no visible benefit.
  // An open golden hour gets its own 1s tick inside GoldenHourClock.
  const now = useNow(15000)

  // Value bands from the whole loaded queue, not the filtered view: if the cuts
  // moved as the operator typed in the search box, a row's badge would change
  // meaning mid-keystroke.
  const bands = useMemo(() => valueBands(complaints), [complaints])

  // Polled, not fetched once. This was keyed on `complaints.length`, so against
  // a static corpus it ran a single time and the WS-client count below it was
  // frozen from page load onward while claiming to read live from /health.
  useEffect(() => {
    let cancelled = false
    const pull = () => endpoints.health()
      .then(h => { if (!cancelled) setHealth(h) })
      .catch(() => { if (!cancelled) setHealth(null) })
    pull()
    const id = setInterval(pull, 15000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const visible = useMemo(() => {
    const rows = complaints.filter(c => c?.ticket_id && !dismissed.has(c.ticket_id))

    const scored = rows.map(c => ({ c, w: severityOf(c, now, bands).weight }))
    scored.sort((a, b) => (b.w - a.w) || ((b.c.stolen_amount || 0) - (a.c.stolen_amount || 0)))

    const needle = q.trim().toLowerCase()
    return scored
      .map(s => s.c)
      .filter(c => {
        if (filter !== 'ALL' && c.fraud_type !== filter) return false
        if (!needle) return true
        // Both the raw id and its displayed short form are searchable, so typing
        // either what is on screen or the full underlying id finds the row.
        return `${c.ticket_id} ${formatTicket(c.ticket_id)} ${c.victim_name} ${c.city} ${c.state} ${c.fraud_type} ${c.victim_account}`
          .toLowerCase().includes(needle)
      })
  }, [complaints, dismissed, q, filter, now, bands])

  const active = useMemo(
    () => visible.find(c => c.ticket_id === selected) || visible[0] || null,
    [visible, selected]
  )

  // Adopt the top of the queue when nothing is selected yet.
  useEffect(() => {
    if (!selected && visible.length) onSelect?.(visible[0].ticket_id)
  }, [selected, visible, onSelect])

  const stats = useMemo(() => {
    let highValue = 0
    let golden = 0
    let atRisk = 0
    for (const c of visible) {
      const s = severityOf(c, now, bands)
      if (s.golden) golden++
      else if (s.weight === 2) highValue++
      atRisk += Number(c.stolen_amount) || 0
    }
    return { highValue, golden, atRisk }
  }, [visible, now, bands])

  // Resolving used to add the ticket to `dismissed` and stop, which dropped it
  // from `visible` but left `selected` pointing at it. The topbar "Active" pill
  // and all three tactical screens then kept naming a complaint the operator had
  // just cleared. The selection has to move with the queue.
  const resolve = useCallback(() => {
    if (!active) return
    const id = active.ticket_id
    const next = visible.find(c => c.ticket_id !== id)
    setDismissed(prev => new Set(prev).add(id))
    setLastResolved(id)
    onSelect?.(next ? next.ticket_id : '')
  }, [active, visible, onSelect])

  // `dismissed` only ever grew, so a misclick cost a page reload to undo.
  const undoResolve = useCallback(() => {
    if (!lastResolved) return
    setDismissed(prev => {
      const nextSet = new Set(prev)
      nextSet.delete(lastResolved)
      return nextSet
    })
    onSelect?.(lastResolved)
    setLastResolved(null)
  }, [lastResolved, onSelect])

  const activeSev = active ? severityOf(active, now, bands) : null

  // The plain-language brief needs the forecast, which lives behind /predict.
  // Fetched per selection and cancelled on change, so switching quickly through
  // the queue cannot land an older response on a newer complaint.
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState('')
  const activeId = active?.ticket_id

  useEffect(() => {
    if (!activeId) { setPrediction(null); return }
    let cancelled = false
    setPredicting(true)
    setPrediction(null)
    setPredictError('')
    endpoints.predictCashout(activeId)
      .then(p => { if (!cancelled) setPrediction(p) })
      .catch(err => {
        if (cancelled) return
        setPrediction(null)
        setPredictError(describeError(err).toLowerCase())
      })
      .finally(() => { if (!cancelled) setPredicting(false) })
    return () => { cancelled = true }
  }, [activeId])

  const jump = (path) => {
    if (!active) return
    navigate(`${path}?c=${encodeURIComponent(active.ticket_id)}`)
  }

  return (
    <div className="p-4 space-y-4">
      {/* ── Command metrics ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="aegis-panel p-3 border-l-4 border-l-blue-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">In View</div>
          <div className="text-2xl font-bold text-white mt-1 mono">{visible.length}</div>
          <div className="text-[11px] text-zinc-500 mt-1">
            {/* This is the loaded page, not a queue depth — the label said
                "Queue Depth" over a number that is really QUEUE_LIMIT. */}
            {health ? `of ${health.active_complaints.toLocaleString('en-IN')} on national feed` : 'Feed total unavailable'}
          </div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-red-500">
          <div className="mono text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">High Value</div>
          <div className="text-2xl font-bold text-red-400 mt-1 mono">{stats.highValue}</div>
          {/* States the rule and the calibrated cut, both computed from the rows
              on screen, so the number can be checked against the queue itself. */}
          <div className="text-[11px] text-red-400/80 mt-1">
            ≥ {amountShort(bands.high)} · top {Math.round(bands.share * 100)}% by loss
          </div>
          {stats.golden > 0 && (
            <div className="text-[11px] text-aegis-green mt-0.5 flex items-center gap-1">
              <Radio size={10} className="animate-pulse-dot" />
              {stats.golden} inside the golden hour
            </div>
          )}
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
                {/* Says what the sort actually does. "SEVERITY RANKED" in red
                    implied every row carried a severity worth alarming about. */}
                <span className="text-[10px] mono text-zinc-400 font-semibold bg-ink-bg border border-ink-border px-2 py-0.5 rounded flex items-center gap-1">
                  <AlertTriangle size={11} /> GOLDEN HOUR FIRST, THEN LOSS
                </span>
              </div>
              <div className="flex items-center gap-2">
                {lastResolved && (
                  <button
                    onClick={undoResolve}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-ink-border bg-ink-bg text-zinc-300 mono text-[11px] hover:text-white hover:border-zinc-600 transition"
                    title={`Restore ${formatTicket(lastResolved)} to the queue`}
                  >
                    <RotateCcw size={12} /> Undo resolve
                  </button>
                )}
                <button
                  onClick={() => setModalOpen(true)}
                  disabled={backendDown}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-aegis-green text-black mono text-[11px] font-bold hover:bg-emerald-400 disabled:opacity-40 transition"
                >
                  <Plus size={14} /> Ingest Complaint
                </button>
              </div>
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
                    bands={bands}
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
                    {/* Same component as the queue row, so the two can never
                        disagree about the same complaint. */}
                    {activeSev && <SevBadge sev={activeSev} />}
                  </div>
                  <div className="text-lg font-bold text-white mono mt-0.5 truncate">{formatTicket(active.ticket_id)}</div>
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

              <CaseStory
                complaint={active}
                sev={activeSev}
                prediction={prediction}
                loading={predicting}
                error={predictError}
                now={now}
              />

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
                  {/* The state of the interception window, said plainly. The
                      console previously implied every complaint still had one. */}
                  <div className={`text-[10px] mt-0.5 ${activeSev?.golden ? 'text-red-400' : 'text-zinc-500'}`}>
                    {activeSev?.golden
                      ? <>window open · <GoldenHourClock startedAt={activeSev.startedAt} /> left</>
                      : 'window closed · historical record'}
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-ink-border space-y-2">
                <div className="mono text-[11px] font-bold text-zinc-300">TACTICAL ACTIONS</div>
                {[
                  ['/map', MapPinned, 'text-red-400', 'hover:border-red-500/60 hover:bg-red-500/10', 'Tactical GIS Map', 'Ranked candidate locations & PCR dispatch'],
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
                // Was 'GNN Embeddings', the same health.embeddings_loaded figure
                // the Graph Corpus card already shows at the top of this screen.
                ['Ranked per case', `Top ${MODEL_STATS.operatingK}`],
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
