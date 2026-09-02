import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  BellRing, Loader2, ServerCrash, ShieldAlert, CheckCircle2, Send, RefreshCw,
} from 'lucide-react'
import { endpoints, describeError } from '../services/api'
import { Panel, Stat } from '../components/Shell'
import { useToast } from '../components/Toast'
import { amountFmt, amountShort, formatTicket, pctFmt } from '../utils/constants'

/**
 * Severity styling.
 *
 * Same grammar as the risk ramp and the triage queue: red and amber carry risk,
 * zinc carries "nothing to do". aegis-accent (#7cf000) appears here only on an
 * ACKNOWLEDGED alert, because green means "handled", never "dangerous".
 */
const SEVERITY = {
  CRITICAL: { bar: 'border-l-red-500', text: 'text-red-300', bg: 'bg-red-500/10',
              ring: 'border-red-500/40' },
  HIGH:     { bar: 'border-l-orange-500', text: 'text-orange-300', bg: 'bg-orange-500/10',
              ring: 'border-orange-500/40' },
  WATCH:    { bar: 'border-l-amber-500', text: 'text-amber-300', bg: 'bg-amber-500/10',
              ring: 'border-amber-500/40' },
}
const sev = (s) => SEVERITY[s] || SEVERITY.WATCH

/**
 * The four dispositions.
 *
 * "False positive" is first in the destructive-honesty sense: it is the one an
 * operator is least motivated to record and the one the system most needs, so it
 * is a first-class button rather than something buried behind "other". Without
 * it nothing ever measures whether an alert was worth raising.
 */
const DISPOSITIONS = ['Dispatched', 'Monitoring', 'False positive', 'Duplicate']

const DELIVERY_TONE = {
  sent: 'text-aegis-accent',
  queued: 'text-zinc-400',
  failed: 'text-amber-300',
  dead: 'text-red-300',
}

function timeAgoShort(iso) {
  if (!iso) return '—'
  const t = new Date(String(iso).replace(' ', 'T'))
  if (Number.isNaN(t.getTime())) return '—'
  const s = Math.max(0, (Date.now() - t.getTime()) / 1000)
  if (s < 90) return `${Math.round(s)}s ago`
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  if (s < 172800) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

/** Live vs historical, as a bar. The same two-segment idiom as ModelPerformance. */
function LiveShareBar({ priorShare }) {
  const live = Math.max(0, Math.min(100, Math.round((1 - (priorShare || 0)) * 100)))
  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-ink-bg border border-ink-border rounded overflow-hidden flex">
          <span className="block h-full bg-red-500" style={{ width: `${live}%` }} />
          <span className="block h-full bg-zinc-600" style={{ width: `${100 - live}%` }} />
        </div>
        <span className="mono tnum text-[11px] text-zinc-400 w-24 text-right">
          {live}% live
        </span>
      </div>
      <div className="text-[10.5px] text-zinc-500 mt-1">
        {live}% of this alert is forecast from open cases · {100 - live}% historical prior
      </div>
    </div>
  )
}

export default function AlertInbox() {
  // useToast() returns the push FUNCTION itself, not an object holding one --
  // see the ToastContext value in components/Toast.jsx, and how RiskHeatmap and
  // TriageFeed both call it. Destructuring { toast } off it yielded undefined,
  // so every toast on this screen threw "toast is not a function". It was
  // invisible in the happy path because the throw lands inside a promise
  // handler, but it broke acknowledgement outright: toast() is the first
  // statement in that .then, so load() and setSelectedId() after it never ran
  // and the inbox did not refresh once an officer closed an alert.
  const toast = useToast()
  const [alerts, setAlerts] = useState([])
  const [summary, setSummary] = useState(null)
  const [detail, setDetail] = useState(null)
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [statusFilter, setStatusFilter] = useState('open')
  const [severityFilter, setSeverityFilter] = useState('ALL')
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    setLoading(true)
    const params = {}
    if (statusFilter !== 'ALL') params.status = statusFilter
    if (severityFilter !== 'ALL') params.severity = severityFilter
    Promise.all([
      endpoints.listAlerts(params),
      endpoints.alertSummary().catch(() => null),
    ])
      .then(([rows, s]) => { setAlerts(rows || []); setSummary(s); setError('') })
      .catch(err => { setAlerts([]); setError(describeError(err)) })
      .finally(() => setLoading(false))
  }, [statusFilter, severityFilter])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!selectedId) { setDetail(null); return }
    let cancelled = false
    endpoints.getAlert(selectedId)
      .then(d => { if (!cancelled) setDetail(d) })
      .catch(() => { if (!cancelled) setDetail(null) })
    return () => { cancelled = true }
  }, [selectedId])

  const runPass = useCallback(() => {
    setBusy(true)
    endpoints.evaluateAlerts()
      .then(r => {
        if (r.degraded) {
          // Not the same as "nothing qualified", and the difference matters.
          toast('No live cases in the current window — nothing to evaluate.', 'error')
        } else {
          toast(`Rule pass over ${r.cells_considered} cells — ${r.raised} raised.`)
        }
        load()
      })
      .catch(err => toast(describeError(err), 'error'))
      .finally(() => setBusy(false))
  }, [load, toast])

  const acknowledge = useCallback((id, disposition) => {
    setBusy(true)
    endpoints.ackAlert(id, disposition)
      .then(() => { toast(`Acknowledged — ${disposition}.`); load(); setSelectedId(id) })
      .catch(err => toast(describeError(err), 'error'))
      .finally(() => setBusy(false))
  }, [load, toast])

  const counts = useMemo(() => {
    const c = { CRITICAL: 0, HIGH: 0, WATCH: 0 }
    alerts.forEach(a => { c[a.severity] = (c[a.severity] || 0) + 1 })
    return c
  }, [alerts])

  const selected = detail || alerts.find(a => a.id === selectedId) || null

  return (
    <div className="p-3 space-y-3 bg-ink-bg">

      <div className="flex items-baseline justify-between flex-wrap gap-x-3 gap-y-1">
        <h1 className="text-[15px] font-semibold text-zinc-100 flex items-center gap-2">
          <BellRing size={15} className="text-orange-400" />
          Alert Inbox
        </h1>
        <p className="text-[11.5px] text-zinc-500">
          Raised automatically by the rule pass · delivery to LEAs, banks and I4C is
          <span className="text-amber-300"> simulated</span> on every channel
        </p>
      </div>

      {/* Every channel is a mocked transport. Saying so here, not only in a
          docstring, because an officer who believes a force was warned when it
          was not is worse off than one who knows it was not. */}
      <div className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2
                      text-[11.5px] text-amber-200 flex items-start gap-2">
        <ShieldAlert size={14} className="mt-px shrink-0" />
        <span>
          <span className="font-semibold">Transport is simulated.</span> SMS, email and the
          CFCFRMS / Samanvaya webhooks record a full delivery attempt — recipient, state,
          retries, provider reference — but send nothing. The queue, retry and
          acknowledgement paths are real; only the wire is not.
        </span>
      </div>

      <Panel
        title="Alert queue"
        right={
          <button onClick={runPass} disabled={busy}
                  className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors
                             disabled:opacity-40">
            {busy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
            run rule pass
          </button>
        }
        bodyClass="p-2.5 space-y-2.5"
      >
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          <Stat label="Critical" value={String(counts.CRITICAL)} tone={counts.CRITICAL ? 'bad' : 'default'} />
          <Stat label="High" value={String(counts.HIGH)} tone={counts.HIGH ? 'warn' : 'default'} />
          <Stat label="Watch" value={String(counts.WATCH)} />
          <Stat label="Delivered" value={String(summary?.deliveries?.sent ?? 0)} tone="good"
                sub={`${summary?.deliveries?.failed ?? 0} failed · ${summary?.deliveries?.dead ?? 0} dead`} />
          <Stat label="Actioned" value={String(summary?.actioned ?? 0)}
                sub="alerts an officer closed" />
          {/* Prominent on purpose: this is the number that tells I4C whether the
              system is worth the officers it costs. */}
          <Stat label="False-positive rate"
                value={pctFmt(summary?.false_positive_rate ?? 0)}
                tone={(summary?.false_positive_rate ?? 0) > 0.3 ? 'bad' : 'default'}
                sub="of alerts closed" />
        </div>

        <div className="flex flex-wrap gap-2">
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}
                  aria-label="Filter by status"
                  className="bg-ink-bg border border-ink-border rounded px-2.5 py-1.5
                             text-[11.5px] text-zinc-300 outline-none">
            <option value="open">Open</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="ALL">All statuses</option>
          </select>
          <select value={severityFilter} onChange={e => setSeverityFilter(e.target.value)}
                  aria-label="Filter by severity"
                  className="bg-ink-bg border border-ink-border rounded px-2.5 py-1.5
                             text-[11.5px] text-zinc-300 outline-none">
            <option value="ALL">All severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="WATCH">Watch</option>
          </select>
        </div>
      </Panel>

      <div className="grid grid-cols-12 gap-3">
        {/* ── Queue ─────────────────────────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-7">
          <Panel title={`${alerts.length} alert(s)`} bodyClass="p-2.5">
            {error ? (
              <div className="py-10 text-center text-[12.5px]">
                <ServerCrash size={22} className="text-red-400 mx-auto mb-2" />
                <div className="text-red-300 font-semibold">Alerts unavailable</div>
                <div className="text-zinc-500 mt-1">{error}</div>
              </div>
            ) : loading ? (
              <div className="py-10 text-center text-zinc-500 text-[12.5px]">
                <Loader2 size={16} className="animate-spin inline" /> loading…
              </div>
            ) : !alerts.length ? (
              <div className="py-10 text-center text-[12.5px] text-zinc-500">
                Nothing raised for these filters. Use <span className="mono text-zinc-300">run
                rule pass</span> to evaluate the current surface.
              </div>
            ) : (
              <div className="space-y-1.5">
                {alerts.map(a => {
                  const S = sev(a.severity)
                  const isSel = a.id === selectedId
                  return (
                    <button key={a.id} onClick={() => setSelectedId(a.id)}
                            className={`w-full text-left border-l-2 ${S.bar} border border-ink-border
                                        ${isSel ? 'bg-ink-panel border-ink-border2' : 'bg-ink-surface'}
                                        rounded px-2.5 py-2 hover:border-ink-border2 transition-colors`}>
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-[10px] font-semibold ${S.text} tracking-wide`}>
                          {a.severity}
                        </span>
                        <span className="mono text-[10.5px] text-zinc-500">
                          {a.rule_id} · {timeAgoShort(a.created_at)}
                        </span>
                      </div>
                      <div className="text-[12.5px] text-zinc-200 mt-0.5">{a.headline}</div>
                      <div className="text-[11px] text-zinc-500 mt-0.5 flex flex-wrap gap-x-3">
                        <span className="mono">{a.cell_id}</span>
                        <span>{a.district}{a.state ? `, ${a.state}` : ''}</span>
                        <span className="mono tnum">{a.case_count} case(s)</span>
                        <span className="mono tnum">{Math.round((1 - a.prior_share) * 100)}% live</span>
                        {a.status === 'acknowledged' && (
                          <span className="text-aegis-accent flex items-center gap-1">
                            <CheckCircle2 size={11} /> {a.disposition}
                          </span>
                        )}
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </Panel>
        </div>

        {/* ── Detail ────────────────────────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-5">
          <Panel title={selected ? selected.id : 'No alert selected'} bodyClass="p-2.5 space-y-3">
            {!selected ? (
              <div className="py-10 text-center text-[12.5px] text-zinc-500">
                Select an alert to see who it reached and what was decided.
              </div>
            ) : (
              <>
                <div>
                  <div className={`text-[10px] font-semibold ${sev(selected.severity).text}`}>
                    {selected.severity} · {selected.rule_id}
                  </div>
                  <div className="text-[13px] text-zinc-100 mt-1">{selected.headline}</div>
                  <div className="text-[11.5px] text-zinc-500 mt-1">
                    <span className="mono">{selected.cell_id}</span> · {selected.district}
                    {selected.state ? `, ${selected.state}` : ''} ·
                    <span className="mono tnum"> {selected.window_start_min}–{selected.window_end_min} min</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <Stat label="At risk" value={amountShort(selected.rupees_at_risk)} tone="bad" />
                  <Stat label="Open cases" value={String(selected.case_count)} />
                </div>

                <LiveShareBar priorShare={selected.prior_share} />

                {/* Provenance. A number with no traceable cases behind it is a
                    number an officer cannot act on or challenge. */}
                {!!(selected.complaint_ids || []).length && (
                  <div>
                    <div className="text-[11px] text-zinc-500 mb-1">Cases behind this alert</div>
                    <div className="flex flex-wrap gap-1">
                      {selected.complaint_ids.slice(0, 12).map(cid => (
                        <Link key={cid} to={`/?c=${encodeURIComponent(cid)}`}
                              className="mono text-[10.5px] px-1.5 py-0.5 rounded border
                                         border-ink-border bg-ink-panel text-zinc-300
                                         hover:border-ink-border2 hover:text-zinc-100">
                          {formatTicket(cid)}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {/* The delivery record. Exposed rather than summarised: a
                    delivery that silently never arrived is the worst thing this
                    system can produce, so it belongs on the screen. */}
                <div>
                  <div className="text-[11px] text-zinc-500 mb-1 flex items-center gap-1.5">
                    <Send size={11} /> Delivery attempts
                  </div>
                  {!(detail?.deliveries || []).length ? (
                    <div className="text-[11.5px] text-zinc-500">No delivery recorded.</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="data-table mono tnum text-[11px]">
                        <thead>
                          <tr><th>Channel</th><th>Recipient</th><th>State</th>
                              <th>Tries</th><th>Reference</th></tr>
                        </thead>
                        <tbody>
                          {detail.deliveries.map(d => (
                            <tr key={d.id}>
                              <td>{d.channel}</td>
                              <td className="max-w-[150px] truncate">{d.recipient}</td>
                              <td className={DELIVERY_TONE[d.state] || 'text-zinc-400'}>
                                {d.state}
                              </td>
                              <td>{d.attempts}</td>
                              <td className="text-zinc-500">{d.provider_ref || d.last_error || '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Acknowledgement. A disposition is required -- there is no
                    "just close it" -- because the disposition IS the outcome
                    record the framework learns from. */}
                {selected.status === 'acknowledged' ? (
                  <div className="rounded border border-aegis-accent/40 bg-ink-panel px-2.5 py-2
                                  text-[11.5px] text-zinc-300 flex items-center gap-2">
                    <CheckCircle2 size={13} className="text-aegis-accent" />
                    <span>
                      <span className="text-zinc-100">{selected.disposition}</span> — closed by{' '}
                      {selected.acknowledged_by} {timeAgoShort(selected.acknowledged_at)}
                    </span>
                  </div>
                ) : (
                  <div>
                    <div className="text-[11px] text-zinc-500 mb-1.5">
                      Acknowledge — a disposition is required
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {DISPOSITIONS.map(d => (
                        <button key={d} disabled={busy}
                                onClick={() => acknowledge(selected.id, d)}
                                className={`px-2 py-1 rounded border text-[11.5px] transition-colors
                                            disabled:opacity-40 ${
                                  d === 'False positive'
                                    ? 'border-amber-500/40 text-amber-200 hover:bg-amber-500/10'
                                    : 'border-ink-border bg-ink-panel text-zinc-300 hover:text-zinc-100'}`}>
                          {d}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}
