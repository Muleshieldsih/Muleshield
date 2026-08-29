import { useEffect, useState, useMemo, useCallback } from 'react'
import { endpoints, describeError } from '../services/api'
import useActiveComplaint from '../hooks/useActiveComplaint'
import { useCountdown } from '../hooks/useCountdown'
import { Panel, Stat } from '../components/Shell'
import { amountFmt, formatTicket } from '../utils/constants'
import {
  ShieldCheck, Radio, MessageCircle, Send, CheckCircle2, Phone,
  Loader2, ServerCrash, AlertTriangle,
} from 'lucide-react'

export default function Interception() {
  const { complaintId, resolving } = useActiveComplaint()
  const [prediction, setPrediction] = useState(null)
  const [complaint, setComplaint] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [selectedAtmId, setSelectedAtmId] = useState('')
  const [freeze, setFreeze] = useState(null)
  const [freezing, setFreezing] = useState(false)
  const [freezeError, setFreezeError] = useState('')
  const [officer, setOfficer] = useState('IO-042')
  const [dispatchStatus, setDispatchStatus] = useState('')
  const [phoneModal, setPhoneModal] = useState(false)
  const [patrolPhone, setPatrolPhone] = useState('9876543210')

  // ── Load prediction + complaint ──────────────────────────────────────────
  useEffect(() => {
    if (resolving) return
    if (!complaintId) { setError('No complaint selected.'); return }

    let cancelled = false
    setLoading(true)
    setError('')
    setFreeze(null)
    setFreezeError('')
    setDispatchStatus('')

    Promise.allSettled([
      endpoints.predictCashout(complaintId),
      endpoints.getComplaint(complaintId),
    ]).then(([predRes, compRes]) => {
      if (cancelled) return

      if (predRes.status === 'fulfilled') {
        setPrediction(predRes.value)
        setSelectedAtmId(predRes.value.ranked_candidates?.[0]?.atm_id || '')
      } else {
        setPrediction(null)
        setError(describeError(predRes.reason))
      }

      setComplaint(compRes.status === 'fulfilled' ? compRes.value : null)
      setLoading(false)
    })

    return () => { cancelled = true }
  }, [complaintId, resolving])

  const atms = useMemo(() => prediction?.ranked_candidates || [], [prediction])
  const activeAtm = atms.find(a => a.atm_id === selectedAtmId) || atms[0] || null
  const { label, remaining, expired } = useCountdown(prediction?.time_to_cashout_minutes)

  const urgentClass = expired
    ? 'text-red-500 animate-blink'
    : remaining < 300 ? 'text-red-400'
    : remaining < 900 ? 'text-amber-400'
    : 'text-aegis-green'

  const stolen = complaint?.stolen_amount ?? prediction?.stolen_amount ?? 0

  // ── Micro-freeze ─────────────────────────────────────────────────────────
  // The request previously omitted `bank`, which the schema required, so every
  // click failed validation and the UI faked a confirmation locally. The bank
  // is now resolved server-side from the account record.
  const doFreeze = useCallback(async () => {
    if (!prediction?.terminal_account || !complaintId) return
    setFreezing(true)
    setFreezeError('')
    try {
      const res = await endpoints.microFreeze({
        account_id: prediction.terminal_account,
        complaint_id: complaintId,
        officer_id: officer || 'OFFICER-001',
      })
      setFreeze(res)
    } catch (err) {
      setFreezeError(describeError(err))
    } finally {
      setFreezing(false)
    }
  }, [prediction, complaintId, officer])

  const alertMessage = useMemo(() => [
    '*CYBER INTERCEPT ALERT - MuleShield AI*',
    '------------------------------',
    `Priority ${activeAtm?.rank ?? '-'} of ${atms.length}: ${activeAtm?.atm_id ?? '-'} (${activeAtm?.bank ?? '-'})`,
    `Location: ${activeAtm?.address ?? '-'}`,
    `GPS: https://maps.google.com/?q=${activeAtm?.lat},${activeAtm?.lon}`,
    `Model rank share: ${((activeAtm?.confidence ?? 0) * 100).toFixed(1)}% (relative, not a certainty)`,
    `Time Remaining: ${label}`,
    `Suspect Mule Account: ${prediction?.terminal_account ?? '-'}`,
    `1930 Ticket: ${formatTicket(complaintId)}`,
    `Amount at risk: ${amountFmt(stolen)}`,
    '------------------------------',
    'Action Required: Search this location for an ATM cashout in progress.',
    'NOTE: A ranked candidate, not a confirmed location.',
  ].join('\n'), [activeAtm, label, prediction, complaintId, stolen])

  const handleWhatsApp = useCallback(() => {
    const phone = patrolPhone.replace(/\D/g, '')
    if (phone.length < 10) {
      setDispatchStatus('Enter a valid 10-digit mobile number before dispatching.')
      return
    }
    window.open(
      `https://api.whatsapp.com/send?phone=91${phone}&text=${encodeURIComponent(alertMessage)}`,
      '_blank', 'noopener,noreferrer'
    )
    setDispatchStatus(`WhatsApp alert dispatched to PCR unit +91 ${phone} at ${new Date().toLocaleTimeString()}.`)
  }, [patrolPhone, alertMessage])

  const handleSMS = useCallback(() => {
    setDispatchStatus(`Flash SMS queued via CCTNS gateway at ${new Date().toLocaleTimeString()} (simulated).`)
  }, [])

  if (error) {
    return (
      <div className="p-3">
        <Panel title={`INTERCEPTION CONTROL — ${formatTicket(complaintId)}`}>
          <div className="p-10 text-center mono text-[12px]">
            <ServerCrash size={26} className="text-red-400 mx-auto mb-2" />
            <div className="text-red-300 font-bold">Interception data unavailable</div>
            <div className="mt-1 text-zinc-500">{error}</div>
          </div>
        </Panel>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      {/* ── Countdown + targets ──────────────────────────────────────────── */}
      <div className="col-span-12 lg:col-span-7 space-y-3">
        <div className="aegis-panel p-4">
          <div className="flex items-center justify-between mono text-[11px] tracking-[0.14em] text-zinc-400 gap-2">
            <span className="truncate">INTERCEPTION CONTROL · {complaintId || '—'}</span>
            {/* Reflects the actual inference state. This was previously a green
                "LIVE" badge rendered unconditionally — it stayed lit while the
                panel below was loading, empty, or showing an error. */}
            <span
              className={`flex items-center gap-1.5 font-bold shrink-0 ${
                prediction ? 'text-aegis-green' : 'text-zinc-500'
              }`}
            >
              <Radio size={13} className={prediction ? 'animate-pulse-dot' : ''} />
              {prediction ? 'ARMED' : loading ? 'COMPUTING' : 'STANDBY'}
            </span>
          </div>

          {loading ? (
            <div className="py-10 grid place-items-center mono text-[12px] text-zinc-400">
              <span className="flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Running inference…</span>
            </div>
          ) : (
            <>
              <div className={`mono text-[46px] font-bold tracking-tight text-center mt-2 ${urgentClass}`}>
                {label}
              </div>
              <div className="text-center mono text-[11px] text-zinc-500">
                {expired ? 'PREDICTED CASHOUT WINDOW ELAPSED' : 'ESTIMATED TIME TO CASHOUT'}
              </div>

              <div className="text-center mono text-[12px] text-zinc-300 font-semibold mt-2">
                Terminal suspect: <span className="text-white font-bold">{prediction?.terminal_account || '—'}</span>
              </div>
              {activeAtm && (
                <div className="text-center mono text-[11px] text-zinc-500 mt-0.5 px-4 truncate">
                  Priority {activeAtm.rank}: <span className="text-red-400 font-bold">{activeAtm.atm_id}</span> · {activeAtm.address}
                </div>
              )}

              <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2 mono text-[11px]">
                {atms.map(a => {
                  const isSel = a.atm_id === activeAtm?.atm_id
                  return (
                    <button
                      key={a.atm_id}
                      onClick={() => setSelectedAtmId(a.atm_id)}
                      className={`rounded-lg border p-2.5 text-left transition-all duration-150 ${
                        isSel
                          ? 'bg-ink-panel border-red-500 ring-1 ring-red-500/30'
                          : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[10px] text-zinc-400">
                        <span className="font-bold">RANK {a.rank}</span>
                        <span className="text-aegis-green font-bold">{(a.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="text-white font-bold text-[12px] mt-1 truncate">{a.atm_id}</div>
                      <div className="text-zinc-500 text-[10px] truncate">{a.bank}</div>
                      <div className="mt-2 h-1 bg-ink-bg border border-ink-border rounded overflow-hidden">
                        <span
                          className="block h-full transition-all duration-300"
                          style={{
                            width: `${Math.max(2, a.confidence * 100)}%`,
                            background: isSel || a.rank === 1 ? '#ff3b3b' : '#7cf000',
                          }}
                        />
                      </div>
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>

        {/* ── Emergency actions ─────────────────────────────────────────── */}
        <Panel title="EMERGENCY LAW ENFORCEMENT ACTIONS" right={freeze ? 'FROZEN ✓' : 'ARMED'}>
          <div className="p-3 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <button
              onClick={doFreeze}
              disabled={!!freeze || freezing || !prediction}
              className={`py-3.5 px-3 rounded-lg border mono text-[12px] flex flex-col items-center justify-center gap-1 font-bold transition ${
                freeze
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-red-500 text-white border-red-600 hover:bg-red-600 active:scale-[0.99] disabled:opacity-50 shadow-[0_0_14px_rgba(255,59,59,0.3)]'
              }`}
            >
              {freezing ? <Loader2 size={18} className="animate-spin" /> : <ShieldCheck size={18} />}
              <span>{freezing ? 'FREEZING IN CBS…' : freeze ? 'ACCOUNT DEBIT FROZEN' : '1-CLICK EMERGENCY FREEZE'}</span>
              <span className="text-[10px] opacity-80 font-normal truncate max-w-full px-2">
                {prediction?.terminal_account || '—'} · {officer}
              </span>
            </button>

            <button
              onClick={() => setPhoneModal(true)}
              disabled={!activeAtm}
              className="py-3.5 px-3 rounded-lg bg-white text-black mono text-[12px] flex flex-col items-center justify-center gap-1 font-bold hover:bg-zinc-200 active:scale-[0.99] disabled:opacity-50 transition"
            >
              <Radio size={18} />
              <span>DISPATCH PATROL UNIT</span>
              <span className="text-[10px] text-zinc-600 font-normal truncate max-w-full px-2">
                {activeAtm ? `${activeAtm.atm_id} · ${activeAtm.bank}` : '—'}
              </span>
            </button>
          </div>

          {freeze && (
            <div className="mx-3 mb-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2 mono text-[11px] text-emerald-300 flex items-start gap-2">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0 mt-0.5" />
              <span>
                <strong>DEBIT HOLD CONFIRMED</strong> at{' '}
                {new Date(freeze.timestamp).toLocaleTimeString()} — {freeze.account} ({freeze.bank})
                <br />
                Reference: <span className="font-bold text-white">{freeze.freeze_reference}</span> ·
                Officer {freeze.officer_id}
              </span>
            </div>
          )}

          {freezeError && (
            <div className="mx-3 mb-3 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 mono text-[11px] text-red-300 flex items-center gap-2">
              <AlertTriangle size={14} className="shrink-0" /> Freeze failed: {freezeError}
            </div>
          )}

          <div className="px-3 pb-3 flex flex-wrap items-center gap-3 mono text-[11px] border-t border-ink-border pt-3">
            <div className="flex items-center gap-1.5">
              <span className="text-zinc-400">Officer ID:</span>
              <input
                value={officer}
                onChange={e => setOfficer(e.target.value)}
                className="bg-ink-panel border border-ink-border rounded px-2 py-1 w-24 outline-none text-zinc-200 focus:border-aegis-green font-bold"
              />
            </div>
            <span className="ml-auto text-zinc-500">NPCI NACH / CBS micro-freeze · simulated</span>
          </div>
        </Panel>
      </div>

      {/* ── Dispatch preview ─────────────────────────────────────────────── */}
      <div className="col-span-12 lg:col-span-5 space-y-3">
        <Panel title="FIELD PATROL DISPATCH" right="PCR VAN">
          <div className="p-3 space-y-3">
            <div className="bg-ink-panel border border-ink-border rounded-lg p-3.5 mono text-[11px] leading-relaxed">
              <div className="text-zinc-400 font-semibold flex items-center justify-between pb-1.5 border-b border-ink-border">
                <span>DESTINATION: Nearest PS / PCR Unit</span>
                <span className="text-red-400 font-bold">FLASH</span>
              </div>
              <div className="text-zinc-300 mt-2 space-y-1">
                <div className="text-red-400 font-bold">CYBER INTERCEPT ALERT</div>
                <div className="truncate">Priority location: <span className="text-white font-bold">{activeAtm?.atm_id || '—'}</span></div>
                <div className="text-zinc-400 line-clamp-2">{activeAtm?.address || '—'}</div>
                <div>
                  GPS: <span className="text-aegis-green font-semibold">
                    {activeAtm ? `${activeAtm.lat.toFixed(4)}°, ${activeAtm.lon.toFixed(4)}°` : '—'}
                  </span>
                </div>
                <div>
                  Confidence: <span className="text-white font-bold">
                    {activeAtm ? `${(activeAtm.confidence * 100).toFixed(1)}%` : '—'}
                  </span> · Countdown: <span className="text-aegis-green font-bold">{label}</span>
                </div>
                <div className="truncate">Mule account: <span className="text-zinc-200 font-semibold">{prediction?.terminal_account || '—'}</span></div>
                <div>Ticket: <span className="text-zinc-400">{complaintId || '—'}</span></div>
              </div>

              <div className="mt-3 pt-3 border-t border-ink-border flex items-center gap-2">
                <button
                  onClick={() => setPhoneModal(true)}
                  disabled={!activeAtm}
                  className="flex-1 px-3 py-2 rounded-lg bg-[#25D366] text-black font-bold flex items-center justify-center gap-1.5 hover:bg-[#20bd5a] disabled:opacity-50 transition active:scale-[0.99]"
                >
                  <MessageCircle size={14} /> WhatsApp Alert
                </button>
                <button
                  onClick={handleSMS}
                  className="px-3 py-2 rounded-lg bg-ink-bg border border-ink-border text-zinc-200 font-semibold hover:border-zinc-500 hover:text-white transition flex items-center gap-1.5"
                >
                  <Send size={13} /> SMS
                </button>
              </div>
            </div>

            {dispatchStatus && (
              <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 mono text-[11px] px-3 py-2 rounded-lg flex items-start gap-2">
                <CheckCircle2 size={14} className="text-emerald-400 shrink-0 mt-0.5" />
                <span>{dispatchStatus}</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <Stat label="Amount at risk" value={amountFmt(stolen)} tone="bad" />
              <Stat
                label={freeze ? 'Funds protected' : 'Recoverable if frozen'}
                value={amountFmt(stolen)}
                tone={freeze ? 'good' : 'warn'}
                sub={freeze ? 'Debit hold active' : 'Pending freeze'}
              />
            </div>
          </div>
        </Panel>
      </div>

      {/* ── Dispatch modal ───────────────────────────────────────────────── */}
      {phoneModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 grid place-items-center p-4" onClick={() => setPhoneModal(false)}>
          <div className="aegis-panel w-full max-w-md p-5 bg-ink-bg border-ink-border2 shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between pb-3 border-b border-ink-border">
              <div className="mono text-[13px] font-bold text-white flex items-center gap-2">
                <Radio size={16} className="text-aegis-green" /> Dispatch Field Unit
              </div>
              <button onClick={() => setPhoneModal(false)} className="text-zinc-400 hover:text-white mono text-[12px] px-1">✕</button>
            </div>
            <div className="space-y-3 mt-4 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border p-3 rounded-lg text-zinc-300 space-y-1">
                <div className="truncate"><strong>Priority {activeAtm?.rank}:</strong> {activeAtm?.atm_id}</div>
                <div className="text-zinc-400 line-clamp-2">{activeAtm?.address}</div>
                <div><strong>Countdown:</strong> {label}</div>
              </div>
              <div>
                <label className="block text-zinc-400 mb-1">PCR duty officer mobile (+91)</label>
                <div className="flex items-center gap-2 bg-ink-panel border border-ink-border rounded px-3 py-2">
                  <Phone size={14} className="text-zinc-500" />
                  <span className="text-zinc-500 font-bold">+91</span>
                  <input
                    type="tel"
                    inputMode="numeric"
                    maxLength={10}
                    value={patrolPhone}
                    onChange={e => setPatrolPhone(e.target.value)}
                    placeholder="10-digit mobile"
                    className="w-full bg-transparent text-white outline-none font-bold"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-ink-border">
                <button onClick={() => setPhoneModal(false)} className="px-4 py-2 rounded border border-ink-border text-zinc-400 hover:text-white">
                  Cancel
                </button>
                <button
                  onClick={() => { setPhoneModal(false); handleWhatsApp() }}
                  className="px-4 py-2 rounded bg-[#25D366] text-black font-bold flex items-center gap-1.5"
                >
                  <MessageCircle size={14} /> Send Alert
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
