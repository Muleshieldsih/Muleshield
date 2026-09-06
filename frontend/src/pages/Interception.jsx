import { useEffect, useState, useMemo, useCallback } from 'react'
import { endpoints, describeError } from '../services/api'
import useActiveComplaint from '../hooks/useActiveComplaint'
import { useCountdown } from '../hooks/useCountdown'
import { Panel, Stat } from '../components/Shell'
import EvidencePanel from '../components/EvidencePanel'
import DossierModal from '../components/DossierModal'
import { amountFmt, formatTicket, shortAccount } from '../utils/constants'
import {
  ShieldCheck, Radio, MessageCircle, Send, CheckCircle2, Phone,
  Loader2, ServerCrash, AlertTriangle, FileText,
} from 'lucide-react'

export default function Interception() {
  const { complaintId, resolving } = useActiveComplaint()
  const [prediction, setPrediction] = useState(null)
  const [complaint, setComplaint] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [selectedAtmId, setSelectedAtmId] = useState('')
  const [dossierOpen, setDossierOpen] = useState(false)
  const [freeze, setFreeze] = useState(null)
  const [freezing, setFreezing] = useState(false)
  const [freezeError, setFreezeError] = useState('')
  const [officer, setOfficer] = useState('IO-042')
  const [dispatchStatus, setDispatchStatus] = useState('')
  // Whether the message above is good news. It was previously always drawn
  // in the light success banner with a tick -- including the validation
  // failure for a short phone number, which told the operator the dispatch
  // had succeeded at the exact moment it had not.
  const [dispatchOk, setDispatchOk] = useState(false)
  // Freezing an account stops a real person's money moving. A single click
  // with no confirmation is the wrong affordance for that, however urgent
  // the case: the one irreversible control on the screen was the only one
  // that asked nothing before acting.
  const [confirmFreeze, setConfirmFreeze] = useState(false)
  // Set after a successful freeze, when the case has been moved on with it.
  const [caseStatus, setCaseStatus] = useState('')
  const [statusWarning, setStatusWarning] = useState('')
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
    setCaseStatus('')
    setStatusWarning('')

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
    ? 'text-red-500'
    : remaining < 300 ? 'text-red-400'
    : remaining < 900 ? 'text-amber-400'
    : 'text-aegis-accent'

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

      // A freeze is an intervention on a live case, so the case has to say so.
      // The account was held and the queue still showed the case as untouched;
      // a colleague opening it would have had no way to know.
      //
      // This uses the existing PATCH endpoint -- no backend behaviour changes.
      // "Intervention Required" rather than "Resolved": the money is held, the
      // case is not finished.
      try {
        const updated = await endpoints.updateCase(complaintId, {
          status: 'Intervention Required',
          actor: officer || 'OFFICER-001',
        })
        setCaseStatus(updated.status)
      } catch {
        // The freeze itself succeeded. Failing to move the case is worth
        // saying, but it must not read as a failed freeze.
        setStatusWarning('Account frozen, but the case status could not be updated.')
      }
    } catch (err) {
      setFreezeError(describeError(err))
    } finally {
      setFreezing(false)
      setConfirmFreeze(false)
    }
  }, [prediction, complaintId, officer])

  const alertMessage = useMemo(() => [
    '*Interception alert — MuleShield AI*',
    '------------------------------',
    `Priority ${activeAtm?.rank ?? '-'} of ${atms.length}: ${activeAtm?.atm_id ?? '-'} (${activeAtm?.bank ?? '-'})`,
    `Location: ${activeAtm?.address ?? '-'}`,
    `GPS: https://maps.google.com/?q=${activeAtm?.lat},${activeAtm?.lon}`,
    `Model rank share: ${((activeAtm?.confidence ?? 0) * 100).toFixed(1)}% (relative, not a certainty)`,
    `Time Remaining: ${label}`,
    `Suspect Mule Account: ${prediction?.terminal_account ?? '-'}`,
    `Case: ${formatTicket(complaintId)}`,
    `Amount at risk: ${amountFmt(stolen)}`,
    '------------------------------',
    'Action Required: Search this location for an ATM cashout in progress.',
    'NOTE: A ranked candidate, not a confirmed location.',
  ].join('\n'), [activeAtm, label, prediction, complaintId, stolen])

  const handleWhatsApp = useCallback(() => {
    const phone = patrolPhone.replace(/\D/g, '')
    if (phone.length < 10) {
      setDispatchOk(false)
      setDispatchStatus('Enter a valid 10-digit mobile number before dispatching.')
      return
    }
    window.open(
      `https://api.whatsapp.com/send?phone=91${phone}&text=${encodeURIComponent(alertMessage)}`,
      '_blank', 'noopener,noreferrer'
    )
    setDispatchOk(true)
    setDispatchStatus(`Alert sent to field unit +91 ${phone} at ${new Date().toLocaleTimeString()}.`)
  }, [patrolPhone, alertMessage])

  const handleSMS = useCallback(() => {
    setDispatchOk(true)
    setDispatchStatus(`SMS queued at ${new Date().toLocaleTimeString()} (simulated).`)
  }, [])

  if (error) {
    return (
      <div className="p-3">
        <Panel title={`Intervention — ${formatTicket(complaintId)}`}>
          <div className="p-10 text-center text-[12.5px]">
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
          <div className="flex items-center justify-between text-[12px] text-zinc-400 gap-2">
            <span className="truncate">Intervention · {formatTicket(complaintId)}</span>
            {/* Reflects the actual inference state. This was previously a lit
                "LIVE" badge rendered unconditionally — it stayed lit while the
                panel below was loading, empty, or showing an error. */}
            <span
              className={`flex items-center gap-1.5 font-bold shrink-0 ${
                prediction ? 'text-aegis-accent' : 'text-zinc-500'
              }`}
            >
              <Radio size={13} />
              {prediction ? 'Ready' : loading ? 'Computing' : 'No case selected'}
            </span>
          </div>

          {loading ? (
            <div className="py-10 grid place-items-center text-[12.5px] text-zinc-400">
              <span className="flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Running inference…</span>
            </div>
          ) : (
            <>
              <div className={`mono tnum text-[44px] font-semibold tracking-tight text-center mt-2 ${urgentClass}`}>
                {label}
              </div>
              <div className="text-center text-[11.5px] text-zinc-500">
                {expired ? 'Predicted window elapsed' : 'Estimated time to cash-out'}
              </div>

              <div className="text-center text-[12.5px] text-zinc-300 mt-2">
                Terminal suspect: <span className="text-white font-bold">{prediction?.terminal_account || '—'}</span>
              </div>
              {activeAtm && (
                <div className="text-center text-[11.5px] text-zinc-500 mt-0.5 px-4 truncate">
                  Priority {activeAtm.rank}: <span className="text-red-400 font-bold">{activeAtm.atm_id}</span> · {activeAtm.address}
                </div>
              )}

              <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11.5px]">
                {atms.map(a => {
                  const isSel = a.atm_id === activeAtm?.atm_id
                  return (
                    <button
                      key={a.atm_id}
                      onClick={() => setSelectedAtmId(a.atm_id)}
                      className={`rounded border p-2.5 text-left transition-all duration-150 ${
                        isSel
                          ? 'bg-ink-panel border-red-500/70'
                          : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[10px] text-zinc-400">
                        <span className="font-bold">RANK {a.rank}</span>
                        <span className="text-aegis-accent font-bold">{(a.confidence * 100).toFixed(1)}%</span>
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
        <Panel title="Intervention" right={freeze ? 'Account frozen' : null}>
          <div className="p-3 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <button
              onClick={() => setConfirmFreeze(true)}
              disabled={!!freeze || freezing || !prediction}
              className={`py-3.5 px-3 rounded border text-[12.5px] flex flex-col items-center justify-center gap-1 font-medium transition-colors ${
                freeze
                  ? 'bg-white/10 border-white/25 text-white'
                  : 'bg-red-500 text-white border-red-600 hover:bg-red-600 disabled:opacity-50 '
              }`}
            >
              {freezing ? <Loader2 size={18} className="animate-spin" /> : <ShieldCheck size={18} />}
              <span>{freezing ? 'Freezing…' : freeze ? 'Account frozen' : 'Freeze account'}</span>
              <span className="text-[10px] opacity-80 font-normal truncate max-w-full px-2">
                {prediction?.terminal_account || '—'} · {officer}
              </span>
            </button>

            <button
              onClick={() => setPhoneModal(true)}
              disabled={!activeAtm}
              className="py-3.5 px-3 rounded border border-ink-border bg-ink-bg text-zinc-200 text-[12.5px] flex flex-col items-center justify-center gap-1 font-medium hover:border-zinc-600 disabled:opacity-50 transition-colors"
            >
              <Radio size={18} />
              <span>Notify field unit</span>
              <span className="text-[10px] text-zinc-600 font-normal truncate max-w-full px-2">
                {activeAtm ? `${activeAtm.atm_id} · ${activeAtm.bank}` : '—'}
              </span>
            </button>
          </div>

          {freeze && (
            <div className="mx-3 mb-3 bg-white/10 border border-white/25 rounded px-3 py-2 text-[11.5px] text-zinc-100 flex items-start gap-2">
              <CheckCircle2 size={15} className="text-white shrink-0 mt-0.5" />
              <span>
                <strong>Debit hold confirmed</strong> at{' '}
                {new Date(freeze.timestamp).toLocaleTimeString()} — {freeze.account} ({freeze.bank})
                <br />
                Reference: <span className="mono font-semibold text-white">{freeze.freeze_reference}</span> ·
                Officer {freeze.officer_id}
                {caseStatus && (
                  <>
                    <br />
                    Case moved to <span className="text-white">{caseStatus}</span>.
                  </>
                )}
                {statusWarning && (
                  <>
                    <br />
                    <span className="text-amber-300">{statusWarning}</span>
                  </>
                )}
              </span>
            </div>
          )}

          {freezeError && (
            <div className="mx-3 mb-3 bg-red-500/10 border border-red-500/30 rounded px-3 py-2 text-[11.5px] text-red-300 flex items-center gap-2">
              <AlertTriangle size={14} className="shrink-0" /> Freeze failed: {freezeError}
            </div>
          )}

          <div className="px-3 pb-3 flex flex-wrap items-center gap-3 mono text-[11px] border-t border-ink-border pt-3">
            <div className="flex items-center gap-1.5">
              <span className="text-zinc-400">Officer</span>
              <input
                value={officer}
                onChange={e => setOfficer(e.target.value)}
                className="bg-ink-panel border border-ink-border rounded px-2 py-1 w-24 outline-none text-zinc-200 focus:border-aegis-accent font-bold"
              />
            </div>
            <span className="ml-auto text-zinc-500">Simulated bank hold — no live NPCI or core-banking call is made</span>
          </div>
        </Panel>

        <Panel title="Field notification" right="Nearest unit">
          <div className="p-3 space-y-3">
            <div className="bg-ink-panel border border-ink-border rounded p-3.5 mono text-[11px] leading-relaxed">
              <div className="text-zinc-400 font-semibold flex items-center justify-between pb-1.5 border-b border-ink-border">
                <span>To: nearest police station</span>
                <span className="text-zinc-500">Priority</span>
              </div>
              <div className="text-zinc-300 mt-2 space-y-1">
                <div className="text-red-400 font-semibold">Interception alert</div>
                <div className="truncate">Priority location: <span className="text-white font-bold">{activeAtm?.atm_id || '—'}</span></div>
                <div className="text-zinc-400 line-clamp-2">{activeAtm?.address || '—'}</div>
                <div>
                  GPS: <span className="text-aegis-accent font-semibold">
                    {activeAtm ? `${activeAtm.lat.toFixed(4)}°, ${activeAtm.lon.toFixed(4)}°` : '—'}
                  </span>
                </div>
                <div>
                  Confidence: <span className="text-white font-bold">
                    {activeAtm ? `${(activeAtm.confidence * 100).toFixed(1)}%` : '—'}
                  </span> · Countdown: <span className="text-aegis-accent font-bold">{label}</span>
                </div>
                <div className="truncate">Mule account: <span className="text-zinc-200 font-semibold">{prediction?.terminal_account || '—'}</span></div>
                <div>Case: <span className="text-zinc-400">{formatTicket(complaintId)}</span></div>
              </div>

              <div className="mt-3 pt-3 border-t border-ink-border flex items-center gap-2">
                <button
                  onClick={() => setPhoneModal(true)}
                  disabled={!activeAtm}
                  className="flex-1 px-3 py-2 rounded bg-ink-panel border border-ink-border text-zinc-200 font-medium flex items-center justify-center gap-1.5 hover:border-zinc-600 disabled:opacity-50 transition"
                >
                  <MessageCircle size={14} /> Send on WhatsApp
                </button>
                <button
                  onClick={handleSMS}
                  className="px-3 py-2 rounded bg-ink-bg border border-ink-border text-zinc-200 font-semibold hover:border-zinc-500 hover:text-white transition flex items-center gap-1.5"
                >
                  <Send size={13} /> SMS
                </button>
              </div>
            </div>

            {dispatchStatus && (
              <div
                className={`text-[11px] px-3 py-2 rounded flex items-start gap-2 border ${
                  dispatchOk
                    ? 'bg-white/10 border-white/25 text-zinc-100'
                    : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                }`}
                role="status"
              >
                {dispatchOk
                  ? <CheckCircle2 size={14} className="text-white shrink-0 mt-0.5" />
                  : <AlertTriangle size={14} className="text-amber-400 shrink-0 mt-0.5" />}
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

      {/* ── Intelligence & Evidence ───────────────────────────────────────── */}
      <div className="col-span-12 lg:col-span-5 flex flex-col space-y-3">
        {/* Evidence sits on the case screen because that is where an officer
            works the case, and above the dispatch preview because what is held
            against a case -- and whether its chain is intact -- is a fact about
            the case rather than an attachment drawer. */}
        {complaintId && (
          <Panel
            title="Intelligence dossier"
            className="shrink-0"
            right={
              <button
                onClick={() => setDossierOpen(true)}
                className="px-2.5 py-1 rounded border border-ink-border text-[11px] text-zinc-300
                           hover:border-zinc-600 hover:text-white transition-colors
                           flex items-center gap-1.5"
              >
                <FileText size={12} /> Produce
              </button>
            }
          >
            <div className="p-3 text-[11.5px] text-zinc-400 leading-relaxed">
              A printable case report — the complaint as filed, every hop of the money
              trail with its IFSC, what the graph engine flagged, the forecast search
              zone and its ranked candidates, the actions taken, and the evidence held.
              Assembled from the store and the shipped model at the moment it is asked
              for, so it cannot describe a state the system is not in.
            </div>
          </Panel>
        )}

        {complaintId && <EvidencePanel caseId={complaintId} className="flex-1 flex flex-col" />}
      </div>

      {/* ── Dispatch modal ───────────────────────────────────────────────── */}
      {/* Confirmation for the one irreversible control on the screen. It names
          the account and states the consequence, so the operator confirms a
          specific act rather than dismissing a generic prompt. */}
      {confirmFreeze && (
        <div
          className="fixed inset-0 bg-black/70 z-50 grid place-items-center p-4"
          onClick={() => !freezing && setConfirmFreeze(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="freeze-title"
        >
          <div
            className="aegis-panel w-full max-w-md p-5 bg-ink-surface"
            onClick={e => e.stopPropagation()}
          >
            <div id="freeze-title" className="text-[14px] font-semibold text-white">
              Freeze account{' '}
              <span className="mono">{shortAccount(prediction?.terminal_account)}</span>?
            </div>
            <p className="text-[12.5px] text-zinc-400 leading-relaxed mt-2">
              This places a debit hold on the account, stopping outgoing transfers
              until a bank officer lifts it. It is recorded against case{' '}
              <span className="mono text-zinc-300">{formatTicket(complaintId)}</span>{' '}
              under officer <span className="mono text-zinc-300">{officer || 'OFFICER-001'}</span>.
            </p>
            <div className="mt-3 rounded border border-ink-border bg-ink-bg px-3 py-2 text-[11.5px] text-zinc-400">
              <div className="flex justify-between gap-2">
                <span>Account</span>
                <span className="mono text-zinc-200">{prediction?.terminal_account || '—'}</span>
              </div>
              <div className="flex justify-between gap-2 mt-1">
                <span>Amount at risk</span>
                <span className="mono tnum text-zinc-200">{amountFmt(stolen)}</span>
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setConfirmFreeze(false)}
                disabled={freezing}
                className="px-3 py-1.5 rounded border border-ink-border text-[12px] text-zinc-300
                           hover:text-white hover:border-zinc-600 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={doFreeze}
                disabled={freezing}
                className="px-3 py-1.5 rounded bg-red-500 border border-red-600 text-[12px] text-white
                           font-medium hover:bg-red-600 disabled:opacity-50 transition-colors
                           flex items-center gap-1.5"
              >
                {freezing && <Loader2 size={13} className="animate-spin" />}
                {freezing ? 'Freezing…' : 'Confirm freeze'}
              </button>
            </div>
          </div>
        </div>
      )}

      {phoneModal && (
        <div className="fixed inset-0 bg-black/70 z-50 grid place-items-center p-4" onClick={() => setPhoneModal(false)}>
          <div className="aegis-panel w-full max-w-md p-5 bg-ink-surface border-ink-border" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between pb-3 border-b border-ink-border">
              <div className="mono text-[13px] font-bold text-white flex items-center gap-2">
                <Radio size={16} className="text-aegis-accent" /> Notify field unit
              </div>
              <button onClick={() => setPhoneModal(false)} className="text-zinc-400 hover:text-white mono text-[12px] px-1">✕</button>
            </div>
            <div className="space-y-3 mt-4 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border p-3 rounded text-zinc-300 space-y-1">
                <div className="truncate"><strong>Priority {activeAtm?.rank}:</strong> {activeAtm?.atm_id}</div>
                <div className="text-zinc-400 line-clamp-2">{activeAtm?.address}</div>
                <div><strong>Countdown:</strong> {label}</div>
              </div>
              <div>
                <label className="block text-zinc-400 mb-1">Field officer mobile (+91)</label>
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
                  className="px-4 py-2 rounded bg-ink-panel border border-ink-border text-zinc-200 font-medium flex items-center gap-1.5"
                >
                  <MessageCircle size={14} /> Send
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    {dossierOpen && complaintId && (
        <DossierModal caseId={complaintId} onClose={() => setDossierOpen(false)} />
      )}
    </div>
  )
}
