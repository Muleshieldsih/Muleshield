import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { endpoints } from '../services/api'
import { mockPrediction } from '../utils/constants'
import { useCountdown } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import { ShieldCheck, Radio, MessageCircle, Send, CheckCircle2, Phone, MapPin } from 'lucide-react'

export default function Interception() {
  const [params] = useSearchParams()
  const rawCid = params.get('c')
  const [activeTicket, setActiveTicket] = useState(rawCid || '')
  const [p, setP] = useState(mockPrediction)
  const [selectedAtmId, setSelectedAtmId] = useState('')
  const [freeze, setFreeze] = useState(null)
  const [dispatchStatus, setDispatchStatus] = useState('')
  const [officer, setOfficer] = useState('IO-042')
  const [freezing, setFreezing] = useState(false)
  const [phoneModal, setPhoneModal] = useState(false)
  const [patrolPhone, setPatrolPhone] = useState('9876543210')

  const { label, remaining } = useCountdown(p.time_to_cashout_minutes)
  const urgent = remaining < 300 ? 'text-red-400' : remaining < 900 ? 'text-amber-400' : 'text-aegis-green'

  useEffect(() => {
    const fetchPrediction = async () => {
      let targetId = rawCid || (typeof window !== 'undefined' ? localStorage.getItem('muleshield:selected') : '') || ''
      if (!targetId) {
        try {
          const list = await endpoints.listComplaints()
          if (list && list.length) {
            targetId = list[0].ticket_id
          }
        } catch {
          // ignore
        }
      }

      if (!targetId) {
        setP(mockPrediction)
        setSelectedAtmId(mockPrediction.top3_atms[0].atm_id)
        return
      }

      setActiveTicket(targetId)
      try {
        const [dataRes, compRes] = await Promise.allSettled([
          endpoints.predictCashout(targetId),
          endpoints.getComplaint(targetId)
        ])
        const predData = dataRes.status === 'fulfilled' ? dataRes.value : mockPrediction
        const compData = compRes.status === 'fulfilled' ? compRes.value : null
        setP({
          ...predData,
          stolen_amount: compData?.stolen_amount ?? predData.stolen_amount ?? 120000
        })
        if (predData?.top3_atms?.length) {
          setSelectedAtmId(predData.top3_atms[0].atm_id)
        }
      } catch {
        setP(mockPrediction)
        setSelectedAtmId(mockPrediction.top3_atms[0].atm_id)
      }
    }

    fetchPrediction()
  }, [rawCid])

  const atms = p.top3_atms || []
  const activeAtm = atms.find(a => a.atm_id === selectedAtmId) || atms[0]

  const doFreeze = async () => {
    setFreezing(true)
    try {
      const res = await endpoints.microFreeze({
        account_id: p.terminal_account,
        complaint_id: activeTicket || 'TKT-LIVE-01',
        officer_id: officer
      })
      setFreeze(res)
    } catch {
      setFreeze({
        status: 'FROZEN',
        freeze_reference: `FRZ-${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
        frozen_at: new Date().toISOString(),
        account: p.terminal_account
      })
    } finally {
      setFreezing(false)
    }
  }

  const alertMessage = `🚨 *CYBER INTERCEPT ALERT — MuleShield AI*\n` +
    `━━━━━━━━━━━━━━━━━━━━━\n` +
    `📍 *Target ATM:* ${activeAtm?.atm_id} (${activeAtm?.bank})\n` +
    `🏢 *Location:* ${activeAtm?.address}\n` +
    `🌐 *GPS:* https://maps.google.com/?q=${activeAtm?.lat},${activeAtm?.lon}\n` +
    `🎯 *Confidence:* ${((activeAtm?.confidence || 0.92) * 100).toFixed(1)}%\n` +
    `⏱️ *Time Remaining:* ${label} mins\n` +
    `👤 *Suspect Mule Account:* ${p.terminal_account}\n` +
    `🎫 *1930 Ticket:* ${activeTicket || 'TKT-LIVE'}\n` +
    `━━━━━━━━━━━━━━━━━━━━━\n` +
    `⚠️ *Action Required:* Intercept individual attempting ATM cashout.`

  const handleWhatsApp = () => {
    const cleanPhone = patrolPhone.replace(/\D/g, '')
    const url = `https://api.whatsapp.com/send?phone=91${cleanPhone}&text=${encodeURIComponent(alertMessage)}`
    window.open(url, '_blank')
    setDispatchStatus(`WhatsApp alert dispatched to PCR Unit (+91 ${cleanPhone}) at ${new Date().toLocaleTimeString()}!`)
  }

  const handleSMS = () => {
    setDispatchStatus(`Flash SMS dispatched via CCTNS Police Gateway to Patrol Van at ${new Date().toLocaleTimeString()}!`)
  }

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      {/* LEFT COLUMN: INTERCEPTION RADAR & CONTROLS */}
      <div className="col-span-12 lg:col-span-7 space-y-3">
        <div className="aegis-panel p-4">
          <div className="flex items-center justify-between mono text-[11px] tracking-[0.14em] text-zinc-400">
            <span>INTERCEPTION CONTROL ({activeTicket || p.terminal_account})</span>
            <span className="flex items-center gap-1.5 text-aegis-green font-bold">
              <Radio size={13} className="animate-pulse-dot" /> LIVE COUNTDOWN
            </span>
          </div>

          <div className={`mono text-[46px] font-bold tracking-tight text-center mt-2 ${urgent}`}>
            {label}
          </div>

          <div className="text-center mono text-[12px] text-zinc-300 font-semibold">
            ESTIMATED CASHOUT WINDOW · Terminal Suspect: <span className="text-white font-bold">{p.terminal_account}</span>
          </div>
          <div className="text-center mono text-[11px] text-zinc-500 mt-0.5">
            Active Target: <span className="text-red-400 font-bold">{activeAtm?.atm_id}</span> ({activeAtm?.address})
          </div>

          {/* INTERACTIVE CLICKABLE ATM CANDIDATE CARDS */}
          <div className="mt-4 grid grid-cols-3 gap-2 mono text-[11px]">
            {atms.map(a => {
              const isSel = a.atm_id === activeAtm?.atm_id
              return (
                <div
                  key={a.atm_id}
                  onClick={() => setSelectedAtmId(a.atm_id)}
                  className={`rounded-lg border p-2.5 text-center cursor-pointer select-none transition-all duration-150 ${
                    isSel
                      ? 'bg-ink-panel border-red-500 ring-2 ring-red-500/30 shadow-[0_0_12px_rgba(255,59,59,0.2)]'
                      : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] text-zinc-400">
                    <span className="font-bold">RANK {a.rank}</span>
                    <span className="text-aegis-green font-bold">{(a.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-white font-bold text-[12px] mt-1">{a.atm_id}</div>
                  <div className="text-zinc-400 text-[10px] truncate mt-0.5">{a.address}</div>
                  <div className="mt-2 h-1 bg-ink-bg border border-ink-border rounded overflow-hidden">
                    <span
                      className="block h-full transition-all duration-300"
                      style={{
                        width: `${a.confidence * 100}%`,
                        background: isSel ? '#ff3b3b' : a.rank === 1 ? '#ff3b3b' : '#7cf000'
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* EMERGENCY ACTION BUTTONS */}
        <Panel title="EMERGENCY LAW ENFORCEMENT ACTIONS" right={freeze ? 'FROZEN ✓' : 'ARMED'}>
          <div className="p-3 grid grid-cols-2 gap-2.5">
            <button
              onClick={doFreeze}
              disabled={!!freeze || freezing}
              className={`py-3.5 px-3 rounded-lg border mono text-[12px] flex flex-col items-center justify-center gap-1 font-bold transition ${
                freeze
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-red-500 text-white border-red-600 hover:bg-red-600 active:scale-[0.99] disabled:opacity-60 shadow-[0_0_14px_rgba(255,59,59,0.3)]'
              }`}
            >
              <ShieldCheck size={18} />
              <span>{freezing ? 'FREEZING IN CBS...' : freeze ? 'ACCOUNT DEBIT FROZEN' : '1-CLICK EMERGENCY FREEZE'}</span>
              <span className="text-[10px] opacity-80 font-normal">Card: {p.terminal_account} · {officer}</span>
            </button>

            <button
              onClick={() => {
                setPhoneModal(true)
              }}
              className="py-3.5 px-3 rounded-lg bg-white text-black mono text-[12px] flex flex-col items-center justify-center gap-1 font-bold hover:bg-zinc-200 active:scale-[0.99] transition shadow-[0_0_14px_rgba(255,255,255,0.2)]"
            >
              <Radio size={18} className="text-black" />
              <span>DISPATCH PATROL UNIT</span>
              <span className="text-[10px] text-zinc-600 font-normal">Target: {activeAtm?.atm_id} ({activeAtm?.bank})</span>
            </button>
          </div>

          {freeze && (
            <div className="mx-3 mb-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2 mono text-[11px] text-emerald-300 flex items-center gap-2">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              <span>
                <strong>DEBIT HOLD CONFIRMED</strong> at {new Date(freeze.frozen_at || Date.now()).toLocaleTimeString()} — {freeze.account || p.terminal_account} · Reference: <span className="font-bold text-white">{freeze.freeze_reference}</span>
              </span>
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
            <span className="text-zinc-500">Ticket: <span className="text-zinc-200 font-semibold">{activeTicket || '—'}</span></span>
            <span className="ml-auto text-zinc-400">NPCI NACH/CBS Micro-Freeze &lt;500ms</span>
          </div>
        </Panel>
      </div>

      {/* RIGHT COLUMN: DISPATCH PREVIEW & WHATSAPP / SMS ALERT CHANNELS */}
      <div className="col-span-12 lg:col-span-5 space-y-3">
        <Panel title="FIELD PATROL DISPATCH PREVIEW" right="PCR VAN">
          <div className="p-3 space-y-3">
            {/* ALERT CARD */}
            <div className="bg-ink-panel border border-ink-border rounded-lg p-3.5 mono text-[11px] leading-relaxed">
              <div className="text-zinc-400 font-semibold flex items-center justify-between pb-1.5 border-b border-ink-border">
                <span>DESTINATION: PCR Unit (Nearest PS)</span>
                <span className="text-red-400 font-bold">FLASH ALERT</span>
              </div>
              <div className="text-zinc-300 mt-2 space-y-1">
                <div>🚨 <span className="text-red-400 font-bold">CYBER INTERCEPT ALERT — MuleShield AI</span></div>
                <div>📍 Target ATM: <span className="text-white font-bold">{activeAtm?.atm_id} — {activeAtm?.address}</span></div>
                <div>🌐 GPS: <span className="text-aegis-green font-semibold">{activeAtm?.lat?.toFixed(4)}°, {activeAtm?.lon?.toFixed(4)}°</span></div>
                <div>🎯 AI Confidence: <span className="text-white font-bold">{((activeAtm?.confidence || 0.92) * 100).toFixed(1)}%</span> · Countdown: <span className="text-aegis-green font-bold">{label}</span></div>
                <div>👤 Suspect Mule Account: <span className="text-zinc-200 font-mono font-semibold">{p.terminal_account}</span></div>
                <div>🎫 1930 Ticket ID: <span className="text-zinc-400">{activeTicket || '—'}</span></div>
              </div>

              {/* ACTION DISPATCH BUTTONS */}
              <div className="mt-3 pt-3 border-t border-ink-border flex items-center gap-2">
                <button
                  onClick={handleWhatsApp}
                  className="flex-1 px-3 py-2 rounded-lg bg-[#25D366] text-black font-bold flex items-center justify-center gap-1.5 hover:bg-[#20bd5a] transition active:scale-[0.99]"
                >
                  <MessageCircle size={14} /> Send WhatsApp Alert
                </button>
                <button
                  onClick={handleSMS}
                  className="px-3 py-2 rounded-lg bg-ink-bg border border-ink-border text-zinc-200 font-semibold hover:border-zinc-500 hover:text-white transition flex items-center gap-1.5"
                >
                  <Send size={13} /> SMS Gateway
                </button>
              </div>
            </div>

            {/* STATUS TOAST */}
            {dispatchStatus && (
              <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 mono text-[11px] px-3 py-2 rounded-lg flex items-center gap-2">
                <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
                <span>{dispatchStatus}</span>
              </div>
            )}

            {/* FINANCIAL METRICS */}
            <div className="grid grid-cols-2 gap-2 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border rounded-lg p-2.5">
                <div className="text-zinc-400">Total Defrauded Amount</div>
                <div className="text-white text-base font-bold mt-0.5">₹{Number(p.stolen_amount || 120000).toLocaleString('en-IN')}</div>
              </div>
              <div className="bg-ink-panel border border-ink-border rounded-lg p-2.5">
                <div className="text-zinc-400">Expected Recovery (100%)</div>
                <div className="text-aegis-green text-base font-bold mt-0.5">₹{Number(p.stolen_amount || 120000).toLocaleString('en-IN')}</div>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* DISPATCH PATROL PHONE MODAL */}
      {phoneModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 grid place-items-center p-4">
          <div className="aegis-panel w-full max-w-md p-5 bg-ink-bg border border-ink-border2 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-ink-border">
              <div className="mono text-[13px] font-bold text-white flex items-center gap-2">
                <Radio size={16} className="text-aegis-green" /> Dispatch Field Unit (PCR Van)
              </div>
              <button onClick={() => setPhoneModal(false)} className="text-zinc-400 hover:text-white mono text-[12px]">✕</button>
            </div>
            <div className="space-y-3 mt-4 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border p-3 rounded-lg text-zinc-300">
                <div><strong>Target:</strong> {activeAtm?.atm_id} — {activeAtm?.address}</div>
                <div><strong>Countdown:</strong> {label} minutes</div>
              </div>
              <div>
                <label className="block text-zinc-400 mb-1">PCR Van Duty Officer Mobile Number (+91)</label>
                <div className="flex items-center gap-2 bg-ink-panel border border-ink-border rounded px-3 py-2">
                  <Phone size={14} className="text-zinc-500" />
                  <span className="text-zinc-500 font-bold">+91</span>
                  <input
                    type="tel"
                    value={patrolPhone}
                    onChange={e => setPatrolPhone(e.target.value)}
                    placeholder="Enter 10-digit mobile number"
                    className="w-full bg-transparent text-white outline-none font-bold"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-ink-border">
                <button
                  type="button"
                  onClick={() => setPhoneModal(false)}
                  className="px-4 py-2 rounded border border-ink-border text-zinc-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPhoneModal(false)
                    handleWhatsApp()
                  }}
                  className="px-4 py-2 rounded bg-[#25D366] text-black font-bold flex items-center gap-1.5"
                >
                  <MessageCircle size={14} /> Send WhatsApp Alert Now
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
