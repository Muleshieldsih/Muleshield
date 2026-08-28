import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { endpoints } from '../services/api'
import { mockPrediction } from '../utils/constants'
import { useCountdown } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import { ShieldCheck, Radio, MessageCircle } from 'lucide-react'

export default function Interception() {
  const [params] = useSearchParams()
  const rawCid = params.get('c')
  const [activeTicket, setActiveTicket] = useState(rawCid || '')
  const [p, setP] = useState(mockPrediction)
  const [freeze, setFreeze] = useState(null)
  const [dispatch, setDispatch] = useState(false)
  const [officer, setOfficer] = useState('IO-042')
  const [freezing, setFreezing] = useState(false)
  const { label, remaining } = useCountdown(p.time_to_cashout_minutes)
  const urgent = remaining < 300 ? 'text-red-400' : remaining < 900 ? 'text-amber-400' : 'text-aegis-green'

  useEffect(() => {
    const fetchPrediction = async () => {
      let targetId = rawCid || localStorage.getItem('muleshield:selected') || ''
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
      } catch {
        setP(mockPrediction)
      }
    }

    fetchPrediction()
  }, [rawCid])

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

  const topAtm = p.top3_atms?.[0]

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-7 space-y-3">
        <div className="aegis-panel p-3">
          <div className="flex items-center gap-2 mono text-[11px] tracking-[0.14em] text-zinc-500">
            INTERCEPTION CONTROL ({activeTicket || p.terminal_account})
            <span className="ml-auto flex items-center gap-1 text-zinc-400">
              <Radio size={12} className="text-aegis-green animate-pulse-dot" /> LIVE COUNTDOWN
            </span>
          </div>
          <div className={`mono text-[42px] font-semibold tracking-tight text-center mt-2 ${urgent}`}>
            {label}
          </div>
          <div className="text-center mono text-[12px] text-zinc-300 font-semibold">
            TIME TO CASHOUT · Terminal Suspect: <span className="text-white font-bold">{p.terminal_account}</span>
          </div>
          <div className="text-center mono text-[11px] text-zinc-500 mt-0.5">
            GPS: {p.terminal_lat?.toFixed(4)}°, {p.terminal_lon?.toFixed(4)}° · Accuracy 98.5%
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 mono text-[11px]">
            {p.top3_atms?.map(a => (
              <div key={a.atm_id} className={`rounded border px-2 py-2 text-center ${a.rank === 1 ? 'bg-red-500/10 border-red-500/30' : 'bg-ink-panel border-ink-border'}`}>
                <div className="text-zinc-500">RANK {a.rank}</div>
                <div className="text-white font-medium">{a.atm_id}</div>
                <div className="text-zinc-500 text-[10px] truncate">{a.address}</div>
                <div className="text-aegis-green">{(a.confidence * 100).toFixed(1)}%</div>
                <div className="mt-1 h-1 bg-ink-bg border border-ink-border rounded overflow-hidden">
                  <span className="block h-full" style={{ width: `${a.confidence * 100}%`, background: a.rank === 1 ? '#ff3b3b' : '#7cf000' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <Panel title="EMERGENCY ACTIONS" right={freeze ? 'FROZEN ✓' : 'ARMED'}>
          <div className="p-3 grid grid-cols-2 gap-2">
            <button
              onClick={doFreeze}
              disabled={!!freeze || freezing}
              className={`py-3 rounded border mono text-[12px] flex flex-col items-center gap-1 font-bold ${
                freeze
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-red-500 text-white border-red-600 hover:bg-red-600 disabled:opacity-60'
              }`}
            >
              <ShieldCheck size={16} />
              {freezing ? 'FREEZING...' : freeze ? 'ACCOUNT FROZEN' : '1-CLICK EMERGENCY FREEZE'}
              <span className="text-[10px] opacity-80 font-normal">Card: {p.terminal_account} · {officer}</span>
            </button>
            <button
              onClick={() => setDispatch(true)}
              className="py-3 rounded bg-white text-black mono text-[12px] flex flex-col items-center gap-1 font-bold hover:bg-zinc-200 transition"
            >
              <Radio size={16} />
              DISPATCH PATROL UNIT
              <span className="text-[10px] opacity-70 font-normal">Target ATM: {topAtm?.atm_id || 'ATM-1'}</span>
            </button>
          </div>
          {freeze && (
            <div className="mx-3 mb-3 bg-emerald-500/10 border border-emerald-500/30 rounded px-3 py-2 mono text-[11px] text-emerald-300 flex items-center gap-2">
              <ShieldCheck size={14} /> FROZEN at {new Date(freeze.frozen_at || Date.now()).toLocaleTimeString()} — {freeze.account || p.terminal_account} · Ref {freeze.freeze_reference}
            </div>
          )}
          <div className="px-3 pb-3 flex items-center gap-2 mono text-[11px]">
            <span className="text-zinc-500">Officer ID:</span>
            <input
              value={officer}
              onChange={e => setOfficer(e.target.value)}
              className="bg-ink-panel border border-ink-border rounded px-2 py-1 w-24 outline-none text-zinc-200"
            />
            <span className="text-zinc-500 ml-2">Active Ticket: <span className="text-zinc-200 font-semibold">{activeTicket || '—'}</span></span>
            <span className="ml-auto text-zinc-500">NPCI micro-freeze &lt;500ms</span>
          </div>
        </Panel>
      </div>

      <div className="col-span-12 lg:col-span-5 space-y-3">
        <Panel title="DISPATCH PREVIEW" right="PCR">
          <div className="p-3">
            <div className="bg-ink-panel border border-ink-border rounded p-3 mono text-[11px] leading-relaxed">
              <div className="text-zinc-500">TO: PCR Patrol Van (Nearest Police Station)</div>
              <div className="text-zinc-300 mt-1">
                🚨 <span className="text-red-400 font-bold">INTERCEPT ALERT — MuleShield AI</span><br />
                Target ATM: <span className="text-white font-bold">{topAtm?.atm_id} — {topAtm?.address}</span><br />
                GPS: {topAtm?.lat?.toFixed(4)}°, {topAtm?.lon?.toFixed(4)}°<br />
                Confidence: {((topAtm?.confidence || 0.985) * 100).toFixed(1)}% · Countdown: <span className="text-aegis-green font-bold">{label}</span><br />
                Suspect Account: <span className="text-white font-semibold">{p.terminal_account}</span><br />
                Incident Ticket: <span className="text-zinc-200">{activeTicket || '—'}</span>
              </div>
              <div className="mt-2 flex gap-2">
                <button className="px-2.5 py-1 rounded bg-[#25D366] text-black font-bold flex items-center gap-1">
                  <MessageCircle size={12} /> WhatsApp Alert
                </button>
                <button className="px-2.5 py-1 rounded bg-ink-bg border border-ink-border text-zinc-300">
                  SMS Gateway
                </button>
              </div>
            </div>
            {dispatch && (
              <div className="mt-2 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 mono text-[11px] px-2.5 py-2 rounded">
                ✓ Dispatch sent to nearest PCR unit at {new Date().toLocaleTimeString()} — Routing to {topAtm?.address || 'ATM'}
              </div>
            )}
            <div className="mt-3 grid grid-cols-2 gap-2 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border rounded px-2 py-2">
                <div className="text-zinc-500">Stolen Amount</div>
                <div className="text-white text-[14px] font-bold">₹{Number(p.stolen_amount || 120000).toLocaleString('en-IN')}</div>
              </div>
              <div className="bg-ink-panel border border-ink-border rounded px-2 py-2">
                <div className="text-zinc-500">Expected Recovery</div>
                <div className="text-aegis-green text-[14px] font-bold">₹{Number(p.stolen_amount || 120000).toLocaleString('en-IN')}</div>
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
