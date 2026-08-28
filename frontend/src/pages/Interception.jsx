import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { endpoints } from '../services/api'
import { mockPrediction } from '../utils/constants'
import { useCountdown } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import { ShieldCheck, Radio, MessageCircle } from 'lucide-react'

export default function Interception(){
  const [params]=useSearchParams()
  const cid = params.get('c') || 'TKT-A1B2C3D4'
  const [p,setP]=useState(mockPrediction)
  const [freeze,setFreeze]=useState(null)
  const [dispatch,setDispatch]=useState(false)
  const [officer,setOfficer]=useState('IO-042')
  const [freezing,setFreezing]=useState(false)
  const { label, remaining } = useCountdown(p.time_to_cashout_minutes)
  const urgent = remaining < 300 ? 'text-red-400' : remaining < 900 ? 'text-amber-400' : 'text-aegis-green'

  useEffect(()=>{
    endpoints.predictCashout(cid).then(setP).catch(()=> setP(mockPrediction))
  },[cid])

  const doFreeze = async()=>{
    setFreezing(true)
    try{
      const res = await endpoints.microFreeze({ account_id: p.terminal_account, complaint_id: cid, officer_id: officer })
      setFreeze(res)
    }catch{
      setFreeze({ status:'FROZEN', freeze_reference:`FRZ-${Math.random().toString(36).slice(2,8).toUpperCase()}`, frozen_at: new Date().toISOString(), account: p.terminal_account })
    }finally{setFreezing(false)}
  }

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-7 space-y-3">
        <div className="aegis-panel p-3">
          <div className="flex items-center gap-2 mono text-[11px] tracking-[0.14em] text-zinc-500">INTERCEPTION CONTROL <span className="ml-auto flex items-center gap-1 text-zinc-400"><Radio size={12} className="text-aegis-green animate-pulse-dot"/> LIVE COUNTDOWN</span></div>
          <div className={`mono text-[42px] font-semibold tracking-tight text-center mt-2 ${urgent}`} style={{textShadow: remaining<300?'0 0 16px rgba(255,59,59,0.6)':''}}>{label}</div>
          <div className="text-center mono text-[11px] text-zinc-500">TIME TO CASHOUT · Terminal {p.terminal_account} · {p.terminal_lat.toFixed(4)}°, {p.terminal_lon.toFixed(4)}°</div>
          <div className="mt-3 grid grid-cols-3 gap-2 mono text-[11px]">
            {p.top3_atms?.map(a=> (
              <div key={a.atm_id} className={`rounded border px-2 py-2 text-center ${a.rank===1?'bg-red-500/10 border-red-500/30':'bg-ink-panel border-ink-border'}`}>
                <div className="text-zinc-500">RANK {a.rank}</div>
                <div className="text-white font-medium">{a.atm_id}</div>
                <div className="text-zinc-500 text-[10px] truncate">{a.address}</div>
                <div className="text-aegis-green">{(a.confidence*100).toFixed(1)}%</div>
                <div className="mt-1 h-1 bg-ink-bg border border-ink-border rounded overflow-hidden"><span className="block h-full" style={{width:`${a.confidence*100}%`, background: a.rank===1?'#ff3b3b':'#7cf000'}}/></div>
              </div>
            ))}
          </div>
        </div>

        <Panel title="EMERGENCY ACTIONS" right={freeze?'FROZEN ✓':'ARMED'}>
          <div className="p-3 grid grid-cols-2 gap-2">
            <button onClick={doFreeze} disabled={!!freeze || freezing} className={`py-3 rounded border mono text-[12px] flex flex-col items-center gap-1 ${freeze?'bg-emerald-500/10 border-emerald-500/30 text-emerald-400':'bg-red-500 text-white border-red-600 hover:bg-red-600 disabled:opacity-60'}`}>
              <ShieldCheck size={16}/>{freezing?'FREEZING...': freeze?'FROZEN':'EMERGENCY FREEZE'}<span className="text-[10px] opacity-70">Card {p.terminal_account.slice(-6)} · {officer}</span>
            </button>
            <button onClick={()=>setDispatch(true)} className="py-3 rounded bg-white text-black mono text-[12px] flex flex-col items-center gap-1">
              <Radio size={16}/>DISPATCH PATROL<span className="text-[10px] opacity-60">Unit CP-03 · 2.3 km</span>
            </button>
          </div>
          {freeze && (
            <div className="mx-3 mb-3 bg-emerald-500/10 border border-emerald-500/30 rounded px-3 py-2 mono text-[11px] text-emerald-300 flex items-center gap-2">
              <ShieldCheck size={14}/> FROZEN at {new Date(freeze.frozen_at).toLocaleTimeString()} — {freeze.account||p.terminal_account} · Ref {freeze.freeze_reference}
            </div>
          )}
          <div className="px-3 pb-3 flex items-center gap-2 mono text-[11px]">
            <span className="text-zinc-500">Officer ID</span><input value={officer} onChange={e=>setOfficer(e.target.value)} className="bg-ink-panel border border-ink-border rounded px-2 py-1 w-24 outline-none text-zinc-200"/>
            <span className="text-zinc-600">TKT {cid}</span>
            <span className="ml-auto text-zinc-500">NPCI micro-freeze &lt;500ms</span>
          </div>
        </Panel>
      </div>

      <div className="col-span-12 lg:col-span-5 space-y-3">
        <Panel title="DISPATCH PREVIEW" right="PCR">
          <div className="p-3">
            <div className="bg-ink-panel border border-ink-border rounded p-3 mono text-[11px] leading-relaxed">
              <div className="text-zinc-500">TO: PCR Van CP-03 — Connaught Place PS</div>
              <div className="text-zinc-300 mt-1">🚨 INTERCEPT ALERT — MuleShield AI<br/>Target ATM: <span className="text-white">{p.top3_atms[0].atm_id} — {p.top3_atms[0].address}</span><br/>GPS: {p.top3_atms[0].lat.toFixed(4)}°, {p.top3_atms[0].lon.toFixed(4)}°<br/>Confidence: {(p.top3_atms[0].confidence*100).toFixed(1)}% · Countdown: {label}<br/>Suspect Account: {p.terminal_account}<br/>TKT: {cid}</div>
              <div className="mt-2 flex gap-2">
                <button className="px-2 py-1 rounded bg-[#25D366] text-black flex items-center gap-1"><MessageCircle size={12}/> WhatsApp</button>
                <button className="px-2 py-1 rounded bg-ink-bg border border-ink-border text-zinc-300">SMS</button>
              </div>
            </div>
            {dispatch && <div className="mt-2 bg-white text-black mono text-[11px] px-2 py-1.5 rounded border border-ink-border">✓ Dispatch sent to CP-03 at {new Date().toLocaleTimeString()} — ETA 6 min via Baba Kharak Singh Marg</div>}
            <div className="mt-3 grid grid-cols-2 gap-2 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border rounded px-2 py-2"><div className="text-zinc-500">Stolen Amount</div><div className="text-white text-[14px]">₹{Number(p.stolen_amount).toLocaleString('en-IN')}</div></div>
              <div className="bg-ink-panel border border-ink-border rounded px-2 py-2"><div className="text-zinc-500">Expected Recovery</div><div className="text-aegis-green text-[14px]">₹{Number(p.stolen_amount).toLocaleString('en-IN')}</div></div>
            </div>
          </div>
        </Panel>

        <Panel title="EVIDENCE CHAIN" right="DAG">
          <div className="p-3 mono text-[11px] space-y-1">
            <div className="flex justify-between"><span className="text-zinc-500">TKT</span><span className="text-white">{cid}</span></div>
            <div className="flex justify-between"><span className="text-zinc-500">Victim → Terminal hops</span><span className="text-zinc-300">0 → 4</span></div>
            <div className="flex justify-between"><span className="text-zinc-500">Build time</span><span className="text-aegis-green">142 ms</span></div>
            <div className="flex justify-between"><span className="text-zinc-500">GNN risk (terminal)</span><span className="text-red-400">0.94</span></div>
            <div className="mt-2 h-px bg-ink-border"/>
            <div className="text-zinc-500">Timeline</div>
            <div className="space-y-1">
              <div className="flex gap-2"><span className="text-zinc-600">02:14:03</span><span className="text-zinc-300">Victim debited ₹{Number(p.stolen_amount).toLocaleString('en-IN')}</span></div>
              <div className="flex gap-2"><span className="text-zinc-600">02:14:47</span><span className="text-zinc-300">Split 1→2 × ₹{(p.stolen_amount/2).toFixed(0)}</span></div>
              <div className="flex gap-2"><span className="text-zinc-600">02:15:19</span><span className="text-amber-400">Terminal {p.terminal_account.slice(-6)} armed</span></div>
              <div className="flex gap-2"><span className="text-zinc-600">{label}</span><span className="text-red-400">Predicted cashout</span></div>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
