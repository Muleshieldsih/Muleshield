import { NavLink, useNavigate } from 'react-router-dom'
import { Shield, Activity, GitBranch, MapPinned, Zap, Search, Bell, Command, Radio, Circle } from 'lucide-react'
import { useState, useEffect } from 'react'

export function Topbar({ wsConnected, complaintId }) {
  const navigate = useNavigate()
  return (
    <div className="h-[48px] flex items-center justify-between gap-3 px-4 border-b border-ink-border bg-ink-bg sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded bg-aegis-green flex items-center justify-center shadow-[0_0_10px_rgba(124,240,0,0.4)]">
          <Shield size={16} className="text-black font-bold" />
        </div>
        <div className="leading-tight">
          <div className="text-[12px] font-bold tracking-[0.14em] text-white mono">MULESHIELD AI</div>
          <div className="text-[10px] mono text-zinc-400">1930 HELPLINE INTERDICTION CONSOLE · MHA / I4C</div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-2 text-[11px] mono bg-ink-panel border border-ink-border rounded-full px-3 py-1">
          <span className={`flex items-center gap-1.5 ${wsConnected ? 'text-aegis-green font-semibold' : 'text-zinc-500'}`}>
            <Circle size={8} className={wsConnected ? 'fill-aegis-green animate-pulse-dot' : ''} />
            {wsConnected ? 'LIVE 1930 TELEMETRY' : 'OFFLINE MODE'}
          </span>
          <span className="text-zinc-600">•</span>
          <span className="text-zinc-400">AI Latency: <span className="text-aegis-green">25.8 ms</span></span>
        </div>

        {complaintId && (
          <div className="hidden sm:block text-[11px] mono text-zinc-300 border border-aegis-green/40 rounded-full px-3 py-1 bg-aegis-green/10">
            Active: <span className="font-bold text-white">{complaintId}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export function Sidebar({ complaintId }) {
  const activeId = complaintId || (typeof window !== 'undefined' ? localStorage.getItem('muleshield:selected') : '') || ''
  const linkWithCid = (path) => path === '/' ? '/' : (path + (activeId ? `?c=${encodeURIComponent(activeId)}` : ''))

  const item = (to, icon, label, sub) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-lg border text-[12px] mono transition ${
          isActive
            ? 'bg-ink-panel border-aegis-green/50 text-white font-bold shadow-[0_0_10px_rgba(124,240,0,0.1)]'
            : 'border-transparent text-zinc-400 hover:text-white hover:bg-ink-panel/50'
        }`
      }
    >
      <span className="w-6 h-6 grid place-items-center rounded bg-ink-surface border border-ink-border">{icon}</span>
      <span className="flex-1 leading-tight">
        {label}
        <br />
        <span className="text-[10px] text-zinc-500 font-normal">{sub}</span>
      </span>
    </NavLink>
  )
  return (
    <div className="w-[210px] shrink-0 border-r border-ink-border bg-ink-bg hidden md:flex flex-col">
      <div className="p-3 space-y-1.5">
        {item(linkWithCid('/'), <Activity size={15} />, 'TRIAGE QUEUE', 'Live 1930 feed')}
        {item(linkWithCid('/map'), <MapPinned size={15} />, 'TACTICAL MAP', 'ATM GPS routing')}
        {item(linkWithCid('/graph'), <GitBranch size={15} />, 'MONEY FLOW', 'Forensic graph')}
        {item(linkWithCid('/intercept'), <Zap size={15} />, 'INTERCEPTION', '1-Click freeze')}
      </div>
      <div className="mt-auto p-3 border-t border-ink-border">
        <div className="aegis-panel p-2.5 bg-ink-panel/40">
          <div className="text-[10px] mono tracking-[0.12em] text-zinc-400 uppercase font-semibold">System SLA Status</div>
          <div className="text-[11px] mono text-aegis-green flex items-center gap-1.5 mt-1 font-bold">
            <Radio size={12} className="animate-pulse-dot" /> 98.5% Top-3 Accuracy
          </div>
          <div className="text-[10px] mono text-zinc-500 mt-1">Countdown MAE: <span className="text-zinc-300">1.4s</span></div>
        </div>
        <div className="text-[10px] mono text-zinc-500 mt-2 text-center">SIH26184 · MHA / I4C</div>
      </div>
    </div>
  )
}

export function Panel({ title, live, right, children, className='' }){
  return (
    <div className={`aegis-panel ${className}`}>
      <div className="aegis-panel-header">
        <div className="flex items-center gap-2 text-[11px] mono tracking-[0.12em] text-zinc-400">
          <span>{title}</span>
          {live && <span className="flex items-center gap-1.5 text-aegis-green normal-case tracking-normal"><span className="w-1.5 h-1.5 rounded-full bg-aegis-green animate-pulse-dot"/>Live Stream</span>}
        </div>
        <div className="text-[11px] mono text-zinc-500">{right}</div>
      </div>
      <div>{children}</div>
    </div>
  )
}
