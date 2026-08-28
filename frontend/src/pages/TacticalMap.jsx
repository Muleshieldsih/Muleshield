import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { endpoints } from '../services/api'
import { mockPrediction } from '../utils/constants'
import { useCountdown } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'

function dot(color) {
  return L.divIcon({
    className: '',
    html: `<span style="display:block;width:12px;height:12px;border-radius:999px;background:${color};box-shadow:0 0 0 6px ${color}22, 0 0 12px ${color};border:2px solid #080a0a"></span>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6]
  })
}

const atmIcon = (rank) => L.divIcon({
  className: '',
  html: `<div style="transform:translate(-50%,-100%);display:flex;flex-direction:column;align-items:center">
    <div style="font:600 10px JetBrains Mono,monospace; color:white;background:${rank === 1 ? '#ff3b3b' : '#1a1f1f'};border:1px solid #2a3333;padding:2px 6px;border-radius:999px;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.5)">ATM-${rank} • ${rank === 1 ? 'TARGET' : '—'}</div>
    <div style="width:10px;height:10px;background:${rank === 1 ? '#ff3b3b' : '#7cf000'};border:2px solid #080a0a;border-radius:999px;margin-top:4px;box-shadow:0 0 10px ${rank === 1 ? '#ff3b3b' : '#7cf000'};"></div>
  </div>`,
  iconSize: [0, 0],
  iconAnchor: [0, 0]
})

function DirectLeafletMap({ center, terminal, atms, top }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return

    // Safely cleanup previous map instance
    if (mapRef.current) {
      mapRef.current.remove()
      mapRef.current = null
    }

    const c = center && center[0] ? center : [28.6139, 77.2090]
    const map = L.map(containerRef.current, {
      center: c,
      zoom: 12,
      zoomControl: true,
    })

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map)

    // Add Terminal Marker
    if (terminal && terminal.lat && terminal.lon) {
      L.marker([terminal.lat, terminal.lon], { icon: dot('#58a6ff') })
        .bindPopup(`<span class="mono text-[11px]">TERMINAL ${terminal.account || 'Mule Account'}</span>`)
        .addTo(map)
    }

    // Add ATM Candidates
    if (atms && atms.length) {
      atms.forEach(a => {
        L.marker([a.lat, a.lon], { icon: atmIcon(a.rank) })
          .bindPopup(`<div class="mono text-[11px]"><div class="font-semibold">${a.atm_id} — ${a.bank}</div><div>${a.address}</div><div>Conf ${(a.confidence * 100).toFixed(1)}%</div></div>`)
          .addTo(map)
      })
    }

    // Add Target Circle and Route Polyline
    if (top && top.lat && top.lon) {
      L.circle([top.lat, top.lon], {
        radius: 600,
        color: '#ff3b3b',
        fillColor: '#ff3b3b',
        fillOpacity: 0.12,
        weight: 1.5,
        dashArray: '6 6'
      }).addTo(map)

      if (terminal && terminal.lat && terminal.lon) {
        L.polyline([[terminal.lat, terminal.lon], [top.lat, top.lon]], {
          color: '#7cf000',
          weight: 2,
          dashArray: '4 6',
          opacity: 0.85
        }).addTo(map)
      }
    }

    mapRef.current = map

    return () => {
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [center?.[0], center?.[1], terminal?.lat, terminal?.lon, terminal?.account, atms?.length, top?.atm_id])

  return <div ref={containerRef} className="h-full w-full" style={{ background: '#080a0a' }} />
}

export default function TacticalMap() {
  const [params] = useSearchParams()
  const rawCid = params.get('c')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTicket, setActiveTicket] = useState(rawCid || '')

  useEffect(() => {
    const fetchActive = async () => {
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
        setPrediction(mockPrediction)
        return
      }

      setActiveTicket(targetId)
      setLoading(true)
      try {
        const data = await endpoints.predictCashout(targetId)
        setPrediction(data)
      } catch {
        setPrediction(mockPrediction)
      } finally {
        setLoading(false)
      }
    }

    fetchActive()
  }, [rawCid])

  const p = prediction || mockPrediction
  const top = p.top3_atms?.[0]
  const { label } = useCountdown(p.time_to_cashout_minutes)

  const center = top ? [top.lat, top.lon] : [28.6139, 77.2090]
  const terminal = {
    lat: p.terminal_lat,
    lon: p.terminal_lon,
    account: p.terminal_account
  }

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-8">
        <Panel title={`TACTICAL GIS — ATM INTERCEPTION (${activeTicket || p.terminal_account})`} right={loading ? 'predicting...' : `${label} remaining`}>
          <div className="h-[62vh] relative">
            <DirectLeafletMap
              center={center}
              terminal={terminal}
              atms={p.top3_atms}
              top={top}
            />
            <div className="absolute top-2 left-2 flex gap-2 mono text-[11px] z-[1000]">
              <span className="px-2 py-1 rounded-full bg-ink-panel border border-ink-border text-zinc-300">Pan-India · 65 cities</span>
              <span className="px-2 py-1 rounded-full bg-red-500/10 border border-red-500/30 text-red-400">{p.top3_atms?.length || 3} targets</span>
            </div>
          </div>
        </Panel>
      </div>
      <div className="col-span-12 lg:col-span-4 space-y-3">
        <div className="aegis-panel p-3">
          <div className="mono text-[11px] tracking-[0.14em] text-zinc-500">TARGET ATM RANKING</div>
          <div className="space-y-2 mt-2">
            {p.top3_atms?.map(a => (
              <div key={a.atm_id} className={`rounded border px-3 py-2.5 ${a.rank === 1 ? 'bg-red-500/10 border-red-500/30' : 'bg-ink-panel border-ink-border'}`}>
                <div className="flex items-center gap-2 mono text-[11px]"><span className={`w-5 h-5 grid place-items-center rounded text-[11px] ${a.rank === 1 ? 'bg-red-500 text-white' : 'bg-ink-bg border border-ink-border text-zinc-400'}`}>{a.rank}</span><span className="text-zinc-200">{a.atm_id}</span><span className="text-zinc-600">· {a.bank}</span><span className="ml-auto text-aegis-green">{(a.confidence * 100).toFixed(1)}%</span></div>
                <div className="mono text-[11px] text-zinc-500 mt-1">{a.address}</div>
                <div className="mono text-[11px] text-zinc-400">{a.lat.toFixed(4)}°, {a.lon.toFixed(4)}° · {a.historical_fraud_count} priors</div>
                <div className="mt-2 h-1.5 bg-ink-bg border border-ink-border rounded overflow-hidden"><span className="block h-full" style={{ width: `${Math.round(a.confidence * 100)}%`, background: a.rank === 1 ? '#ff3b3b' : '#7cf000' }} /></div>
              </div>
            ))}
          </div>
          <div className="mono text-[11px] text-zinc-400 mt-3 font-semibold">
            Terminal Node: <span className="text-white font-bold">{p.terminal_account}</span>
          </div>
          <div className="mono text-[11px] text-zinc-500 mt-1">Inference {p.inference_time_ms} ms · Target {top?.atm_id}</div>
          <div className="grid grid-cols-2 gap-2 mt-3 mono text-[11px]">
            <div className="rounded border border-ink-border bg-ink-panel px-2 py-2"><div className="text-zinc-500">Time to Cashout</div><div className="text-[16px] text-white">{label}</div></div>
            <div className="rounded border border-ink-border bg-ink-panel px-2 py-2"><div className="text-zinc-500">Interception Conf</div><div className="text-[16px] text-aegis-green">{(p.interception_confidence * 100).toFixed(1)}%</div></div>
          </div>
        </div>
      </div>
    </div>
  )
}
