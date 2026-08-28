import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { endpoints } from '../services/api'
import { mockPrediction } from '../utils/constants'
import { useCountdown } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import { Navigation, MapPin } from 'lucide-react'

function dot(color) {
  return L.divIcon({
    className: '',
    html: `<span style="display:block;width:12px;height:12px;border-radius:999px;background:${color};box-shadow:0 0 0 6px ${color}22, 0 0 12px ${color};border:2px solid #080a0a"></span>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6]
  })
}

const atmIcon = (rank, isSelected) => L.divIcon({
  className: '',
  html: `<div style="transform:translate(-50%,-100%);display:flex;flex-direction:column;align-items:center;cursor:pointer">
    <div style="font:600 10px JetBrains Mono,monospace; color:white;background:${isSelected ? '#ff3b3b' : rank === 1 ? '#e11d48' : '#1e2323'};border:${isSelected ? '2px solid #ffffff' : '1px solid #3a4242'};padding:3px 8px;border-radius:999px;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.6)">
      ATM-${rank} • ${isSelected ? 'TARGET ACTIVE' : rank === 1 ? 'TOP RANK' : 'CANDIDATE'}
    </div>
    <div style="width:12px;height:12px;background:${isSelected ? '#ff3b3b' : rank === 1 ? '#ff3b3b' : '#7cf000'};border:2px solid #ffffff;border-radius:999px;margin-top:4px;box-shadow:0 0 12px ${isSelected ? '#ff3b3b' : '#7cf000'};"></div>
  </div>`,
  iconSize: [0, 0],
  iconAnchor: [0, 0]
})

function DirectLeafletMap({ center, terminal, atms = [], selectedAtmId, onSelectAtm }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const circleRef = useRef(null)
  const polylineRef = useRef(null)
  const markersRef = useRef([])

  // Initialize Map once
  useEffect(() => {
    if (!containerRef.current) return

    if (mapRef.current) {
      mapRef.current.remove()
      mapRef.current = null
    }

    const c = center && center[0] ? center : [28.6139, 77.2090]
    const map = L.map(containerRef.current, {
      center: c,
      zoom: 13,
      zoomControl: true,
    })

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map)

    mapRef.current = map

    return () => {
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [])

  // Update Markers, Target Circle, and Route Polyline when data or selectedAtm changes
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    // Clean previous markers, circle, polyline
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []

    if (circleRef.current) {
      circleRef.current.remove()
      circleRef.current = null
    }
    if (polylineRef.current) {
      polylineRef.current.remove()
      polylineRef.current = null
    }

    // Terminal marker
    if (terminal && terminal.lat && terminal.lon) {
      const termMarker = L.marker([terminal.lat, terminal.lon], { icon: dot('#58a6ff') })
        .bindPopup(`<div class="mono text-[11px] p-1"><div class="font-bold text-blue-400">TERMINAL MULE ACCOUNT</div><div>${terminal.account || 'Account'}</div><div>GPS: ${terminal.lat.toFixed(4)}°, ${terminal.lon.toFixed(4)}°</div></div>`)
        .addTo(map)
      markersRef.current.push(termMarker)
    }

    const activeAtm = atms.find(a => a.atm_id === selectedAtmId) || atms[0]

    // ATM candidate markers
    atms.forEach(a => {
      const isSel = a.atm_id === activeAtm?.atm_id
      const marker = L.marker([a.lat, a.lon], { icon: atmIcon(a.rank, isSel), zIndexOffset: isSel ? 1000 : 100 })
        .bindPopup(`
          <div class="mono text-[11px] p-1">
            <div class="font-bold text-red-400">${a.atm_id} — ${a.bank} (Rank #${a.rank})</div>
            <div class="text-zinc-700">${a.address}</div>
            <div class="mt-1 text-emerald-600 font-bold">Confidence: ${(a.confidence * 100).toFixed(1)}%</div>
            <div class="text-zinc-500">${a.historical_fraud_count || 0} historical incident priors</div>
          </div>
        `)
        .addTo(map)

      marker.on('click', () => {
        if (onSelectAtm) onSelectAtm(a.atm_id)
      })

      if (isSel) {
        marker.openPopup()
      }

      markersRef.current.push(marker)
    })

    // Targeting Circle and Travel Route
    if (activeAtm && activeAtm.lat && activeAtm.lon) {
      const circle = L.circle([activeAtm.lat, activeAtm.lon], {
        radius: 600,
        color: '#ff3b3b',
        fillColor: '#ff3b3b',
        fillOpacity: 0.15,
        weight: 2,
        dashArray: '6 6'
      }).addTo(map)
      circleRef.current = circle

      if (terminal && terminal.lat && terminal.lon) {
        const line = L.polyline([[terminal.lat, terminal.lon], [activeAtm.lat, activeAtm.lon]], {
          color: '#7cf000',
          weight: 2.5,
          dashArray: '6 6',
          opacity: 0.9
        }).addTo(map)
        polylineRef.current = line
      }

      // Smoothly pan camera to the selected ATM
      map.flyTo([activeAtm.lat, activeAtm.lon], 13, { duration: 0.8 })
    }
  }, [terminal?.lat, terminal?.lon, terminal?.account, atms, selectedAtmId, onSelectAtm])

  return <div ref={containerRef} className="h-full w-full" style={{ background: '#080a0a' }} />
}

export default function TacticalMap() {
  const [params] = useSearchParams()
  const rawCid = params.get('c')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTicket, setActiveTicket] = useState(rawCid || '')
  const [selectedAtmId, setSelectedAtmId] = useState('')

  useEffect(() => {
    const fetchActive = async () => {
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
        setPrediction(mockPrediction)
        setSelectedAtmId(mockPrediction.top3_atms[0].atm_id)
        return
      }

      setActiveTicket(targetId)
      setLoading(true)
      try {
        const data = await endpoints.predictCashout(targetId)
        setPrediction(data)
        if (data?.top3_atms?.length) {
          setSelectedAtmId(data.top3_atms[0].atm_id)
        }
      } catch {
        setPrediction(mockPrediction)
        setSelectedAtmId(mockPrediction.top3_atms[0].atm_id)
      } finally {
        setLoading(false)
      }
    }

    fetchActive()
  }, [rawCid])

  const p = prediction || mockPrediction
  const atms = p.top3_atms || []
  const activeAtm = atms.find(a => a.atm_id === selectedAtmId) || atms[0]
  const { label } = useCountdown(p.time_to_cashout_minutes)

  const center = activeAtm ? [activeAtm.lat, activeAtm.lon] : [28.6139, 77.2090]
  const terminal = {
    lat: p.terminal_lat,
    lon: p.terminal_lon,
    account: p.terminal_account
  }

  // Calculate distance between terminal account and selected ATM
  const distanceKm = (activeAtm && terminal.lat && terminal.lon)
    ? (L.latLng(terminal.lat, terminal.lon).distanceTo(L.latLng(activeAtm.lat, activeAtm.lon)) / 1000).toFixed(2)
    : '0.85'

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      {/* MAP VIEW */}
      <div className="col-span-12 lg:col-span-8">
        <Panel
          title={`TACTICAL GIS — ATM INTERCEPTION (${activeTicket || p.terminal_account})`}
          right={loading ? 'predicting...' : `${label} remaining`}
        >
          <div className="h-[64vh] relative">
            <DirectLeafletMap
              center={center}
              terminal={terminal}
              atms={atms}
              selectedAtmId={selectedAtmId}
              onSelectAtm={setSelectedAtmId}
            />
            <div className="absolute top-3 left-3 flex flex-wrap gap-2 mono text-[11px] z-[1000]">
              <span className="px-2.5 py-1 rounded-full bg-ink-panel/90 border border-ink-border text-zinc-200 backdrop-blur-sm">
                Pan-India Grid · 65 Cities
              </span>
              <span className="px-2.5 py-1 rounded-full bg-red-500/20 border border-red-500/40 text-red-400 font-semibold backdrop-blur-sm">
                Active Target: {activeAtm?.atm_id} (Rank #{activeAtm?.rank || 1})
              </span>
            </div>
            <div className="absolute bottom-3 left-3 bg-ink-panel/90 border border-ink-border px-3 py-1.5 rounded-lg mono text-[11px] text-zinc-400 z-[1000] backdrop-blur-sm flex items-center gap-2">
              <Navigation size={13} className="text-aegis-green" />
              <span>Click any ATM card on the right or pin on the map to switch live targeting</span>
            </div>
          </div>
        </Panel>
      </div>

      {/* RIGHT COLUMN: INTERACTIVE ATM RANKING CARDS */}
      <div className="col-span-12 lg:col-span-4 space-y-3">
        <div className="aegis-panel p-3">
          <div className="flex items-center justify-between pb-2 border-b border-ink-border">
            <span className="mono text-[11px] tracking-[0.14em] text-zinc-400 font-semibold uppercase">
              Target ATM Ranking (Top-3)
            </span>
            <span className="mono text-[10px] text-aegis-green font-bold">XGBoost 98.5% Conf</span>
          </div>

          <div className="space-y-2 mt-3">
            {atms.map(a => {
              const isSel = a.atm_id === activeAtm?.atm_id
              return (
                <div
                  key={a.atm_id}
                  onClick={() => setSelectedAtmId(a.atm_id)}
                  className={`p-3 rounded-lg border cursor-pointer select-none transition-all duration-150 ${
                    isSel
                      ? 'bg-ink-panel border-red-500 ring-2 ring-red-500/30 shadow-[0_0_14px_rgba(255,59,59,0.2)]'
                      : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between mono text-[11px]">
                    <div className="flex items-center gap-2">
                      <span className={`w-5 h-5 grid place-items-center rounded text-[10px] font-bold ${
                        isSel ? 'bg-red-500 text-white' : a.rank === 1 ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-ink-bg border border-ink-border text-zinc-400'
                      }`}>
                        {a.rank}
                      </span>
                      <span className="font-bold text-white">{a.atm_id}</span>
                      <span className="text-zinc-400">· {a.bank}</span>
                    </div>
                    <span className="text-aegis-green font-bold">{(a.confidence * 100).toFixed(1)}%</span>
                  </div>

                  <div className="mono text-[11px] text-zinc-300 mt-1.5 font-medium flex items-center gap-1">
                    <MapPin size={12} className="text-red-400 shrink-0" />
                    <span className="truncate">{a.address}</span>
                  </div>

                  <div className="mono text-[10px] text-zinc-500 mt-1 flex justify-between">
                    <span>{a.lat.toFixed(4)}°, {a.lon.toFixed(4)}°</span>
                    <span>{a.historical_fraud_count} incident priors</span>
                  </div>

                  <div className="mt-2 h-1.5 bg-ink-bg border border-ink-border rounded overflow-hidden">
                    <span
                      className="block h-full transition-all duration-300"
                      style={{
                        width: `${Math.round(a.confidence * 100)}%`,
                        background: isSel ? '#ff3b3b' : a.rank === 1 ? '#ff3b3b' : '#7cf000'
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-3 pt-3 border-t border-ink-border space-y-1.5 mono text-[11px]">
            <div className="flex justify-between text-zinc-400">
              <span>Terminal Account:</span>
              <span className="text-white font-bold">{p.terminal_account}</span>
            </div>
            <div className="flex justify-between text-zinc-400">
              <span>Selected Target ATM:</span>
              <span className="text-red-400 font-bold">{activeAtm?.atm_id} ({activeAtm?.bank})</span>
            </div>
            <div className="flex justify-between text-zinc-400">
              <span>Terminal → ATM Distance:</span>
              <span className="text-aegis-green font-semibold">{distanceKm} km</span>
            </div>
            <div className="flex justify-between text-zinc-400">
              <span>Inference Time:</span>
              <span className="text-zinc-300">{p.inference_time_ms} ms</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 mt-3 mono text-[11px]">
            <div className="rounded-lg border border-ink-border bg-ink-panel p-2.5">
              <div className="text-zinc-500">Estimated Cashout</div>
              <div className="text-lg font-bold text-white mt-0.5">{label}</div>
            </div>
            <div className="rounded-lg border border-ink-border bg-ink-panel p-2.5">
              <div className="text-zinc-500">Interception Conf</div>
              <div className="text-lg font-bold text-aegis-green mt-0.5">
                {activeAtm ? (activeAtm.confidence * 100).toFixed(1) : (p.interception_confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
