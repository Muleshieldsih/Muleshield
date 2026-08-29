import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { endpoints, describeError } from '../services/api'
import useActiveComplaint from '../hooks/useActiveComplaint'
import { useCountdown } from '../hooks/useCountdown'
import { Panel, Stat } from '../components/Shell'
import { amountFmt, formatTicket } from '../utils/constants'
import { Navigation, MapPin, Loader2, ServerCrash, CloudOff } from 'lucide-react'

const INDIA_CENTER = [22.9734, 78.6569]

function terminalIcon() {
  return L.divIcon({
    className: '',
    html: `<span style="display:block;width:13px;height:13px;border-radius:999px;background:#58a6ff;
      box-shadow:0 0 0 4px rgba(88,166,255,0.18);border:2px solid #080a0a"></span>`,
    iconSize: [13, 13],
    iconAnchor: [6.5, 6.5],
  })
}

function atmIcon(rank, isSelected) {
  const bg = isSelected ? '#ff3b3b' : rank === 1 ? '#e11d48' : '#1e2323'
  const dot = isSelected || rank === 1 ? '#ff3b3b' : '#7cf000'
  const label = isSelected ? 'TARGET ACTIVE' : rank === 1 ? 'TOP RANK' : 'CANDIDATE'
  return L.divIcon({
    className: '',
    html: `<div style="transform:translate(-50%,-100%);display:flex;flex-direction:column;align-items:center;cursor:pointer">
      <div style="font:600 10px 'JetBrains Mono',monospace;color:#fff;background:${bg};
        border:${isSelected ? '2px solid #fff' : '1px solid #3a4242'};padding:3px 8px;border-radius:999px;
        white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.6)">ATM-${rank} • ${label}</div>
      <div style="width:12px;height:12px;background:${dot};border:2px solid #fff;border-radius:999px;
        margin-top:4px;box-shadow:0 1px 3px rgba(0,0,0,0.5)"></div>
    </div>`,
    iconSize: [0, 0],
    iconAnchor: [0, 0],
  })
}

function TacticalLeafletMap({ terminal, atms, zone, selectedAtmId, onSelectAtm, onTilesFailed }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layerRef = useRef(null)
  const selectRef = useRef(onSelectAtm)
  const failedRef = useRef(false)

  useEffect(() => { selectRef.current = onSelectAtm }, [onSelectAtm])

  // ── Create the map exactly once ──────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: INDIA_CENTER,
      zoom: 5,
      zoomControl: true,
      attributionControl: false,
    })

    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      crossOrigin: true,
    })

    // A hackathon venue's wifi frequently blocks or throttles tile servers. The
    // map must stay usable — markers, ranges and routing lines all render on the
    // dark canvas without a basemap.
    let errors = 0
    tiles.on('tileerror', () => {
      errors += 1
      if (errors >= 4 && !failedRef.current) {
        failedRef.current = true
        onTilesFailed?.()
      }
    })
    tiles.addTo(map)

    layerRef.current = L.layerGroup().addTo(map)
    mapRef.current = map

    // Leaflet mis-sizes itself inside a flex/grid parent that is still settling.
    const ro = new ResizeObserver(() => map.invalidateSize())
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      map.remove()
      mapRef.current = null
      layerRef.current = null
    }
  }, [onTilesFailed])

  // ── Redraw overlays whenever the prediction or selection changes ─────────
  useEffect(() => {
    const map = mapRef.current
    const layer = layerRef.current
    if (!map || !layer) return

    layer.clearLayers()

    const active = atms.find(a => a.atm_id === selectedAtmId) || atms[0]
    const hasTerminal = Number.isFinite(terminal?.lat) && Number.isFinite(terminal?.lon)

    if (hasTerminal) {
      L.marker([terminal.lat, terminal.lon], { icon: terminalIcon() })
        .bindPopup(
          `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px;line-height:1.5">
            <b style="color:#1d4ed8">Terminal mule account</b><br/>${terminal.account || '—'}<br/>
            ${terminal.lat.toFixed(4)}°, ${terminal.lon.toFixed(4)}°
          </div>`
        )
        .addTo(layer)
    }

    atms.forEach(a => {
      if (!Number.isFinite(a.lat) || !Number.isFinite(a.lon)) return
      const isSel = a.atm_id === active?.atm_id
      const marker = L.marker([a.lat, a.lon], {
        icon: atmIcon(a.rank, isSel),
        zIndexOffset: isSel ? 1000 : 100,
      })
        .bindPopup(
          `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px;line-height:1.5">
            <b style="color:#b91c1c">${a.atm_id} — ${a.bank}</b> (Rank #${a.rank})<br/>
            ${a.address}<br/>
            <span style="color:#374151">Rank share ${(a.confidence * 100).toFixed(1)}%
              · ${a.historical_fraud_count} prior incidents</span><br/>
            <span style="color:#6b7280">${
              a.opening_time ? `Open ${a.opening_time}–${a.closing_time}` : 'Hours unknown'
            }</span>
          </div>`
        )
        .on('click', () => selectRef.current?.(a.atm_id))
        .addTo(layer)

      if (isSel) marker.openPopup()
    })

    // The predicted SEARCH ZONE - the problem statement's actual deliverable.
    // This used to be a fixed 600 m ring drawn around the top ATM, which showed
    // nothing the model had computed. It is now the area the ranker's own
    // probability distribution puts the withdrawal in.
    if (zone && Number.isFinite(zone.lat) && Number.isFinite(zone.radius_km)) {
      L.circle([zone.lat, zone.lon], {
        radius: zone.radius_km * 1000,
        color: '#ff8c42', fillColor: '#ff8c42',
        fillOpacity: 0.10, weight: 2, dashArray: '8 6',
      })
        .bindPopup(
          `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px;line-height:1.5">
            <b style="color:#b45309">Search zone</b><br/>
            radius ${zone.radius_km.toFixed(2)} km<br/>
            ${zone.atm_count} ATM(s) to cover<br/>
            ${(zone.probability_mass * 100).toFixed(0)}% of predicted probability
          </div>`
        )
        .addTo(layer)
    }

    if (active && Number.isFinite(active.lat) && Number.isFinite(active.lon)) {
      L.circle([active.lat, active.lon], {
        radius: 400, color: '#ff3b3b', fillColor: '#ff3b3b',
        fillOpacity: 0.14, weight: 2, dashArray: '6 6',
      }).addTo(layer)

      if (hasTerminal) {
        L.polyline([[terminal.lat, terminal.lon], [active.lat, active.lon]], {
          color: '#7cf000', weight: 2.5, dashArray: '6 6', opacity: 0.9,
        }).addTo(layer)

        // Frame both ends of the run rather than only the ATM.
        const bounds = L.latLngBounds([[terminal.lat, terminal.lon],
                                       [active.lat, active.lon]])
        if (zone && Number.isFinite(zone.lat)) {
          bounds.extend(L.latLng(zone.lat, zone.lon))
        }
        map.fitBounds(bounds, { padding: [70, 70], maxZoom: 15, animate: true })
      } else {
        map.flyTo([active.lat, active.lon], 14, { duration: 0.8 })
      }
    }
  }, [terminal?.lat, terminal?.lon, terminal?.account, atms, zone, selectedAtmId])

  return <div ref={containerRef} className="h-full w-full" style={{ background: '#080a0a' }} />
}

/** Great-circle distance in km. */
/**
 * Is this ATM open at the hour the withdrawal is expected?
 *
 * The directory carries opening and closing times for every machine and the
 * console had never used them, so a location shut at the predicted hour ranked
 * beside one that was open and an officer could be sent to a closed branch.
 *
 * Returns null when the hours are unknown -- an absent answer, not a false one.
 */
function openAt(atm, minutesFromNow) {
  if (!atm?.opening_time || !atm?.closing_time) return null
  const at = new Date(Date.now() + (Number(minutesFromNow) || 0) * 60000)
  const mins = at.getHours() * 60 + at.getMinutes()
  const toMins = (hhmm) => {
    const [h, m] = String(hhmm).split(':').map(Number)
    return Number.isFinite(h) ? h * 60 + (m || 0) : null
  }
  const open = toMins(atm.opening_time)
  const close = toMins(atm.closing_time)
  if (open == null || close == null) return null
  // A window that ends before it starts runs past midnight.
  return close > open ? mins >= open && mins < close : mins >= open || mins < close
}

function haversineKm(aLat, aLon, bLat, bLon) {
  const R = 6371
  const toRad = (d) => (d * Math.PI) / 180
  const dLat = toRad(bLat - aLat)
  const dLon = toRad(bLon - aLon)
  const h = Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

export default function TacticalMap() {
  const { complaintId, resolving } = useActiveComplaint()
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedAtmId, setSelectedAtmId] = useState('')
  const [tilesOffline, setTilesOffline] = useState(false)

  useEffect(() => {
    if (resolving) return
    if (!complaintId) { setError('No complaint selected.'); return }

    let cancelled = false
    setLoading(true)
    setError('')

    endpoints.predictCashout(complaintId)
      .then(data => {
        if (cancelled) return
        setPrediction(data)
        setSelectedAtmId(data.ranked_candidates?.[0]?.atm_id || '')
      })
      .catch(err => {
        if (cancelled) return
        setPrediction(null)
        setError(describeError(err))
      })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [complaintId, resolving])

  const atms = useMemo(() => prediction?.ranked_candidates || [], [prediction])
  const activeAtm = atms.find(a => a.atm_id === selectedAtmId) || atms[0] || null
  const { label, remaining } = useCountdown(prediction?.time_to_cashout_minutes)

  const terminal = useMemo(() => ({
    lat: prediction?.terminal_lat,
    lon: prediction?.terminal_lon,
    account: prediction?.terminal_account,
  }), [prediction])

  const onTilesFailed = useCallback(() => setTilesOffline(true), [])

  const distanceKm = activeAtm && Number.isFinite(terminal.lat)
    ? haversineKm(terminal.lat, terminal.lon, activeAtm.lat, activeAtm.lon).toFixed(2)
    : '—'

  const urgencyTone = remaining < 300 ? 'bad' : remaining < 900 ? 'warn' : 'good'

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-8">
        <Panel
          title={`Cash-out locations — ${formatTicket(complaintId)}`}
          right={loading ? 'predicting…' : prediction ? `${label} remaining` : ''}
        >
          <div className="h-[64vh] relative">
            {error ? (
              <div className="absolute inset-0 grid place-items-center px-6 text-center text-[12.5px] z-[1200] bg-ink-bg">
                <div>
                  <ServerCrash size={26} className="text-red-400 mx-auto mb-2" />
                  <div className="text-red-300 font-bold">Prediction unavailable</div>
                  <div className="mt-1 text-zinc-500">{error}</div>
                </div>
              </div>
            ) : (
              <>
                <TacticalLeafletMap
                  terminal={terminal}
                  atms={atms}
                  zone={prediction?.search_zone}
                  selectedAtmId={selectedAtmId}
                  onSelectAtm={setSelectedAtmId}
                  onTilesFailed={onTilesFailed}
                />

                {loading && (
                  <div className="absolute inset-0 grid place-items-center bg-ink-bg/70 z-[1100] text-[12.5px] text-zinc-300">
                    <span className="flex items-center gap-2">
                      <Loader2 size={14} className="animate-spin" /> Ranking cash-out locations…
                    </span>
                  </div>
                )}

                <div className="absolute top-3 left-3 flex flex-wrap gap-2 text-[11.5px] z-[1000] max-w-[calc(100%-24px)]">
                  <span className="px-2.5 py-1 rounded bg-ink-panel border border-ink-border text-zinc-300">
                    ATM directory
                  </span>
                  {activeAtm && (
                    <span className="px-2.5 py-1 rounded bg-red-500/15 border border-red-500/40 text-red-300 font-medium">
                      Priority {activeAtm.rank}: {activeAtm.atm_id}
                    </span>
                  )}
                  {tilesOffline && (
                    <span className="px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/40 text-amber-300 font-medium flex items-center gap-1.5">
                      <CloudOff size={11} /> Basemap offline — positions still accurate
                    </span>
                  )}
                </div>

                <div className="absolute bottom-3 left-3 bg-ink-panel border border-ink-border px-3 py-1.5 rounded text-[11px] text-zinc-400 z-[1000] flex items-center gap-2">
                  <Navigation size={13} className="text-aegis-green" />
                  Click a pin or a card to retarget
                </div>
              </>
            )}
          </div>
        </Panel>
      </div>

      <div className="col-span-12 lg:col-span-4 space-y-3">
        <div className="aegis-panel p-3">
          <div className="flex items-center justify-between pb-2 border-b border-ink-border">
            <span className="text-[12px] font-medium text-zinc-300">
              Ranked locations
            </span>
            <span className="text-[11px] text-zinc-500">XGBoost v2</span>
          </div>

          <div className="space-y-2 mt-3">
            {atms.length === 0 && !loading && (
              <div className="text-[12.5px] text-zinc-500 py-4 text-center">No ranked locations for this case.</div>
            )}
            {atms.map(a => {
              const isSel = a.atm_id === activeAtm?.atm_id
              const d = Number.isFinite(terminal.lat)
                ? haversineKm(terminal.lat, terminal.lon, a.lat, a.lon).toFixed(1)
                : null
              const openNow = openAt(a, prediction?.time_to_cashout_minutes)
              return (
                <button
                  key={a.atm_id}
                  onClick={() => setSelectedAtmId(a.atm_id)}
                  className={`w-full text-left p-3 rounded border transition-all duration-150 ${
                    isSel
                      ? 'bg-ink-panel border-red-500/70'
                      : 'bg-ink-surface/60 border-ink-border hover:bg-ink-panel hover:border-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11.5px] gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`w-5 h-5 grid place-items-center rounded text-[10px] font-bold shrink-0 ${
                        isSel ? 'bg-red-500 text-white'
                          : a.rank === 1 ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                          : 'bg-ink-bg border border-ink-border text-zinc-400'
                      }`}>{a.rank}</span>
                      <span className="mono font-semibold text-zinc-100 truncate">{a.atm_id}</span>
                    </div>
                    <span className="mono tnum text-zinc-300 shrink-0">{(a.confidence * 100).toFixed(1)}%</span>
                  </div>

                  <div className="text-[11.5px] text-zinc-400 mt-1">{a.bank}</div>
                  <div className="text-[11.5px] text-zinc-300 mt-1 flex items-start gap-1">
                    <MapPin size={12} className="text-red-400 shrink-0 mt-0.5" />
                    <span className="line-clamp-2">{a.address}</span>
                  </div>

                  <div className="text-[11px] text-zinc-500 mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
                    <span className="mono tnum">{d != null ? `${d} km away` : '—'}</span>
                    <span>{a.historical_fraud_count} prior incidents</span>
                    {a.opening_time && (
                      <span className="mono tnum">{a.opening_time}–{a.closing_time}</span>
                    )}
                    {openNow === false && (
                      <span className="text-amber-400 border border-amber-500/40 bg-amber-500/10
                                       rounded px-1.5 py-0.5">
                        closed at the expected time
                      </span>
                    )}
                  </div>

                  <div className="mt-2 h-1.5 bg-ink-bg border border-ink-border rounded overflow-hidden">
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

          {prediction && (
            <>
              <div className="mt-3 pt-3 border-t border-ink-border space-y-1.5 text-[11.5px]">
                {[
                  ['Terminal account', prediction.terminal_account],
                  ['Selected target', activeAtm ? `${activeAtm.atm_id}` : '—'],
                  ['Mule → ATM', `${distanceKm} km`],
                  ['Stolen amount', amountFmt(prediction.stolen_amount)],
                  ['Inference time', `${prediction.inference_time_ms} ms`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2 text-zinc-400">
                    <span className="shrink-0">{k}:</span>
                    <span className="text-zinc-200 font-semibold truncate text-right">{v}</span>
                  </div>
                ))}
              </div>

              {prediction.search_zone && (
                <div className="mt-3 rounded border border-amber-500/40 bg-amber-500/10 p-2.5">
                  <div className="text-[11px] text-zinc-400">
                    Search zone
                  </div>
                  <div className="mono tnum text-[15px] font-semibold text-amber-200 mt-0.5">
                    {prediction.search_zone.radius_km.toFixed(2)} km radius
                  </div>
                  <div className="text-[11px] text-zinc-400 mt-0.5">
                    {prediction.search_zone.atm_count} ATM
                    {prediction.search_zone.atm_count === 1 ? '' : 's'} to cover ·
                    {' '}{(prediction.search_zone.probability_mass * 100).toFixed(0)}% probability mass
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2 mt-3">
                <Stat
                  label="Cashout in"
                  value={label}
                  tone={urgencyTone}
                  sub={
                    prediction.time_to_cashout_low != null
                      ? `${prediction.time_to_cashout_low.toFixed(0)}-${prediction.time_to_cashout_high.toFixed(0)} min band`
                      : undefined
                  }
                />
                <Stat
                  label="Rank-1 share"
                  value={`${((activeAtm?.confidence ?? prediction.interception_confidence) * 100).toFixed(1)}%`}
                  tone="good"
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
