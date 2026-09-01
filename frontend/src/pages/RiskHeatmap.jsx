import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {
  Flame, Loader2, ServerCrash, ShieldAlert, ChevronRight, ChevronDown,
  History, Radio,
} from 'lucide-react'
import { endpoints, describeError } from '../services/api'
import { Panel, Stat } from '../components/Shell'
import { useToast } from '../components/Toast'
import {
  MODEL_STATS, HOTSPOT_CURVE, FRAUD_TYPES,
  amountShort, amountFmt, formatTicket,
} from '../utils/constants'

const INDIA_CENTER = [22.9734, 78.6569]

/**
 * The intensity ramp.
 *
 * No green, at any band. #7cf000 (aegis-accent) means "healthy / live"
 * everywhere else in this console -- the websocket pip, the open-for-business
 * tags -- so a green hotspot would say the opposite of what it means. Zinc, not
 * green, is the bottom of the ramp: a quiet cell is uninteresting, not good.
 *
 * The critical band is the DARKEST red rather than the brightest, which is the
 * house grammar; it is given a light stroke so it still reads as the hottest
 * thing on a dark basemap instead of disappearing into it.
 */
const RAMP = [
  { key: 'critical', fill: '#7f1d1d', stroke: '#fca5a5', label: 'Critical' },
  { key: 'high',     fill: '#dc2626', stroke: '#fecaca', label: 'High' },
  { key: 'elevated', fill: '#f97316', stroke: '#fed7aa', label: 'Elevated' },
  { key: 'moderate', fill: '#f59e0b', stroke: '#fde68a', label: 'Moderate' },
  { key: 'low',      fill: '#52525b', stroke: '#a1a1aa', label: 'Low' },
]

const WINDOW_CHOICES = [
  { id: '0-30',   label: '0–30 min',     start: 0,  end: 30 },
  { id: '30-60',  label: '30–60 min',    start: 30, end: 60 },
  { id: '60-120', label: '60–120 min',   start: 60, end: 120 },
  { id: 'all',    label: 'Next 2 hours', start: 0,  end: 120 },
]

/**
 * Quintile cuts over the UNFILTERED national cell list.
 *
 * This is the whole reason the cuts are computed here rather than inside the map
 * from whatever it was handed. If the thresholds were recomputed over the cells
 * left after a state filter, the busiest cell in the quietest state would paint
 * critical-red purely because it is a local maximum, and an officer would read
 * national danger off a screen showing a quiet district. A colour has to mean
 * the same thing at every zoom level and under every filter, so it is anchored
 * to the national surface exactly once.
 *
 * (TriageFeed.jsx:504 makes the same argument for its risk bands.)
 */
function nationalCuts(cells) {
  const scores = cells.map(c => c.score).filter(Number.isFinite).sort((a, b) => a - b)
  if (!scores.length) return []
  const at = q => scores[Math.min(scores.length - 1, Math.floor(q * scores.length))]
  return [at(0.95), at(0.85), at(0.70), at(0.50)]
}

function bandFor(score, cuts) {
  if (!cuts.length) return RAMP[4]
  for (let i = 0; i < cuts.length; i++) if (score >= cuts[i]) return RAMP[i]
  return RAMP[4]
}

/** Area proportional to intensity, clamped so one huge cell cannot eat the map. */
function radiusFor(score) {
  // Area proportional to forecast rupees, on an ABSOLUTE scale rather than one
  // normalised to whatever is currently on screen.
  //
  // The adaptive version looked better in isolation -- the biggest marker always
  // filled the same space -- and that was the problem. Drilling from national
  // into one state re-normalised every radius, so a cell that had just been a
  // small dot became a large one without its forecast changing. An operator
  // reading size as severity would have been misled by the act of zooming in.
  // Absolute sizing spans 5-26 px across roughly Rs 4k to Rs 1M, so a marker
  // means the same thing at every scope.
  return Math.max(5, Math.min(26, Math.sqrt(Math.max(0, score)) * 0.08))
}

/** Roll cells up to one marker per group, at the score-weighted centroid. */
function rollUp(cells, key) {
  const groups = new Map()
  cells.forEach(c => {
    const name = c[key] || '—'
    if (!groups.has(name)) {
      groups.set(name, {
        name, score: 0, conditional_rupees: 0, prior_rupees: 0,
        atm_count: 0, cellIds: [], caseIds: new Set(), wLat: 0, wLon: 0,
      })
    }
    const g = groups.get(name)
    const w = Math.max(c.score, 0)
    g.score += c.score
    g.conditional_rupees += c.conditional_rupees
    g.prior_rupees += c.prior_rupees
    g.atm_count += c.atm_count
    g.cellIds.push(c.cell_id)
    ;(c.complaint_ids || []).forEach(id => g.caseIds.add(id))
    g.wLat += c.lat * w
    g.wLon += c.lon * w
  })
  return [...groups.values()].map(g => ({
    ...g,
    // Score-weighted, so a state's marker sits on its risk rather than on its
    // geographic middle. Falls back to nothing drawable when every cell is 0.
    lat: g.score > 0 ? g.wLat / g.score : 0,
    lon: g.score > 0 ? g.wLon / g.score : 0,
    case_count: g.caseIds.size,
    prior_share: (g.conditional_rupees + g.prior_rupees) > 0
      ? g.prior_rupees / (g.conditional_rupees + g.prior_rupees) : 1,
  })).sort((a, b) => b.score - a.score)
}

// ── Map ────────────────────────────────────────────────────────────────────

function HotspotMap({ points, cuts, onSelect, focusKey }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layerRef = useRef(null)
  const selectRef = useRef(onSelect)

  useEffect(() => { selectRef.current = onSelect }, [onSelect])

  // ── Create the map exactly once ──────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: INDIA_CENTER, zoom: 5, zoomControl: true, attributionControl: false,
    })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, crossOrigin: true,
    }).addTo(map)

    layerRef.current = L.layerGroup().addTo(map)
    mapRef.current = map

    // Leaflet mis-sizes itself inside a flex/grid parent that is still settling,
    // which is what leaves a half-drawn grey map on first paint.
    const ro = new ResizeObserver(() => map.invalidateSize())
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      map.remove()
      mapRef.current = null
      layerRef.current = null
    }
  }, [])

  // ── Redraw overlays; the map instance itself is never rebuilt ───────────
  useEffect(() => {
    const map = mapRef.current
    const layer = layerRef.current
    if (!map || !layer) return

    layer.clearLayers()

    points.forEach(p => {
      if (!Number.isFinite(p.lat) || !Number.isFinite(p.lon)) return
      if (p.lat === 0 && p.lon === 0) return
      const band = bandFor(p.score, cuts)
      const live = Math.round((1 - (p.prior_share || 0)) * 100)

      // circleMarker, not a heat plugin: it is a vector in the overlay pane, and
      // index.css inverts the TILE pane only. A canvas heat layer would be
      // inverted along with the basemap and come out cyan.
      L.circleMarker([p.lat, p.lon], {
        radius: radiusFor(p.score),
        fillColor: band.fill,
        color: band.stroke,
        weight: band.key === 'critical' ? 2 : 1,
        opacity: 0.9,
        fillOpacity: 0.55,
      })
        .bindPopup(
          `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px;line-height:1.55">
            <b style="color:#b91c1c">${p.name || p.cell_id}</b> — ${band.label}<br/>
            <b>${amountShort(p.score)}</b> forecast · ${p.atm_count} ATM(s)<br/>
            <span style="color:#374151">${live}% from ${p.case_count} open case(s)
              · ${100 - live}% historical prior</span>
          </div>`
        )
        // Hover reads, click commits. Scanning a national surface by clicking
        // every marker is not scanning -- the popup steals focus, recentres the
        // map and has to be dismissed. A tooltip lets an officer sweep the map
        // and read forecast, district and case count without changing anything.
        .bindTooltip(
          `<div style="font-family:Inter,system-ui,sans-serif;font-size:11.5px;line-height:1.5">
             <b>${p.name || p.cell_id}</b><br/>
             <span style="opacity:.75">${p.district || p.state || '—'}</span><br/>
             <b>${amountShort(p.score)}</b> forecast · ${p.case_count} case(s)<br/>
             <span style="opacity:.75">${live}% live · ${p.atm_count} ATM(s)</span>
           </div>`,
          { direction: 'top', offset: [0, -4], opacity: 1, className: 'hotspot-tip' }
        )
        .on('click', () => selectRef.current?.(p))
        .addTo(layer)
    })
  }, [points, cuts])

  // ── Frame the current scope, without touching the instance ─────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const usable = points.filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon)
                                      && !(p.lat === 0 && p.lon === 0))
    if (!usable.length) return
    if (usable.length === 1) {
      map.flyTo([usable[0].lat, usable[0].lon], 11, { duration: 0.7 })
      return
    }
    map.fitBounds(L.latLngBounds(usable.map(p => [p.lat, p.lon])),
                  { padding: [50, 50], maxZoom: 12, animate: true })
  }, [focusKey]) // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={containerRef} className="h-full w-full" style={{ background: '#080a0a' }} />
}

// ── Decomposition ──────────────────────────────────────────────────────────

/**
 * The two-segment bar. This is the claim the whole screen exists to make, so it
 * is rendered as a proportion rather than asserted in prose: a reader can see
 * that the forecast is driven by cases filed minutes ago and not by a density
 * map of where crime happened last year.
 *
 * Plain divs with a width style, following ModelPerformance.jsx -- no charting
 * library is added for one stacked bar.
 */
function DecompositionBar({ conditional, prior }) {
  const total = conditional + prior
  const condPct = total > 0 ? (conditional / total) * 100 : 0
  const priorPct = 100 - condPct
  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded bg-ink-bg border border-ink-border">
        <div style={{ width: `${condPct}%` }} className="bg-red-500" title="Open cases" />
        <div style={{ width: `${priorPct}%` }} className="bg-zinc-700" title="Historical prior" />
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[11px]">
        <span className="text-red-300">
          <Radio size={10} className="inline mb-px mr-1" />
          <span className="mono tnum font-semibold">{condPct.toFixed(0)}%</span> from open cases
        </span>
        <span className="text-zinc-500">
          <History size={10} className="inline mb-px mr-1" />
          <span className="mono tnum">{priorPct.toFixed(0)}%</span> historical prior
        </span>
      </div>
    </div>
  )
}

// ── Screen ─────────────────────────────────────────────────────────────────

export default function RiskHeatmap() {
  const toast = useToast()

  const [surface, setSurface] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [windowId, setWindowId] = useState('all')
  const [asOf, setAsOf] = useState('')
  const [categories, setCategories] = useState([])

  const [scopeState, setScopeState] = useState('')
  const [scopeDistrict, setScopeDistrict] = useState('')
  const [selected, setSelected] = useState(null)

  const [atms, setAtms] = useState([])
  const [showCurve, setShowCurve] = useState(false)
  const [raising, setRaising] = useState(false)

  const win = WINDOW_CHOICES.find(w => w.id === windowId) || WINDOW_CHOICES[3]

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    endpoints.listHotspots({
      windowStartMin: win.start,
      windowEndMin: win.end,
      fraudTypes: categories,
      // datetime-local yields 'YYYY-MM-DDTHH:mm', which the backend's tolerant
      // parse_ts accepts unchanged.
      asOf: asOf || undefined,
    })
      .then(data => { if (!cancelled) { setSurface(data); setSelected(null) } })
      .catch(err => { if (!cancelled) setError(describeError(err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [win.start, win.end, categories, asOf])

  useEffect(() => {
    endpoints.listAtmIntel(12).then(setAtms).catch(() => setAtms([]))
  }, [])

  const allCells = surface?.cells || []

  // Anchored to the national surface, before any scope filter. See nationalCuts.
  const cuts = useMemo(() => nationalCuts(allCells), [allCells])

  const scoped = useMemo(() => allCells.filter(c =>
    (!scopeState || c.state === scopeState) &&
    (!scopeDistrict || c.district === scopeDistrict)
  ), [allCells, scopeState, scopeDistrict])

  // National view rolls up to states; inside a state the cells themselves are
  // the deployable unit, so they are drawn individually.
  const points = useMemo(() => {
    if (!scopeState) return rollUp(scoped, 'state')
    if (!scopeDistrict) return rollUp(scoped, 'district')
    return scoped.map(c => ({ ...c, name: c.cell_id }))
  }, [scoped, scopeState, scopeDistrict])

  const focusKey = `${scopeState}|${scopeDistrict}|${windowId}|${allCells.length}`

  const handleSelect = useCallback(point => {
    if (point.cell_id) { setSelected(point); return }
    // A roll-up marker: descend one level rather than selecting it.
    if (!scopeState) setScopeState(point.name)
    else setScopeDistrict(point.name)
  }, [scopeState])

  const toggleCategory = (t) =>
    setCategories(cur => cur.includes(t) ? cur.filter(x => x !== t) : [...cur, t])

  async function raiseForReview() {
    if (!selected) return
    setRaising(true)
    try {
      await endpoints.raiseAlert({
        cell_id: selected.cell_id,
        window_start_min: win.start,
        window_end_min: win.end,
        score: selected.score,
        rupees_at_risk: selected.conditional_rupees + selected.prior_rupees,
        prior_share: selected.prior_share,
        complaint_ids: selected.complaint_ids || [],
      })
      toast('Raised for review. It is now open in the alert queue.')
    } catch (err) {
      // 404 means the alert service is not deployed on this build, which is a
      // different thing from a failure -- reporting "raised" here would be a
      // claim the officer would then act on.
      if (err?.response?.status === 404) {
        toast('Alert queue is not enabled on this build yet.', 'error')
      } else {
        toast(describeError(err), 'error')
      }
    } finally {
      setRaising(false)
    }
  }

  const districts = useMemo(() => [...new Set(
    allCells.filter(c => c.state === scopeState).map(c => c.district)
  )].sort(), [allCells, scopeState])

  return (
    <div className="p-3 space-y-3 bg-ink-bg">

      {/* ── Page header ─────────────────────────────────────────────────
          Names the thing precisely. "Forecast", not "heatmap": a heatmap is a
          picture of what has already happened, and this screen exists because
          that picture already exists elsewhere. The window is stated in the
          subtitle because a forecast without a horizon is not actionable. */}
      <div className="flex items-baseline justify-between flex-wrap gap-x-3 gap-y-1">
        <h1 className="text-[15px] font-semibold text-zinc-100 flex items-center gap-2">
          <Flame size={15} className="text-red-400" />
          Tactical Risk Forecast
        </h1>
        <p className="text-[11.5px] text-zinc-500">
          Forward cash-out intensity aggregated over open complaints in the
          golden hour <span className="mono tnum text-zinc-400">(0–120 min)</span>
        </p>
      </div>

      {/* Governance. The screen forecasts; it does not act. */}
      <div className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2
                      text-[11.5px] text-amber-200 flex items-start gap-2">
        <ShieldAlert size={14} className="mt-px shrink-0" />
        <span>
          <span className="font-semibold">Forecast only.</span> No unit is dispatched and no
          account is frozen without an officer acknowledging an alert.
        </span>
      </div>

      {surface?.degraded && (
        <div className="rounded border border-zinc-600 bg-ink-panel px-3 py-2
                        text-[11.5px] text-zinc-300">
          <span className="font-semibold text-zinc-100">No live cases in this window.</span>{' '}
          What is drawn below is historical density only — it is not a forecast. Move the
          <span className="mono text-zinc-200"> as of</span> control to a time when complaints
          were arriving.
        </div>
      )}

      {/* ── Model card ─────────────────────────────────────────────────── */}
      <Panel
        title="Forward cash-out forecast — held-out performance"
        right={
          <button onClick={() => setShowCurve(v => !v)}
                  className="hover:text-zinc-300 transition-colors">
            {showCurve ? <ChevronDown size={13} className="inline" />
                       : <ChevronRight size={13} className="inline" />} precision curve
          </button>
        }
        bodyClass="p-2.5"
      >
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          <Stat label="Hit rate @5 cells" value={MODEL_STATS.hotspotHitRate} tone="info" />
          <Stat label="PAI @5" value={MODEL_STATS.hotspotPai} tone="info"
                sub={`vs ${MODEL_STATS.hotspotBaselinePai} historical density`} />
          <Stat label="Advantage over density" value={MODEL_STATS.hotspotAdvantage} tone="good"
                sub="the density-map baseline" />
          <Stat label="Median lead time" value={MODEL_STATS.hotspotLeadTime}
                sub={`${MODEL_STATS.hotspotActionable} actionable`} />
          <Stat label="Stolen ₹ covered @5" value={MODEL_STATS.hotspotRupeesCovered}
                sub={`${MODEL_STATS.hotspotAtmShare} of ATMs flagged`} />
          {/* The comparison we lose, on screen rather than in a footnote -- the
              same rule constants.js already applies to top-5 containment. */}
          <Stat label="Nearest-cell hit rate" value={MODEL_STATS.hotspotNearestCellHitRate}
                tone="warn" sub="distance alone beats us on hit rate" />
        </div>

        {showCurve && (
          <div className="mt-2.5 overflow-x-auto">
            <table className="data-table mono tnum">
              <thead>
                <tr>
                  <th>k cells</th>
                  <th className="text-right">Coverage</th>
                  <th className="text-right">Precision</th>
                  <th className="text-right">False cells / hit</th>
                  <th className="text-right">PAI</th>
                  <th className="text-right">₹ covered</th>
                  <th className="text-right">ATMs flagged</th>
                </tr>
              </thead>
              <tbody>
                {HOTSPOT_CURVE.filter(r => [1, 3, 5, 10].includes(r.k)).map(r => (
                  <tr key={r.k} className="cursor-default">
                    <td className="text-zinc-100 font-semibold">{r.k}</td>
                    <td className="text-right">{r.coverage}</td>
                    <td className="text-right">{r.precision}</td>
                    <td className="text-right text-amber-400">{r.falseCellsPerHit}</td>
                    <td className="text-right">{r.pai}</td>
                    <td className="text-right">{r.rupeesCovered}</td>
                    <td className="text-right">{r.atmShare}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[10.5px] text-zinc-500 mt-1.5 px-2">
              {MODEL_STATS.hotspotNTestCashouts} held-out cash-outs ·
              {' '}{MODEL_STATS.hotspotCells} cells at {surface?.cell_radius_km ?? 12} km ·
              prior capped at {MODEL_STATS.hotspotPriorWeight}
            </div>
          </div>
        )}
      </Panel>

      {/* ── The three drill-down axes ──────────────────────────────────── */}
      <Panel title="Scope" bodyClass="p-2.5 space-y-2.5 bg-ink-surface">
        {/* Axis 1: location */}
        <div className="flex items-center gap-1.5 text-[12px] flex-wrap">
          <button onClick={() => { setScopeState(''); setScopeDistrict(''); setSelected(null) }}
                  className={`px-2 py-1 rounded border transition-colors ${
                    scopeState ? 'border-ink-border text-zinc-400 hover:text-zinc-100'
                               : 'border-aegis-accent/50 bg-ink-panel text-zinc-100'}`}>
            National
          </button>
          {scopeState && <>
            <ChevronRight size={12} className="text-zinc-600" />
            <button onClick={() => { setScopeDistrict(''); setSelected(null) }}
                    className={`px-2 py-1 rounded border transition-colors ${
                      scopeDistrict ? 'border-ink-border text-zinc-400 hover:text-zinc-100'
                                    : 'border-aegis-accent/50 bg-ink-panel text-zinc-100'}`}>
              {scopeState}
            </button>
          </>}
          {scopeDistrict && <>
            <ChevronRight size={12} className="text-zinc-600" />
            <button onClick={() => setSelected(null)}
                    className={`px-2 py-1 rounded border transition-colors ${
                      selected ? 'border-ink-border text-zinc-400 hover:text-zinc-100'
                               : 'border-aegis-accent/50 bg-ink-panel text-zinc-100'}`}>
              {scopeDistrict}
            </button>
          </>}
          {selected && <>
            <ChevronRight size={12} className="text-zinc-600" />
            <span className="px-2 py-1 rounded border border-red-500/50 bg-ink-panel
                             text-zinc-100 mono text-[11px]">{selected.cell_id}</span>
          </>}

          {scopeState && !scopeDistrict && districts.length > 0 && (
            <select value={scopeDistrict}
                    onChange={e => { setScopeDistrict(e.target.value); setSelected(null) }}
                    className="ml-auto bg-ink-panel border border-ink-border rounded
                               px-2 py-1 text-[11px] text-zinc-300">
              <option value="">Jump to district…</option>
              {districts.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          )}
        </div>

        {/* Axis 2: time */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-zinc-500 w-16 shrink-0">Window</span>
          {WINDOW_CHOICES.map(w => (
            <button key={w.id} onClick={() => setWindowId(w.id)}
                    className={`px-2.5 py-1 rounded border mono text-[11px] transition-colors ${
                      windowId === w.id
                        ? 'border-ink-border bg-ink-panel text-zinc-100 font-semibold'
                        : 'border-ink-border2 bg-ink-surface text-zinc-400 hover:text-zinc-100'}`}>
              {w.label}
            </button>
          ))}
          <span className="text-[11px] text-zinc-500 ml-2">As of</span>
          {/* Not cosmetic: the corpus is generated ahead of a demo, so "now" can
              legitimately hold no open complaints. The response always echoes the
              timestamp it actually used, printed beside the map title. */}
          <input type="datetime-local" value={asOf} onChange={e => setAsOf(e.target.value)}
                 className="bg-ink-panel border border-ink-border rounded px-2 py-1
                            text-[11px] text-zinc-300 mono" />
          {asOf && (
            <button onClick={() => setAsOf('')}
                    className="text-[11px] text-zinc-500 hover:text-zinc-100">clear</button>
          )}
        </div>

        {/* Axis 3: crime category */}
        <div className="flex items-start gap-2 flex-wrap">
          <span className="text-[11px] text-zinc-500 w-16 shrink-0 pt-1">Category</span>
          <div className="flex gap-1.5 flex-wrap flex-1">
            {FRAUD_TYPES.map(t => (
              <button key={t} onClick={() => toggleCategory(t)}
                      className={`px-2 py-0.5 rounded-full border text-[11px] transition-colors ${
                        categories.includes(t)
                          ? 'border-red-500/60 bg-red-500/15 text-red-200'
                          : 'border-ink-border bg-ink-panel text-zinc-400 hover:text-zinc-100'}`}>
                {t}
              </button>
            ))}
            {categories.length > 0 && (
              <button onClick={() => setCategories([])}
                      className="px-2 py-0.5 text-[11px] text-zinc-500 hover:text-zinc-100">
                all categories
              </button>
            )}
          </div>
        </div>
      </Panel>

      {/* ── Map + decomposition ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <Panel
          className="lg:col-span-2"
          title={scopeDistrict ? `${scopeDistrict} — ${surface?.cell_radius_km ?? 12} km cells`
                : scopeState ? `${scopeState} — districts`
                : 'National forward surface'}
          right={surface ? `${points.length} shown · as of ${String(surface.as_of).replace('T', ' ')}` : ''}
          bodyClass="p-0"
        >
          <div className="h-[460px] relative">
            {loading && (
              <div className="absolute inset-0 z-[500] grid place-items-center bg-ink-bg/70">
                <Loader2 size={18} className="animate-spin text-zinc-400" />
              </div>
            )}
            {error ? (
              <div className="h-full grid place-items-center text-center px-6">
                <div>
                  <ServerCrash size={20} className="text-red-400 mx-auto" />
                  <div className="text-[12px] text-zinc-300 mt-2">{error}</div>
                </div>
              </div>
            ) : (
              <HotspotMap points={points} cuts={cuts}
                          onSelect={handleSelect} focusKey={focusKey} />
            )}
          </div>
          <div className="flex items-center gap-3 px-2.5 py-1.5 border-t border-ink-border
                          text-[10.5px] text-zinc-500 flex-wrap">
            <span>Intensity</span>
            {RAMP.map(b => (
              <span key={b.key} className="flex items-center gap-1">
                <span style={{ background: b.fill, borderColor: b.stroke }}
                      className="inline-block w-2.5 h-2.5 rounded-full border" />
                {b.label}
              </span>
            ))}
            <span className="ml-auto">
              Bands are national quintiles — a colour means the same thing at every zoom.
            </span>
          </div>
        </Panel>

        <Panel title={selected ? 'Cell decomposition' : 'Ranked'} bodyClass="p-2.5">
          {!selected ? (
            <div className="space-y-1">
              <div className="text-[11px] text-zinc-500 mb-1.5">
                {scopeDistrict ? 'Select a cell on the map.' : 'Select to drill down.'}
              </div>
              {points.slice(0, 12).map(p => {
                const band = bandFor(p.score, cuts)
                return (
                  <button key={p.cell_id || p.name} onClick={() => handleSelect(p)}
                          className="w-full flex items-center gap-2 px-2 py-1.5 rounded
                                     border border-ink-border bg-ink-panel
                                     hover:border-zinc-600 transition-colors text-left">
                    <span style={{ background: band.fill, borderColor: band.stroke }}
                          className="w-2 h-2 rounded-full border shrink-0" />
                    <span className="text-[11.5px] text-zinc-200 truncate flex-1">
                      {p.name || p.cell_id}
                    </span>
                    <span className="mono tnum text-[11px] text-zinc-100">
                      {amountShort(p.score)}
                    </span>
                  </button>
                )
              })}
              {!points.length && !loading && (
                <div className="text-[11.5px] text-zinc-500">No cells in this scope.</div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="mono tnum text-[19px] font-semibold text-zinc-100">
                  {amountFmt(Math.round(selected.score))}
                </div>
                <div className="text-[11px] text-zinc-500">
                  forecast in {win.label.toLowerCase()} · {selected.cell_id}
                  {selected.city ? `, ${selected.city}` : ''}
                </div>
              </div>

              <DecompositionBar conditional={selected.conditional_rupees}
                                prior={selected.prior_rupees} />

              <div className="grid grid-cols-3 gap-2">
                <Stat label="Open cases" value={selected.case_count} tone="info" />
                <Stat label="ATMs in cell" value={selected.atm_count} />
                <Stat label="Prior share"
                      value={`${Math.round((selected.prior_share || 0) * 100)}%`}
                      tone={selected.prior_share > 0.15 ? 'warn' : 'default'} />
              </div>

              <div>
                <div className="text-[11px] text-zinc-500 mb-1">Cases driving this cell</div>
                <div className="space-y-1">
                  {(selected.complaint_ids || []).slice(0, 8).map(id => (
                    <Link key={id} to={`/?c=${encodeURIComponent(id)}`}
                          className="block px-2 py-1 rounded border border-ink-border
                                     bg-ink-panel mono text-[11px] text-zinc-300
                                     hover:text-zinc-100 hover:border-zinc-600 transition-colors">
                      {formatTicket(id)}
                    </Link>
                  ))}
                  {!(selected.complaint_ids || []).length && (
                    <div className="text-[11px] text-zinc-500">
                      No open case contributes here — this cell is historical prior only.
                    </div>
                  )}
                </div>
              </div>

              {/* The ONLY action on this screen. No freeze, no dispatch. */}
              <button onClick={raiseForReview} disabled={raising}
                      className="w-full bg-ink-panel border border-ink-border
                                 hover:border-zinc-500 text-zinc-200 text-[12px]
                                 font-medium py-1.5 px-3 rounded-lg
                                 disabled:opacity-50 transition-colors">
                {raising ? 'Raising…' : 'Raise for review'}
              </button>
              <div className="text-[10.5px] text-zinc-500">
                Opens an alert for an officer to acknowledge. It does not freeze an account
                or task a unit.
              </div>
            </div>
          )}
        </Panel>
      </div>

      {/* ── Recurring machines ─────────────────────────────────────────── */}
      <Panel title="Recurring cash-out machines" right="all-time, across cases" bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>ATM</th>
                <th>Bank</th>
                <th>District</th>
                <th className="text-right">Repeat incidents</th>
                <th className="text-right">Distinct cases</th>
                <th className="text-right">Total volume</th>
                <th className="text-right">Last seen</th>
              </tr>
            </thead>
            <tbody>
              {atms.map(a => (
                <tr key={a.atm_id} className="cursor-default">
                  <td className="mono text-zinc-100">{a.atm_id}</td>
                  <td>{a.bank_name || '—'}</td>
                  <td>{[a.district, a.state].filter(Boolean).join(', ') || '—'}</td>
                  <td className="text-right mono tnum">{a.cashouts}</td>
                  <td className="text-right mono tnum">{a.distinct_complaints}</td>
                  <td className="text-right mono tnum">{amountShort(a.total_amount)}</td>
                  <td className="text-right mono text-zinc-500">
                    {String(a.last_seen || '—').slice(0, 16).replace('T', ' ')}
                  </td>
                </tr>
              ))}
              {!atms.length && (
                <tr><td colSpan={7} className="text-zinc-500">
                  No recurring machines loaded.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="text-[10.5px] text-zinc-500 px-1 pb-2 flex items-center gap-1.5">
        <Flame size={11} />
        Cells are {surface?.cell_radius_km ?? 12} km clusters of cash-out points, not districts.
        The historical prior is capped at {MODEL_STATS.hotspotPriorWeight} of the surface and
        reported per cell, so a density map cannot be read as a forecast.
      </div>
    </div>
  )
}
