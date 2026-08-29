import { useEffect, useState, useMemo, useCallback } from 'react'
import ReactFlow, { Background, Controls, MiniMap, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'
import { endpoints, describeError } from '../services/api'
import useActiveComplaint from '../hooks/useActiveComplaint'
import { Panel } from '../components/Shell'
import { amountFmt, formatTicket, shortAccount } from '../utils/constants'
import { AlertTriangle, Split, Target, Loader2, ServerCrash, Activity } from 'lucide-react'

const NODE_COLOR = { victim: '#58a6ff', mule: '#ff8c42', terminal: '#ff3b3b' }
const NODE_FILL = {
  victim: 'rgba(88,166,255,0.12)',
  mule: 'rgba(255,140,66,0.10)',
  terminal: 'rgba(255,59,59,0.14)',
}

/** Colour ramp for a 0–1 GNN mule probability. */
function riskColor(r) {
  if (r >= 0.85) return '#ff3b3b'
  if (r >= 0.5) return '#ff8c42'
  if (r >= 0.2) return '#e8c547'
  return '#7cf000'
}

/**
 * Lay the DAG out left-to-right, one column per hop depth. Flagged accounts get
 * a coloured ring so velocity / fund-splitting detections are visible on the
 * canvas itself, not just in the side panel.
 */
function buildLayout(nodes = [], edges = [], anomalies = {}) {
  const velocity = new Set(anomalies.velocity_flagged || [])
  const split = new Set(anomalies.fund_split_flagged || [])

  const byDepth = new Map()
  nodes.forEach(n => {
    const d = n.hop_depth ?? 0
    if (!byDepth.has(d)) byDepth.set(d, [])
    byDepth.get(d).push(n)
  })

  const COL_W = 230
  const ROW_H = 96
  const flowNodes = []

  const depths = [...byDepth.keys()].sort((a, b) => a - b)
  const tallest = Math.max(...depths.map(d => byDepth.get(d).length), 1)

  depths.forEach((depth, col) => {
    const arr = byDepth.get(depth)
    // Centre each column vertically against the tallest one.
    const offset = ((tallest - arr.length) * ROW_H) / 2
    arr.forEach((n, i) => {
      const type = n.node_type || 'mule'
      const color = NODE_COLOR[type] || NODE_COLOR.mule
      const risk = Number(n.risk_score) || 0
      const flagged = velocity.has(n.id) || split.has(n.id)
      const header = type === 'victim' ? 'VICTIM' : type === 'terminal' ? 'TERMINAL' : `HOP-${depth}`

      flowNodes.push({
        id: n.id,
        position: { x: 40 + col * COL_W, y: 40 + offset + i * ROW_H },
        data: {
          label: `${header} · ${n.bank || '—'}\n${shortAccount(n.id)}${
            type === 'victim' ? '' : `\nGNN risk ${(risk * 100).toFixed(1)}%`
          }`,
        },
        style: {
          background: NODE_FILL[type] || NODE_FILL.mule,
          border: `1px solid ${color}88`,
          outline: flagged ? '2px solid #ffd23f' : 'none',
          outlineOffset: '2px',
          color: '#e6edf3',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '10px',
          lineHeight: 1.5,
          whiteSpace: 'pre-line',
          textAlign: 'left',
          borderRadius: '8px',
          padding: '8px 10px',
          width: 178,
          boxShadow: 'none',
        },
      })
    })
  })

  const flowEdges = (edges || []).map((e, i) => ({
    id: e.id || `edge-${i}`,
    source: e.source,
    target: e.target,
    label: e.label || amountFmt(e.amount),
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#4b5563', width: 16, height: 16 },
    style: { stroke: '#3a4242', strokeWidth: 1.5 },
    labelStyle: { fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fill: '#9ca3af' },
    labelBgStyle: { fill: '#0f1111', fillOpacity: 0.95 },
    labelBgPadding: [4, 2],
    labelBgBorderRadius: 3,
  }))

  return { flowNodes, flowEdges }
}

export default function ForensicGraph() {
  const { complaintId, resolving } = useActiveComplaint()
  const [data, setData] = useState(null)
  const [selected, setSelected] = useState(null)
  const [topMules, setTopMules] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (resolving) return
    if (!complaintId) { setError('No complaint selected.'); return }

    let cancelled = false
    setLoading(true)
    setError('')

    Promise.allSettled([
      endpoints.getGraph(complaintId),
      endpoints.getEmbeddings(complaintId, 5),
    ]).then(([graphRes, embRes]) => {
      if (cancelled) return

      if (graphRes.status === 'fulfilled') {
        const d = graphRes.value
        setData(d)
        const terminal = d.nodes?.find(n => n.node_type === 'terminal')
        setSelected(terminal || d.nodes?.[0] || null)
      } else {
        setData(null)
        setError(describeError(graphRes.reason))
      }

      setTopMules(embRes.status === 'fulfilled' ? (embRes.value.top_mules || []) : [])
      setLoading(false)
    })

    return () => { cancelled = true }
  }, [complaintId, resolving])

  const anomalies = data?.anomalies || {}
  const { flowNodes, flowEdges } = useMemo(
    () => (data ? buildLayout(data.nodes, data.edges, anomalies) : { flowNodes: [], flowEdges: [] }),
    [data, anomalies]
  )

  const onNodeClick = useCallback((_, node) => {
    const found = data?.nodes?.find(n => n.id === node.id)
    if (found) setSelected(found)
  }, [data])

  const selectedFlags = useMemo(() => {
    if (!selected) return []
    const out = []
    if ((anomalies.velocity_flagged || []).includes(selected.id)) out.push('VELOCITY')
    if ((anomalies.fund_split_flagged || []).includes(selected.id)) out.push('FUND SPLIT')
    if ((anomalies.terminal_leaves || []).includes(selected.id)) out.push('TERMINAL LEAF')
    return out
  }, [selected, anomalies])

  const risk = Number(selected?.risk_score) || 0

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-9">
        <Panel
          title={`Transaction trail — ${formatTicket(complaintId)}`}
          right={
            loading
              ? 'building…'
              : data
              ? `${data.node_count} nodes · ${data.edge_count} edges · ${data.build_time_ms} ms`
              : ''
          }
        >
          <div className="h-[64vh] bg-ink-bg relative">
            {loading && (
              <div className="absolute inset-0 grid place-items-center z-10 bg-ink-bg/70 mono text-[12px] text-zinc-400">
                <span className="flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Building transaction trail…</span>
              </div>
            )}

            {!loading && error && (
              <div className="absolute inset-0 grid place-items-center px-6 text-center mono text-[12px] text-zinc-400">
                <div>
                  <ServerCrash size={26} className="text-red-400 mx-auto mb-2" />
                  <div className="text-red-300 font-bold">Graph unavailable</div>
                  <div className="mt-1 text-zinc-500">{error}</div>
                  <div className="mt-1 text-zinc-600">
                    Complaints from the historical dataset always carry a ledger; a live complaint
                    gets one at ingestion.
                  </div>
                </div>
              </div>
            )}

            {!error && (
              <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                onNodeClick={onNodeClick}
                fitView
                fitViewOptions={{ padding: 0.18 }}
                minZoom={0.2}
                proOptions={{ hideAttribution: true }}
              >
                <Background gap={16} size={1} color="#1e2323" />
                <Controls showInteractive={false} />
                <MiniMap
                  pannable
                  zoomable
                  style={{ background: '#0f1111', border: '1px solid #1e2323' }}
                  maskColor="rgba(8,10,10,0.7)"
                  nodeColor={n => NODE_COLOR[data?.nodes?.find(x => x.id === n.id)?.node_type] || '#ff8c42'}
                />
              </ReactFlow>
            )}
          </div>

          <div className="px-3 py-2 flex flex-wrap items-center gap-x-3 gap-y-1 mono text-[11px] border-t border-ink-border bg-ink-surface/50">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded" style={{ background: NODE_COLOR.victim }} /> Victim</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded" style={{ background: NODE_COLOR.mule }} /> Layering mule</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded" style={{ background: NODE_COLOR.terminal }} /> Terminal cashout</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded border-2 border-[#ffd23f]" /> Anomaly flagged</span>
            <span className="ml-auto text-zinc-500">Select an account to inspect</span>
          </div>
        </Panel>
      </div>

      <div className="col-span-12 lg:col-span-3 space-y-3">
        {/* ── Node inspector ─────────────────────────────────────────────── */}
        <div className="aegis-panel p-3">
          <div className="mono text-[11px] tracking-[0.14em] text-zinc-400 font-semibold uppercase">Account detail</div>
          {!selected ? (
            <div className="mono text-[12px] text-zinc-500 mt-3">Select a node on the graph.</div>
          ) : (
            <div className="mt-3 space-y-2 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border rounded px-2.5 py-2">
                <div className="text-zinc-500">Account</div>
                <div className="text-white font-bold break-all">{selected.id}</div>
                <div className="text-zinc-400 mt-1">
                  {selected.bank} · Hop {selected.hop_depth} ·{' '}
                  <span className="uppercase" style={{ color: NODE_COLOR[selected.node_type] }}>
                    {selected.node_type}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="bg-ink-panel border border-ink-border rounded px-2.5 py-2">
                  <div className="text-zinc-500">Amount received</div>
                  <div className="text-white font-semibold">{amountFmt(selected.amount)}</div>
                </div>
                <div className="bg-ink-panel border border-ink-border rounded px-2.5 py-2">
                  <div className="text-zinc-500">Mule probability</div>
                  <div className="font-bold" style={{ color: riskColor(risk) }}>
                    {(risk * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="h-1.5 bg-ink-bg border border-ink-border rounded overflow-hidden">
                <span
                  className="block h-full transition-all duration-300"
                  style={{ width: `${Math.round(risk * 100)}%`, background: riskColor(risk) }}
                />
              </div>
              <div className="text-[10px] text-zinc-600 leading-relaxed">
                How likely this account is a money mule, scored from who it moves money
                with rather than its own activity alone.
              </div>

              <div className="bg-ink-panel border border-ink-border rounded px-2.5 py-2">
                <div className="text-zinc-500">GPS</div>
                <div className="text-zinc-300">
                  {Number(selected.lat).toFixed(4)}°, {Number(selected.lon).toFixed(4)}°
                </div>
              </div>

              {selectedFlags.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-0.5">
                  {selectedFlags.map(f => (
                    <span key={f} className="px-2 py-1 rounded border border-amber-500/40 bg-amber-500/10 text-amber-300 text-[10px] font-bold">
                      {f}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Real anomaly detections ────────────────────────────────────── */}
        <Panel title="Risk indicators" right="Rule-based">
          <div className="p-3 mono text-[11px] space-y-2">
            {[
              [AlertTriangle, 'text-amber-400', 'Velocity', anomalies.velocity_count ?? 0, anomalies.velocity_rule],
              [Split, 'text-red-400', 'Fund splitting', anomalies.fund_split_count ?? 0, anomalies.fund_split_rule],
              [Target, 'text-aegis-green', 'Terminal leaves', anomalies.terminal_count ?? 0, 'out-degree 0 = cashout candidate'],
            ].map(([Icon, tone, label, count, rule]) => (
              <div key={label} className="bg-ink-panel border border-ink-border rounded px-2.5 py-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-zinc-300 flex items-center gap-1.5">
                    <Icon size={12} className={tone} /> {label}
                  </span>
                  <span className={`${tone} font-bold`}>{count} flagged</span>
                </div>
                {rule && <div className="text-[9px] text-zinc-600 mt-0.5">{rule}</div>}
              </div>
            ))}
            <div className="text-[10px] text-zinc-600 pt-0.5 leading-relaxed">
              Detected on this complaint's sub-graph using the same thresholds as
              <span className="text-zinc-500"> engine/graph_engine.py</span>.
            </div>
          </div>
        </Panel>

        {/* ── Highest-risk accounts ──────────────────────────────────────── */}
        <Panel title="Highest-risk accounts" right="Model score">
          <div className="p-3 space-y-1.5 mono text-[11px]">
            {topMules.length === 0 ? (
              <div className="text-zinc-500 text-[11px]">No embedding data for this complaint.</div>
            ) : (
              topMules.map((m, i) => (
                <button
                  key={m.account_id}
                  onClick={() => {
                    const n = data?.nodes?.find(x => x.id === m.account_id)
                    if (n) setSelected(n)
                  }}
                  className="w-full text-left bg-ink-panel border border-ink-border rounded px-2.5 py-1.5 hover:border-aegis-green/40 transition"
                >
                  <div className="flex justify-between items-center gap-2">
                    <span className="text-zinc-300 truncate">
                      <span className="text-zinc-600">{i + 1}.</span> {shortAccount(m.account_id)}
                    </span>
                    <span className="font-bold shrink-0" style={{ color: riskColor(m.risk_score) }}>
                      {(m.risk_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-[9px] text-zinc-600 mt-0.5 flex justify-between">
                    <span className="truncate">{m.bank}</span>
                    <span>hop {m.hop_depth} · ‖h‖ {m.embedding_norm.toFixed(1)}</span>
                  </div>
                </button>
              ))
            )}
            <div className="flex items-center gap-1.5 text-[10px] text-zinc-600 pt-0.5">
              <Activity size={11} /> Ranked by model mule probability
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
