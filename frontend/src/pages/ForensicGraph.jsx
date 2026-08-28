import { useEffect, useState, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactFlow, { Background, Controls, MiniMap } from 'reactflow'
import 'reactflow/dist/style.css'
import { endpoints } from '../services/api'
import { Panel } from '../components/Shell'

function buildLayout(nodes, edges){
  // dagre-like layered layout minimal (no dagre dep) — group by hop_depth
  const layers = new Map()
  nodes.forEach(n=> {
    const d = n.hop_depth ?? (n.node_type==='victim'?0: n.node_type==='terminal'?4:1)
    if(!layers.has(d)) layers.set(d, [])
    layers.get(d).push(n)
  })
  const xs = {0: 60, 1: 280, 2: 500, 3: 720, 4: 940}
  const flowNodes = []
  layers.forEach((arr, depth)=>{
    const x = xs[depth] ?? depth*220+60
    const gap = 86
    const startY = 80
    arr.forEach((n,i)=>{
      const color = n.node_type==='victim' ? '#58a6ff' : n.node_type==='terminal' ? '#ff3b3b' : '#ff8c42'
      flowNodes.push({
        id: n.id,
        position: { x, y: startY + i*gap },
        data: { label: `${n.label?.split('\n')[0]||n.node_type}\n${n.id.slice(-6)}` },
        style: {
          background: n.node_type==='terminal' ? 'rgba(255,59,59,0.12)' : n.node_type==='victim' ? 'rgba(88,166,255,0.12)' : 'rgba(255,140,66,0.10)',
          border: `1px solid ${color}55`,
          color: '#e6edf3',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '10px',
          borderRadius: '10px',
          padding: '8px 10px',
          width: 160,
          boxShadow: n.node_type==='terminal' ? '0 0 14px rgba(255,59,59,0.35)' : 'none'
        }
      })
    })
  })
  const flowEdges = edges.map(e=> ({
    id: e.id, source: e.source, target: e.target,
    label: e.label, animated: true,
    style:{ stroke: '#2a3333', strokeWidth: 1 },
    labelStyle:{ fontFamily:'JetBrains Mono', fontSize:'10px', fill:'#9ca3af' },
    labelBgStyle:{ fill:'#0f1111', fillOpacity:0.9 }
  }))
  return { flowNodes, flowEdges }
}

export default function ForensicGraph(){
  const [params]=useSearchParams()
  const cid = params.get('c')
  const [data,setData]=useState(null)
  const [selected,setSelected]=useState(null)
  const [err,setErr]=useState('')

  useEffect(()=>{
    const id = cid || 'TKT-A1B2C3D4'
    endpoints.getGraph(id).then(d=> setData(d)).catch(e=>{
      setErr(e?.response?.data?.detail || e.message)
      // fallback mock DAG
      setData({
        complaint_id:id, node_count:6, edge_count:5, build_time_ms:142,
        nodes:[
          {id:'ACC-VICTIM-001', label:'VICTIM\nRohan', node_type:'victim', hop_depth:0, bank:'HDFC', amount:120000, lat:28.6, lon:77.2, risk_score:0},
          {id:'ACC-HDFC-4821', label:'HOP-1\nHDFC', node_type:'mule', hop_depth:1, bank:'HDFC', amount:60000, lat:28.61, lon:77.21, risk_score:0.42},
          {id:'ACC-SBI-9932', label:'HOP-1\nSBI', node_type:'mule', hop_depth:1, bank:'SBI', amount:60000, lat:28.62, lon:77.22, risk_score:0.51},
          {id:'ACC-KOTAK-2211', label:'HOP-2\nKotak', node_type:'mule', hop_depth:2, bank:'Kotak', amount:29500, lat:28.615, lon:77.215, risk_score:0.78},
          {id:'ACC-PNB-0041', label:'TERMINAL\nPNB', node_type:'terminal', hop_depth:4, bank:'PNB', amount:29500, lat:28.612, lon:77.208, risk_score:0.94},
          {id:'ACC-UCO-8812', label:'TERMINAL\nUCO', node_type:'terminal', hop_depth:4, bank:'UCO', amount:29500, lat:28.63, lon:77.23, risk_score:0.87},
        ],
        edges:[
          {id:'e1', source:'ACC-VICTIM-001', target:'ACC-HDFC-4821', label:'₹60,000', amount:60000},
          {id:'e2', source:'ACC-VICTIM-001', target:'ACC-SBI-9932', label:'₹60,000', amount:60000},
          {id:'e3', source:'ACC-HDFC-4821', target:'ACC-KOTAK-2211', label:'₹29,500', amount:29500},
          {id:'e4', source:'ACC-KOTAK-2211', target:'ACC-PNB-0041', label:'₹29,500', amount:29500},
          {id:'e5', source:'ACC-SBI-9932', target:'ACC-UCO-8812', label:'₹29,500', amount:29500},
        ]
      })
    })
  },[cid])

  const { flowNodes, flowEdges } = useMemo(()=> data? buildLayout(data.nodes, data.edges): {flowNodes:[], flowEdges:[]}, [data])

  return (
    <div className="grid grid-cols-12 gap-3 p-3">
      <div className="col-span-12 lg:col-span-9">
        <Panel title={`MONEY-FLOW DAG — ${data?.complaint_id||'—'}`} right={data? `${data.node_count} nodes · ${data.edge_count} edges · ${data.build_time_ms} ms`:''}>
          <div className="h-[62vh] bg-ink-bg">
            <ReactFlow nodes={flowNodes} edges={flowEdges} onNodeClick={(_,n)=> setSelected(data.nodes.find(x=>x.id===n.id))} fitView fitViewOptions={{padding:0.2}}>
              <Background gap={16} size={1} color="#1e2323"/>
              <Controls />
              <MiniMap style={{background:'#0f1111', border:'1px solid #1e2323'}} maskColor="rgba(8,10,10,0.7)"/>
            </ReactFlow>
          </div>
          {err && <div className="px-3 py-1.5 mono text-[11px] text-amber-300 border-t border-ink-border bg-amber-500/10">Fallback mock — {err}</div>}
          <div className="px-3 py-2 flex gap-2 mono text-[11px] border-t border-ink-border">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#58a6ff]"/> Victim</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#ff8c42]"/> Mule L1-3</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#ff3b3b] animate-pulse-dot"/> Terminal</span>
            <span className="ml-auto text-zinc-500">Click node → inspect</span>
          </div>
        </Panel>
      </div>
      <div className="col-span-12 lg:col-span-3 space-y-3">
        <div className="aegis-panel p-3">
          <div className="mono text-[11px] tracking-[0.14em] text-zinc-500">NODE INSPECTOR</div>
          {!selected ? <div className="mono text-[12px] text-zinc-500 mt-3">Select a node on the graph.</div> : (
            <div className="mt-3 space-y-2 mono text-[11px]">
              <div className="bg-ink-panel border border-ink-border rounded px-2.5 py-2">
                <div className="text-zinc-400">Account</div><div className="text-white font-medium">{selected.id}</div>
                <div className="text-zinc-500 mt-1">{selected.bank} · Hop {selected.hop_depth} · {selected.node_type}</div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-ink-panel border border-ink-border rounded px-2 py-2"><div className="text-zinc-500">Amount</div><div className="text-white">₹{Number(selected.amount).toLocaleString('en-IN')}</div></div>
                <div className="bg-ink-panel border border-ink-border rounded px-2 py-2"><div className="text-zinc-500">Risk</div><div className={selected.risk_score>0.8?'text-red-400': selected.risk_score>0.5?'text-amber-400':'text-emerald-400'}>{(Number(selected.risk_score)||0).toFixed(2)}</div></div>
              </div>
              <div className="bg-ink-panel border border-ink-border rounded px-2 py-2"><div className="text-zinc-500">GPS</div><div className="text-zinc-300">{selected.lat?.toFixed(4)}°, {selected.lon?.toFixed(4)}°</div></div>
              <div className="h-1.5 bg-ink-panel border border-ink-border rounded overflow-hidden"><span className="block h-full bg-aegis-green" style={{width:`${Math.round((selected.risk_score||0)*100)}%`}}/></div>
              <div className="flex gap-2"><span className="px-2 py-1 rounded border border-ink-border bg-ink-panel text-zinc-400">Terminal: {selected.node_type==='terminal'? 'YES':'NO'}</span><span className="px-2 py-1 rounded border border-ink-border bg-ink-panel text-zinc-400">GNN {((selected.risk_score||0)*100).toFixed(1)}%</span></div>
            </div>
          )}
        </div>
        <Panel title="VELOCITY & SPLIT ANOMALIES" right="GNN">
          <div className="p-3 mono text-[11px] space-y-2">
            <div className="flex justify-between bg-ink-panel border border-ink-border rounded px-2 py-1.5"><span className="text-zinc-400">Velocity (&gt;2 / 5m)</span><span className="text-amber-400">2 flagged</span></div>
            <div className="flex justify-between bg-ink-panel border border-ink-border rounded px-2 py-1.5"><span className="text-zinc-400">Fund split (1→3+)</span><span className="text-red-400">1 flagged</span></div>
            <div className="flex justify-between bg-ink-panel border border-ink-border rounded px-2 py-1.5"><span className="text-zinc-400">Terminal leaves</span><span className="text-aegis-green">{data?.nodes.filter(n=>n.node_type==='terminal').length||2}</span></div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
