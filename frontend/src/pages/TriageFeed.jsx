import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { endpoints } from '../services/api'
import { mockComplaints, amountFmt } from '../utils/constants'
import { timeAgo } from '../hooks/useCountdown'
import { Panel } from '../components/Shell'
import { ShieldAlert, Plus, Search, Filter, ArrowUpRight, MapPinned, GitBranch, Zap, CheckCircle2, Clock } from 'lucide-react'

function levelOf(iso) {
  const mins = (Date.now() - new Date(iso).getTime()) / 60000
  if (mins < 15) return { label: 'CRITICAL (<15m)', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', dot: 'bg-red-500' }
  if (mins < 30) return { label: 'WARNING (15-30m)', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', dot: 'bg-amber-500' }
  return { label: 'MONITORING', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30', dot: 'bg-emerald-500' }
}

export default function TriageFeed({ complaints: liveComplaints, onSelect }) {
  const [remote, setRemote] = useState([])
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [modalOpen, setModalOpen] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [newComplaint, setNewComplaint] = useState({
    victim_name: '',
    victim_bank: 'SBI',
    victim_account: '',
    fraud_type: 'UPI Fraud',
    stolen_amount: 75000,
    city: 'Delhi',
    state: 'Delhi'
  })
  const [selectedComplaint, setSelectedComplaint] = useState(null)
  const navigate = useNavigate()

  const loadData = () => {
    endpoints.listComplaints().then(setRemote).catch(() => setRemote([]))
  }

  useEffect(() => {
    loadData()
  }, [])

  const complaints = useMemo(() => {
    const merged = [...(liveComplaints || []), ...remote, ...mockComplaints]
    const map = new Map()
    merged.forEach(c => { if (c?.ticket_id && !map.has(c.ticket_id)) map.set(c.ticket_id, c) })
    const arr = Array.from(map.values()).sort((a, b) => new Date(b.complaint_timestamp) - new Date(a.complaint_timestamp))
    return arr.filter(c => {
      if (filter !== 'ALL' && c.fraud_type !== filter) return false
      if (!q) return true
      const s = `${c.ticket_id} ${c.victim_name} ${c.city} ${c.fraud_type} ${c.victim_account}`.toLowerCase()
      return s.includes(q.toLowerCase())
    })
  }, [remote, liveComplaints, q, filter])

  // Select first complaint if none selected
  useEffect(() => {
    if (complaints.length && !selectedComplaint) {
      setSelectedComplaint(complaints[0])
      if (onSelect) onSelect(complaints[0].ticket_id)
    }
  }, [complaints, selectedComplaint, onSelect])

  const handleSelect = (c) => {
    setSelectedComplaint(c)
    if (onSelect) onSelect(c.ticket_id)
  }

  const handleIngest = async (e) => {
    e.preventDefault()
    setIngesting(true)
    try {
      const payload = {
        ...newComplaint,
        stolen_amount: Number(newComplaint.stolen_amount) || 50000,
        victim_account: newComplaint.victim_account || `ACC-${Math.floor(10000000 + Math.random() * 90000000)}`
      }
      const res = await endpoints.ingestComplaint(payload)
      setModalOpen(false)
      loadData()
      handleSelect(res)
    } catch (err) {
      alert(`Ingestion failed: ${err.message}`)
    } finally {
      setIngesting(false)
    }
  }

  const criticalCount = complaints.filter(c => levelOf(c.complaint_timestamp).label.includes('CRITICAL')).length
  const totalAmount = complaints.reduce((sum, c) => sum + (c.stolen_amount || 0), 0)

  return (
    <div className="p-4 space-y-4">
      {/* TOP STATS CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="aegis-panel p-3 border-l-4 border-l-blue-500">
          <div className="mono text-[11px] text-zinc-500 uppercase tracking-wider">Active 1930 Complaints</div>
          <div className="text-2xl font-bold text-white mt-1 mono">{complaints.length}</div>
          <div className="text-[11px] text-zinc-400 mt-1">Live queue from NCRP feed</div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-red-500">
          <div className="mono text-[11px] text-zinc-500 uppercase tracking-wider">Critical (Golden Hour)</div>
          <div className="text-2xl font-bold text-red-400 mt-1 mono">{criticalCount}</div>
          <div className="text-[11px] text-red-400/80 mt-1">&lt; 15 mins since reporting</div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-emerald-500">
          <div className="mono text-[11px] text-zinc-500 uppercase tracking-wider">Total At-Risk Funds</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1 mono">{amountFmt(totalAmount)}</div>
          <div className="text-[11px] text-emerald-400/80 mt-1">Target for rapid interdiction</div>
        </div>
        <div className="aegis-panel p-3 border-l-4 border-l-purple-500">
          <div className="mono text-[11px] text-zinc-500 uppercase tracking-wider">AI Pipeline Latency</div>
          <div className="text-2xl font-bold text-purple-400 mt-1 mono">25.8 ms</div>
          <div className="text-[11px] text-zinc-400 mt-1">GraphSAGE GNN + XGBoost v2</div>
        </div>
      </div>

      {/* MAIN TWO COLUMN VIEW */}
      <div className="grid grid-cols-12 gap-4">
        {/* LEFT COLUMN: FILTERABLE COMPLAINT LIST */}
        <div className="col-span-12 lg:col-span-7 space-y-3">
          <div className="aegis-panel p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <ShieldAlert size={16} className="text-aegis-green" />
                <span className="mono text-[12px] font-bold tracking-wider text-white">1930 HELPLINE TRIAGE QUEUE</span>
              </div>
              <button
                onClick={() => setModalOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-aegis-green text-black mono text-[11px] font-bold hover:bg-emerald-400 transition"
              >
                <Plus size={14} /> Ingest 1930 Complaint
              </button>
            </div>

            {/* SEARCH & FILTERS */}
            <div className="flex flex-wrap gap-2 mb-3">
              <div className="flex-1 min-w-[200px] flex items-center gap-2 bg-ink-bg border border-ink-border rounded px-2.5 py-1.5 text-zinc-200">
                <Search size={13} className="text-zinc-500" />
                <input
                  type="text"
                  placeholder="Search by Ticket ID, Victim Name, City..."
                  value={q}
                  onChange={e => setQ(e.target.value)}
                  className="bg-transparent outline-none text-[12px] mono w-full placeholder:text-zinc-600"
                />
              </div>
              <select
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="bg-ink-bg border border-ink-border rounded px-2.5 py-1.5 text-[11px] mono text-zinc-300 outline-none"
              >
                <option value="ALL">All Fraud Types</option>
                <option value="UPI Fraud">UPI Fraud</option>
                <option value="Digital Arrest">Digital Arrest</option>
                <option value="Job Scam">Job Scam</option>
                <option value="Investment Scam">Investment Scam</option>
                <option value="KYC Fraud">KYC Fraud</option>
              </select>
            </div>

            {/* COMPLAINTS SCROLLABLE LIST */}
            <div className="space-y-2 max-h-[56vh] overflow-y-auto pr-1">
              {complaints.map(c => {
                const lvl = levelOf(c.complaint_timestamp)
                const isSel = selectedComplaint?.ticket_id === c.ticket_id
                return (
                  <div
                    key={c.ticket_id}
                    onClick={() => handleSelect(c)}
                    className={`p-3 rounded border cursor-pointer transition ${
                      isSel ? 'bg-ink-panel border-aegis-green/60 shadow-[0_0_12px_rgba(124,240,0,0.15)]' : 'bg-ink-surface/50 border-ink-border hover:bg-ink-panel'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${lvl.dot} animate-pulse-dot`} />
                        <span className="mono text-[12px] font-bold text-white">{c.ticket_id}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] mono border ${lvl.bg} ${lvl.color}`}>{lvl.label}</span>
                      </div>
                      <span className="mono text-[13px] font-bold text-aegis-green">{amountFmt(c.stolen_amount)}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 mt-2 text-[11px] mono text-zinc-400">
                      <div>Victim: <span className="text-zinc-200">{c.victim_name}</span> ({c.victim_bank})</div>
                      <div className="text-right">{c.city}, {c.state}</div>
                      <div>Type: <span className="text-zinc-300">{c.fraud_type}</span></div>
                      <div className="text-right text-zinc-500">{timeAgo(c.complaint_timestamp)}</div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: ACTIVE COMPLAINT ACTION PANEL */}
        <div className="col-span-12 lg:col-span-5 space-y-3">
          {selectedComplaint ? (
            <div className="aegis-panel p-4 space-y-4">
              <div className="border-b border-ink-border pb-3">
                <div className="mono text-[10px] uppercase tracking-wider text-zinc-500">Selected Incident Overview</div>
                <div className="text-lg font-bold text-white mono mt-0.5">{selectedComplaint.ticket_id}</div>
                <div className="text-[12px] mono text-zinc-400 mt-1">
                  Victim: <span className="text-white font-medium">{selectedComplaint.victim_name}</span> • {selectedComplaint.victim_bank} ({selectedComplaint.victim_account})
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] mono">
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-500">Stolen Amount</div>
                  <div className="text-base font-bold text-red-400 mt-0.5">{amountFmt(selectedComplaint.stolen_amount)}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-500">Incident Category</div>
                  <div className="text-sm font-semibold text-white mt-1">{selectedComplaint.fraud_type}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-500">Location</div>
                  <div className="text-sm text-zinc-200 mt-1">{selectedComplaint.city}, {selectedComplaint.state}</div>
                </div>
                <div className="bg-ink-bg p-2.5 rounded border border-ink-border">
                  <div className="text-zinc-500">Time Reported</div>
                  <div className="text-sm text-zinc-200 mt-1">{timeAgo(selectedComplaint.complaint_timestamp)}</div>
                </div>
              </div>

              <div className="pt-2 border-t border-ink-border space-y-2">
                <div className="mono text-[11px] font-bold text-zinc-300">TACTICAL INTERACTION ACTIONS:</div>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    onClick={() => navigate(`/map?c=${selectedComplaint.ticket_id}`)}
                    className="flex items-center justify-between p-2.5 rounded bg-ink-bg border border-ink-border hover:border-red-500/60 hover:bg-red-500/10 text-left transition group"
                  >
                    <div className="flex items-center gap-2">
                      <MapPinned size={16} className="text-red-400" />
                      <div>
                        <div className="mono text-[11px] font-bold text-white">View Tactical GIS Map</div>
                        <div className="mono text-[10px] text-zinc-500">Pinpoint predicted ATM & dispatch PCR units</div>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-zinc-500 group-hover:text-white" />
                  </button>

                  <button
                    onClick={() => navigate(`/graph?c=${selectedComplaint.ticket_id}`)}
                    className="flex items-center justify-between p-2.5 rounded bg-ink-bg border border-ink-border hover:border-blue-500/60 hover:bg-blue-500/10 text-left transition group"
                  >
                    <div className="flex items-center gap-2">
                      <GitBranch size={16} className="text-blue-400" />
                      <div>
                        <div className="mono text-[11px] font-bold text-white">View Forensic Money-Flow Graph</div>
                        <div className="mono text-[10px] text-zinc-500">Inspect multi-hop mule layering & GNN risk scores</div>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-zinc-500 group-hover:text-white" />
                  </button>

                  <button
                    onClick={() => navigate(`/intercept?c=${selectedComplaint.ticket_id}`)}
                    className="flex items-center justify-between p-2.5 rounded bg-ink-bg border border-ink-border hover:border-aegis-green/60 hover:bg-aegis-green/10 text-left transition group"
                  >
                    <div className="flex items-center gap-2">
                      <Zap size={16} className="text-aegis-green" />
                      <div>
                        <div className="mono text-[11px] font-bold text-white">1-Click Emergency Card Freeze</div>
                        <div className="mono text-[10px] text-zinc-500">Lock terminal debit accounts before ATM cashout</div>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-zinc-500 group-hover:text-white" />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="aegis-panel p-8 text-center text-zinc-500 mono text-[12px]">
              Select a complaint from the queue to view tactical details.
            </div>
          )}
        </div>
      </div>

      {/* INGEST COMPLAINT MODAL */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 grid place-items-center p-4">
          <div className="aegis-panel w-full max-w-lg p-5 bg-ink-bg border border-ink-border2 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-ink-border">
              <div className="mono text-[13px] font-bold text-white flex items-center gap-2">
                <Plus size={16} className="text-aegis-green" /> Ingest 1930 Cybercrime Complaint
              </div>
              <button onClick={() => setModalOpen(false)} className="text-zinc-400 hover:text-white mono text-[12px]">✕</button>
            </div>
            <form onSubmit={handleIngest} className="space-y-3 mt-4 mono text-[11px]">
              <div>
                <label className="block text-zinc-400 mb-1">Victim Full Name</label>
                <input
                  required
                  type="text"
                  placeholder="e.g. Ramesh Chandra"
                  value={newComplaint.victim_name}
                  onChange={e => setNewComplaint({ ...newComplaint, victim_name: e.target.value })}
                  className="w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-zinc-400 mb-1">Victim Bank</label>
                  <select
                    value={newComplaint.victim_bank}
                    onChange={e => setNewComplaint({ ...newComplaint, victim_bank: e.target.value })}
                    className="w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green"
                  >
                    <option value="SBI">State Bank of India (SBI)</option>
                    <option value="HDFC">HDFC Bank</option>
                    <option value="ICICI">ICICI Bank</option>
                    <option value="PNB">Punjab National Bank</option>
                    <option value="Axis">Axis Bank</option>
                    <option value="Kotak">Kotak Mahindra Bank</option>
                    <option value="BOI">Bank of India</option>
                  </select>
                </div>
                <div>
                  <label className="block text-zinc-400 mb-1">Stolen Amount (₹)</label>
                  <input
                    required
                    type="number"
                    min="1000"
                    placeholder="e.g. 75000"
                    value={newComplaint.stolen_amount}
                    onChange={e => setNewComplaint({ ...newComplaint, stolen_amount: e.target.value })}
                    className="w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-zinc-400 mb-1">Fraud Category</label>
                  <select
                    value={newComplaint.fraud_type}
                    onChange={e => setNewComplaint({ ...newComplaint, fraud_type: e.target.value })}
                    className="w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green"
                  >
                    <option value="UPI Fraud">UPI Fraud</option>
                    <option value="Digital Arrest">Digital Arrest</option>
                    <option value="Job Scam">Job Scam</option>
                    <option value="Investment Scam">Investment Scam</option>
                    <option value="KYC Fraud">KYC Fraud</option>
                  </select>
                </div>
                <div>
                  <label className="block text-zinc-400 mb-1">City / Region</label>
                  <input
                    required
                    type="text"
                    placeholder="e.g. Mumbai"
                    value={newComplaint.city}
                    onChange={e => setNewComplaint({ ...newComplaint, city: e.target.value })}
                    className="w-full bg-ink-panel border border-ink-border rounded px-3 py-2 text-white outline-none focus:border-aegis-green"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-ink-border">
                <button
                  type="button"
                  onClick={() => setModalOpen(false)}
                  className="px-4 py-2 rounded border border-ink-border text-zinc-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={ingesting}
                  className="px-4 py-2 rounded bg-aegis-green text-black font-bold hover:bg-emerald-400 disabled:opacity-50"
                >
                  {ingesting ? 'Broadcasting...' : 'Ingest & Trigger AI'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
