import { useCallback, useEffect, useState } from 'react'
import { Loader2, ShieldAlert, KeyRound, Check, X, Copy } from 'lucide-react'
import { Panel, Stat } from '../components/Shell'
import AuditList from '../components/AuditList'
import { endpoints, describeError } from '../services/api'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../components/Toast'
import { amountShort } from '../utils/constants'

/**
 * The landing screen after sign-in.
 *
 * Everything else in this console answers a question about ONE case. This answers
 * the one I4C asks across a district: which machines keep coming back. The models
 * have always consumed that history — `atm_prior_count` and
 * `historical_hotspot_density` are ranking features — but until now nothing put
 * it in front of a person.
 */
export default function Dashboard() {
  const { user } = useAuth()
  const toast = useToast()

  const [atms, setAtms] = useState([])
  const [audit, setAudit] = useState([])
  const [health, setHealth] = useState(null)
  const [requests, setRequests] = useState([])
  const [issued, setIssued] = useState(null)      // {username, token} shown once
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const loadRequests = useCallback(() => {
    if (!user?.is_admin) return
    endpoints.listResetRequests('pending').then(setRequests).catch(() => {})
  }, [user])

  useEffect(() => {
    let cancelled = false
    Promise.allSettled([
      endpoints.listAtmIntel(12),
      endpoints.listAudit(8),
      endpoints.health(),
    ]).then(([a, b, c]) => {
      if (cancelled) return
      if (a.status === 'fulfilled') setAtms(a.value || [])
      else setError(describeError(a.reason))
      if (b.status === 'fulfilled') setAudit(b.value || [])
      if (c.status === 'fulfilled') setHealth(c.value)
      setLoading(false)
    })
    return () => { cancelled = true }
  }, [])

  useEffect(loadRequests, [loadRequests])

  async function approve(id, username) {
    try {
      const r = await endpoints.approveReset(id)
      setIssued({ username, token: r.reset_token })
      loadRequests()
    } catch (err) {
      toast.push(describeError(err), 'error')
    }
  }

  async function deny(id) {
    try {
      await endpoints.denyReset(id)
      loadRequests()
      toast.push('Request denied', 'success')
    } catch (err) {
      toast.push(describeError(err), 'error')
    }
  }

  const repeat = atms.filter(a => a.distinct_complaints > 1).length

  return (
    <div className="p-4 space-y-4">

      <div>
        <h1 className="text-[15px] font-semibold text-white">
          {greeting()}, {user?.display_name || 'Officer'}
        </h1>
        <p className="text-[12px] text-zinc-500 mt-0.5">
          Cross-case intelligence and recent activity.
        </p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-300 px-3 py-2 rounded text-[12px]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Complaints loaded" value={fmt(health?.complaints_loaded)} />
        <Stat label="Accounts scored" value={fmt(health?.embeddings_loaded)} />
        <Stat label="ATMs seen in 2+ cases" value={repeat ? `${repeat} of ${atms.length}` : '—'}
              tone={repeat ? 'warn' : 'default'} sub="in the top slice below" />
        <Stat label="Feed" value={health ? 'Connected' : 'Offline'}
              tone={health ? 'good' : 'bad'} />
      </div>

      {user?.is_admin && requests.length > 0 && (
        <Panel title="Password reset requests"
               right={<span className="text-[11px] text-amber-400">{requests.length} awaiting</span>}>
          <div className="p-3 space-y-2">
            {issued && (
              <div className="bg-aegis-green/10 border border-aegis-green/30 rounded px-3 py-2.5">
                <div className="text-[11px] text-zinc-400 mb-1.5">
                  Token for <span className="mono text-white">{issued.username}</span> — shown once,
                  valid 30 minutes. Hand it over by a channel you trust.
                </div>
                <div className="flex items-center gap-2">
                  <code className="mono text-[12px] text-aegis-green break-all flex-1">
                    {issued.token}
                  </code>
                  <button
                    onClick={() => {
                      navigator.clipboard?.writeText(issued.token)
                      toast.push('Token copied', 'success')
                    }}
                    className="shrink-0 p-1.5 rounded border border-ink-border text-zinc-400 hover:text-white"
                    title="Copy">
                    <Copy size={12} />
                  </button>
                </div>
              </div>
            )}
            {requests.map(r => (
              <div key={r.id}
                   className="flex items-center justify-between bg-ink-panel border border-ink-border rounded px-3 py-2">
                <div className="min-w-0">
                  <div className="text-[12.5px] text-white">
                    <span className="mono">{r.username}</span>
                    <span className="text-zinc-500"> · {r.display_name}</span>
                  </div>
                  <div className="text-[10.5px] text-zinc-600">
                    requested {new Date(r.requested_at).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button onClick={() => approve(r.id, r.username)}
                          className="px-2.5 py-1 rounded bg-aegis-green text-black text-[11px] font-bold flex items-center gap-1 hover:bg-emerald-400">
                    <Check size={11} /> Approve
                  </button>
                  <button onClick={() => deny(r.id)}
                          className="px-2.5 py-1 rounded border border-ink-border text-zinc-400 text-[11px] hover:text-white flex items-center gap-1">
                    <X size={11} /> Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel
        title="Recurring cash-out ATMs"
        right={<span className="text-[11px] text-zinc-500">ranked by distinct cases</span>}>
        {loading ? (
          <div className="p-6 grid place-items-center">
            <Loader2 size={16} className="animate-spin text-zinc-600" />
          </div>
        ) : atms.length === 0 ? (
          <div className="p-4 text-[12px] text-zinc-500">No cash-out history loaded.</div>
        ) : (
          <>
            <div className="px-3 pt-3 text-[11.5px] text-zinc-500 leading-relaxed">
              A machine appearing across many separate complaints is a crew&rsquo;s habit,
              not a coincidence. Ranked by <span className="text-zinc-300">distinct cases</span> rather
              than raw withdrawals — one chain pooling four ways into a single terminal
              is one case, not four.
            </div>
            <div className="overflow-x-auto p-3">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ATM</th>
                    <th>Location</th>
                    <th className="text-right">Cases</th>
                    <th className="text-right">Withdrawals</th>
                    <th className="text-right">Total</th>
                    <th className="text-right">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {atms.map(a => (
                    <tr key={a.atm_id}>
                      <td className="mono text-zinc-300">{a.atm_id}</td>
                      <td className="text-zinc-400">
                        {[a.city, a.state].filter(Boolean).join(', ') || '—'}
                      </td>
                      <td className="text-right tnum">
                        <span className={a.distinct_complaints >= 10
                          ? 'text-red-400 font-semibold'
                          : a.distinct_complaints > 1 ? 'text-amber-400' : 'text-zinc-400'}>
                          {a.distinct_complaints}
                        </span>
                      </td>
                      <td className="text-right tnum text-zinc-400">{a.cashouts}</td>
                      <td className="text-right tnum text-zinc-300">
                        {amountShort(a.total_amount)}
                      </td>
                      <td className="text-right tnum text-zinc-500">
                        {a.cashout_risk_score ? a.cashout_risk_score.toFixed(2) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Panel>

      <Panel title="Recent activity"
             right={<span className="text-[11px] text-zinc-500">all cases</span>}>
        <AuditList entries={audit} showCase
                   emptyHint="Nothing recorded yet. Actions appear here as they happen." />
      </Panel>
    </div>
  )
}

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

function fmt(n) {
  return typeof n === 'number' ? n.toLocaleString('en-IN') : '—'
}
