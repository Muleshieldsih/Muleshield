import { memo } from 'react'
import { Cpu, User } from 'lucide-react'

/**
 * The audit trail.
 *
 * Rendered from GET /api/v1/audit and nothing else. Entries appear because an
 * action produced one: a status change, an assignment, a note, an account
 * freeze. Nothing is added here to make the list look busier — an audit log
 * that contains anything other than what happened is worse than no audit log,
 * because it can be believed.
 *
 * Analyst actions and system actions are distinguished on the row rather than
 * left to be inferred from the actor string. "Who did this, a person or the
 * software" is the first question anyone asks of a trail, and on a screen where
 * an account can be frozen it is the one that matters.
 */

const SYSTEM_ACTORS = new Set(['SYSTEM', 'UNKNOWN', ''])

function stamp(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString('en-IN', {
    day: '2-digit', month: 'short',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).replace(',', ' ·')
}

const AuditList = memo(function AuditList({ entries = [], showCase = false, emptyHint }) {
  if (!entries.length) {
    return (
      <div className="p-4 text-[12.5px] text-zinc-500 leading-relaxed">
        {emptyHint || 'No recorded activity yet.'}
      </div>
    )
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Actor</th>
          <th>Action</th>
          <th>Object</th>
          {showCase && <th>Case</th>}
          <th className="text-right">Result</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(e => {
          const isSystem = SYSTEM_ACTORS.has(String(e.actor).toUpperCase())
          return (
            <tr key={e.id} className="cursor-default">
              <td className="mono tnum whitespace-nowrap text-zinc-400">{stamp(e.timestamp)}</td>
              <td className="whitespace-nowrap">
                <span className="inline-flex items-center gap-1.5">
                  {isSystem
                    ? <Cpu size={11} className="text-zinc-500 shrink-0" />
                    : <User size={11} className="text-blue-400 shrink-0" />}
                  <span className={isSystem ? 'text-zinc-500' : 'mono text-zinc-200'}>
                    {isSystem ? 'System' : e.actor}
                  </span>
                </span>
              </td>
              <td className="text-zinc-300 whitespace-nowrap">{e.action}</td>
              <td className="mono text-zinc-400 max-w-[280px] truncate" title={e.object}>
                {e.object}
              </td>
              {showCase && (
                <td className="mono text-zinc-500 max-w-[140px] truncate" title={e.case_id}>
                  {e.case_id || '—'}
                </td>
              )}
              <td className="text-right whitespace-nowrap">
                <span className={e.result === 'ok' ? 'text-zinc-500' : 'text-amber-400'}>
                  {e.result === 'ok' ? 'Completed' : e.result}
                </span>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
})

export default AuditList
