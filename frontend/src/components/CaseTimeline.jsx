import { memo, useMemo, useState } from 'react'
import { ArrowRight, Banknote, FileText, ShieldCheck, UserPlus, Landmark } from 'lucide-react'
import { amountFmt, shortAccount, CITY_STATE_MAP } from '../utils/constants'

/**
 * What happened on this case, in order.
 *
 * Built entirely from records that already exist: the complaint's own
 * timestamp, the transfers on the ledger, and the audit trail. Nothing is
 * synthesised to fill the list out — a case with three transfers shows three,
 * and a case nobody has touched shows only the report and the money moving.
 * An invented "case escalated for review" would read well and mean nothing.
 *
 * Two clocks are being merged here and they are not the same clock. The
 * complaint and its transfers are when the FRAUD happened; audit entries are
 * when an ANALYST did something, which is now. On the seed corpus those are
 * months apart, so the list is grouped rather than interleaved — presenting
 * them as one continuous sequence would imply a response time that never
 * happened.
 */

const KIND = {
  report:   { icon: FileText,    tone: 'text-blue-400' },
  transfer: { icon: ArrowRight,  tone: 'text-zinc-500' },
  cashout:  { icon: Banknote,    tone: 'text-red-400' },
  freeze:   { icon: ShieldCheck, tone: 'text-red-400' },
  assign:   { icon: UserPlus,    tone: 'text-zinc-400' },
  note:     { icon: FileText,    tone: 'text-zinc-400' },
  status:   { icon: Landmark,    tone: 'text-zinc-400' },
}

function auditKind(action = '') {
  const a = action.toLowerCase()
  if (a.includes('froze')) return 'freeze'
  if (a.includes('assign')) return 'assign'
  if (a.includes('note')) return 'note'
  return 'status'
}

/** "29 Aug 2026 · 14:32" — one format everywhere, so events can be compared. */
function stamp(iso) {
  const d = new Date(String(iso).replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).replace(',', ' ·')
}

function Row({ event, expanded, onToggle }) {
  const { icon: Icon, tone } = KIND[event.kind] || KIND.transfer
  const canExpand = !!event.detail
  return (
    <li className="relative pl-6">
      {/* The rail. A plain 1px line, not a decorated track. */}
      <span className="absolute left-[7px] top-0 bottom-0 w-px bg-ink-border" aria-hidden="true" />
      <span className="absolute left-0 top-[5px] w-3.5 h-3.5 rounded-full bg-ink-bg
                       border border-ink-border grid place-items-center">
        <Icon size={9} className={tone} />
      </span>

      <button
        onClick={canExpand ? onToggle : undefined}
        disabled={!canExpand}
        aria-expanded={canExpand ? expanded : undefined}
        className={`w-full text-left pb-3 ${canExpand ? 'cursor-pointer' : 'cursor-default'}`}
      >
        <div className="mono tnum text-[10.5px] text-zinc-500">{stamp(event.at)}</div>
        <div className="text-[12.5px] text-zinc-200 mt-0.5 leading-snug">{event.title}</div>
        {event.sub && (
          <div className="text-[11.5px] text-zinc-500 mt-0.5 leading-snug">{event.sub}</div>
        )}

        {canExpand && expanded && (
          <div className="mt-1.5 rounded border border-ink-border bg-ink-bg px-2.5 py-2
                          grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            {event.detail.map(([k, v]) => (
              <div key={k} className="flex flex-col min-w-0">
                <span className="text-zinc-500">{k}</span>
                <span className="mono text-zinc-300 truncate" title={String(v)}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </button>
    </li>
  )
}

const CaseTimeline = memo(function CaseTimeline({ complaint, transactions = [], audit = [] }) {
  const [open, setOpen] = useState(null)

  const { fraud, activity } = useMemo(() => {
    const fraudEvents = []

    if (complaint?.complaint_timestamp) {
      fraudEvents.push({
        id: 'reported',
        at: complaint.complaint_timestamp,
        kind: 'report',
        title: `Complaint reported — ${amountFmt(complaint.stolen_amount)}`,
        sub: `${complaint.victim_name} · ${complaint.victim_bank} · ${complaint.city}`,
        detail: [
          ['Victim account', complaint.victim_account],
          ['Category', complaint.fraud_type],
          ['Location', `${complaint.city}, ${CITY_STATE_MAP[complaint.city] || complaint.state}`],
          ['Status', complaint.status || 'New'],
        ],
      })
    }

    transactions.forEach(t => {
      fraudEvents.push({
        id: t.txn_id,
        at: t.timestamp,
        kind: t.is_terminal ? 'cashout' : 'transfer',
        title: t.is_terminal
          ? `Cash-out — ${amountFmt(t.amount)}`
          : `Transferred ${amountFmt(t.amount)} to ${shortAccount(t.dst_account)}`,
        sub: t.is_terminal
          ? `Withdrawn from ${shortAccount(t.dst_account)} · ${t.city || 'location unknown'}`
          : `${shortAccount(t.src_account)} → ${shortAccount(t.dst_account)} · hop ${t.hop_depth}`,
        detail: [
          ['Reference', t.txn_id],
          ['Amount', amountFmt(t.amount)],
          ['From', t.src_account],
          ['To', t.dst_account],
          ['Bank', t.bank_name || '—'],
          ['IFSC', t.ifsc_code || '—'],
          ['Elapsed', t.minutes_from_first === 0 ? 'start' : `+${t.minutes_from_first} min`],
          ['Cash-out point', t.cashout_atm_id || 'not a cash-out'],
        ],
      })
    })

    const activityEvents = audit.map(e => ({
      id: e.id,
      at: e.timestamp,
      kind: auditKind(e.action),
      title: e.action,
      sub: `${e.actor === 'SYSTEM' ? 'System' : e.actor} · ${e.object}`,
      detail: [
        ['Entry', e.id],
        ['Actor', e.actor],
        ['Action', e.action],
        ['Object', e.object],
        ['Result', e.result],
      ],
    }))

    const byTime = (a, b) => String(a.at).localeCompare(String(b.at))
    return {
      fraud: fraudEvents.sort(byTime),
      // Audit comes back newest-first; the timeline reads forwards.
      activity: activityEvents.sort(byTime),
    }
  }, [complaint, transactions, audit])

  if (!fraud.length && !activity.length) {
    return <div className="p-4 text-[12.5px] text-zinc-500">No recorded events for this case.</div>
  }

  const section = (label, note, events) => (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-[11px] text-zinc-400 font-medium">{label}</span>
        <span className="text-[10.5px] text-zinc-600">{note}</span>
      </div>
      <ol className="space-y-0">
        {events.map(e => (
          <Row
            key={e.id}
            event={e}
            expanded={open === e.id}
            onToggle={() => setOpen(prev => (prev === e.id ? null : e.id))}
          />
        ))}
      </ol>
    </div>
  )

  return (
    <div className="p-3 space-y-4">
      {fraud.length > 0 && section('Fraud sequence', 'as recorded on the ledger', fraud)}
      {activity.length > 0 && section('Case activity', 'actions taken in this console', activity)}
    </div>
  )
})

export default CaseTimeline
