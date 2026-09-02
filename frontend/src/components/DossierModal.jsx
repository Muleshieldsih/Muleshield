import { useEffect, useState } from 'react'
import { FileText, Loader2, Printer, X } from 'lucide-react'
import { endpoints, describeError } from '../services/api'
import { useToast } from './Toast'

/**
 * Police intelligence dossier — deliverable (c)'s middle clause.
 *
 * The problem statement asks for a secure interface to *"alerts, intelligence
 * reports, and evidence documentation"*. Alerts have an inbox and evidence has a
 * custody chain; the report in the middle had nothing behind it until now.
 *
 * Structurally a sibling of EvidencePanel's CertificateModal, and deliberately
 * so: a document that leaves this building should look like the other document
 * that leaves this building. Black on white inside a dark console, printed by
 * the browser rather than by a PDF library, with the console chrome hidden by
 * the `@media print` block in index.css keyed to `.case-dossier`.
 *
 * ORDERING IS AN ARGUMENT.
 * The forecast section leads with the search ZONE and only then lists the five
 * ranked candidates. SearchZone's own docstring records that the zone is the
 * deliverable and the ranked list is the tactical drill-down; a page that opened
 * with a single ATM id would invite an officer to read a candidate as a
 * prediction, which is exactly what the caveats at the bottom deny.
 */

function fmtWhen(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

function fmtCoord(v) {
  return typeof v === 'number' ? v.toFixed(5) : '—'
}

function Field({ label, children }) {
  return (
    <>
      <div className="text-zinc-600">{label}</div>
      <div className="font-medium">{children || '—'}</div>
    </>
  )
}

function Section({ title, note, children }) {
  return (
    <div className="mt-5">
      <div className="font-semibold text-[12px] mb-1">{title}</div>
      {note && <div className="text-[10.5px] text-zinc-600 mb-1.5">{note}</div>}
      {children}
    </div>
  )
}

export default function DossierModal({ caseId, onClose }) {
  const toast = useToast()
  const [doc, setDoc] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    let alive = true
    setDoc(null); setErr('')
    endpoints.caseDossier(caseId)
      .then(d => { if (alive) setDoc(d) })
      .catch(e => {
        if (!alive) return
        const m = describeError(e)
        setErr(m)
        toast(m, 'error')
      })
    return () => { alive = false }
  }, [caseId, toast])

  const c = doc?.complaint || {}
  const fc = doc?.forecast
  const det = doc?.detections || {}
  const cert = doc?.evidence || {}
  const intact = cert?.verification?.intact !== false

  return (
    <div className="fixed inset-0 z-[3000] bg-black/70 flex items-start justify-center
                    overflow-y-auto p-4 no-print-backdrop">
      <div className="case-dossier bg-white text-black rounded max-w-3xl w-full my-6 shadow-2xl">
        <div className="no-print flex items-center justify-between px-5 py-3 border-b border-zinc-300">
          <div className="text-[12px] font-semibold text-zinc-700 flex items-center gap-1.5">
            <FileText size={13} /> Intelligence dossier — {caseId}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => window.print()}
              disabled={!doc}
              className="px-3 py-1.5 rounded border border-zinc-300 text-[11.5px]
                         text-zinc-700 hover:bg-zinc-100 disabled:opacity-40
                         flex items-center gap-1.5"
            >
              <Printer size={13} /> Print
            </button>
            <button
              onClick={onClose}
              aria-label="Close dossier"
              className="p-1.5 rounded border border-zinc-300 text-zinc-600 hover:bg-zinc-100"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {!doc && !err && (
          <div className="px-8 py-16 flex items-center justify-center gap-2 text-[12px] text-zinc-600">
            <Loader2 size={15} className="animate-spin" />
            Assembling dossier — running the forecast and verifying evidence…
          </div>
        )}

        {err && (
          <div className="px-8 py-10 text-[12px] text-red-700">{err}</div>
        )}

        {doc && (
          <div className="px-8 py-6 text-[12px] leading-relaxed">
            {/* ── Title ─────────────────────────────────────────────────── */}
            <div className="text-center border-b border-zinc-300 pb-4">
              <div className="text-[15px] font-bold uppercase tracking-wide">
                Police Intelligence Dossier
              </div>
              <div className="text-[11px] text-zinc-700 mt-2">
                {doc.system?.operator} · {doc.system?.name} · {doc.system?.problem_statement}
              </div>
              <div className="text-[11px] text-zinc-600 mt-1">
                Case {doc.case_id} · produced {fmtWhen(doc.produced_at)} by {doc.produced_by}
              </div>
            </div>

            {/* ── Complaint ─────────────────────────────────────────────── */}
            <Section title="1 · Complaint as reported (NCRP / 1930)">
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11.5px]">
                <Field label="Ticket">{c.ticket_id}</Field>
                <Field label="Reported">{fmtWhen(c.complaint_timestamp)}</Field>
                <Field label="Complainant">{c.victim_name}</Field>
                <Field label="Fraud type">{c.fraud_type}</Field>
                <Field label="Bank">{c.victim_bank}</Field>
                <Field label="Account">{c.victim_account}</Field>
                <Field label="Amount">{c.stolen_amount_text}</Field>
                <Field label="Place">{[c.city, c.state].filter(Boolean).join(', ')}</Field>
                <Field label="Status">{c.status}</Field>
                <Field label="Assigned to">{c.assignee}</Field>
              </div>
            </Section>

            {/* ── Money trail ───────────────────────────────────────────── */}
            <Section
              title={`2 · Money trail — ${doc.money_trail_count} hop(s)`}
              note={`Banks involved: ${(doc.banks_involved || []).join(', ') || '—'}`}
            >
              <table className="w-full text-[10.5px] border border-zinc-300">
                <thead>
                  <tr className="bg-zinc-100 text-left">
                    <th className="border border-zinc-300 px-2 py-1">Hop</th>
                    <th className="border border-zinc-300 px-2 py-1">Time</th>
                    <th className="border border-zinc-300 px-2 py-1">From → To</th>
                    <th className="border border-zinc-300 px-2 py-1">Bank / IFSC</th>
                    <th className="border border-zinc-300 px-2 py-1">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {(doc.money_trail || []).map(r => (
                    <tr key={r.txn_id} className={r.is_terminal ? 'bg-zinc-50 font-medium' : ''}>
                      <td className="border border-zinc-300 px-2 py-1 align-top">
                        {r.hop_depth}{r.is_terminal ? ' · terminal' : ''}
                      </td>
                      <td className="border border-zinc-300 px-2 py-1 align-top">
                        {fmtWhen(r.timestamp)}
                      </td>
                      <td className="border border-zinc-300 px-2 py-1 align-top font-mono">
                        <div>{r.src_account}</div>
                        <div className="text-zinc-600">→ {r.dst_account}</div>
                      </td>
                      <td className="border border-zinc-300 px-2 py-1 align-top">
                        <div>{r.bank_name}</div>
                        <div className="text-zinc-600 font-mono">{r.ifsc_code}</div>
                        <div className="text-zinc-600">
                          {[r.city, r.state].filter(Boolean).join(', ')}
                        </div>
                      </td>
                      <td className="border border-zinc-300 px-2 py-1 align-top text-right">
                        {r.amount_text}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>

            {/* ── Detections ────────────────────────────────────────────── */}
            <Section
              title="3 · Automated detections"
              note={`Rules applied — velocity: ${det.velocity_rule || '—'}; fund splitting: ${det.fund_split_rule || '—'}`}
            >
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11.5px]">
                <Field label={`Velocity anomalies (${det.velocity_count ?? 0})`}>
                  <span className="font-mono">
                    {(det.velocity_flagged || []).join(', ') || 'none'}
                  </span>
                </Field>
                <Field label={`Fund splitting (${det.fund_split_count ?? 0})`}>
                  <span className="font-mono">
                    {(det.fund_split_flagged || []).join(', ') || 'none'}
                  </span>
                </Field>
                <Field label={`Terminal accounts (${det.terminal_count ?? 0})`}>
                  <span className="font-mono">
                    {(det.terminal_leaves || []).join(', ') || 'none'}
                  </span>
                </Field>
              </div>
            </Section>

            {/* ── Forecast: zone first, candidates second ───────────────── */}
            <Section title="4 · Forecast cash-out">
              {!fc && (
                <div className="text-[11.5px] text-zinc-700">
                  No transaction ledger has been received for this case, so no chain
                  could be traced and no forecast is available.
                </div>
              )}
              {fc && (
                <>
                  <div className="border border-zinc-300 bg-zinc-50 px-3 py-2 text-[11.5px]">
                    <div className="font-semibold">Priority search zone</div>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-1">
                      <Field label="Centre">
                        {fmtCoord(fc.search_zone?.lat)}, {fmtCoord(fc.search_zone?.lon)}
                      </Field>
                      <Field label="Radius">
                        {fc.search_zone?.radius_km != null
                          ? `${fc.search_zone.radius_km} km` : '—'}
                      </Field>
                      <Field label="Machines in zone">{fc.search_zone?.atm_count}</Field>
                      <Field label="Probability mass">
                        {fc.search_zone?.probability_mass != null
                          ? `${(fc.search_zone.probability_mass * 100).toFixed(1)}%` : '—'}
                      </Field>
                      <Field label="Expected in">
                        {fc.time_to_cashout_minutes != null
                          ? `${Math.round(fc.time_to_cashout_minutes)} min` : '—'}
                        {fc.time_to_cashout_low != null && fc.time_to_cashout_high != null
                          ? ` (band ${Math.round(fc.time_to_cashout_low)}–${Math.round(fc.time_to_cashout_high)} min)`
                          : ''}
                      </Field>
                      <Field label="Terminal account">
                        <span className="font-mono">{fc.terminal_account}</span>
                      </Field>
                    </div>
                  </div>

                  <div className="font-semibold text-[11.5px] mt-3 mb-1">
                    Ranked search candidates (in order)
                  </div>
                  <table className="w-full text-[10.5px] border border-zinc-300">
                    <thead>
                      <tr className="bg-zinc-100 text-left">
                        <th className="border border-zinc-300 px-2 py-1">#</th>
                        <th className="border border-zinc-300 px-2 py-1">ATM</th>
                        <th className="border border-zinc-300 px-2 py-1">Address</th>
                        <th className="border border-zinc-300 px-2 py-1">Coordinates</th>
                        <th className="border border-zinc-300 px-2 py-1">Hours</th>
                        <th className="border border-zinc-300 px-2 py-1">Rel. score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(fc.ranked_candidates || []).map(a => (
                        <tr key={a.atm_id}>
                          <td className="border border-zinc-300 px-2 py-1 align-top">{a.rank}</td>
                          <td className="border border-zinc-300 px-2 py-1 align-top">
                            <div className="font-mono">{a.atm_id}</div>
                            <div className="text-zinc-600">{a.bank}</div>
                          </td>
                          <td className="border border-zinc-300 px-2 py-1 align-top">
                            <div>{a.address}</div>
                            <div className="text-zinc-600">
                              {[a.district, a.state].filter(Boolean).join(', ')}
                            </div>
                          </td>
                          <td className="border border-zinc-300 px-2 py-1 align-top font-mono">
                            {fmtCoord(a.lat)}, {fmtCoord(a.lon)}
                          </td>
                          <td className="border border-zinc-300 px-2 py-1 align-top">
                            {a.opening_time || '—'}–{a.closing_time || '—'}
                          </td>
                          <td className="border border-zinc-300 px-2 py-1 align-top text-right">
                            {a.confidence != null ? (a.confidence * 100).toFixed(1) + '%' : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </Section>

            {/* ── Actions ───────────────────────────────────────────────── */}
            <Section title="5 · Actions taken on this case" note="Oldest first.">
              {(doc.actions || []).length === 0 && (
                <div className="text-[11.5px] text-zinc-700">
                  No recorded action on this case.
                </div>
              )}
              {(doc.actions || []).length > 0 && (
                <table className="w-full text-[10.5px] border border-zinc-300">
                  <thead>
                    <tr className="bg-zinc-100 text-left">
                      <th className="border border-zinc-300 px-2 py-1">When</th>
                      <th className="border border-zinc-300 px-2 py-1">Officer</th>
                      <th className="border border-zinc-300 px-2 py-1">Action</th>
                      <th className="border border-zinc-300 px-2 py-1">Object</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(doc.actions || []).map(a => (
                      <tr key={a.id}>
                        <td className="border border-zinc-300 px-2 py-1 align-top">
                          {fmtWhen(a.timestamp)}
                        </td>
                        <td className="border border-zinc-300 px-2 py-1 align-top">{a.actor}</td>
                        <td className="border border-zinc-300 px-2 py-1 align-top">{a.action}</td>
                        <td className="border border-zinc-300 px-2 py-1 align-top font-mono break-all">
                          {a.object}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Section>

            {/* ── Evidence ──────────────────────────────────────────────── */}
            <Section
              title="6 · Evidence held"
              note={`${cert.statute || ''} — ${cert.artefact_count ?? 0} artefact(s) on the custody chain.`}
            >
              {!intact && (
                <div className="border border-red-600 bg-red-50 text-red-800 px-3 py-2 text-[11.5px] mb-2">
                  Integrity check FAILED. This dossier must not be produced as evidence
                  until the custody chain is investigated.
                </div>
              )}
              {(cert.artefacts || []).length === 0 && (
                <div className="text-[11.5px] text-zinc-700">
                  No artefacts have been collected against this case.
                </div>
              )}
              {(cert.artefacts || []).length > 0 && (
                <table className="w-full text-[10.5px] border border-zinc-300">
                  <thead>
                    <tr className="bg-zinc-100 text-left">
                      <th className="border border-zinc-300 px-2 py-1">#</th>
                      <th className="border border-zinc-300 px-2 py-1">Artefact</th>
                      <th className="border border-zinc-300 px-2 py-1">SHA-256</th>
                      <th className="border border-zinc-300 px-2 py-1">Re-verified</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(cert.artefacts || []).map(a => (
                      <tr key={a.id} className={a.withdrawn ? 'text-zinc-500' : ''}>
                        <td className="border border-zinc-300 px-2 py-1 align-top">{a.seq}</td>
                        <td className="border border-zinc-300 px-2 py-1 align-top">
                          <div className={a.withdrawn ? 'line-through' : ''}>{a.filename}</div>
                          <div className="text-zinc-600">{a.collected_by}</div>
                        </td>
                        <td className="border border-zinc-300 px-2 py-1 align-top font-mono break-all">
                          {a.sha256}
                        </td>
                        <td className="border border-zinc-300 px-2 py-1 align-top">
                          {a.reverified_ok ? 'Matches' : `FAILED — ${a.reverification_note || ''}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="text-[10.5px] text-zinc-600 mt-1.5">
                A full s.63 certificate for these records is available from the case
                screen and should accompany this dossier when records are produced.
              </div>
            </Section>

            {/* ── Caveats ───────────────────────────────────────────────── */}
            <Section title="7 · What this document is not">
              <ul className="list-disc pl-5 space-y-1 text-[11px] text-zinc-700">
                {(doc.caveats || []).map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </Section>

            {/* ── Signatures ────────────────────────────────────────────── */}
            <div className="mt-8 pt-6 border-t border-zinc-300 grid grid-cols-2 gap-8 text-[11px]">
              <div>
                <div className="h-10 border-b border-zinc-500" />
                <div className="mt-1 text-zinc-700">Signature of the producing officer</div>
                <div className="text-zinc-600">{doc.produced_by}</div>
              </div>
              <div>
                <div className="h-10 border-b border-zinc-500" />
                <div className="mt-1 text-zinc-700">Date and place</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
