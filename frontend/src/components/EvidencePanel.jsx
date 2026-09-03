import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Paperclip, ShieldCheck, ShieldAlert, Download, Ban, FileText,
  Loader2, Printer, X,
} from 'lucide-react'
import { endpoints, describeError } from '../services/api'
import { useToast } from './Toast'
import { Panel } from './Shell'

/**
 * Evidence documentation — deliverable (c)'s last clause.
 *
 * The problem statement asks for a *"secure interface for investigators to
 * access alerts, intelligence reports, and evidence documentation"*. The first
 * two shipped; this is the third, and COMPLIANCE_AUDIT.md finding 4.5 recorded
 * that nothing in the repository could accept, hold or account for a file.
 *
 * What is on screen is deliberately not a file list. Three things an officer
 * has to be able to see without asking anybody:
 *
 *   - **the hash**, rendered, so it can be read off and compared against the
 *     copy they were handed;
 *   - **the chain state**, at the top, because a broken chain is the single
 *     most important fact about a case's evidence and must not be something you
 *     find by scrolling;
 *   - **the fact that withdrawal is not deletion** — withdrawn artefacts stay
 *     on the list, struck through, with the reason visible.
 *
 * `useToast()` returns the push FUNCTION, not an object holding one. See
 * components/Toast.jsx, and REMEDIATION_AUDIT.md §4.5 for what destructuring it
 * cost on the alert inbox.
 */

const KINDS = [
  ['bank_statement', 'Bank statement'],
  ['screenshot', 'Screenshot'],
  ['fir_copy', 'FIR copy'],
  ['transaction_export', 'Transaction export'],
  ['device_image', 'Device image'],
  ['correspondence', 'Correspondence'],
  ['other', 'Other'],
]
const KIND_LABEL = Object.fromEntries(KINDS)

function fmtBytes(n) {
  const b = Number(n) || 0
  if (b >= 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`
  return `${b} B`
}

function fmtWhen(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

export default function EvidencePanel({ caseId, className = '', bodyClass = '' }) {
  const toast = useToast()
  const fileRef = useRef(null)

  const [items, setItems] = useState([])
  const [check, setCheck] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [kind, setKind] = useState('bank_statement')
  const [description, setDescription] = useState('')
  const [source, setSource] = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)

  const [certificate, setCertificate] = useState(null)
  const [busyId, setBusyId] = useState('')
  // Withdrawal asks in the row rather than through window.prompt(). The freeze
  // control on this screen already established that an action with consequences
  // gets a deliberate second step in the console's own chrome; a native dialog
  // is a different affordance, blocks the page, and has nowhere to say what
  // withdrawal does and does not mean.
  const [withdrawing, setWithdrawing] = useState(null)   // { id, reason }

  const load = useCallback(() => {
    if (!caseId) return
    setLoading(true)
    setError('')
    Promise.all([
      endpoints.listEvidence(caseId),
      endpoints.verifyEvidence(caseId),
    ])
      .then(([rows, verification]) => { setItems(rows); setCheck(verification) })
      .catch(err => setError(describeError(err)))
      .finally(() => setLoading(false))
  }, [caseId])

  useEffect(() => { load() }, [load])

  const collect = useCallback(() => {
    const file = fileRef.current?.files?.[0]
    if (!file) { toast('Choose a file to collect.', 'error'); return }

    const form = new FormData()
    form.append('file', file)
    form.append('kind', kind)
    form.append('description', description)
    form.append('source', source)

    setUploading(true)
    setProgress(0)
    endpoints.collectEvidence(caseId, form, e => {
      if (e.total) setProgress(Math.round((e.loaded / e.total) * 100))
    })
      .then(item => {
        toast(`${item.filename} taken into custody as ${item.id}.`)
        if (fileRef.current) fileRef.current.value = ''
        setDescription('')
        setSource('')
        load()
      })
      .catch(err => toast(describeError(err), 'error'))
      .finally(() => { setUploading(false); setProgress(0) })
  }, [caseId, kind, description, source, toast, load])

  const download = useCallback((item) => {
    setBusyId(item.id)
    endpoints.downloadEvidence(item.id)
      .then(({ blob }) => {
        // The endpoint is Tier A, so the bytes arrive through axios with the
        // bearer attached rather than through a plain <a href>.
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = item.filename
        document.body.appendChild(a)
        a.click()
        a.remove()
        URL.revokeObjectURL(url)
      })
      .catch(err => toast(describeError(err), 'error'))
      .finally(() => setBusyId(''))
  }, [toast])

  const confirmWithdraw = useCallback(() => {
    const { id, reason } = withdrawing || {}
    if (!id) return
    if ((reason || '').trim().length < 4) {
      toast('A withdrawal needs a stated reason.', 'error')
      return
    }
    setBusyId(id)
    endpoints.withdrawEvidence(id, reason.trim())
      .then(() => {
        toast(`${id} withdrawn. The artefact stays on the record.`)
        load()
      })
      .catch(err => toast(describeError(err), 'error'))
      .finally(() => { setBusyId(''); setWithdrawing(null) })
  }, [withdrawing, toast, load])

  const openCertificate = useCallback(() => {
    endpoints.evidenceCertificate(caseId)
      .then(setCertificate)
      .catch(err => toast(describeError(err), 'error'))
  }, [caseId, toast])

  const intact = check?.intact !== false
  const count = items.filter(i => !i.withdrawn).length

  return (
    <>
      <Panel
        title="Evidence"
        right={
          <span className="flex items-center gap-1.5">
            <Paperclip size={12} />
            {count} artefact{count === 1 ? '' : 's'}
          </span>
        }
        className={className}
        bodyClass={`p-3 flex-1 flex flex-col min-h-0 ${bodyClass}`}
      >
        <div className="flex-1 flex flex-col space-y-3 min-h-0">
          {/* Chain state first. A broken chain is the most important fact about
              a case's evidence and must not be something you find by scrolling. */}
          {check && (
            <div
              className={`flex items-start gap-2 rounded border px-3 py-2 text-[11.5px] ${
                intact
                  ? 'border-ink-border bg-ink-panel text-zinc-400'
                  : 'border-red-500/50 bg-red-500/10 text-red-300'
              }`}
            >
              {intact
                ? <ShieldCheck size={14} className="text-aegis-accent shrink-0 mt-px" />
                : <ShieldAlert size={14} className="shrink-0 mt-px" />}
              <div className="min-w-0">
                {intact ? (
                  <>
                    <span className="text-zinc-300 font-medium">Chain of custody intact.</span>{' '}
                    {check.items} artefact{check.items === 1 ? '' : 's'} re-hashed and
                    each link reproduced from the one before it.
                  </>
                ) : (
                  <>
                    <span className="font-semibold">Integrity check FAILED.</span>{' '}
                    {check.content_ok === false && 'An artefact no longer matches the hash recorded when it was collected. '}
                    {check.chain_ok === false && 'The chain does not reproduce, so the set has been altered, reordered or added to. '}
                    This case&rsquo;s evidence should not be produced until it is explained.
                  </>
                )}
              </div>
            </div>
          )}

          {/* ── Collect ─────────────────────────────────────────────────── */}
          <div className="rounded border border-ink-border bg-ink-panel p-3 space-y-2">
            <div className="text-[11px] text-zinc-500">
              Take an artefact into custody. Its SHA-256 is recorded at this moment
              and re-checked on every read.
            </div>
            <input
              ref={fileRef}
              type="file"
              aria-label="Artefact"
              className="block w-full text-[11.5px] text-zinc-400 file:mr-3 file:rounded
                         file:border file:border-ink-border file:bg-ink-surface
                         file:px-3 file:py-1.5 file:text-[11.5px] file:text-zinc-300
                         hover:file:text-white"
            />
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="sr-only">Kind</span>
                <select
                  aria-label="Artefact kind"
                  value={kind}
                  onChange={e => setKind(e.target.value)}
                  className="w-full bg-ink-surface border border-ink-border rounded
                             px-2 py-1.5 text-[11.5px] text-zinc-300"
                >
                  {KINDS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                </select>
              </label>
              <input
                aria-label="Source"
                value={source}
                onChange={e => setSource(e.target.value)}
                placeholder="Source (bank, victim, device)"
                className="w-full bg-ink-surface border border-ink-border rounded
                           px-2 py-1.5 text-[11.5px] text-zinc-300 placeholder:text-zinc-600"
              />
            </div>
            <input
              aria-label="Description"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="What this is, in one line"
              className="w-full bg-ink-surface border border-ink-border rounded
                         px-2 py-1.5 text-[11.5px] text-zinc-300 placeholder:text-zinc-600"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={collect}
                disabled={uploading}
                className="px-3 py-1.5 rounded border border-ink-border bg-ink-surface
                           text-[11.5px] text-zinc-200 hover:text-white
                           disabled:opacity-50 flex items-center gap-1.5"
              >
                {uploading
                  ? <><Loader2 size={13} className="animate-spin" /> {progress}%</>
                  : <><Paperclip size={13} /> Collect artefact</>}
              </button>
              <button
                onClick={openCertificate}
                disabled={!items.length}
                className="px-3 py-1.5 rounded border border-ink-border text-[11.5px]
                           text-zinc-400 hover:text-white disabled:opacity-40
                           flex items-center gap-1.5"
              >
                <FileText size={13} /> s.63 certificate
              </button>
            </div>
          </div>

          {/* ── The record ──────────────────────────────────────────────── */}
          {error && <div className="text-[11.5px] text-red-300">{error}</div>}

          {loading && !items.length ? (
            <div className="text-[11.5px] text-zinc-500 py-6 text-center flex-1 flex items-center justify-center">Loading…</div>
          ) : !items.length ? (
            <div className="text-[11.5px] text-zinc-500 py-6 text-center flex-1 flex items-center justify-center">
              No artefacts held against this case.
            </div>
          ) : (
            <div className="overflow-x-auto overflow-y-auto flex-1 min-h-0">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-zinc-500 text-left border-b border-ink-border">
                    <th className="py-1.5 pr-2 font-medium">#</th>
                    <th className="py-1.5 pr-2 font-medium">Artefact</th>
                    <th className="py-1.5 pr-2 font-medium">SHA-256</th>
                    <th className="py-1.5 pr-2 font-medium">Collected</th>
                    <th className="py-1.5 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map(item => (
                    <tr
                      key={item.id}
                      className={`border-b border-ink-border/60 align-top ${
                        item.withdrawn ? 'opacity-50' : ''
                      }`}
                    >
                      <td className="py-2 pr-2 mono tnum text-zinc-500">{item.seq}</td>
                      <td className="py-2 pr-2 min-w-0">
                        <div className={`text-zinc-200 truncate max-w-[220px] ${
                          item.withdrawn ? 'line-through' : ''
                        }`}>
                          {item.filename}
                        </div>
                        <div className="text-zinc-500 mono text-[10px]">
                          {item.id} · {KIND_LABEL[item.kind] || item.kind} · {fmtBytes(item.size_bytes)}
                        </div>
                        {item.description && (
                          <div className="text-zinc-500 text-[10.5px] max-w-[240px]">
                            {item.description}
                          </div>
                        )}
                        {item.withdrawn && (
                          <div className="text-red-300/80 text-[10.5px] max-w-[240px]">
                            Withdrawn by {item.withdrawn_by}: {item.withdrawn_reason}
                          </div>
                        )}
                      </td>
                      <td className="py-2 pr-2 mono text-[10px] text-zinc-400 break-all max-w-[120px]">
                        {item.sha256.slice(0, 16)}…
                      </td>
                      <td className="py-2 pr-2 text-zinc-400">
                        <div className="truncate max-w-[150px]">{item.collected_by}</div>
                        <div className="text-zinc-600 text-[10px]">{fmtWhen(item.collected_at)}</div>
                      </td>
                      <td className="py-2 text-right whitespace-nowrap">
                        <button
                          onClick={() => download(item)}
                          disabled={busyId === item.id}
                          title="Download (hash-checked first)"
                          aria-label={`Download ${item.filename}`}
                          className="p-1 rounded border border-ink-border text-zinc-400
                                     hover:text-white disabled:opacity-40"
                        >
                          <Download size={12} />
                        </button>
                        {!item.withdrawn && (
                          <button
                            onClick={() => setWithdrawing({ id: item.id, reason: '' })}
                            disabled={busyId === item.id}
                            title="Withdraw (not a delete)"
                            aria-label={`Withdraw ${item.filename}`}
                            className="ml-1 p-1 rounded border border-ink-border text-zinc-400
                                       hover:text-red-300 disabled:opacity-40"
                          >
                            <Ban size={12} />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                  {withdrawing && items.some(i => i.id === withdrawing.id) && (
                    <tr>
                      <td colSpan={5} className="py-2">
                        <div className="rounded border border-red-500/40 bg-red-500/5 p-2.5 space-y-2">
                          <div className="text-[11px] text-zinc-300">
                            Withdraw <span className="mono">{withdrawing.id}</span>?
                            <span className="text-zinc-500">
                              {' '}It is <strong className="text-zinc-300">not deleted</strong> — the
                              artefact, its hash and its place in the chain stay on the record,
                              with this reason attached.
                            </span>
                          </div>
                          <input
                            autoFocus
                            aria-label="Reason for withdrawal"
                            value={withdrawing.reason}
                            onChange={e => setWithdrawing(w => ({ ...w, reason: e.target.value }))}
                            onKeyDown={e => { if (e.key === 'Enter') confirmWithdraw() }}
                            placeholder="Reason (required)"
                            className="w-full bg-ink-surface border border-ink-border rounded
                                       px-2 py-1.5 text-[11.5px] text-zinc-200
                                       placeholder:text-zinc-600"
                          />
                          <div className="flex justify-end gap-2">
                            <button
                              onClick={() => setWithdrawing(null)}
                              className="px-3 py-1 rounded border border-ink-border text-[11.5px]
                                         text-zinc-400 hover:text-white"
                            >
                              Cancel
                            </button>
                            <button
                              onClick={confirmWithdraw}
                              disabled={busyId === withdrawing.id}
                              className="px-3 py-1 rounded border border-red-500/50 bg-red-500/10
                                         text-[11.5px] text-red-200 hover:text-white
                                         disabled:opacity-50"
                            >
                              Withdraw
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          <div className="text-[10.5px] text-zinc-600 leading-relaxed mt-auto pt-2.5 border-t border-ink-border shrink-0">
            Withdrawal is not deletion: the artefact, its hash and its position in
            the chain stay on the record with the reason attached. The chain shows
            that this set is internally consistent; it is not anchored outside this
            system, which a deployment would need to add.
          </div>
        </div>
      </Panel>

      {certificate && (
        <CertificateModal cert={certificate} onClose={() => setCertificate(null)} />
      )}
    </>
  )
}

function CertificateModal({ cert, onClose }) {
  const intact = cert?.verification?.intact !== false
  return (
    <div className="fixed inset-0 z-[3000] bg-black/70 flex items-start justify-center
                    overflow-y-auto p-4 no-print-backdrop">
      <div className="evidence-certificate bg-white text-black rounded max-w-3xl w-full my-6 shadow-2xl">
        <div className="no-print flex items-center justify-between px-5 py-3 border-b border-zinc-300">
          <div className="text-[12px] font-semibold text-zinc-700">
            Certificate — {cert.statute}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => window.print()}
              className="px-3 py-1.5 rounded border border-zinc-300 text-[11.5px]
                         text-zinc-700 hover:bg-zinc-100 flex items-center gap-1.5"
            >
              <Printer size={13} /> Print
            </button>
            <button
              onClick={onClose}
              aria-label="Close certificate"
              className="p-1.5 rounded border border-zinc-300 text-zinc-600 hover:bg-zinc-100"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <div className="px-8 py-6 text-[12px] leading-relaxed">
          <div className="text-center border-b border-zinc-300 pb-4">
            <div className="text-[15px] font-bold uppercase tracking-wide">
              Certificate under {cert.statute}
            </div>
            <div className="text-[11px] text-zinc-600 mt-1">{cert.statute_note}</div>
            <div className="text-[11px] text-zinc-700 mt-2">
              {cert.system?.operator} · {cert.system?.name} · {cert.system?.problem_statement}
            </div>
          </div>

          {!intact && (
            <div className="mt-4 border border-red-600 bg-red-50 text-red-800 px-3 py-2 text-[11.5px]">
              <strong>Integrity check failed at the time this certificate was produced.</strong>{' '}
              One or more artefacts no longer match the record. This document states
              the failure rather than suppressing it, and the affected artefacts are
              marked below. It must not be produced as though the set were intact.
            </div>
          )}

          <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-[11.5px]">
            <div><span className="text-zinc-600">Case:</span> {cert.case_id}</div>
            <div><span className="text-zinc-600">Produced:</span> {fmtWhen(cert.produced_at)}</div>
            <div><span className="text-zinc-600">Custodian:</span> {cert.custodian}</div>
            <div><span className="text-zinc-600">Artefacts:</span> {cert.artefact_count}
              {cert.withdrawn_count > 0 && ` (+${cert.withdrawn_count} withdrawn)`}</div>
            {cert.complaint && (
              <>
                <div><span className="text-zinc-600">Complainant:</span> {cert.complaint.victim_name}</div>
                <div><span className="text-zinc-600">Bank:</span> {cert.complaint.victim_bank}</div>
                <div><span className="text-zinc-600">Offence:</span> {cert.complaint.fraud_type}</div>
                <div><span className="text-zinc-600">Place:</span> {cert.complaint.city}, {cert.complaint.state}</div>
              </>
            )}
          </div>

          <ol className="mt-5 space-y-2 list-decimal pl-5 text-[11.5px]">
            {(cert.statements || []).map((s, i) => <li key={i}>{s}</li>)}
          </ol>

          <div className="mt-5">
            <div className="font-semibold text-[12px] mb-1">Schedule of electronic records</div>
            <table className="w-full text-[10.5px] border border-zinc-300">
              <thead>
                <tr className="bg-zinc-100 text-left">
                  <th className="border border-zinc-300 px-2 py-1">#</th>
                  <th className="border border-zinc-300 px-2 py-1">Artefact</th>
                  <th className="border border-zinc-300 px-2 py-1">SHA-256</th>
                  <th className="border border-zinc-300 px-2 py-1">Collected by / at</th>
                  <th className="border border-zinc-300 px-2 py-1">Re-verified</th>
                </tr>
              </thead>
              <tbody>
                {(cert.artefacts || []).map(a => (
                  <tr key={a.id} className={a.withdrawn ? 'text-zinc-500' : ''}>
                    <td className="border border-zinc-300 px-2 py-1 align-top">{a.seq}</td>
                    <td className="border border-zinc-300 px-2 py-1 align-top">
                      <div className={a.withdrawn ? 'line-through' : ''}>{a.filename}</div>
                      <div className="text-zinc-600">{a.id} · {a.kind} · {fmtBytes(a.size_bytes)}</div>
                      {a.description && <div className="text-zinc-600">{a.description}</div>}
                      {a.source && <div className="text-zinc-600">Source: {a.source}</div>}
                      {a.withdrawn && (
                        <div className="text-zinc-700">Withdrawn: {a.withdrawn_reason}</div>
                      )}
                    </td>
                    <td className="border border-zinc-300 px-2 py-1 align-top font-mono break-all">
                      {a.sha256}
                    </td>
                    <td className="border border-zinc-300 px-2 py-1 align-top">
                      <div>{a.collected_by}</div>
                      <div className="text-zinc-600">{fmtWhen(a.collected_at)}</div>
                    </td>
                    <td className="border border-zinc-300 px-2 py-1 align-top">
                      {a.reverified_ok
                        ? 'Matches'
                        : `FAILED — ${a.reverification_note || 'does not match'}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-5">
            <div className="font-semibold text-[12px] mb-1">Stated limitations</div>
            <ul className="list-disc pl-5 space-y-1 text-[11px] text-zinc-700">
              {(cert.limitations || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>

          <div className="mt-8 pt-6 border-t border-zinc-300 grid grid-cols-2 gap-8 text-[11px]">
            <div>
              <div className="h-10 border-b border-zinc-500" />
              <div className="mt-1 text-zinc-700">Signature of the custodian</div>
              <div className="text-zinc-600">{cert.custodian}</div>
            </div>
            <div>
              <div className="h-10 border-b border-zinc-500" />
              <div className="mt-1 text-zinc-700">Date and place</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
