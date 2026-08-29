export const FRAUD_TYPES = [
  'UPI Fraud',
  'Digital Arrest',
  'Job Scam',
  'Investment Scam',
  'KYC Fraud',
  'Romance Scam',
  'Electricity Bill Scam',
]

export const BANKS = [
  'State Bank of India',
  'HDFC Bank',
  'ICICI Bank',
  'Axis Bank',
  'Punjab National Bank',
  'Bank of Baroda',
  'Canara Bank',
  'Kotak Mahindra Bank',
  'Union Bank of India',
  'Indian Bank',
  'YES Bank',
  'IndusInd Bank',
]

export const CITIES = [
  'Delhi', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata',
  'Pune', 'Jaipur', 'Lucknow', 'Gurgaon', 'Jamtara', 'Deoghar',
]

/** Benchmarked model figures — generated, never typed. */
// These previously read 98.5% Top-3 / 1.2 s MAE / 0.9996 F1. Those numbers were
// retracted by the leakage audit (OVERNIGHT_ML_AUDIT.md) and were still being
// displayed in the console long after every document had been corrected. They
// survived because they were hand-written here, so nothing tied them to a
// trained model.
//
// Now they are not written here at all. model_stats.json is generated from
// data/metrics.json, which only the training and evaluation scripts write:
//
//   python engine/train_gnn.py            -> detection
//   python scripts/evaluate_baselines.py  -> detection_baselines
//   python engine/train_xgb.py            -> location  (+ refreshes this file)
//   python scripts/export_metrics.py      -> refresh without retraining
//
// Raw numbers live in the JSON; this module owns how they are displayed, so
// there is exactly one place that decides what '87.4%' looks like.
import stats from '../data/model_stats.json'

export const MODEL_STATS = {
  zoneContainment: pctFmt(stats.zoneContainment),
  zoneBaseline: pctFmt(stats.zoneBaselineNearest3),
  searchCost: `${Math.round(stats.zoneMedianAtms).toLocaleString('en-IN')} of ${stats.atmTotal.toLocaleString('en-IN')}`,
  zoneRadius: `${stats.zoneMedianRadiusKm.toFixed(1)} km`,
  countdownMae: `${stats.countdownMae.toFixed(1)} min`,
  countdownBaseline: `${stats.countdownBaseline.toFixed(1)} min`,
  gnnF1: stats.gnnF1.toFixed(4),
  gnnBaseline: stats.gnnBaseline.toFixed(4),
  gnnBaselineModel: stats.gnnBaselineModel,

  // The operating point the product actually ships: K ranked candidate ATMs.
  // These sat in model_stats.json unread, so the headline figure appeared
  // nowhere in the console it describes.
  //
  // Deliberately no baseline beside top5Containment. The distance-only baseline
  // is 0.7217 against our 0.7136 -- we do NOT beat distance-sorting at K=5, and
  // a "vs" here would either be a false win or a bare loss with no room for the
  // Bayes-ceiling context that explains it. The claim we make on screen is the
  // search-space reduction, which is true and is the point of the system. The
  // full comparison lives in README.md and OVERNIGHT_ML_AUDIT.md.
  operatingK: stats.operatingK,
  top5Containment: pctFmt(stats.top5Containment),
  top5Reduction: pctFmt(stats.top5SearchReduction),
  atmTotal: stats.atmTotal.toLocaleString('en-IN'),
}

export function amountFmt(n) {
  const v = Number(n)
  if (!Number.isFinite(v)) return '₹—'
  return `₹${v.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

/** Compact Indian-format amount: ₹2.5L, ₹1.2Cr — for dense cards and tickers. */
export function amountShort(n) {
  const v = Number(n)
  if (!Number.isFinite(v)) return '₹—'
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)}Cr`
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2)}L`
  if (v >= 1e3) return `₹${(v / 1e3).toFixed(0)}K`
  return `₹${v.toFixed(0)}`
}

export function pctFmt(p) {
  const v = Number(p)
  return Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—'
}

/**
 * A case reference for display.
 *
 * The store holds two id formats: seed complaints carry a raw UUID
 * (`bb295c33-b740-48d6-b88c-bf8a123af907`), complaints ingested during the
 * session carry `TKT-` plus eight hex characters. Shown side by side in one
 * queue they read as records from two different systems.
 *
 * Both collapse to one reference shaped like something the 1930 helpline would
 * actually issue. It is derived from the id alone, deterministically, so the
 * same case renders identically in the queue, the topbar, the graph title and
 * an alert message — an earlier version formatted in some places and not
 * others, and the same case appeared under two identities.
 *
 * Display only. Routing, the API and localStorage always use the untouched
 * ticket_id.
 *
 * (The brief's example embeds a year and month. That would need the complaint's
 * timestamp, which several call sites do not have — the topbar holds an id and
 * nothing else — and deriving it where available would reintroduce exactly the
 * two-identities problem this exists to remove.)
 */
export function formatTicket(id) {
  const s = String(id || '')
  if (!s) return '—'
  // FNV-1a: small, stable, and no dependency. Not a security hash; it only has
  // to be deterministic and spread ids across the range.
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return `1930-${String(h % 1000000).padStart(6, '0')}`
}

/** Shorten a long account number for dense table cells, keeping the tail. */
export function shortAccount(id) {
  const s = String(id || '')
  if (s.length <= 14) return s
  return `…${s.slice(-11)}`
}
