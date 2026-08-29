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

export function formatTicket(id) {
  if (!id) return '—'
  return String(id)
}

/** Shorten a long account number for dense table cells, keeping the tail. */
export function shortAccount(id) {
  const s = String(id || '')
  if (s.length <= 14) return s
  return `…${s.slice(-11)}`
}
