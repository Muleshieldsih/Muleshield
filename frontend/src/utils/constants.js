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

/** Benchmarked model figures, from engine/train_xgb.py + train_gnn.py. */
// Measured figures, each with the naive baseline it beats. Regenerate with
// `python engine/train_xgb.py` and `python scripts/evaluate_baselines.py`.
//
// These previously read 98.5% Top-3 / 1.2 s MAE / 0.9996 F1. Those numbers were
// retracted by the leakage audit (OVERNIGHT_ML_AUDIT.md) and were still being
// displayed in the console long after every document had been corrected.
export const MODEL_STATS = {
  zoneContainment: '87.4%',
  zoneBaseline: '78.5%',
  searchCost: '8 of 1,000',
  countdownMae: '11.8 min',
  countdownBaseline: '15.0 min',
  gnnF1: '0.8955',
  gnnBaseline: '0.8463',
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
