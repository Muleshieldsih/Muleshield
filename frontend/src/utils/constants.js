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
export const MODEL_STATS = {
  top3Accuracy: '98.5%',
  countdownMae: '1.2 s',
  gnnF1: '0.9996',
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
