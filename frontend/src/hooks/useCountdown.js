import { useEffect, useRef, useState } from 'react'

/**
 * Countdown to a cashout deadline.
 *
 * Anchored to an absolute deadline rather than decremented once per tick:
 * a `setInterval` that subtracts 1 drifts whenever the tab is backgrounded or
 * the main thread is busy, which on a 30-minute interception clock can be
 * minutes of error by the end of a demo.
 */
export function useCountdown(minutes) {
  const deadlineRef = useRef(null)
  const [remaining, setRemaining] = useState(() => Math.max(0, Math.round((Number(minutes) || 0) * 60)))

  // Re-anchor whenever the predicted window changes.
  useEffect(() => {
    const secs = Math.max(0, Math.round((Number(minutes) || 0) * 60))
    deadlineRef.current = Date.now() + secs * 1000
    setRemaining(secs)
  }, [minutes])

  useEffect(() => {
    const tick = () => {
      if (deadlineRef.current == null) return
      const left = Math.max(0, Math.round((deadlineRef.current - Date.now()) / 1000))
      setRemaining(left)
    }
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  const ss = String(remaining % 60).padStart(2, '0')
  return { remaining, mm, ss, label: `${mm}:${ss}`, expired: remaining === 0 }
}

export function timeAgo(iso, now = Date.now()) {
  if (!iso) return '—'
  try {
    const t = new Date(iso).getTime()
    if (Number.isNaN(t)) return '—'
    const s = Math.floor((now - t) / 1000)
    if (s < 0) return 'just now'
    if (s < 60) return `${s}s ago`
    if (s < 3600) return `${Math.floor(s / 60)}m ago`
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`
    return `${Math.floor(s / 86400)}d ago`
  } catch {
    return '—'
  }
}

/**
 * Shared 1-second clock. Every screen showing "3m ago" needs a repaint tick;
 * without a shared hook each page spins up its own interval and they drift
 * out of phase with one another.
 */
export function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}
