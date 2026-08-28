import { useEffect, useState } from 'react'

export function useCountdown(minutes) {
  const [remaining, setRemaining] = useState(() => Math.max(0, Math.round((minutes || 0) * 60)))
  useEffect(() => { setRemaining(Math.max(0, Math.round((minutes || 0) * 60))) }, [minutes])
  useEffect(() => {
    if (remaining <= 0) return
    const id = setInterval(() => setRemaining(s => s - 1 > 0 ? s - 1 : 0), 1000)
    return () => clearInterval(id)
  }, [remaining])
  const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  const ss = String(remaining % 60).padStart(2, '0')
  return { remaining, mm, ss, label: `${mm}:${ss}` }
}

export function timeAgo(iso) {
  try {
    const diff = Date.now() - new Date(iso).getTime()
    const s = Math.floor(diff / 1000)
    if (s < 60) return `${s}s ago`
    if (s < 3600) return `${Math.floor(s / 60)}m ago`
    return `${Math.floor(s / 3600)}h ago`
  } catch { return '—' }
}
