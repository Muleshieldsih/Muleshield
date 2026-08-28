import { useEffect, useState } from 'react'

export function useCountdown(minutes) {
  const targetSeconds = Math.max(0, Math.round((Number(minutes) || 0) * 60))
  const [remaining, setRemaining] = useState(targetSeconds)

  useEffect(() => {
    setRemaining(Math.max(0, Math.round((Number(minutes) || 0) * 60)))
  }, [minutes])

  useEffect(() => {
    const id = setInterval(() => {
      setRemaining(prev => (prev > 0 ? prev - 1 : 0))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  const ss = String(remaining % 60).padStart(2, '0')
  return { remaining, mm, ss, label: `${mm}:${ss}` }
}

export function timeAgo(iso) {
  if (!iso) return '—'
  try {
    const diff = Date.now() - new Date(iso).getTime()
    const s = Math.floor(diff / 1000)
    if (s < 0 || isNaN(s)) return 'just now'
    if (s < 60) return `${s}s ago`
    if (s < 3600) return `${Math.floor(s / 60)}m ago`
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`
    return `${Math.floor(s / 86400)}d ago`
  } catch {
    return '—'
  }
}
