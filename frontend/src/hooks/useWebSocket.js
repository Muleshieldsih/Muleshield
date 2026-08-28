import { useEffect, useRef, useState, useCallback } from 'react'
import { wsUrl } from '../services/api'

export default function useWebSocket(onMessage) {
  const [status, setStatus] = useState('connecting')
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const retryRef = useRef(0)
  const timerRef = useRef(null)
  const unmountedRef = useRef(false)

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    // Clear any existing reconnect timer
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }

    try {
      const ws = new WebSocket(wsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        if (unmountedRef.current) { ws.close(); return }
        setStatus('open')
        setConnected(true)
        retryRef.current = 0
      }

      ws.onclose = () => {
        if (unmountedRef.current) return
        setStatus('closed')
        setConnected(false)
        const delay = Math.min(1000 * Math.pow(1.5, retryRef.current++), 8000)
        timerRef.current = setTimeout(connect, delay)
      }

      ws.onerror = () => {
        if (unmountedRef.current) return
        setStatus('error')
        try { ws.close() } catch {}
      }

      ws.onmessage = (ev) => {
        if (unmountedRef.current) return
        try {
          const data = JSON.parse(ev.data)
          if (onMessage) onMessage(data)
        } catch {
          // ignore non-json messages
        }
      }
    } catch {
      setStatus('error')
      setConnected(false)
    }
  }, [onMessage])

  useEffect(() => {
    unmountedRef.current = false
    connect()
    return () => {
      unmountedRef.current = true
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      try {
        wsRef.current?.close()
      } catch {}
    }
  }, [connect])

  return { status, connected }
}
