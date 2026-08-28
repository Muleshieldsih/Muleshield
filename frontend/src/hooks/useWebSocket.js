import { useEffect, useRef, useState, useCallback } from 'react'
import { wsUrl } from '../services/api'

export default function useWebSocket(onMessage) {
  const [status, setStatus] = useState('connecting')
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const retryRef = useRef(0)

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(wsUrl())
      wsRef.current = ws
      ws.onopen = () => { setStatus('open'); setConnected(true); retryRef.current = 0 }
      ws.onclose = () => {
        setStatus('closed'); setConnected(false)
        const delay = Math.min(1000 * Math.pow(1.6, retryRef.current++), 8000)
        setTimeout(connect, delay)
      }
      ws.onerror = () => { setStatus('error'); try{ws.close()}catch{} }
      ws.onmessage = (ev) => {
        try { const data = JSON.parse(ev.data); onMessage && onMessage(data) } catch {}
      }
    } catch { setStatus('error') }
  }, [onMessage])

  useEffect(() => { connect(); return () => { try{wsRef.current?.close()}catch{} } }, [connect])
  return { status, connected }
}
