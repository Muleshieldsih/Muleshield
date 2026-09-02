import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { CheckCircle2, AlertTriangle, X } from 'lucide-react'

/**
 * Transient confirmations.
 *
 * Every action in the console previously reported itself with a permanent
 * inline banner that stayed until the component remounted, so the record of a
 * freeze from ten minutes ago sat beside the case an analyst had since moved
 * on to. An action needs to be acknowledged and then get out of the way.
 *
 * Written rather than installed: this is one small component against a
 * dependency, and the styling has to match a console that already has its own
 * panel chrome.
 */

const ToastContext = createContext(() => {})

export function useToast() {
  return useContext(ToastContext)
}

const LIFETIME_MS = 5000

export function ToastProvider({ children }) {
  const [items, setItems] = useState([])

  const dismiss = useCallback(id => {
    setItems(prev => prev.filter(t => t.id !== id))
  }, [])

  const push = useCallback((message, tone = 'success') => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    setItems(prev => [...prev.slice(-3), { id, message, tone }])
    return id
  }, [])

  useEffect(() => {
    if (!items.length) return undefined
    const timers = items.map(t => setTimeout(() => dismiss(t.id), LIFETIME_MS))
    return () => timers.forEach(clearTimeout)
  }, [items, dismiss])

  const value = useMemo(() => push, [push])

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* aria-live so the confirmation reaches a screen reader; an analyst
          driving this by keyboard should not have to look for it. */}
      <div
        className="fixed bottom-8 right-4 z-[2000] flex flex-col gap-2 pointer-events-none"
        aria-live="polite"
        role="status"
      >
        {items.map(t => (
          <div
            key={t.id}
            className={`animate-toast-in pointer-events-auto flex items-start gap-2 rounded border
                        px-3 py-2 text-[12px] shadow-lg max-w-sm ${
              t.tone === 'error'
                ? 'bg-ink-surface border-red-500/40 text-red-300'
                : 'bg-ink-surface border-aegis-accent/40 text-zinc-200'
            }`}
          >
            {t.tone === 'error'
              ? <AlertTriangle size={14} className="text-red-400 shrink-0 mt-0.5" />
              : <CheckCircle2 size={14} className="text-aegis-accent shrink-0 mt-0.5" />}
            <span className="flex-1 leading-snug">{t.message}</span>
            <button
              onClick={() => dismiss(t.id)}
              className="text-zinc-500 hover:text-zinc-200 shrink-0 transition-colors"
              aria-label="Dismiss notification"
            >
              <X size={13} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
