import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { endpoints } from '../services/api'

export const SELECTED_KEY = 'muleshield:selected'

export function readStoredComplaintId() {
  if (typeof window === 'undefined') return ''
  try {
    return localStorage.getItem(SELECTED_KEY) || ''
  } catch {
    return ''
  }
}

export function clearComplaintId() {
  if (typeof window === 'undefined') return
  try {
    localStorage.removeItem(SELECTED_KEY)
  } catch {
    /* private browsing */
  }
}

export function storeComplaintId(id) {
  if (typeof window === 'undefined' || !id) return
  try {
    localStorage.setItem(SELECTED_KEY, id)
  } catch {
    /* private browsing — selection just won't persist */
  }
}

/**
 * Resolve which complaint the Map / Graph / Interception screens should show.
 *
 * Order: ?c= in the URL → last selection in localStorage → newest complaint
 * from the backend. Returns `null` for the id only while still resolving, so
 * callers can distinguish "loading" from "nothing to show".
 */
export default function useActiveComplaint() {
  const [params] = useSearchParams()
  const urlCid = params.get('c') || ''
  const [complaintId, setComplaintId] = useState(() => urlCid || readStoredComplaintId())
  const [resolving, setResolving] = useState(!(urlCid || readStoredComplaintId()))

  useEffect(() => {
    let cancelled = false

    const newestComplaint = async () => {
      const list = await endpoints.listComplaints(1)
      return list?.length ? list[0].ticket_id : ''
    }

    const resolve = async () => {
      const known = urlCid || readStoredComplaintId()
      setResolving(true)

      try {
        if (known) {
          // Verify it still exists before adopting it.
          //
          // A stored id outlives the data it points at: a ticket from a previous
          // dataset, or one that was only ever in the removed demo fixtures, stays
          // in localStorage forever. Trusting it blindly left every screen stuck on
          // "complaint not found" while the live queue beside them was populated —
          // which reads as a broken app rather than a stale selection.
          try {
            await endpoints.getComplaint(known)
            if (!cancelled) {
              setComplaintId(known)
              storeComplaintId(known)
            }
            return
          } catch (err) {
            // Only a genuine 404 invalidates it. A network failure or a backend
            // that is still booting must not discard a valid selection.
            if (err?.response?.status !== 404) {
              if (!cancelled) setComplaintId(known)
              return
            }
            clearComplaintId()
          }
        }

        const newest = await newestComplaint()
        if (!cancelled && newest) {
          setComplaintId(newest)
          storeComplaintId(newest)
        }
      } catch {
        /* leave empty — the page renders its own disconnected state */
      } finally {
        if (!cancelled) setResolving(false)
      }
    }

    resolve()
    return () => { cancelled = true }
  }, [urlCid])

  return { complaintId, resolving }
}
