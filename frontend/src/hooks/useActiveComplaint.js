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

    const resolve = async () => {
      const known = urlCid || readStoredComplaintId()
      if (known) {
        if (!cancelled) {
          setComplaintId(known)
          setResolving(false)
          storeComplaintId(known)
        }
        return
      }

      setResolving(true)
      try {
        const list = await endpoints.listComplaints(1)
        if (!cancelled && list?.length) {
          setComplaintId(list[0].ticket_id)
          storeComplaintId(list[0].ticket_id)
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
