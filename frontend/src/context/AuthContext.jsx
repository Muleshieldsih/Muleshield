import { createContext, useContext, useCallback, useEffect, useState } from 'react'
import { endpoints, describeError } from '../services/api'
import { getToken, setToken, clearToken, setExpiryHandler } from '../services/auth'

/**
 * Who is signed in.
 *
 * `ready` is the one non-obvious piece: on a page load with a stored token we do
 * not yet know whether that token is still good, and rendering the login screen
 * in the meantime would flash it in front of an officer who is perfectly signed
 * in. So the app shows nothing until /auth/me has answered once.
 */
const AuthContext = createContext(null)

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)
  const [expired, setExpired] = useState(false)

  // Restore a session on load. A stored token that the server no longer accepts
  // is cleared by the response interceptor, which lands us in the same place as
  // never having had one.
  useEffect(() => {
    let cancelled = false
    if (!getToken()) {
      setReady(true)
      return
    }
    endpoints.me()
      .then(me => { if (!cancelled) setUser(me) })
      .catch(err => {
        if (cancelled) return
        // Only a REFUSAL ends the session. This used to discard the token on any
        // failure at all, which meant a request aborted by navigating during
        // boot -- or one transient network blip -- signed the officer out and
        // dropped them on the login form mid-shift.
        //
        // scripts/smoke_ui.py found it: every route after the first reported
        // three controls and a clean sweep of a login screen, because /auth/me
        // had been aborted by the sweep navigating away.
        //
        // A 401/403 is the server saying the token is no good, and that is worth
        // clearing for. Everything else -- no response, an abort, a 5xx -- means
        // we do not know, and destroying a valid session on "we do not know" is
        // the wrong default.
        const status = err?.response?.status
        if (status === 401 || status === 403) {
          clearToken()
          setUser(null)
        }
      })
      .finally(() => { if (!cancelled) setReady(true) })
    return () => { cancelled = true }
  }, [])

  // The axios interceptor cannot call a hook, so it calls through here instead.
  useEffect(() => {
    setExpiryHandler(() => {
      setUser(null)
      setExpired(true)
    })
    return () => setExpiryHandler(null)
  }, [])

  const login = useCallback(async (username, password, { persist = true } = {}) => {
    // Deliberately not try/catch'd into a boolean: the caller needs the message,
    // and the backend's 423 lockout text is the most useful thing we can show.
    const data = await endpoints.login(username, password)
    setToken(data.access_token, { persist })
    setUser(data.user)
    setExpired(false)
    return data.user
  }, [])

  const logout = useCallback(async () => {
    try {
      await endpoints.logout()
    } catch {
      // A failed logout must still sign the officer out locally. The session row
      // expires on its own; leaving them looking at a console they think they
      // have left is the worse outcome.
    }
    clearToken()
    setUser(null)
    setExpired(false)
  }, [])

  return (
    <AuthContext.Provider value={{ user, ready, expired, login, logout, describeError }}>
      {children}
    </AuthContext.Provider>
  )
}
