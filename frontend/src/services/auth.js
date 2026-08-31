/**
 * Where the bearer token lives on the client.
 *
 * localStorage, mirroring the `muleshield:selected` convention already used by
 * hooks/useActiveComplaint.js — including its try/catch wrappers, because a
 * private window throws on access rather than returning null, and an officer in
 * a locked-down browser must still get a login screen rather than a blank page.
 *
 * Not a cookie: backend/main.py sets `allow_origins: ["*"]` together with
 * `allow_credentials: True`, a combination browsers refuse for credentialed
 * requests. A header sidesteps CORS entirely and leaves the working console
 * untouched.
 *
 * The trade being made: localStorage is readable by any script that gets injected
 * into this origin. It is mitigated by a 12-hour expiry and by the fact that
 * sessions are revocable server-side — logout kills the row, so a stolen token
 * does not outlive the shift. sessionStorage would be marginally safer and would
 * sign the officer out on every new tab, which is worse in a control room.
 */

export const TOKEN_KEY = 'muleshield:token'

// Mirrored in memory so a page load can decide "signed in or not" synchronously,
// without a flash of the login screen while a hook hydrates.
let cached = read()

function read() {
  try {
    return window.localStorage.getItem(TOKEN_KEY) || ''
  } catch {
    return ''
  }
}

export function getToken() {
  return cached
}

export function setToken(token) {
  cached = token || ''
  try {
    if (cached) window.localStorage.setItem(TOKEN_KEY, cached)
    else window.localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* private browsing: the in-memory copy still carries this session */
  }
}

export function clearToken() {
  setToken('')
}

/**
 * A single global handler for "the server says this token is no longer good".
 *
 * The axios interceptor lives outside React and must not import a hook, so the
 * AuthContext registers itself here instead and the interceptor just fires it.
 */
let onExpired = null

export function setExpiryHandler(fn) {
  onExpired = fn
}

export function notifyExpired() {
  if (onExpired) onExpired()
}
