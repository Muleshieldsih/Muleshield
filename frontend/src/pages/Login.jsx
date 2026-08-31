import { useState } from 'react'
import { Shield, Loader2, AlertCircle, ArrowLeft, KeyRound } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { endpoints, describeError } from '../services/api'

/**
 * Sign-in, and the password-reset request behind it.
 *
 * Rendered OUTSIDE the console shell (see App.jsx) so it is full-bleed and, more
 * importantly, so Layout never mounts while signed out — Layout fetches the case
 * queue and opens a WebSocket on mount, and both would fail loudly behind the
 * login form.
 */

const INPUT = 'w-full bg-ink-panel border border-ink-border rounded px-3 py-2 ' +
  'text-[12.5px] text-white outline-none focus:border-aegis-green transition-colors ' +
  'placeholder:text-zinc-600'
const LABEL = 'text-[10.5px] font-medium text-zinc-500 tracking-[0.07em] uppercase'
const PRIMARY = 'w-full px-4 py-2 rounded bg-aegis-green text-black font-bold ' +
  'text-[12.5px] hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed ' +
  'flex items-center justify-center gap-2 transition-colors'

function Banner({ tone = 'error', children }) {
  const skin = tone === 'error'
    ? 'bg-red-500/10 border-red-500/30 text-red-300'
    : 'bg-aegis-green/10 border-aegis-green/30 text-aegis-green'
  return (
    <div className={`flex items-start gap-2 border rounded px-2.5 py-2 ${skin}`}>
      <AlertCircle size={13} className="shrink-0 mt-0.5" />
      <div className="text-[11.5px] leading-snug">{children}</div>
    </div>
  )
}

export default function Login() {
  const { login, expired } = useAuth()
  const [mode, setMode] = useState('signin')   // 'signin' | 'forgot' | 'reset'

  return (
    <div className="min-h-screen bg-ink-bg text-zinc-200 flex flex-col items-center justify-center px-4">

      <div className="flex flex-col items-center gap-2.5 mb-6">
        <div className="flex items-center gap-2.5">
          <Shield size={26} className="text-aegis-green" strokeWidth={1.8} />
          <div className="text-[20px] font-bold tracking-[-0.02em] text-white">MuleShield AI</div>
        </div>
        <div className="text-[11.5px] text-zinc-500">
          Cash-out interception console · 1930 helpline
        </div>
      </div>

      {mode === 'signin' && <SignIn onLogin={login} expired={expired} onForgot={() => setMode('forgot')} />}
      {mode === 'forgot' && <Forgot onBack={() => setMode('signin')} onHaveToken={() => setMode('reset')} />}
      {mode === 'reset' && <Reset onDone={() => setMode('signin')} onBack={() => setMode('signin')} />}

      <div className="mt-6 flex flex-col items-center gap-1.5">
        <div className="mono text-[10.5px] text-zinc-600 tracking-[0.04em]">
          SIH26184 · MINISTRY OF HOME AFFAIRS / I4C
        </div>
        <div className="text-[10.5px] text-zinc-700">
          Authorised personnel only. Access is logged.
        </div>
      </div>
    </div>
  )
}

function SignIn({ onLogin, expired, onForgot }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await onLogin(username.trim(), password)
      // No navigate() here: App swaps the whole tree once `user` is set.
    } catch (err) {
      setError(describeError(err))
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="w-[384px] aegis-panel p-5.5" style={{ padding: '22px' }}>
      <div className="flex flex-col gap-0.5 mb-4">
        <div className="text-[15px] font-semibold text-white">Sign in</div>
        <div className="text-[11.5px] text-zinc-500">
          Every action you take is recorded against your name.
        </div>
      </div>

      {expired && !error && (
        <div className="mb-3.5">
          <Banner>Your session expired. Sign in again.</Banner>
        </div>
      )}
      {error && <div className="mb-3.5"><Banner>{error}</Banner></div>}

      <div className="flex flex-col gap-3.5">
        <div className="flex flex-col gap-1.5">
          <label className={LABEL} htmlFor="username">Officer ID</label>
          <input
            id="username"
            className={`${INPUT} mono`}
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            required
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className={LABEL} htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            className={INPUT}
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </div>

        <button type="submit" className={PRIMARY} disabled={busy || !username || !password}>
          {busy && <Loader2 size={13} className="animate-spin" />}
          {busy ? 'Signing in' : 'Sign in'}
        </button>

        <div className="flex justify-end">
          <button type="button" onClick={onForgot}
                  className="text-[11.5px] text-aegis-green hover:text-emerald-300">
            Forgot password?
          </button>
        </div>
      </div>
    </form>
  )
}

function Forgot({ onBack, onHaveToken }) {
  const [username, setUsername] = useState('')
  const [sent, setSent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(e) {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      await endpoints.forgotPassword(username.trim())
      setSent(true)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="w-[384px] aegis-panel" style={{ padding: '22px' }}>
      <div className="flex flex-col gap-0.5 mb-4">
        <div className="text-[15px] font-semibold text-white">Reset password</div>
        <div className="text-[11.5px] text-zinc-500 leading-relaxed">
          An administrator reviews the request and issues you a single-use token.
        </div>
      </div>

      {error && <div className="mb-3.5"><Banner>{error}</Banner></div>}

      {sent ? (
        <div className="flex flex-col gap-3.5">
          <Banner tone="ok">
            Request raised. An administrator will action it and give you a token.
          </Banner>
          <button type="button" onClick={onHaveToken} className={PRIMARY}>
            <KeyRound size={13} /> I have a token
          </button>
          <button type="button" onClick={onBack}
                  className="text-[11.5px] text-zinc-500 hover:text-white flex items-center justify-center gap-1.5">
            <ArrowLeft size={12} /> Back to sign in
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-3.5">
          <div className="flex flex-col gap-1.5">
            <label className={LABEL} htmlFor="forgot-user">Officer ID</label>
            <input
              id="forgot-user"
              className={`${INPUT} mono`}
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </div>
          <button type="submit" className={PRIMARY} disabled={busy || !username}>
            {busy && <Loader2 size={13} className="animate-spin" />}
            Request reset
          </button>
          <button type="button" onClick={onBack}
                  className="text-[11.5px] text-zinc-500 hover:text-white flex items-center justify-center gap-1.5">
            <ArrowLeft size={12} /> Back to sign in
          </button>
        </div>
      )}
    </form>
  )
}

function Reset({ onDone, onBack }) {
  const [token, setToken] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      await endpoints.resetPassword(token.trim(), password)
      setDone(true)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  if (done) {
    return (
      <div className="w-[384px] aegis-panel flex flex-col gap-3.5" style={{ padding: '22px' }}>
        <div className="text-[15px] font-semibold text-white">Password updated</div>
        <Banner tone="ok">
          You have been signed out of every device. Sign in with the new password.
        </Banner>
        <button type="button" onClick={onDone} className={PRIMARY}>Back to sign in</button>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="w-[384px] aegis-panel" style={{ padding: '22px' }}>
      <div className="flex flex-col gap-0.5 mb-4">
        <div className="text-[15px] font-semibold text-white">Set a new password</div>
        <div className="text-[11.5px] text-zinc-500 leading-relaxed">
          Tokens are single-use and expire 30 minutes after approval.
        </div>
      </div>

      {error && <div className="mb-3.5"><Banner>{error}</Banner></div>}

      <div className="flex flex-col gap-3.5">
        <div className="flex flex-col gap-1.5">
          <label className={LABEL} htmlFor="reset-token">Reset token</label>
          <input id="reset-token" className={`${INPUT} mono`} value={token}
                 onChange={e => setToken(e.target.value)} autoFocus required />
        </div>
        <div className="flex flex-col gap-1.5">
          <label className={LABEL} htmlFor="new-pw">New password</label>
          <input id="new-pw" type="password" className={INPUT} value={password}
                 onChange={e => setPassword(e.target.value)}
                 autoComplete="new-password" placeholder="Minimum eight characters"
                 minLength={8} required />
        </div>
        <button type="submit" className={PRIMARY}
                disabled={busy || !token || password.length < 8}>
          {busy && <Loader2 size={13} className="animate-spin" />}
          Set new password
        </button>
        <button type="button" onClick={onBack}
                className="text-[11.5px] text-zinc-500 hover:text-white flex items-center justify-center gap-1.5">
          <ArrowLeft size={12} /> Back to sign in
        </button>
      </div>
    </form>
  )
}
