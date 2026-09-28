import { useState } from 'react'
import { Loader2, AlertCircle, ArrowLeft, KeyRound, Eye, EyeOff, Lock, CheckCircle2, Circle } from 'lucide-react'
import Mark from '../components/Mark'
import { useAuth } from '../context/AuthContext'
import { endpoints, describeError } from '../services/api'

/**
 * Sign-in, and the password-reset request behind it.
 *
 * Rendered OUTSIDE the console shell (see App.jsx) so it is full-bleed and, more
 * importantly, so Layout never mounts while signed out — Layout fetches the case
 * queue and opens a WebSocket on mount, and both would fail loudly behind the
 * login form.
 *
 * Two halves: the mark standing in its own light, and a glass pane carrying the
 * form. The split is the argument — this console is one thing, entered at one
 * door — and it is why the sign-in screen looks nothing like the dense grey
 * panels behind it. There is no data on this screen to respect, only the
 * product's face.
 *
 * Below `lg` the left half is dropped rather than stacked. Squashed onto a
 * phone the mark becomes decoration above a form, which is worse than absent,
 * so the wordmark and the provenance line move into the pane instead.
 */

/** The lit plate everything sits on: two soft pools and grain so the large soft
 *  fields do not band.
 *
 *  The construction lines that used to run through here are gone. They read as a
 *  drafting grid on a wide desktop, but the layout collapses to one column below
 *  lg -- and stacked, every line converged on the mark and turned it into the hub
 *  of a starburst. The mark is a reticle; giving it rays makes it a sun. */
function Backdrop() {
  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(900px 620px at 26% 40%, rgba(190, 198, 206, 0.075), transparent 68%),' +
            'radial-gradient(760px 560px at 88% 84%, rgba(160, 168, 180, 0.05), transparent 70%),' +
            'linear-gradient(158deg, #0b0b0c 0%, #060607 52%, #0a0f0f 100%)',
        }}
      />


      <svg
        className="absolute inset-0 w-full h-full"
        style={{ opacity: 0.15, mixBlendMode: 'overlay' }}
        preserveAspectRatio="none"
      >
        <filter id="login-grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch" />
        </filter>
        <rect width="100%" height="100%" filter="url(#login-grain)" />
      </svg>
    </div>
  )
}

function Wordmark({ size = 17 }) {
  return (
    <div className="flex items-baseline gap-[5px]" style={{ fontSize: size }}>
      <span className="font-medium tracking-[0.01em] text-white">MuleShield</span>
      <span className="font-extralight tracking-[0.01em] text-[#8f8f94]">AI</span>
    </div>
  )
}

function Provenance() {
  return (
    <div className="flex flex-col gap-1">
      <span className="mono text-[9.5px] text-[#525256] tracking-[0.1em]">SIH26184 · MHA / I4C</span>
      <span className="text-[9.5px] text-[#3f3f43]">Authorised personnel only. Access is logged.</span>
    </div>
  )
}

/** Left half: the mark, sitting in light rather than on top of black. */
function MarkPanel() {
  return (
    <div className="relative hidden lg:flex items-center justify-center p-10">
      <div
        className="absolute rounded-full"
        style={{
          width: 460,
          height: 460,
          background: 'radial-gradient(circle, rgba(220, 224, 232, 0.085), transparent 66%)',
        }}
      />
      <Mark size={300} className="relative text-white" pip="#ffffff" />

      <div className="absolute top-10 left-11"><Wordmark /></div>
      <div className="absolute bottom-10 left-11"><Provenance /></div>
    </div>
  )
}

/** A failure has to survive the glass, so it gets a tinted plate of its own
 *  rather than coloured text floating on the pane. */
function Banner({ tone = 'error', children }) {
  const skin = tone === 'error'
    ? 'border-red-400/25 bg-red-500/[0.09] text-red-200'
    : 'border-white/20 bg-white/[0.07] text-zinc-100'
  const Icon = tone === 'error' ? AlertCircle : CheckCircle2
  return (
    <div className={`flex items-start gap-2.5 border rounded-[10px] px-3.5 py-2.5 ${skin}`}>
      <Icon size={14} className="shrink-0 mt-px" />
      <div className="text-[11.5px] leading-snug">{children}</div>
    </div>
  )
}

function Field({ label, htmlFor, children }) {
  return (
    <div className="flex flex-col gap-[9px]">
      <label className="login-label" htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  )
}

/** The password field carries a reveal, because an officer typing a shift
 *  password under pressure will mistype it, and the alternative is a lockout
 *  somebody else has to undo. */
function PasswordField({ id, value, onChange, label = 'Password', autoComplete = 'current-password', ...rest }) {
  const [shown, setShown] = useState(false)
  return (
    <Field label={label} htmlFor={id}>
      <div className="login-field">
        <input
          id={id}
          type={shown ? 'text' : 'password'}
          className="login-input"
          value={value}
          onChange={onChange}
          autoComplete={autoComplete}
          required
          {...rest}
        />
        <button
          type="button"
          onClick={() => setShown(s => !s)}
          className="shrink-0 text-[#717176] hover:text-white transition-colors"
          aria-label={shown ? 'Hide password' : 'Show password'}
        >
          {shown ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>
    </Field>
  )
}

function Title({ children, sub }) {
  return (
    <>
      <h1 className="login-title mb-[9px]" style={{ fontSize: 'clamp(38px, 5.2vw, 62px)' }}>
        {children}
      </h1>
      <div className="text-[11.5px] text-[#717176] tracking-[0.01em] mb-6">{sub}</div>
    </>
  )
}

function DemoCredentialsBox({ onAutoFill }) {
  return (
    <div className="mb-7 p-3.5 rounded-[10px] border border-emerald-500/25 bg-emerald-500/[0.07] backdrop-blur-sm transition-all duration-200">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <KeyRound size={13} className="text-emerald-400 shrink-0" />
          <span className="text-[11px] font-medium tracking-wide uppercase text-emerald-400">
            SIH Evaluator / Jury Access
          </span>
        </div>
        <button
          type="button"
          onClick={onAutoFill}
          className="text-[10px] font-medium tracking-wide uppercase px-2 py-0.5 rounded border border-emerald-400/40 bg-emerald-400/10 hover:bg-emerald-400/20 text-emerald-300 transition-colors cursor-pointer"
        >
          Auto-fill
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-zinc-300">
        <div>
          <span className="text-[#88888e]">Officer ID:</span>{' '}
          <code className="font-mono text-white bg-black/40 px-1.5 py-0.5 rounded border border-white/10">officer</code>
        </div>
        <div>
          <span className="text-[#88888e]">Password:</span>{' '}
          <code className="font-mono text-white bg-black/40 px-1.5 py-0.5 rounded border border-white/10">password123</code>
        </div>
      </div>
    </div>
  )
}

function BackLink({ onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="login-quiet mt-6 flex items-center gap-1.5 text-[10.5px]"
    >
      <ArrowLeft size={12} /> Back to sign in
    </button>
  )
}

export default function Login() {
  const { login, expired } = useAuth()
  const [mode, setMode] = useState('signin')   // 'signin' | 'forgot' | 'reset'

  return (
    <div className="relative min-h-screen w-full overflow-hidden text-[#ececee]">
      <Backdrop />

      <div className="relative min-h-screen grid grid-cols-1 lg:grid-cols-[44fr_56fr]">
        <MarkPanel />

        <section className="login-glass relative flex flex-col justify-center px-6 sm:px-12 py-16 lg:px-[clamp(48px,6vw,100px)] lg:border-l lg:border-white/10">
          <div className="hidden lg:flex absolute top-9 right-12 items-center gap-[7px]">
            <span className="text-[10.5px] text-[#5e5e62]">No account?</span>
            <span className="text-[10.5px] text-[#9d9da0]">Issued by your administrator</span>
          </div>

          <div className="w-full max-w-[640px]">
            <div className="lg:hidden mb-10"><Wordmark size={16} /></div>

            {mode === 'signin' && (
              <SignIn onLogin={login} expired={expired} onForgot={() => setMode('forgot')} />
            )}
            {mode === 'forgot' && (
              <Forgot onBack={() => setMode('signin')} onHaveToken={() => setMode('reset')} />
            )}
            {mode === 'reset' && (
              <Reset onDone={() => setMode('signin')} onBack={() => setMode('signin')} />
            )}

            <div className="login-rule mt-10" />

            <div className="mt-[18px] flex items-start gap-[9px]">
              <Lock size={12} className="text-[#525256] shrink-0 mt-0.5" />
              <span className="text-[10.5px] text-[#525256] leading-[1.5]">
                Every freeze and dispatch is recorded against the officer who ordered it.
              </span>
            </div>

            <div className="lg:hidden mt-10"><Provenance /></div>
          </div>
        </section>
      </div>
    </div>
  )
}

function SignIn({ onLogin, expired, onForgot }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [keepSignedIn, setKeepSignedIn] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const handleAutoFill = () => {
    setUsername('officer')
    setPassword('password123')
    setError('')
  }

  async function submit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await onLogin(username.trim(), password, { persist: keepSignedIn })
      // No navigate() here: App swaps the whole tree once `user` is set.
    } catch (err) {
      setError(describeError(err))
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit}>
      <Title sub="Cash-out interception console · 1930 helpline">Sign in</Title>

      <DemoCredentialsBox onAutoFill={handleAutoFill} />

      {(error || expired) && (
        <div className="mb-7">
          <Banner>{error || 'Your session expired. Sign in again.'}</Banner>
        </div>
      )}

      {/* Two columns, because the pair is one act. They stack on a narrow pane. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-[26px]">
        <Field label="Officer ID" htmlFor="username">
          <div className="login-field">
            <input
              id="username"
              className="login-input mono"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </div>
        </Field>

        <PasswordField
          id="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />
      </div>

      <div className="mt-4 flex items-center justify-between gap-4">
        {/* Real, not decorative: unchecked, the token goes to sessionStorage and
            dies with the tab. See services/auth.js. */}
        <button
          type="button"
          role="checkbox"
          aria-checked={keepSignedIn}
          onClick={() => setKeepSignedIn(v => !v)}
          className="flex items-center gap-2 group"
        >
          {keepSignedIn
            ? <CheckCircle2 size={13} className="text-white shrink-0" />
            : <Circle size={13} className="text-[#717176] shrink-0" />}
          <span className="text-[10.5px] text-[#7e7e82] group-hover:text-zinc-300 transition-colors">
            Keep me signed in this shift
          </span>
        </button>

        <button type="button" onClick={onForgot} className="login-quiet text-[10.5px]">
          Forgot password?
        </button>
      </div>

      <button
        type="submit"
        className="login-submit mt-[30px]"
        disabled={busy || !username || !password}
      >
        {busy && <Loader2 size={13} className="animate-spin" />}
        {busy ? 'SIGNING IN' : 'SIGN IN'}
      </button>
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
    <form onSubmit={submit}>
      <Title sub="An administrator reviews the request and issues you a single-use token.">
        Reset password
      </Title>

      {error && <div className="mb-7"><Banner>{error}</Banner></div>}

      {sent ? (
        <>
          <Banner tone="ok">
            Request raised. An administrator will action it and give you a token.
          </Banner>
          <button type="button" onClick={onHaveToken} className="login-submit mt-[30px]">
            <KeyRound size={13} /> I HAVE A TOKEN
          </button>
          <BackLink onClick={onBack} />
        </>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-[26px]">
            <Field label="Officer ID" htmlFor="forgot-user">
              <div className="login-field">
                <input
                  id="forgot-user"
                  className="login-input mono"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  autoComplete="username"
                  autoFocus
                  required
                />
              </div>
            </Field>
          </div>
          <button type="submit" className="login-submit mt-[30px]" disabled={busy || !username}>
            {busy && <Loader2 size={13} className="animate-spin" />}
            REQUEST RESET
          </button>
          <BackLink onClick={onBack} />
        </>
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
      <div>
        <Title sub="Tokens are single-use and expire 30 minutes after approval.">
          Password updated
        </Title>
        <Banner tone="ok">
          You have been signed out of every device. Sign in with the new password.
        </Banner>
        <button type="button" onClick={onDone} className="login-submit mt-[30px]">
          BACK TO SIGN IN
        </button>
      </div>
    )
  }

  return (
    <form onSubmit={submit}>
      <Title sub="Tokens are single-use and expire 30 minutes after approval.">
        Set a new password
      </Title>

      {error && <div className="mb-7"><Banner>{error}</Banner></div>}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-[26px]">
        <Field label="Reset token" htmlFor="reset-token">
          <div className="login-field">
            <input
              id="reset-token"
              className="login-input mono"
              value={token}
              onChange={e => setToken(e.target.value)}
              autoFocus
              required
            />
          </div>
        </Field>

        <PasswordField
          id="new-pw"
          label="New password"
          autoComplete="new-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="Minimum eight characters"
          minLength={8}
        />
      </div>

      <button
        type="submit"
        className="login-submit mt-[30px]"
        disabled={busy || !token || password.length < 8}
      >
        {busy && <Loader2 size={13} className="animate-spin" />}
        SET NEW PASSWORD
      </button>
      <BackLink onClick={onBack} />
    </form>
  )
}
