import { useState } from 'react'
import type { FormEvent } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { errorMessage, useI18n } from '../i18n'
import { ErrorNotice, Icon } from '../ui'

export default function LoginPage() {
  const { login, token } = useAuth(), { t, locale } = useI18n()
  const [username, setUsername] = useState(''), [password, setPassword] = useState(''), [busy, setBusy] = useState(false), [error, setError] = useState('')
  if (token) return <Navigate to="/admin" replace/>
  async function submit(event: FormEvent) { event.preventDefault(); if (busy) return; setBusy(true); setError(''); try { await login(username.trim(), password); setPassword('') } catch (error) { setError(errorMessage(error, locale)) } finally { setBusy(false) } }
  return <div className="login-layout"><div className="form-card"><span className="login-symbol"><Icon name="shield"/></span><h1>{t('signIn')}</h1><p className="form-subtitle">{t('signInHint')}</p><form onSubmit={submit} className="space-y-5" aria-busy={busy}><div><label htmlFor="admin-username">{t('username')}</label><input id="admin-username" required maxLength={128} value={username} onChange={event => setUsername(event.target.value)} autoComplete="username" disabled={busy}/></div><div><label htmlFor="admin-password">{t('password')}</label><input id="admin-password" required type="password" maxLength={256} value={password} onChange={event => setPassword(event.target.value)} autoComplete="current-password" disabled={busy}/></div>{error && <ErrorNotice>{error}</ErrorNotice>}<button className="button-primary w-full" disabled={busy}>{busy ? t('loggingIn') : t('login')}<Icon name="arrow"/></button></form><p className="field-hint mt-5">{t('sessionHint')}</p></div><Link to="/" className="text-button">← {t('back')}</Link></div>
}
