import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { errorMessage, useI18n } from '../i18n'
import { ErrorNotice, Icon, Loading } from '../ui'

export default function AccountPage() {
  const { citizen, citizenToken, citizenLogin, citizenRegister, citizenLogout } = useAuth()
  const { locale } = useI18n()
  const navigate = useNavigate()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [phone, setPhone] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const bn = locale === 'bn'

  if (citizenToken && !citizen) return <div className="login-layout"><Loading /></div>

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError('')
    try {
      if (mode === 'login') await citizenLogin(email.trim(), password)
      else await citizenRegister({
        email: email.trim(),
        password,
        ...(fullName.trim() ? { full_name: fullName.trim() } : {}),
        ...(phone.trim() ? { phone: phone.trim() } : {}),
      })
      setPassword('')
      navigate('/account/complaints')
    } catch (cause) {
      setError(errorMessage(cause, locale))
    } finally { setBusy(false) }
  }

  if (citizen) return <div className="login-layout">
    <section className="form-card" aria-labelledby="account-title">
      <span className="login-symbol"><Icon name="building" /></span>
      <p className="eyebrow">{bn ? 'আপনার নাগরিক অ্যাকাউন্ট' : 'Your citizen account'}</p>
      <h1 id="account-title">{citizen.full_name || citizen.email}</h1>
      <div className="mt-5 space-y-3 rounded-2xl bg-teal-50 p-4 text-sm text-brand">
        <p><strong>{bn ? 'ইমেইল' : 'Email'}:</strong> {citizen.email}</p>
        {citizen.full_name && <p><strong>{bn ? 'নাম' : 'Name'}:</strong> {citizen.full_name}</p>}
        {citizen.phone && <p><strong>{bn ? 'ফোন' : 'Phone'}:</strong> {citizen.phone}</p>}
        <p className="field-hint">{bn ? 'এই তথ্যগুলো আপনার অভিযোগের লেখা থেকে আলাদা করে রাখা হয়।' : 'Your profile details are stored separately from complaint text.'}</p>
      </div>
      <Link className="button-primary mt-5 w-full" to="/account/complaints">
        {bn ? 'আমার অভিযোগগুলো দেখুন' : 'View my complaints'} <Icon name="arrow" />
      </Link>
      <button type="button" className="text-button mt-5" onClick={() => { citizenLogout(); setPassword('') }}>
        {bn ? 'লগআউট' : 'Sign out'}
      </button>
    </section>
    <Link to="/" className="text-button">← {bn ? 'অভিযোগের ফর্মে ফিরুন' : 'Back to the complaint form'}</Link>
  </div>

  return <div className="login-layout">
    <section className="form-card" aria-labelledby="account-title">
      <span className="login-symbol"><Icon name="building" /></span>
      <p className="eyebrow">{bn ? 'নাগরিক পোর্টাল' : 'Citizen portal'}</p>
      <h1 id="account-title">{mode === 'login' ? (bn ? 'আপনার অ্যাকাউন্টে প্রবেশ করুন' : 'Sign in to your account') : (bn ? 'নতুন অ্যাকাউন্ট খুলুন' : 'Create an account')}</h1>
      <p className="form-subtitle">{mode === 'login'
        ? (bn ? 'আপনার আগের অভিযোগ ও অগ্রগতি দেখতে লগইন করুন।' : 'Sign in to see your earlier complaints and their progress.')
        : (bn ? 'অ্যাকাউন্ট ছাড়াও অভিযোগ করা যায়। অ্যাকাউন্ট করলে এই ইমেইলে জমা অভিযোগের ইতিহাস দেখতে পারবেন।' : 'You can submit without an account. An account lets you see reports submitted while signed in.')}</p>
      {mode === 'register' && <div className="privacy-note mt-4"><Icon name="shield"/><p>{bn ? 'ডেমো অ্যাকাউন্টে ইমেইল যাচাই বা পাসওয়ার্ড পুনরুদ্ধার নেই। অনন্য পাসওয়ার্ড এবং ব্যক্তিগত ব্যবহারের জন্য তৈরি করা ইমেইল ব্যবহার করুন।' : 'Demo accounts do not have email verification or password recovery. Use a unique password and an email address you are comfortable storing.'}</p></div>}
      <div className="language-toggle my-5" role="group" aria-label={bn ? 'অ্যাকাউন্টের ধরন' : 'Account action'}>
        <button type="button" aria-pressed={mode === 'login'} onClick={() => { setMode('login'); setError('') }}>{bn ? 'লগইন' : 'Sign in'}</button>
        <button type="button" aria-pressed={mode === 'register'} onClick={() => { setMode('register'); setError('') }}>{bn ? 'নিবন্ধন' : 'Create account'}</button>
      </div>
      <form onSubmit={submit} className="space-y-5" aria-busy={busy}>
        {mode === 'register' && <>
          <div><label htmlFor="citizen-name">{bn ? 'নাম' : 'Name'} <span className="optional-label">({bn ? 'ঐচ্ছিক' : 'optional'})</span></label><input id="citizen-name" maxLength={120} autoComplete="name" value={fullName} onChange={event => setFullName(event.target.value)} disabled={busy} /></div>
          <div><label htmlFor="citizen-phone">{bn ? 'ফোন' : 'Phone'} <span className="optional-label">({bn ? 'ঐচ্ছিক' : 'optional'})</span></label><input id="citizen-phone" maxLength={32} autoComplete="tel" value={phone} onChange={event => setPhone(event.target.value)} disabled={busy} /></div>
        </>}
        <div><label htmlFor="citizen-email">{bn ? 'ইমেইল' : 'Email'}</label><input id="citizen-email" type="email" required maxLength={254} autoComplete="email" value={email} onChange={event => setEmail(event.target.value)} disabled={busy} /></div>
        <div><label htmlFor="citizen-password">{bn ? 'পাসওয়ার্ড' : 'Password'}</label><input id="citizen-password" type="password" required minLength={mode === 'register' ? 10 : 1} maxLength={256} autoComplete={mode === 'register' ? 'new-password' : 'current-password'} value={password} onChange={event => setPassword(event.target.value)} disabled={busy} /><p className="field-hint mt-1">{mode === 'register' && (bn ? 'কমপক্ষে ১০ অক্ষর।' : 'At least 10 characters.')}</p></div>
        {error && <ErrorNotice>{error}</ErrorNotice>}
        <button className="button-primary w-full" disabled={busy}>{busy
          ? (bn ? 'একটু অপেক্ষা করুন…' : 'Please wait…')
          : mode === 'login' ? (bn ? 'লগইন করুন' : 'Sign in') : (bn ? 'অ্যাকাউন্ট তৈরি করুন' : 'Create account')}
          <Icon name="arrow" />
        </button>
      </form>
      <p className="field-hint mt-5">{bn ? 'অ্যাকাউন্ট ছাড়া করা অভিযোগ এখানে দেখা যাবে না। সেই অভিযোগের tracking ID দিয়ে অগ্রগতি দেখুন।' : 'Reports submitted anonymously are not added to this account. Use their tracking ID to check progress.'}</p>
    </section>
    <div className="flex flex-wrap justify-between gap-3">
      <Link to="/" className="text-button">← {bn ? 'অভিযোগের ফর্ম' : 'Complaint form'}</Link>
      <Link to="/admin/login" className="text-button">{bn ? 'কর্মকর্তা লগইন' : 'Staff sign in'} →</Link>
    </div>
  </div>
}
