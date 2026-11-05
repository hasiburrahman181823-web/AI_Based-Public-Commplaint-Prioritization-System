import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { Complaint, Lang } from '../api'
import { useAuth } from '../auth'
import { errorMessage, label, useI18n } from '../i18n'
import { detectLanguage } from '../language'
import { Badge, ErrorNotice, Icon } from '../ui'
import { operatorPublished } from '../privacy'

export default function PublicPage() {
  const { t, locale } = useI18n()
  const { citizen, citizenToken } = useAuth()
  const [text, setText] = useState(''), [area, setArea] = useState(''), [contact, setContact] = useState('')
  const [language, setLanguage] = useState<'auto' | Lang>('auto'), [busy, setBusy] = useState(false)
  const [error, setError] = useState(''), [receipt, setReceipt] = useState<Complaint | null>(null), [copyState, setCopyState] = useState('')
  const successHeading = useRef<HTMLHeadingElement>(null)
  useEffect(() => { if (receipt) successHeading.current?.focus() }, [receipt])
  const detected = language === 'auto' ? detectLanguage(text) : language
  const count = Array.from(text.normalize('NFKC').trim()).length
  async function submit(event: FormEvent) {
    event.preventDefault(); if (busy) return
    setError('')
    if (!count || count > 1000) { setError(locale === 'bn' ? '১ থেকে ১০০০ অক্ষরে অভিযোগ লিখুন।' : 'Describe your complaint in 1–1,000 characters.'); return }
    setBusy(true)
    try {
      const body = { text: text.trim(), ...(area.trim() ? { location: area.trim() } : {}), ...(contact.trim() ? { contact: contact.trim() } : {}), ...(language !== 'auto' ? { lang: language } : {}) }
      setReceipt(await (citizenToken ? api.submit(body, citizenToken) : api.submit(body)))
    }
    catch (error) { setError(errorMessage(error, locale)) } finally { setBusy(false) }
  }
  async function copyId() { try { await navigator.clipboard.writeText(receipt!.id); setCopyState(t('copied')) } catch { setCopyState(t('copyFailed')) } }
  function reset() { setText(''); setArea(''); setContact(''); setReceipt(null); setCopyState(''); setLanguage('auto') }
  return <div className="public-layout">
    <section className="public-intro" aria-labelledby="public-title">
      <div className="eyebrow">{t('local')}</div>
      <h1 id="public-title">{t('heroStart')}<br/><span>{t('heroEnd')}</span></h1>
      <p className="intro-copy">{t('intro')}</p>
      <p className="demo-disclaimer">This is an AI decision-support demo, not an official government system.</p>
      <div className="trust-row"><span>{t('noAccount')}</span><span>{t('privateContact')}</span></div>
      <div className="how-card"><h2>{t('how')}</h2><ol>{(['step1', 'step2', 'step3'] as const).map((step, index) => <li key={step}><span className="step-number">{index + 1}</span><div><h3>{t(step)}</h3><p>{t(`${step}Hint`)}</p></div></li>)}</ol></div>
      <div className="public-note"><Icon name="pin"/><p>{locale === 'bn' ? 'আপনার এলাকার জন্য, আপনার ভাষায়।' : 'Rooted in your community. Open to every voice.'}</p></div>
    </section>
    <section className="form-card" aria-labelledby="form-title">
      {receipt ? <div className="receipt" role="status">
        <span className="success-mark"><Icon name="check"/></span>
        <p className="eyebrow">{locale === 'bn' ? 'অভিযোগ জমা হয়েছে' : 'Complaint submitted'}</p>
        <h2 ref={successHeading} tabIndex={-1} id="form-title">{t('received')}</h2><p>{t('receivedHint')}</p>
        <div className="tracking-code"><span>{t('trackingId')}</span><code>{receipt.id}</code><button type="button" className="button-secondary" onClick={copyId}><Icon name="copy"/>{t('copy')}</button><p aria-live="polite">{copyState}</p></div>
        <Link className="button-primary w-full" to={`/track/${receipt.id}`}>{t('viewProgress')}<Icon name="arrow"/></Link>
        {citizenToken && <Link className="text-button mt-4 block text-center" to="/account/complaints">{locale === 'bn' ? 'আপনার অ্যাকাউন্টে অভিযোগটি দেখুন' : 'See this report in your account'} →</Link>}
        <button className="text-button mt-5" onClick={reset}>{t('another')}</button>
      </div> : <>
        <h2 id="form-title">{t('formTitle')}</h2><p className="form-subtitle">{t('formHint')}</p>
        {citizenToken
          ? <div className="privacy-note mb-5"><Icon name="shield"/><p>{locale === 'bn' ? <>{citizen?.email ? citizen.email + ' হিসেবে ' : ''}জমা হবে এবং আপনার <Link className="text-button" to="/account/complaints">অভিযোগের তালিকায়</Link> দেখা যাবে।</> : <>Submitting{citizen?.email ? ' as ' + citizen.email : ''}. This will appear in <Link className="text-button" to="/account/complaints">your complaint history</Link>.</>}</p></div>
          : <p className="field-hint mb-5">{locale === 'bn' ? <>চাইলে <Link className="text-button" to="/account">অ্যাকাউন্ট খুলুন বা লগইন করুন</Link>—তাহলে জমা অভিযোগের ইতিহাস দেখতে পারবেন। লগইন ছাড়া জমা দেওয়াও চালু আছে।</> : <>You can <Link className="text-button" to="/account">sign in or create an account</Link> to keep a history. Anonymous submission is still available.</>}</p>}
        <form onSubmit={submit} aria-busy={busy} className="space-y-6">
          <div><div className="mb-2 flex flex-wrap items-center justify-between gap-2"><label htmlFor="complaint-text">{t('complaint')}<span className="sr-only"> ({t('required')})</span></label>{detected && <Badge tone="teal">{t('detected')}: {label(detected, locale)}</Badge>}</div>
            <textarea id="complaint-text" required maxLength={1000} rows={6} value={text} onChange={event => setText(event.target.value)} placeholder={t('complaintHint')} aria-describedby="text-count language-hint" disabled={busy}/>
            <div className="mt-2 flex items-center justify-between gap-3"><span className="text-xs text-brand">বাংলা · English · Banglish</span><span id="text-count" className="text-xs text-brand">{new Intl.NumberFormat(locale === 'bn' ? 'bn-BD' : 'en').format(count)} / 1,000 {t('chars')}</span></div>
            <details className="language-detail mt-2"><summary>{t('textLanguage')}</summary><select aria-label={t('textLanguage')} value={language} onChange={event => setLanguage(event.target.value as 'auto' | Lang)} disabled={busy}><option value="auto">{t('auto')}</option><option value="bn">{label('bn', locale)}</option><option value="en">{label('en', locale)}</option><option value="banglish">{label('banglish', locale)}</option></select><p id="language-hint">{t('detectionHint')}</p></details>
          </div>
          <div><label htmlFor="complaint-area">{t('area')} <span className="optional-label">({t('optional')})</span></label><div className="input-icon"><Icon name="pin"/><input id="complaint-area" maxLength={256} value={area} onChange={event => setArea(event.target.value)} placeholder={t('areaHint')} disabled={busy} list="area-options" autoComplete="off"/></div><datalist id="area-options"><option value="Mirpur 10"/><option value="Mohammadpur"/><option value="Uttara sector 4"/><option value="Dhanmondi"/><option value="মিরপুর ১০"/></datalist></div>
          <div><label htmlFor="complaint-contact">{t('contact')} <span className="optional-label">({t('optional')})</span></label><input id="complaint-contact" maxLength={256} value={contact} onChange={event => setContact(event.target.value)} autoComplete="off" aria-describedby="contact-hint" disabled={busy}/><p id="contact-hint" className="field-hint">{t('contactHint')}</p></div>
          <div className="privacy-note"><Icon name="shield"/><p>{t('privacy')}</p></div>
          <div className="text-xs leading-6"><Link className="text-button" to="/privacy">{t('privacyNotice')} →</Link>{!operatorPublished && <p className="mt-2">{t('demoPrivacy')}</p>}</div>
          {error && <ErrorNotice>{error}</ErrorNotice>}
          <button type="submit" className="button-primary w-full" disabled={busy}>{busy ? <><span className="spinner"/>{t('sending')}</> : <>{t('submit')}<Icon name="arrow"/></>}</button>
          <p className="text-center text-xs text-brand">{t('sendHint')}</p>
        </form>
      </>}
    </section>
  </div>
}
