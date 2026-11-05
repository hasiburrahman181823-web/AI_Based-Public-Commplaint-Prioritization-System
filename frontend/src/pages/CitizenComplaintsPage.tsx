import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { api } from '../api'
import type { Complaint, Page } from '../api'
import { useAuth } from '../auth'
import { errorMessage, useI18n } from '../i18n'
import { ErrorNotice, Icon, Loading } from '../ui'

const PAGE_SIZE = 10

export default function CitizenComplaintsPage() {
  const { citizen, citizenToken } = useAuth()
  const { locale } = useI18n()
  const [page, setPage] = useState(1)
  const [data, setData] = useState<Page<Complaint> | null>(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const bn = locale === 'bn'

  useEffect(() => {
    if (!citizenToken) return
    const controller = new AbortController()
    setBusy(true)
    setError('')
    api.citizenComplaints({ page, page_size: PAGE_SIZE }, citizenToken, controller.signal)
      .then(setData)
      .catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause, locale)) })
      .finally(() => { if (!controller.signal.aborted) setBusy(false) })
    return () => controller.abort()
  }, [citizenToken, page, locale, revision])

  if (!citizenToken) return <Navigate to="/account" replace />

  const formatDate = (value: string) => new Intl.DateTimeFormat(bn ? 'bn-BD' : 'en-GB', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
  const status = (value: string) => ({
    submitted: bn ? 'জমা হয়েছে' : 'Submitted',
    in_review: bn ? 'পর্যালোচনা চলছে' : 'In review',
    confirmed: bn ? 'নিশ্চিত' : 'Confirmed',
    in_progress: bn ? 'কাজ চলছে' : 'In progress',
    resolved: bn ? 'সমাধান হয়েছে' : 'Resolved',
    rejected: bn ? 'গ্রহণ করা যায়নি' : 'Not accepted',
  }[value] || value)
  const priority = (value: string) => ({
    urgent: bn ? 'জরুরি' : 'Urgent',
    high: bn ? 'উচ্চ' : 'High',
    moderate: bn ? 'মাঝারি' : 'Moderate',
    medium: bn ? 'মাঝারি' : 'Medium',
    low: bn ? 'কম' : 'Low',
  }[value] || value)

  return <div className="tracking-layout">
    <Link to="/account" className="text-button">← {bn ? 'অ্যাকাউন্ট' : 'Account'}</Link>
    <div className="mb-6 mt-4 flex flex-wrap items-end justify-between gap-4">
      <div><p className="eyebrow">{citizen?.email}</p><h1>{bn ? 'আমার অভিযোগগুলো' : 'My complaints'}</h1><p className="form-subtitle">{bn ? 'আপনার অ্যাকাউন্টে লগইন থাকা অবস্থায় জমা দেওয়া অভিযোগগুলো এখানে দেখা যাবে।' : 'Reports submitted while you were signed in to this account appear here.'}</p></div>
      <Link className="button-primary" to="/">{bn ? 'নতুন অভিযোগ' : 'New complaint'} <Icon name="arrow" /></Link>
    </div>
    <div className="privacy-note mb-5"><Icon name="shield"/><p>{bn ? 'লগইন ছাড়া জমা দেওয়া অভিযোগ এই তালিকায় যুক্ত হয় না। সেই অভিযোগের tracking ID ব্যবহার করুন।' : 'Reports submitted without signing in are not linked to this account. Use their tracking ID instead.'}</p></div>
    {busy && <Loading />}
    {!busy && error && <ErrorNotice>{error}<button type="button" className="text-button ml-3" onClick={() => setRevision(value => value + 1)}>{bn ? 'আবার চেষ্টা' : 'Try again'}</button></ErrorNotice>}
    {!busy && !error && data?.items.length === 0 && <section className="form-card text-center"><h2>{bn ? 'এখনো কোনো অভিযোগ নেই' : 'No complaints yet'}</h2><p className="form-subtitle">{bn ? 'অভিযোগ জমা দিতে শুরু করুন।' : 'Start by submitting a complaint.'}</p><Link to="/" className="button-primary mt-4">{bn ? 'অভিযোগ জমা দিন' : 'Submit a complaint'} <Icon name="arrow" /></Link></section>}
    {!busy && !error && data?.items.map(item => <article key={item.id} className="form-card mb-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="rounded-full border border-teal-300 bg-teal-50 px-3 py-1 text-xs font-semibold text-brand">{priority(item.priority_label)}</span>
        <span className="text-xs text-brand">{status(item.status)}</span>
      </div>
      <p className="leading-7 text-brand" lang={item.lang === 'bn' ? 'bn' : 'en'}>{item.text}</p>
      <dl className="mt-4 grid gap-x-5 gap-y-2 border-t border-teal-100 pt-4 text-sm sm:grid-cols-2">
        <div><dt className="field-hint">{bn ? 'বিভাগ' : 'Category'}</dt><dd className="text-brand">{item.category || (bn ? 'পর্যালোচনাধীন' : 'Awaiting review')}</dd></div>
        <div><dt className="field-hint">{bn ? 'এলাকা' : 'Area'}</dt><dd className="text-brand">{item.location || item.area_id?.replace(/^dhaka_/, '').replaceAll('_', ' ') || (bn ? 'উল্লেখ নেই' : 'Not specified')}</dd></div>
        <div><dt className="field-hint">{bn ? 'জমার সময়' : 'Submitted'}</dt><dd className="text-brand">{formatDate(item.created_at)}</dd></div>
        {item.contact && <div><dt className="field-hint">{bn ? 'যোগাযোগ' : 'Contact'}</dt><dd className="text-brand">{item.contact}</dd></div>}
        {item.timestamp && <div><dt className="field-hint">{bn ? 'ঘটনার সময়' : 'Incident time'}</dt><dd className="text-brand">{formatDate(item.timestamp)}</dd></div>}
        {item.severity && <div><dt className="field-hint">{bn ? 'তীব্রতা' : 'Severity'}</dt><dd className="text-brand">{item.severity}</dd></div>}
      </dl>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-teal-100 pt-4">
        <span className="text-xs text-brand">{bn ? 'ট্র্যাকিং আইডি' : 'Tracking ID'}: <code>{item.id}</code></span>
        <Link className="text-button" to={'/track/' + item.id}>{bn ? 'অগ্রগতি দেখুন' : 'View progress'} →</Link>
      </div>
    </article>)}
    {!busy && !error && (data?.total || 0) > PAGE_SIZE && <nav className="flex items-center justify-between gap-3" aria-label={bn ? 'পাতা' : 'Pages'}>
      <span className="text-sm text-brand">{bn ? 'পৃষ্ঠা' : 'Page'} {page} / {Math.ceil((data?.total || 0) / PAGE_SIZE)}</span>
      <div className="flex gap-2"><button className="button-secondary" disabled={page <= 1 || busy} onClick={() => setPage(value => value - 1)}>{bn ? 'আগের' : 'Previous'}</button><button className="button-secondary" disabled={page * PAGE_SIZE >= (data?.total || 0) || busy} onClick={() => setPage(value => value + 1)}>{bn ? 'পরের' : 'Next'}</button></div>
    </nav>}
  </div>
}
