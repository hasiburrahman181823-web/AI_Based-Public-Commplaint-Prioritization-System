import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { api, ApiError } from '../api'
import { useAuth } from '../auth'
import { errorMessage, useI18n } from '../i18n'
import Charts from '../components/Charts'
import ExplanationPanel from '../components/ExplanationPanel'

type Complaint = Awaited<ReturnType<typeof api.getComplaint>>
type ComplaintPage = Awaited<ReturnType<typeof api.listComplaints>>
type Group = Awaited<ReturnType<typeof api.getGroup>>
type GroupPage = Awaited<ReturnType<typeof api.listGroups>>
type Analytics = Awaited<ReturnType<typeof api.analytics>>
type Tab = 'queue' | 'review' | 'groups' | 'analytics'
const STATUSES = ['submitted', 'in_review', 'confirmed', 'in_progress', 'resolved', 'rejected'] as const
const fieldClass = 'w-full rounded-xl border border-teal-300 bg-white px-3 py-2.5 text-sm text-brand outline-none transition focus:border-teal-700 focus:ring-2 focus:ring-teal-700/15 disabled:opacity-60'
const primaryButton = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-teal-100 px-4 py-2.5 text-sm font-semibold text-brand transition hover:bg-teal-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 disabled:cursor-wait disabled:opacity-50'
const secondaryButton = 'inline-flex min-h-10 items-center justify-center gap-2 rounded-xl border border-teal-300 bg-white px-3 py-2 text-sm font-medium text-brand transition hover:bg-teal-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 disabled:cursor-not-allowed disabled:opacity-50'

function useLabels() {
  const { locale } = useI18n()
  const tr = (en: string, bn: string) => locale === 'bn' ? bn : en
  const priority = (value: string) => ({ urgent: tr('Urgent', 'জরুরি'), high: tr('High', 'উচ্চ'), moderate: tr('Moderate', 'মাঝারি'), medium: tr('Medium', 'মাঝারি'), low: tr('Low', 'কম') }[value] || value)
  const status = (value: string) => ({ submitted: tr('Submitted', 'জমা হয়েছে'), in_review: tr('In review', 'পর্যালোচনাধীন'), confirmed: tr('Confirmed', 'নিশ্চিত'), in_progress: tr('In progress', 'কাজ চলছে'), resolved: tr('Resolved', 'সমাধান হয়েছে'), rejected: tr('Rejected', 'বাতিল') }[value] || value)
  const language = (value: string) => ({ bn: tr('Bangla', 'বাংলা'), en: tr('English', 'ইংরেজি'), banglish: tr('Banglish', 'বাংলিশ') }[value] || value)
  const count = (value: number) => new Intl.NumberFormat(locale === 'bn' ? 'bn-BD' : 'en-GB', { maximumFractionDigits: 1 }).format(value)
  const date = (value: string) => new Intl.DateTimeFormat(locale === 'bn' ? 'bn-BD' : 'en-GB', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
  const area = (value: string | null | undefined) => value ? value.replace(/^dhaka_/, '').replaceAll('_', ' ') : tr('Area not specified', 'এলাকা উল্লেখ নেই')
  const failure = (error: unknown) => {
    if (error && typeof error === 'object' && 'status' in error && error.status === 409) return tr('This record changed while you were reviewing it. Refresh and check the latest version before saving again.', 'আপনার পর্যালোচনার সময়ে তথ্য পরিবর্তিত হয়েছে। আবার সংরক্ষণের আগে রিফ্রেশ করে সর্বশেষ তথ্য দেখুন।')
    if (error && typeof error === 'object' && 'status' in error && error.status === 401) return tr('Your session has expired. Sign in again to continue.', 'আপনার সেশনের মেয়াদ শেষ। চালিয়ে যেতে আবার লগইন করুন।')
    if (error && typeof error === 'object' && 'status' in error && error.status === 429) return tr('Too many requests. Wait a moment, then try again.', 'অনেক অনুরোধ পাঠানো হয়েছে। একটু অপেক্ষা করে আবার চেষ্টা করুন।')
    return error instanceof ApiError ? errorMessage(error, locale) : error instanceof Error ? error.message : errorMessage(error, locale)
  }
  return { locale, tr, priority, status, language, count, date, area, failure }
}

function SmallIcon({ kind, className = 'size-4' }: { kind: 'arrow' | 'close' | 'refresh' | 'check' | 'group' | 'list' | 'alert' | 'chart'; className?: string }) {
  const paths = {
    arrow: 'M5 12h14m-5-5 5 5-5 5', close: 'm6 6 12 12M6 18 18 6', refresh: 'M20 7v5h-5M4 17v-5h5M6.1 7a7 7 0 0 1 11.6-2L20 8M4 16l2.3 3A7 7 0 0 0 18 17',
    check: 'm5 12 4 4L19 6', group: 'M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2m20 0v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8',
    list: 'M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01', alert: 'M12 8v4m0 4h.01M10.3 3.9 2.1 18a2 2 0 0 0 1.7 3h16.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z', chart: 'M4 19h16M7 15V9m5 6V4m5 11v-5',
  }
  return <svg aria-hidden="true" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d={paths[kind]} /></svg>
}

function PriorityBadge({ value }: { value: string }) {
  const { priority } = useLabels()
  const colors: Record<string, string> = { urgent: 'border-teal-200 bg-teal-50 text-brand', high: 'border-teal-200 bg-teal-50 text-brand', moderate: 'border-teal-200 bg-teal-50 text-brand', medium: 'border-teal-200 bg-teal-50 text-brand', low: 'border-teal-200 bg-teal-50 text-brand' }
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${colors[value] || colors.low}`}><span className="size-1.5 rounded-full bg-current" aria-hidden="true" />{priority(value)}</span>
}

function Confidence({ value }: { value: number | null | undefined }) {
  const { tr, count } = useLabels()
  return <span title={tr('Calibrated category/severity confidence; not priority or duplicate probability.', 'ক্যালিব্রেটেড বিভাগ/তীব্রতার আস্থা; এটি অগ্রাধিকার বা সদৃশ হওয়ার সম্ভাবনা নয়।')} className="inline-flex items-center rounded-lg bg-teal-100 px-2 py-1 text-xs font-medium tabular-nums text-brand">{typeof value === 'number' ? `${tr('Confidence', 'আস্থা')} ${count(value * 100)}%` : tr('Unscored', 'স্কোর হয়নি')}</span>
}

function Alert({ children, retry }: { children: ReactNode; retry?: () => void }) {
  const { tr } = useLabels()
  return <div role="alert" className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-teal-200 bg-teal-50 p-4 text-sm leading-6 text-brand"><p className="min-w-0 flex-1">{children}</p>{retry && <button type="button" onClick={retry} className="font-semibold underline underline-offset-4">{tr('Refresh', 'রিফ্রেশ')}</button>}</div>
}

function Spinner() {
  const { tr } = useLabels()
  return <div role="status" className="flex items-center justify-center gap-3 py-16 text-sm text-brand"><span aria-hidden="true" className="size-5 animate-spin rounded-full border-2 border-teal-200 border-t-teal-700 motion-reduce:animate-none" />{tr('Loading reports…', 'অভিযোগ লোড হচ্ছে…')}</div>
}

function Drawer({ title, onClose, children, wide = false }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const panel = useRef<HTMLDivElement>(null)
  const close = useRef(onClose)
  close.current = onClose
  const { tr } = useLabels()
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    panel.current?.querySelector<HTMLElement>('button')?.focus()
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); close.current(); return }
      if (event.key !== 'Tab') return
      const items = [...(panel.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex="0"]') || [])].filter(item => item.getClientRects().length > 0)
      const first = items[0], last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
    document.addEventListener('keydown', keydown)
    return () => { document.body.style.overflow = overflow; document.removeEventListener('keydown', keydown); previous?.focus() }
  }, [])
  return <div className="fixed inset-0 z-50 flex justify-end">
    <button type="button" tabIndex={-1} aria-label={tr('Close details', 'বিস্তারিত বন্ধ করুন')} onClick={onClose} className="absolute inset-0 bg-teal-950/35 backdrop-blur-[2px]" />
    <div ref={panel} role="dialog" aria-modal="true" aria-labelledby="detail-panel-title" className={`relative flex h-full w-full flex-col bg-teal-50 shadow-2xl ${wide ? 'max-w-6xl' : 'max-w-xl'}`}>
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-teal-200 bg-white px-5 py-4 sm:px-7"><h2 id="detail-panel-title" className="text-lg font-semibold text-brand">{title}</h2><button type="button" onClick={onClose} className={secondaryButton} aria-label={tr('Close details', 'বিস্তারিত বন্ধ করুন')}><SmallIcon kind="close" /></button></header>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-5 sm:p-7">{children}</div>
    </div>
  </div>
}

function ComplaintDetail({ complaint, token, updated, reload }: { complaint: Complaint; token: string; updated: (value: Complaint) => void; reload: () => void }) {
  const { tr, status, language, date, failure, count } = useLabels()
  const [form, setForm] = useState({ status: complaint.status, score: String(complaint.priority_score), label: complaint.priority_label, category: complaint.category || '', severity: complaint.severity || '', reason: '' })
  const [saving, setSaving] = useState(false)
  const savingRef = useRef(false)
  const [error, setError] = useState<unknown>(null)
  const [saved, setSaved] = useState(false)
  useEffect(() => { setForm({ status: complaint.status, score: String(complaint.priority_score), label: complaint.priority_label, category: complaint.category || '', severity: complaint.severity || '', reason: '' }) }, [complaint])
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (savingRef.current) return
    const submitter = (event.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null
    savingRef.current = true
    setSaving(true); setError(null); setSaved(false)
    try {
      const result = await api.patchComplaint(complaint.id, {
        expected_version: complaint.version, reason: form.reason.trim(), status: submitter?.value === 'confirm' ? 'confirmed' : form.status,
        priority_score: Number(form.score), priority_label: form.label.trim(),
        ...(form.category.trim() ? { category: form.category.trim() } : {}),
        ...(form.severity.trim() ? { severity: form.severity.trim() } : {}),
      }, token)
      updated(result); setSaved(true)
    } catch (value) { setError(value) } finally { savingRef.current = false; setSaving(false) }
  }
  return <div className="space-y-7">
    <section className="rounded-2xl border border-teal-200 bg-white p-5">
      <div className="flex flex-wrap items-center gap-2"><PriorityBadge value={complaint.priority_label} /><Confidence value={complaint.confidence} /><span className="rounded-lg bg-teal-100 px-2 py-1 text-xs text-brand">{language(complaint.lang)}</span></div>
      <p className="mt-4 whitespace-pre-wrap break-words text-base leading-7 text-brand" lang={complaint.lang === 'bn' ? 'bn' : 'en'}>{complaint.text}</p>
      {complaint.analysis && (('category' in complaint.analysis && complaint.analysis.category !== complaint.category) || ('severity' in complaint.analysis && complaint.analysis.severity !== complaint.severity)) && <p className="mt-3 rounded-lg bg-teal-50 px-3 py-2 text-xs leading-5 text-brand">{tr('Category or severity was reviewed. The confidence badge belongs to the original automatic prediction, not the human correction.', 'বিভাগ বা তীব্রতা পর্যালোচনা করা হয়েছে। আস্থার মান মূল স্বয়ংক্রিয় পূর্বাভাসের জন্য, মানব সংশোধনের জন্য নয়।')}</p>}
      <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-teal-100 pt-4 text-xs">
        <div><dt className="text-brand">{tr('Area', 'এলাকা')}</dt><dd className="mt-1 break-words font-medium text-brand">{complaint.location || tr('Not specified', 'উল্লেখ নেই')}</dd></div>
        <div><dt className="text-brand">{tr('Status', 'অবস্থা')}</dt><dd className="mt-1 font-medium text-brand">{status(complaint.status)}</dd></div>
        <div><dt className="text-brand">{tr('Received', 'জমার সময়')}</dt><dd className="mt-1 font-medium text-brand">{date(complaint.created_at)}</dd></div>
        <div><dt className="text-brand">{tr('Priority score', 'অগ্রাধিকার স্কোর')}</dt><dd className="mt-1 font-semibold tabular-nums text-brand">{count(complaint.priority_score)}</dd></div>
        {complaint.contact && <div className="col-span-2"><dt className="text-brand">{tr('Private contact · administrator only', 'ব্যক্তিগত যোগাযোগ · শুধু প্রশাসকের জন্য')}</dt><dd className="mt-1 break-all font-medium text-brand">{complaint.contact}</dd></div>}
        <div className="col-span-2"><dt className="text-brand">{tr('Tracking ID', 'ট্র্যাকিং আইডি')}</dt><dd className="mt-1 break-all font-mono text-[11px] text-brand">{complaint.id}</dd></div>
      </dl>
    </section>
    <ExplanationPanel analysis={complaint.analysis} currentScore={complaint.priority_score} currentLabel={complaint.priority_label} />
    <section className="rounded-2xl border border-teal-200 bg-white p-5">
      <h3 className="text-base font-semibold text-brand">{tr('Review & update', 'পর্যালোচনা ও হালনাগাদ')}</h3>
      <p className="mt-1 text-xs leading-5 text-brand">{tr('Your reason and changes are saved as feedback. Automatic evidence is preserved.', 'আপনার কারণ ও পরিবর্তন ভবিষ্যৎ পর্যালোচনার জন্য সংরক্ষিত হবে। স্বয়ংক্রিয় প্রমাণ অক্ষত থাকবে।')}</p>
      <form onSubmit={save} className="mt-4 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1.5 text-xs font-medium text-brand">{tr('Status', 'অবস্থা')}<select className={fieldClass} value={form.status} onChange={event => setForm({ ...form, status: event.target.value as Complaint['status'] })}>{STATUSES.map(value => <option key={value} value={value}>{status(value)}</option>)}</select></label>
          <label className="space-y-1.5 text-xs font-medium text-brand">{tr('Priority label', 'অগ্রাধিকারের লেবেল')}<input className={fieldClass} value={form.label} required maxLength={32} list="priority-labels" onChange={event => setForm({ ...form, label: event.target.value })} /><datalist id="priority-labels">{[...new Set(['low', 'moderate', 'high', 'urgent', complaint.priority_label])].map(value => <option key={value} value={value} />)}</datalist></label>
          <label className="space-y-1.5 text-xs font-medium text-brand">{tr('Priority score', 'অগ্রাধিকার স্কোর')}<input className={fieldClass} type="number" inputMode="decimal" min="0" max="1000" step="any" value={form.score} required onChange={event => setForm({ ...form, score: event.target.value })} /></label>
          <label className="space-y-1.5 text-xs font-medium text-brand">{tr('Severity', 'তীব্রতা')}<input className={fieldClass} value={form.severity} required={Boolean(complaint.severity)} maxLength={64} onChange={event => setForm({ ...form, severity: event.target.value })} /></label>
        </div>
        <label className="block space-y-1.5 text-xs font-medium text-brand">{tr('Category', 'বিভাগ')}<input className={fieldClass} value={form.category} required={Boolean(complaint.category)} maxLength={128} onChange={event => setForm({ ...form, category: event.target.value })} /></label>
        <label className="block space-y-1.5 text-xs font-medium text-brand">{tr('Reason for review', 'পর্যালোচনার কারণ')}<textarea className={fieldClass} rows={3} minLength={3} maxLength={2000} required value={form.reason} onChange={event => setForm({ ...form, reason: event.target.value })} /></label>
        {error !== null && <Alert retry={reload}>{failure(error)}</Alert>}
        {saved && <p role="status" className="rounded-xl bg-teal-50 px-3 py-2 text-sm text-brand">{tr('Review saved. The queue has been updated.', 'পর্যালোচনা সংরক্ষিত। তালিকা হালনাগাদ হয়েছে।')}</p>}
        <div className="flex flex-wrap gap-2"><button type="submit" name="decision" value="save" disabled={saving} className={primaryButton}><SmallIcon kind="check" />{saving ? tr('Saving…', 'সংরক্ষণ হচ্ছে…') : tr('Save review', 'পর্যালোচনা সংরক্ষণ')}</button><button type="submit" name="decision" value="confirm" disabled={saving} className={secondaryButton}>{tr('Confirm priority', 'অগ্রাধিকার নিশ্চিত করুন')}</button></div>
      </form>
    </section>
  </div>
}

function GroupDetails({ groupId, token, changed }: { groupId: string; token: string; changed: () => void }) {
  const { tr, language, failure, date, count } = useLabels()
  const [group, setGroup] = useState<Group | null>(null)
  const [members, setMembers] = useState<Complaint[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [revision, setRevision] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<unknown>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [reason, setReason] = useState('')
  const [mergeIds, setMergeIds] = useState('')
  const [saving, setSaving] = useState(false)
  const savingRef = useRef(false)
  const [notice, setNotice] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError(null)
    Promise.all([api.getGroup(groupId, token, controller.signal), api.listComplaints({ group_id: groupId, page, page_size: 30 }, token, controller.signal)])
      .then(([value, response]) => { if (!controller.signal.aborted) { setGroup(value); setMembers(response.items); setTotal(response.total) } })
      .catch(value => { if (!controller.signal.aborted) setError(value) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [groupId, token, revision, page])
  async function action(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!group || group.id !== groupId || loading || savingRef.current) return
    const submitter = (event.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null
    const operation = submitter?.value || 'confirm'
    savingRef.current = true
    setSaving(true); setError(null); setNotice('')
    try {
      if (operation === 'split') {
        if (!selected.length || selected.length >= group.size) throw new Error(tr('Select at least one report and leave at least one in the original group.', 'অন্তত একটি অভিযোগ বাছুন এবং মূল দলে অন্তত একটি রাখুন।'))
        await api.splitGroup(group.id, { expected_version: group.version, reason: reason.trim(), complaint_ids: selected }, token)
        setNotice(tr('Selected reports were moved to a separate reviewed group.', 'বাছাইকৃত অভিযোগ আলাদা পর্যালোচিত দলে সরানো হয়েছে।'))
      } else {
        const ids = [...new Set(mergeIds.split(/[\s,]+/).map(value => value.trim()).filter(Boolean))]
        await api.confirmGroup(group.id, { expected_version: group.version, reason: reason.trim(), ...(ids.length ? { complaint_ids: ids } : {}) }, token)
        setNotice(ids.length ? tr('Reviewed membership was merged and confirmed.', 'পর্যালোচিত সদস্যপদ একত্র করে নিশ্চিত করা হয়েছে।') : tr('Group confirmed.', 'দল নিশ্চিত করা হয়েছে।'))
      }
      setSelected([]); setReason(''); setMergeIds(''); setRevision(value => value + 1); changed()
    } catch (value) { setError(value) } finally { savingRef.current = false; setSaving(false) }
  }
  const toggle = (id: string) => setSelected(previous => previous.includes(id) ? previous.filter(value => value !== id) : [...previous, id])
  if (loading && (!group || group.id !== groupId)) return <Spinner />
  return <div className="space-y-5">
    {error !== null && <Alert retry={() => setRevision(value => value + 1)}>{failure(error)}</Alert>}
    {group && group.id === groupId && <>
      <div className="rounded-xl border border-teal-100 bg-teal-50 p-4"><p className="text-sm font-semibold text-brand">{count(group.size)} {tr('reports in this incident group', 'অভিযোগ এই ঘটনার দলে')}</p><p className="mt-1 text-xs leading-5 text-brand">{tr('Compare the submitted versions. Similar wording alone does not establish the same incident; check area, date and service context.', 'জমা দেওয়া ভাষার সংস্করণ তুলনা করুন। শুধু লেখার মিল একই ঘটনা প্রমাণ করে না; এলাকা, সময় ও সেবার প্রেক্ষাপট যাচাই করুন।')}</p></div>
      <div className="grid gap-3 lg:grid-cols-3">{(['bn', 'en', 'banglish'] as const).map(lang => <section key={lang} className="min-w-0 rounded-xl border border-teal-200 bg-white p-3"><h3 className="mb-3 flex items-center justify-between text-sm font-semibold text-brand">{language(lang)}<span className="rounded-lg bg-teal-100 px-2 py-0.5 text-xs font-normal text-brand">{members.filter(row => row.lang === lang).length}</span></h3>
        <div className="space-y-3">{members.filter(row => row.lang === lang).map(member => <label key={member.id} className={`block cursor-pointer rounded-xl border p-3 ${selected.includes(member.id) ? 'border-teal-600 bg-teal-50' : 'border-teal-200 bg-teal-50/50'}`}><span className="flex items-start gap-2"><input className="mt-1 size-4 shrink-0 accent-teal-800" type="checkbox" checked={selected.includes(member.id)} onChange={() => toggle(member.id)} aria-label={`${tr('Select report', 'অভিযোগ বাছুন')} ${member.id}`} /><span className="min-w-0 whitespace-pre-wrap break-words text-sm leading-6 text-brand" lang={lang === 'bn' ? 'bn' : 'en'}>{member.text}</span></span><span className="mt-3 block text-xs leading-5 text-brand">{member.location || tr('No area specified', 'এলাকা নেই')}<br />{date(member.timestamp || member.created_at)}</span><span className="mt-2 block break-all font-mono text-[10px] text-brand">{member.id}</span></label>)}
          {!members.some(row => row.lang === lang) && <p className="py-6 text-xs leading-5 text-brand">{tr('No submitted report in this language on this page.', 'এই পাতায় এই ভাষায় জমা দেওয়া অভিযোগ নেই।')}</p>}
        </div></section>)}</div>
      {total > 30 && <div className="flex items-center justify-between gap-2"><button className={secondaryButton} disabled={page <= 1 || loading} onClick={() => setPage(value => value - 1)}>{tr('Previous members', 'আগের সদস্য')}</button><span className="text-xs text-brand">{tr('Page', 'পাতা')} {count(page)} / {count(Math.ceil(total / 30))}</span><button className={secondaryButton} disabled={page * 30 >= total || loading} onClick={() => setPage(value => value + 1)}>{tr('Next members', 'পরের সদস্য')}</button></div>}
      <form onSubmit={action} className="space-y-4 rounded-2xl border border-teal-200 bg-white p-5">
        <h3 className="text-base font-semibold text-brand">{tr('Review group membership', 'দলের সদস্যপদ পর্যালোচনা')}</h3>
        <label className="block space-y-1.5 text-xs font-medium text-brand">{tr('Add reviewed report IDs (optional)', 'পর্যালোচিত অভিযোগের আইডি যোগ করুন (ঐচ্ছিক)')}<textarea rows={2} className={fieldClass} maxLength={38000} value={mergeIds} onChange={event => setMergeIds(event.target.value)} placeholder={tr('Paste tracking IDs, separated by commas or new lines', 'কমা বা নতুন লাইনে ট্র্যাকিং আইডি লিখুন')} /></label>
        <p className="text-xs leading-5 text-brand">{tr('Adding IDs moves those reports into this group. Confirm only after checking their incident evidence.', 'আইডি যোগ করলে অভিযোগগুলো এই দলে চলে আসবে। ঘটনার প্রমাণ যাচাই করে নিশ্চিত করুন।')}</p>
        <label className="block space-y-1.5 text-xs font-medium text-brand">{tr('Reason for group decision', 'দলের সিদ্ধান্তের কারণ')}<textarea rows={3} className={fieldClass} minLength={3} maxLength={2000} required value={reason} onChange={event => setReason(event.target.value)} /></label>
        {notice && <p role="status" className="rounded-xl bg-teal-50 p-3 text-sm text-brand">{notice}</p>}
        <div className="flex flex-wrap gap-2"><button type="submit" value="confirm" disabled={saving || loading} className={primaryButton}><SmallIcon kind="check" />{saving ? tr('Saving…', 'সংরক্ষণ হচ্ছে…') : mergeIds.trim() ? tr('Merge & confirm', 'একত্র করে নিশ্চিত করুন') : tr('Confirm group', 'দল নিশ্চিত করুন')}</button><button type="submit" value="split" disabled={saving || loading || selected.length === 0 || selected.length >= group.size} className={secondaryButton}>{tr('Split selected', 'বাছাইকৃতগুলো আলাদা করুন')} ({count(selected.length)})</button></div>
        <p className="text-xs leading-5 text-brand">{tr('A split protects the reviewed partitions from automatic regrouping. Your reason is retained in the audit trail.', 'আলাদা করা হলে পর্যালোচিত দলগুলো স্বয়ংক্রিয়ভাবে একত্র হবে না। আপনার কারণ নিরীক্ষার রেকর্ডে থাকবে।')}</p>
      </form>
    </>}
  </div>
}

export default function AdminDashboard() {
  const { token, logout } = useAuth()
  const { tr, priority, status, language, count, date, area, failure } = useLabels()
  const [tab, setTab] = useState<Tab>('queue')
  const [filters, setFilters] = useState({ category: '', priority: '', area: '', status: '' })
  const [applied, setApplied] = useState(filters)
  const [page, setPage] = useState(1)
  const [revision, setRevision] = useState(0)
  const [complaints, setComplaints] = useState<ComplaintPage | null>(null)
  const [groups, setGroups] = useState<GroupPage | null>(null)
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<unknown>(null)
  const [analyticsError, setAnalyticsError] = useState<unknown>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<Complaint | null>(null)
  const [detailError, setDetailError] = useState<unknown>(null)
  const [detailRevision, setDetailRevision] = useState(0)
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null)
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null)
  const refresh = () => setRevision(value => value + 1)
  useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    setAnalyticsError(null)
    api.analytics(token, controller.signal).then(value => { if (!controller.signal.aborted) setAnalytics(value) }).catch(value => { if (!controller.signal.aborted) setAnalyticsError(value) })
    return () => controller.abort()
  }, [token, revision])
  useEffect(() => {
    if (!token || tab === 'analytics') { setLoading(false); setError(null); return }
    const controller = new AbortController()
    setLoading(true); setError(null)
    const params = Object.fromEntries(Object.entries(applied).filter(([, value]) => value.trim()))
    const request = tab === 'groups' ? api.listGroups({ page, page_size: 10 }, token, controller.signal).then(value => { if (!controller.signal.aborted) setGroups(value) })
      : api.listComplaints({ ...params, page, page_size: 15, ...(tab === 'review' ? { review_required: true } : {}) }, token, controller.signal).then(value => { if (!controller.signal.aborted) setComplaints(value) })
    request.catch(value => { if (!controller.signal.aborted) setError(value) }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [token, tab, page, applied, revision])
  useEffect(() => {
    if (!selectedId || !token) return
    const controller = new AbortController()
    setDetail(null); setDetailError(null)
    api.getComplaint(selectedId, token, controller.signal).then(value => { if (!controller.signal.aborted) setDetail(value) }).catch(value => { if (!controller.signal.aborted) setDetailError(value) })
    return () => controller.abort()
  }, [selectedId, token, detailRevision])
  const currentTotal = tab === 'groups' ? groups?.total || 0 : complaints?.total || 0
  const pageSize = tab === 'groups' ? 10 : 15
  const tabs: { key: Tab; label: string; icon: 'list' | 'alert' | 'group' | 'chart' }[] = [
    { key: 'queue', label: tr('Priority queue', 'অগ্রাধিকার তালিকা'), icon: 'list' }, { key: 'review', label: tr('Needs review', 'পর্যালোচনা প্রয়োজন'), icon: 'alert' },
    { key: 'groups', label: tr('Duplicate groups', 'সদৃশ অভিযোগের দল'), icon: 'group' }, { key: 'analytics', label: tr('Analytics', 'পরিসংখ্যান'), icon: 'chart' },
  ]
  if (!token) return null
  return <div className="mx-auto max-w-[1440px] space-y-6 px-4 py-7 sm:px-6 lg:px-9 lg:py-9">
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div><p className="text-xs font-semibold uppercase tracking-[.16em] text-brand">{tr('SERVICE OPERATIONS', 'সেবা পরিচালনা')}</p><h1 className="mt-2 text-2xl font-semibold tracking-tight text-brand sm:text-3xl">{tr('Every report. A clearer next step.', 'প্রতি অভিযোগে, এগিয়ে যাওয়ার পথ।')}</h1><p className="mt-2 max-w-xl text-sm leading-6 text-brand">{tr('Prioritize urgent needs, connect related reports, and keep people informed.', 'জরুরি প্রয়োজনকে অগ্রাধিকার দিন, সম্পর্কিত অভিযোগ যুক্ত করুন এবং মানুষকে অবগত রাখুন।')}</p></div>
      <div className="flex items-center gap-2"><button type="button" onClick={refresh} disabled={loading} className={secondaryButton}><SmallIcon kind="refresh" />{tr('Refresh', 'রিফ্রেশ')}</button><button type="button" onClick={logout} className="min-h-10 rounded-xl px-3 text-sm font-medium text-brand hover:bg-teal-100 hover:text-brand">{tr('Sign out', 'লগআউট')}</button></div>
    </header>
    {analyticsError !== null && <Alert retry={refresh}>{failure(analyticsError)}</Alert>}
    <section aria-label={tr('All-system summary', 'পুরো ব্যবস্থার সারসংক্ষেপ')} className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {[
        { title: tr('Total reports', 'মোট অভিযোগ'), value: analytics?.total_complaints, icon: 'list' as const, note: tr('Across all areas', 'সব এলাকা মিলিয়ে') },
        { title: tr('Needs human review', 'মানব পর্যালোচনা প্রয়োজন'), value: analytics?.needs_human_review, icon: 'alert' as const, note: tr('Awaiting a decision', 'সিদ্ধান্তের অপেক্ষায়') },
        { title: tr('Recorded groups', 'দলের রেকর্ড'), value: analytics?.total_groups, icon: 'group' as const, note: tr('Includes single and emptied groups', 'একক ও পর্যালোচনায় খালি হওয়া দলসহ') },
        { title: tr('Resolved reports', 'সমাধান হওয়া অভিযোগ'), value: analytics?.by_status.resolved || (analytics ? 0 : undefined), icon: 'check' as const, note: tr('Current resolved status', 'বর্তমানে সমাধান হয়েছে') },
      ].map((metric, index) => <div key={index} className={`rounded-2xl border p-4 sm:p-5 ${index === 1 ? 'border-teal-800 bg-teal-100 text-brand' : 'border-teal-200 bg-white text-brand'}`}>
        <div className="flex items-center justify-between gap-2"><p className="text-xs font-medium text-brand">{metric.title}</p><SmallIcon kind={metric.icon} className="size-4 text-brand" /></div>
        <p className="mt-4 text-3xl font-semibold tracking-tight tabular-nums">{metric.value === undefined ? '—' : count(metric.value)}</p><p className="mt-1 text-xs text-brand">{metric.note}</p>
      </div>)}
    </section>
    <div className="flex overflow-x-auto border-b border-teal-200" role="tablist" aria-label={tr('Workspace views', 'কাজের দৃশ্য')}>
      {tabs.map((item, index) => <button key={item.key} type="button" role="tab" id={`tab-${item.key}`} aria-controls="workspace-panel" aria-selected={tab === item.key} tabIndex={tab === item.key ? 0 : -1}
        onKeyDown={event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; setTab(tabs[next].key); setPage(1); document.getElementById(`tab-${tabs[next].key}`)?.focus() } }}
        onClick={() => { setTab(item.key); setPage(1); setExpandedGroup(null) }} className={`flex min-h-12 shrink-0 items-center gap-2 border-b-2 px-4 text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-teal-700 ${tab === item.key ? 'border-teal-800 text-brand' : 'border-transparent text-brand hover:text-brand'}`}><SmallIcon kind={item.icon} />{item.label}{item.key === 'review' && (analytics?.needs_human_review || 0) > 0 && <span className="rounded-md bg-teal-100 px-1.5 py-0.5 text-[10px] text-brand">{count(analytics!.needs_human_review)}</span>}</button>)}
    </div>
    <section id="workspace-panel" role="tabpanel" aria-labelledby={`tab-${tab}`} className="space-y-5">
      {(tab === 'queue' || tab === 'review') && <form onSubmit={event => { event.preventDefault(); setApplied({ ...filters }); setPage(1) }} className="flex flex-wrap items-end gap-3 rounded-2xl border border-teal-200 bg-white p-4">
        <label className="min-w-32 flex-1 space-y-1.5 text-xs font-medium text-brand">{tr('Category', 'বিভাগ')}<select className={fieldClass} value={filters.category} onChange={event => setFilters({ ...filters, category: event.target.value })}><option value="">{tr('All categories', 'সব বিভাগ')}</option>{Object.keys(analytics?.by_category || {}).filter(value => value !== 'unknown').sort().map(value => <option key={value} value={value}>{value}</option>)}</select></label>
        <label className="min-w-28 flex-1 space-y-1.5 text-xs font-medium text-brand">{tr('Priority', 'অগ্রাধিকার')}<select className={fieldClass} value={filters.priority} onChange={event => setFilters({ ...filters, priority: event.target.value })}><option value="">{tr('All priorities', 'সব অগ্রাধিকার')}</option>{[...new Set(['urgent', 'high', 'moderate', 'low', ...Object.keys(analytics?.by_priority || {})])].map(value => <option key={value} value={value}>{priority(value)}</option>)}</select></label>
        <label className="min-w-32 flex-1 space-y-1.5 text-xs font-medium text-brand">{tr('Area', 'এলাকা')}<input className={fieldClass} value={filters.area} maxLength={256} list="area-options" onChange={event => setFilters({ ...filters, area: event.target.value })} placeholder={tr('All areas', 'সব এলাকা')} /><datalist id="area-options">{Object.keys(analytics?.by_area || {}).filter(value => value !== 'unknown').map(value => <option key={value} value={value}>{area(value)}</option>)}</datalist></label>
        <label className="min-w-32 flex-1 space-y-1.5 text-xs font-medium text-brand">{tr('Status', 'অবস্থা')}<select className={fieldClass} value={filters.status} onChange={event => setFilters({ ...filters, status: event.target.value })}><option value="">{tr('All statuses', 'সব অবস্থা')}</option>{STATUSES.map(value => <option key={value} value={value}>{status(value)}</option>)}</select></label>
        <button className={primaryButton} type="submit">{tr('Apply filters', 'ফিল্টার প্রয়োগ')}</button><button className={secondaryButton} type="button" onClick={() => { const empty = { category: '', priority: '', area: '', status: '' }; setFilters(empty); setApplied(empty); setPage(1) }}>{tr('Reset', 'রিসেট')}</button>
      </form>}
      {error !== null && <Alert retry={refresh}>{failure(error)}</Alert>}
      {tab === 'analytics' ? analytics ? <Charts data={analytics} /> : analyticsError === null && <Spinner /> : loading ? <Spinner /> : tab === 'groups' ? <div className="space-y-3">
        <p className="text-xs text-brand">{tr('Proposed groups include single reports. Expand a group to compare the submitted languages.', 'প্রস্তাবিত দলে একক অভিযোগও আছে। জমা দেওয়া ভাষাগুলো তুলনা করতে দল খুলুন।')}</p>
        {groups?.items.map(group => <article key={group.id} className="overflow-hidden rounded-2xl border border-teal-200 bg-white">
          <button type="button" aria-expanded={expandedGroup === group.id} aria-controls={`group-${group.id}`} onClick={() => setExpandedGroup(expandedGroup === group.id ? null : group.id)} className="flex w-full items-center gap-3 p-4 text-left transition hover:bg-teal-50 sm:p-5"><span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-brand"><SmallIcon kind="group" className="size-5" /></span><span className="min-w-0 flex-1"><span className="block text-sm font-semibold text-brand">{count(group.size)} {tr('reports', 'অভিযোগ')}<span className="ml-2 rounded-lg bg-teal-100 px-2 py-0.5 text-[10px] font-normal text-brand">{group.status === 'confirmed' ? tr('Confirmed', 'নিশ্চিত') : group.status === 'reviewed_split' ? tr('Reviewed split', 'পর্যালোচিত পৃথক দল') : tr('Proposed', 'প্রস্তাবিত')}</span></span><span className="mt-1 block truncate font-mono text-[10px] text-brand">{group.id}</span></span><span className="hidden text-xs text-brand sm:block">{date(group.updated_at)}</span><SmallIcon kind="arrow" className={`size-4 shrink-0 text-brand transition ${expandedGroup === group.id ? 'rotate-90' : ''}`} /></button>
          {expandedGroup === group.id && <div id={`group-${group.id}`} className="border-t border-teal-100 bg-teal-50/40 p-4 sm:p-5"><GroupDetails groupId={group.id} token={token} changed={refresh} /></div>}
        </article>)}
        {!groups?.items.length && error === null && <Empty title={tr('No incident groups yet', 'এখনো ঘটনার দল নেই')} description={tr('Groups appear as reports are received.', 'অভিযোগ জমা হলে দল দেখা যাবে।')} />}
      </div> : <div className="overflow-hidden rounded-2xl border border-teal-200 bg-white">
        <div className="flex items-center justify-between border-b border-teal-100 px-5 py-4"><h2 className="text-sm font-semibold text-brand">{tab === 'review' ? tr('Reports needing a decision', 'সিদ্ধান্ত প্রয়োজন এমন অভিযোগ') : tr('Highest priority first', 'সর্বোচ্চ অগ্রাধিকার আগে')}</h2><span className="text-xs tabular-nums text-brand">{count(complaints?.total || 0)} {tr('reports', 'অভিযোগ')}</span></div>
        <div className="hidden overflow-x-auto md:block"><table className="w-full text-left text-sm"><caption className="sr-only">{tr('Complaint priority queue sorted by priority score', 'অগ্রাধিকার স্কোর অনুযায়ী অভিযোগের তালিকা')}</caption><thead className="border-b border-teal-100 bg-teal-50/80 text-[11px] font-medium uppercase tracking-wider text-brand"><tr><th scope="col" className="px-5 py-3">{tr('Complaint', 'অভিযোগ')}</th><th scope="col" className="px-4 py-3">{tr('Priority', 'অগ্রাধিকার')}</th><th scope="col" className="px-4 py-3">{tr('Area', 'এলাকা')}</th><th scope="col" className="px-4 py-3">{tr('Status', 'অবস্থা')}</th><th scope="col" className="px-4 py-3"><span className="sr-only">{tr('Actions', 'পদক্ষেপ')}</span></th></tr></thead><tbody>{complaints?.items.map(row => <tr key={row.id} className="border-b border-teal-100 transition last:border-0 hover:bg-teal-50/30"><td className="max-w-sm px-5 py-4"><button type="button" onClick={() => setSelectedId(row.id)} aria-label={`${tr('Open complaint', 'অভিযোগ খুলুন')} ${row.id}`} className="w-full text-left"><span className="line-clamp-2 leading-6 font-medium text-brand" lang={row.lang === 'bn' ? 'bn' : 'en'}>{row.text}</span><span className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-brand"><span>{row.category || tr('Awaiting classification', 'শ্রেণিবিভাগের অপেক্ষায়')}</span><span aria-hidden="true">·</span><span>{language(row.lang)}</span>{row.review_required && <span className="rounded-md bg-teal-50 px-1.5 text-brand">{tr('Needs review', 'পর্যালোচনা প্রয়োজন')}</span>}</span></button></td><td className="px-4 py-4"><PriorityBadge value={row.priority_label} /><p className="mt-1.5 text-[11px] tabular-nums text-brand">{tr('Score', 'স্কোর')} {count(row.priority_score)}</p></td><td className="max-w-40 px-4 py-4 text-xs text-brand"><span className="line-clamp-2">{row.location || area(row.area_id)}</span></td><td className="px-4 py-4 text-xs whitespace-nowrap text-brand">{status(row.status)}</td><td className="px-4 py-4"><button type="button" onClick={() => setSelectedId(row.id)} className="flex size-9 items-center justify-center rounded-lg border border-teal-200 text-brand hover:bg-teal-50" aria-label={`${tr('Review report', 'অভিযোগ পর্যালোচনা')} ${row.id}`}><SmallIcon kind="arrow" /></button></td></tr>)}</tbody></table></div>
        <div className="divide-y divide-slate-100 md:hidden">{complaints?.items.map(row => <button key={row.id} type="button" onClick={() => setSelectedId(row.id)} aria-label={`${tr('Open complaint', 'অভিযোগ খুলুন')} ${row.id}`} className="block w-full space-y-3 p-4 text-left hover:bg-teal-50"><span className="flex items-center justify-between gap-2"><PriorityBadge value={row.priority_label} /><span className="text-xs text-brand">{language(row.lang)}</span></span><span className="line-clamp-3 text-sm leading-6 font-medium text-brand" lang={row.lang === 'bn' ? 'bn' : 'en'}>{row.text}</span><span className="flex flex-wrap items-center justify-between gap-2 text-xs text-brand"><span>{row.location || area(row.area_id)}</span><span>{status(row.status)}</span></span>{row.review_required && <span className="inline-block rounded-lg bg-teal-50 px-2 py-1 text-[11px] text-brand">{tr('Needs review', 'পর্যালোচনা প্রয়োজন')}</span>}</button>)}</div>
        {!complaints?.items.length && error === null && <Empty title={tr('Your queue is clear', 'তালিকায় কোনো অভিযোগ নেই')} description={tr('No reports match these filters. Try changing the filters or check back after new submissions.', 'এই ফিল্টারে অভিযোগ পাওয়া যায়নি। ফিল্টার বদলান বা নতুন অভিযোগ জমা হলে আবার দেখুন।')} />}
      </div>}
      {tab !== 'analytics' && currentTotal > pageSize && <nav aria-label={tr('Result pages', 'ফলাফলের পাতা')} className="flex items-center justify-between gap-3"><p className="text-xs text-brand">{tr('Page', 'পাতা')} {count(page)} / {count(Math.ceil(currentTotal / pageSize))}</p><div className="flex gap-2"><button className={secondaryButton} disabled={loading || page <= 1} onClick={() => setPage(value => value - 1)}>{tr('Previous', 'আগের পাতা')}</button><button className={secondaryButton} disabled={loading || page * pageSize >= currentTotal} onClick={() => setPage(value => value + 1)}>{tr('Next', 'পরের পাতা')}</button></div></nav>}
    </section>
    {selectedId && <Drawer title={tr('Complaint review', 'অভিযোগ পর্যালোচনা')} onClose={() => setSelectedId(null)}>{detailError !== null ? <Alert retry={() => setDetailRevision(value => value + 1)}>{failure(detailError)}</Alert> : detail?.id === selectedId ? <><ComplaintDetail complaint={detail} token={token} reload={() => setDetailRevision(value => value + 1)} updated={value => { setDetail(previous => previous?.id === value.id ? value : previous); refresh() }} />{detail.group_id && <button type="button" onClick={() => { setSelectedId(null); setSelectedGroup(detail.group_id) }} className={`${secondaryButton} mt-5 w-full`}><SmallIcon kind="group" />{tr('Review related report group', 'সম্পর্কিত অভিযোগের দল দেখুন')}</button>}</> : <Spinner />}</Drawer>}
    {selectedGroup && <Drawer wide title={tr('Incident group review', 'ঘটনার দল পর্যালোচনা')} onClose={() => setSelectedGroup(null)}><GroupDetails key={selectedGroup} groupId={selectedGroup} token={token} changed={refresh} /></Drawer>}
  </div>
}

function Empty({ title, description }: { title: string; description: string }) {
  return <div className="flex flex-col items-center px-6 py-16 text-center"><span className="mb-4 flex size-12 items-center justify-center rounded-2xl bg-teal-50 text-brand"><SmallIcon kind="check" className="size-6" /></span><h3 className="text-base font-semibold text-brand">{title}</h3><p className="mt-2 max-w-sm text-sm leading-6 text-brand">{description}</p></div>
}
