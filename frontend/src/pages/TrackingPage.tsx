import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import type { Tracking } from '../api'
import { dateLabel, errorMessage, label, useI18n } from '../i18n'
import { Badge, ErrorNotice, Icon, Loading } from '../ui'

export default function TrackingPage() {
  const { id = '' } = useParams(), navigate = useNavigate(), { locale, t } = useI18n()
  const [input, setInput] = useState(id), [refresh, setRefresh] = useState(0)
  const [record, setRecord] = useState<Tracking | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState('')
  useEffect(() => {
    setInput(id); setRecord(null); setError(''); setBusy(false)
    if (!id) return
    if (!/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(id)) { setError(t('invalidId')); return }
    const controller = new AbortController(); setBusy(true)
    api.track(id, controller.signal).then(value => { if (!controller.signal.aborted) setRecord(value) }).catch(error => { if (!controller.signal.aborted) setError(errorMessage(error, locale)) }).finally(() => { if (!controller.signal.aborted) setBusy(false) })
    return () => controller.abort()
  }, [id, refresh, locale])
  function submit(event: FormEvent) { event.preventDefault(); const value = input.trim(); if (value === id) setRefresh(value => value + 1); else navigate('/track/' + encodeURIComponent(value)) }
  const statusIndex = record ? ['submitted', 'in_review', 'confirmed', 'in_progress', 'resolved'].indexOf(record.status) : -1
  return <div className="tracking-layout">
    <div className="eyebrow">{t('track')}</div><h1>{t('trackTitle')}</h1><p className="intro-copy">{t('trackHint')}</p>
    <form onSubmit={submit} className="tracking-search"><div className="flex-1"><label htmlFor="tracking-id">{t('trackingId')}</label><input id="tracking-id" value={input} onChange={event => setInput(event.target.value)} required maxLength={36} autoComplete="off" spellCheck={false}/></div><button className="button-primary" disabled={busy}><Icon name="search"/>{t('trackAction')}</button></form>
    {busy && <Loading text={t('tracking')}/>}{error && <ErrorNotice>{error}</ErrorNotice>}
    {record && !busy && <section className="tracking-result" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">{t('status')}</p><h2>{label(record.status, locale)}</h2></div><Badge tone={record.status === 'resolved' ? 'teal' : 'amber'}>{label(record.priority_label, locale)} {t('priority')}</Badge></div>
      <p className="mt-3 break-all font-mono text-xs text-brand">{record.id}</p>
      {record.status !== 'rejected' && <ol className="status-timeline">{(['submitted', 'in_review', 'confirmed', 'in_progress', 'resolved'] as const).map((status, index) => <li key={status} className={index <= statusIndex ? 'done' : ''} aria-current={status === record.status ? 'step' : undefined}><span>{index < statusIndex ? <Icon name="check"/> : index + 1}</span>{label(status, locale)}</li>)}</ol>}
      <dl className="tracking-facts"><div><dt>{t('category')}</dt><dd>{label(record.category, locale)}</dd></div><div><dt>{t('area')}</dt><dd>{label(record.area_id, locale)}</dd></div><div><dt>{t('submittedAt')}</dt><dd>{dateLabel(record.created_at, locale)}</dd></div><div><dt>{t('updatedAt')}</dt><dd>{dateLabel(record.updated_at, locale)}</dd></div></dl>
      {record.review_required && <div className="mt-5"><Badge tone="amber">{t('reviewNeeded')}</Badge></div>}
      <div className="mt-6 border-t border-slate-200 pt-5"><h3 className="font-semibold">{t('next')}</h3><p className="mt-2 text-sm leading-7 text-brand">{t('nextHint')}</p><button className="button-secondary mt-4" onClick={() => setRefresh(value => value + 1)}>{t('refresh')}</button></div>
    </section>}
    <p className="tracking-privacy"><Icon name="shield"/>{t('trackingPrivacy')}</p>
  </div>
}
