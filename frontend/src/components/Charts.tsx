import { useId } from 'react'
import { useI18n } from '../i18n'
import { api } from '../api'

type Analytics = Awaited<ReturnType<typeof api.analytics>>
const BRAND = '#076657'
const PRIORITY_ORDER = ['urgent', 'high', 'moderate', 'medium', 'low']

export default function Charts({ data }: { data: Analytics }) {
  const patternPrefix = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const { locale } = useI18n()
  const tr = (en: string, bn: string) => locale === 'bn' ? bn : en
  const count = (value: number) => new Intl.NumberFormat(locale === 'bn' ? 'bn-BD' : 'en-GB').format(value)
  const priority = (value: string) => ({ urgent: tr('Urgent', 'জরুরি'), high: tr('High', 'উচ্চ'), moderate: tr('Moderate', 'মাঝারি'), medium: tr('Medium', 'মাঝারি'), low: tr('Low', 'কম') }[value] || value)
  const area = (value: string) => value === 'unknown' ? tr('Not specified', 'উল্লেখ নেই') : value.replace(/^dhaka_/, '').replaceAll('_', ' ')
  const days = [...new Set(data.priority_over_time.map(row => row.date))].sort().slice(-14)
  const priorities = [...new Set(data.priority_over_time.map(row => row.priority))].sort((a, b) => {
    const left = PRIORITY_ORDER.indexOf(a), right = PRIORITY_ORDER.indexOf(b)
    return (left < 0 ? PRIORITY_ORDER.length : left) - (right < 0 ? PRIORITY_ORDER.length : right) || a.localeCompare(b)
  })
  const dayRows = days.map(date => ({ date, values: priorities.map(label => data.priority_over_time.filter(row => row.date === date && row.priority === label).reduce((sum, row) => sum + row.count, 0)) }))
  const patternKind = (value: string) => value === 'urgent' ? 0 : value === 'high' ? 1 : value === 'moderate' || value === 'medium' ? 2 : 3
  const patternId = (value: string) => `${patternPrefix}-priority-${patternKind(value)}`
  const legendPattern = (value: string) => {
    const kind = patternKind(value)
    return kind === 0 ? { backgroundColor: BRAND }
      : kind === 1 ? { backgroundImage: `repeating-linear-gradient(135deg, ${BRAND} 0 2px, white 2px 5px)` }
        : kind === 2 ? { backgroundImage: `repeating-linear-gradient(0deg, ${BRAND} 0 1px, white 1px 4px)` }
          : { backgroundImage: `radial-gradient(${BRAND} 1px, white 1px)`, backgroundSize: '4px 4px' }
  }
  const maximum = Math.max(1, ...dayRows.map(row => row.values.reduce((sum, value) => sum + value, 0)))
  const axisTicks = Math.min(4, maximum)
  const axisStep = Math.ceil(maximum / axisTicks)
  const axisMaximum = axisTicks * axisStep
  const state = (value: string) => ({ submitted: tr('Submitted', 'জমা হয়েছে'), in_review: tr('In review', 'পর্যালোচনাধীন'), confirmed: tr('Confirmed', 'নিশ্চিত'), in_progress: tr('In progress', 'কাজ চলছে'), resolved: tr('Resolved', 'সমাধান হয়েছে'), rejected: tr('Rejected', 'বাতিল') }[value] || value)

  function BarChart({ title, values, label = (value: string) => value }: { title: string; values: Record<string, number>; label?: (value: string) => string }) {
    const sorted = Object.entries(values).sort((a, b) => b[1] - a[1]).slice(0, 8)
    const max = Math.max(1, ...sorted.map(([, value]) => value))
    return <section className="rounded-2xl border border-teal-200 bg-white p-5 sm:p-6">
      <h3 className="text-base font-semibold text-brand">{title}</h3>
      <p className="mt-1 text-xs text-brand">{tr('Largest 8 groups · all complaints', 'বৃহত্তম ৮টি দল · সব অভিযোগ')}</p>
      {sorted.length ? <ul className="mt-6 space-y-4">{sorted.map(([key, value]) => <li key={key}>
        <div className="mb-1.5 flex items-center justify-between gap-3 text-sm"><span className="truncate text-brand" title={label(key)}>{label(key)}</span><span className="font-semibold tabular-nums text-brand">{count(value)}</span></div>
        <div aria-hidden="true" className="h-2 overflow-hidden rounded-full bg-teal-100"><div className="chart-bar h-full rounded-full bg-brand" style={{ width: `${value / max * 100}%` }} /></div>
      </li>)}</ul> : <p className="mt-10 text-sm text-brand">{tr('No submissions yet.', 'এখনো অভিযোগ জমা হয়নি।')}</p>}
    </section>
  }

  return <div className="space-y-5">
    <div className="rounded-xl border border-teal-100 bg-teal-50 px-4 py-3 text-sm leading-6 text-brand">{tr('All-system overview. Charts use current complaint labels and original submission dates in UTC; they do not reconstruct past priority changes.', 'পুরো ব্যবস্থার সারসংক্ষেপ। চার্টে বর্তমান অভিযোগের লেবেল ও জমার মূল তারিখ (UTC) দেখানো হয়েছে; অতীতের অগ্রাধিকার পরিবর্তন নয়।')}</div>
    <div className="grid gap-5 lg:grid-cols-2">
      <BarChart title={tr('Complaints by category', 'বিভাগ অনুযায়ী অভিযোগ')} values={data.by_category} label={value => value === 'unknown' ? tr('Awaiting classification', 'শ্রেণিবিভাগের অপেক্ষায়') : value} />
      <BarChart title={tr('Complaints by area', 'এলাকা অনুযায়ী অভিযোগ')} values={data.by_area} label={area} />
    </div>
    <section className="rounded-2xl border border-teal-200 bg-white p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="text-base font-semibold text-brand">{tr('Priority over time', 'সময়ের সাথে অগ্রাধিকার')}</h3><p className="mt-1 text-xs text-brand">{tr('Latest 14 reporting days · submission date, UTC', 'সর্বশেষ ১৪টি জমার দিন · জমার তারিখ, UTC')}</p></div>
        <div className="flex flex-wrap gap-3">{priorities.map(label => <span key={label} className="inline-flex items-center gap-1.5 text-xs text-brand"><span aria-hidden="true" className="size-3 rounded-sm border border-brand" style={legendPattern(label)} />{priority(label)}</span>)}</div>
      </div>
      {dayRows.length ? <>
        <svg role="img" aria-labelledby="priority-timeline-title priority-timeline-description" viewBox="0 0 640 235" className="mt-6 w-full">
          <title id="priority-timeline-title">{tr('Complaint counts by priority and submission date', 'অগ্রাধিকার ও জমার তারিখ অনুযায়ী অভিযোগের সংখ্যা')}</title>
          <desc id="priority-timeline-description">{tr('Stacked bars use green patterns to distinguish priorities. An accessible data table is available below.', 'অগ্রাধিকার আলাদা করতে সবুজ নকশাযুক্ত স্তূপীকৃত বার ব্যবহার করা হয়েছে। নিচে পাঠযোগ্য তথ্যসারণি আছে।')}</desc>
          <defs>
            <pattern id={`${patternPrefix}-priority-0`} patternUnits="userSpaceOnUse" width="6" height="6"><rect width="6" height="6" fill={BRAND}/></pattern>
            <pattern id={`${patternPrefix}-priority-1`} patternUnits="userSpaceOnUse" width="6" height="6"><rect width="6" height="6" fill="white"/><path d="m-1 1 2-2M0 6 6 0m-1 7 2-2" stroke={BRAND} strokeWidth="2"/></pattern>
            <pattern id={`${patternPrefix}-priority-2`} patternUnits="userSpaceOnUse" width="5" height="5"><rect width="5" height="5" fill="white"/><path d="M0 1h5" stroke={BRAND} strokeWidth="1.5"/></pattern>
            <pattern id={`${patternPrefix}-priority-3`} patternUnits="userSpaceOnUse" width="5" height="5"><rect width="5" height="5" fill="white"/><circle cx="2.5" cy="2.5" r="1" fill={BRAND}/></pattern>
          </defs>
          {Array.from({ length: axisTicks + 1 }, (_, step) => <g key={step}><line x1="40" x2="630" y1={190 - step * 168 / axisTicks} y2={190 - step * 168 / axisTicks} stroke="#076657" strokeOpacity="0.15" strokeDasharray="3 4" /><text x="30" y={194 - step * 168 / axisTicks} textAnchor="end" fill="#076657" fontSize="10">{count(axisStep * step)}</text></g>)}
          {dayRows.map((row, index) => {
            const slot = 580 / dayRows.length
            const barWidth = Math.max(4, Math.min(36, slot - 7))
            let stacked = 0
            return <g key={row.date}>{row.values.map((value, priorityIndex) => {
              const height = value / axisMaximum * 168
              const y = 190 - stacked - height
              stacked += height
              return <rect className="chart-bar" key={priorities[priorityIndex]} x={45 + index * slot + (slot - barWidth) / 2} y={y} width={barWidth} height={height} rx="2" fill={`url(#${patternId(priorities[priorityIndex])})`} stroke={BRAND} strokeWidth="0.6"><title>{`${row.date} · ${priority(priorities[priorityIndex])}: ${count(value)}`}</title></rect>
            })}<text x={45 + index * slot + slot / 2} y="211" textAnchor="middle" fill="#076657" fontSize="9">{new Intl.DateTimeFormat(locale === 'bn' ? 'bn-BD' : 'en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' }).format(new Date(row.date + 'T00:00:00Z'))}</text></g>
          })}
        </svg>
        <details className="mt-2 text-sm text-brand"><summary className="cursor-pointer font-medium text-brand">{tr('View chart data', 'চার্টের তথ্য দেখুন')}</summary><div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><caption className="sr-only">{tr('Priority counts by UTC submission date', 'UTC জমার তারিখ অনুযায়ী অগ্রাধিকারের সংখ্যা')}</caption><thead><tr className="border-b border-teal-200"><th className="p-2">{tr('Date', 'তারিখ')}</th>{priorities.map(label => <th key={label} className="p-2">{priority(label)}</th>)}</tr></thead><tbody>{dayRows.map(row => <tr key={row.date} className="border-b border-teal-100"><th scope="row" className="p-2 font-normal">{row.date}</th>{row.values.map((value, index) => <td key={index} className="p-2 tabular-nums">{count(value)}</td>)}</tr>)}</tbody></table></div></details>
      </> : <p className="py-12 text-center text-sm text-brand">{tr('A timeline appears when complaints arrive.', 'অভিযোগ জমা হলে সময়ভিত্তিক চার্ট দেখা যাবে।')}</p>}
    </section>
    <div className="grid gap-5 lg:grid-cols-2">
      <BarChart title={tr('Current priority distribution', 'বর্তমান অগ্রাধিকারের বণ্টন')} values={data.by_priority} label={priority} />
      <section className="rounded-2xl border border-teal-200 bg-white p-5 sm:p-6">
        <h3 className="text-base font-semibold text-brand">{tr('Review workload', 'পর্যালোচনার কাজ')}</h3>
        <p className="mt-1 text-xs text-brand">{tr('Human-review flag by current status', 'বর্তমান অবস্থাভেদে মানব পর্যালোচনার প্রয়োজন')}</p>
        {data.review_workload.length ? <div className="mt-5 overflow-x-auto"><table className="w-full text-left text-sm"><caption className="sr-only">{tr('Review workload by status', 'অবস্থাভেদে পর্যালোচনার কাজ')}</caption><thead><tr className="border-b border-teal-200 text-xs text-brand"><th scope="col" className="py-2 font-medium">{tr('Status', 'অবস্থা')}</th><th scope="col" className="px-2 py-2 font-medium">{tr('Needs review', 'পর্যালোচনা প্রয়োজন')}</th><th scope="col" className="py-2 text-right font-medium">{tr('Count', 'সংখ্যা')}</th></tr></thead><tbody>{data.review_workload.map((row, index) => <tr key={index} className="border-b border-teal-100 last:border-0"><th scope="row" className="py-3 font-normal text-brand">{state(row.status)}</th><td className="px-2 py-3 text-xs text-brand">{row.review_required ? tr('Yes', 'হ্যাঁ') : tr('No', 'না')}</td><td className="py-3 text-right font-semibold tabular-nums text-brand">{count(row.count)}</td></tr>)}</tbody></table></div> : <p className="mt-10 text-sm text-brand">{tr('No review workload yet.', 'এখনো পর্যালোচনার কাজ নেই।')}</p>}
      </section>
    </div>
  </div>
}
