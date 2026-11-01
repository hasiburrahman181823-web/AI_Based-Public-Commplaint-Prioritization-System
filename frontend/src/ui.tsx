import type { ReactNode } from 'react'
import { useI18n } from './i18n'

export function Icon({ name, className = '' }: { name: 'building' | 'arrow' | 'check' | 'shield' | 'pin' | 'send' | 'search' | 'copy'; className?: string }) {
  const paths = {
    building: <><path d="M3 21V8l9-5 9 5v13M8 21V10h8v11M12 3V1M3 21h18"/><path d="M11 14h2m-2 3h2"/></>,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6"/>, check: <path d="m5 12 4 4L19 6"/>,
    shield: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/></>,
    pin: <><path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/></>,
    send: <><path d="m21 3-7 18-4-7-7-4 18-7ZM10 14 21 3"/></>,
    search: <><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/></>,
    copy: <><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/></>,
  }
  return <svg aria-hidden="true" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}
export function Loading({ text }: { text?: string }) { const { t } = useI18n(); return <div role="status" className="flex items-center gap-3 py-5 text-sm text-brand"><span className="spinner" aria-hidden="true"/>{text || t('loading')}</div> }
export function ErrorNotice({ children }: { children: ReactNode }) { return <div role="alert" className="error-notice">{children}</div> }
export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'teal' | 'amber' | 'red' }) { return <span className={`badge badge-${tone}`}>{children}</span> }
