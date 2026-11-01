import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { AuthProvider, useAuth } from '../auth'
import { I18nProvider, useI18n } from '../i18n'
import { detectLanguage } from '../language'

function LocaleProbe() {
  const { locale, setLocale, t } = useI18n()
  return <><p>{t('submit')}</p><button onClick={() => setLocale(locale === 'en' ? 'bn' : 'en')}>Switch language</button></>
}

function SessionProbe() {
  const { token, login, logout } = useAuth()
  return <><p>{token ? 'Signed in' : 'Signed out'}</p>
    <button onClick={() => void login('admin', 'synthetic-password')}>Log in</button>
    <button onClick={logout}>Log out</button></>
}

describe('language and local preference', () => {
  it('detects representative Bangla, Banglish and English locally', () => {
    expect(detectLanguage('মিরপুর ১০-এ পানি নেই')).toBe('bn')
    expect(detectLanguage('pani nai 3 din dhore')).toBe('banglish')
    expect(detectLanguage('There has been no water for three days')).toBe('en')
    expect(detectLanguage('')).toBeNull()
    expect(detectLanguage('https://example.org/pani/nai')).toBe('en')
  })

  it('changes visible language and the document language, remembering only the locale', async () => {
    const first = render(<I18nProvider><LocaleProbe /></I18nProvider>)
    expect(screen.getByText('Send complaint')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Switch language' }))
    expect(screen.getByText('অভিযোগ পাঠান')).toBeVisible()
    await waitFor(() => expect(document.documentElement.lang).toBe('bn'))
    expect(localStorage.getItem('nagarik:locale')).toBe('bn')
    first.unmount()
    render(<I18nProvider><LocaleProbe /></I18nProvider>)
    expect(screen.getByText('অভিযোগ পাঠান')).toBeVisible()
    expect(Object.keys(localStorage)).toEqual(['nagarik:locale'])
  })
})

describe('administrator memory-only sessions', () => {
  it('keeps credentials out of storage and signs out after a reload/remount', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ access_token: 'synthetic-secret-token', expires_in: 1800, token_type: 'bearer' }), { status: 200 })))
    const first = render(<AuthProvider><SessionProbe /></AuthProvider>)
    fireEvent.click(screen.getByRole('button', { name: 'Log in' }))
    expect(await screen.findByText('Signed in')).toBeVisible()
    expect(localStorage.length).toBe(0)
    expect(sessionStorage.length).toBe(0)
    first.unmount()
    render(<AuthProvider><SessionProbe /></AuthProvider>)
    expect(screen.getByText('Signed out')).toBeVisible()
  })

  it('clears the current session on an unauthorized-event signal', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ access_token: 'synthetic-secret-token', expires_in: 1800, token_type: 'bearer' }), { status: 200 })))
    render(<AuthProvider><SessionProbe /></AuthProvider>)
    fireEvent.click(screen.getByRole('button', { name: 'Log in' }))
    expect(await screen.findByText('Signed in')).toBeVisible()
    fireEvent(window, new Event('nagarik:unauthorized'))
    expect(screen.getByText('Signed out')).toBeVisible()
  })
})
