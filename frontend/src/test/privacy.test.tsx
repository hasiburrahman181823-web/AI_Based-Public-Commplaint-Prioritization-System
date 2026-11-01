import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { I18nProvider, useI18n } from '../i18n'
import PrivacyPage from '../pages/PrivacyPage'
import { privacyContactUrl } from '../privacy'

describe('privacy operator links', () => {
  it('allows explicit HTTPS and email contact without credential or script URLs', () => {
    expect(privacyContactUrl('https://example.org/privacy')).toBe('https://example.org/privacy')
    expect(privacyContactUrl('mailto:privacy@example.org')).toBe('mailto:privacy@example.org')
    for (const value of [undefined, '', 'javascript:alert(1)', 'http://example.org', '//example.org', 'https://user:password@example.org', 'mailto:privacy@example.org?bcc=other@example.org', 'mailto:user%0aBcc:other@example.org']) expect(privacyContactUrl(value)).toBeNull()
  })
})

function Language() { const { setLocale } = useI18n(); return <button onClick={() => setLocale('bn')}>Switch Bangla</button> }
describe('privacy notice', () => {
  it('makes demo state and manual retention/deletion limits explicit in both languages', () => {
    render(<MemoryRouter><I18nProvider><Language/><PrivacyPage/></I18nProvider></MemoryRouter>)
    expect(screen.getByText(/Demo notice:/)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Retention policy' })).toBeVisible()
    expect(screen.getByText(/90 days after their latest update/)).toBeVisible()
    expect(screen.getByText(/preview and apply the deletion tool/)).toBeVisible()
    expect(screen.getByText(/not an automatic deletion feature/)).toBeVisible()
    expect(screen.getByText(/Live complaints are not automatically approved research data/)).toBeVisible()
    expect(screen.queryByRole('link', { name: 'Contact the privacy operator' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Switch Bangla'))
    expect(screen.getByRole('heading', { name: 'কতদিন তথ্য রাখা হয়' })).toBeVisible()
    expect(screen.getByText(/সর্বশেষ হালনাগাদের ৯০ দিন/)).toBeVisible()
    expect(screen.getByText(/স্বয়ংক্রিয় তথ্য মোছার সুবিধা নয়/)).toBeVisible()
  })
})
