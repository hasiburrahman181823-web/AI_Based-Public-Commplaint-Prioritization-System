/** Public operator details are build-time settings, never credentials. */
export function privacyContactUrl(value: string | undefined): string | null {
  if (!value) return null
  const contact = value.trim()
  if (contact.length > 512 || /[\u0000-\u0020\u007f]/.test(contact)) return null
  try {
    if (/[\r\n]/.test(decodeURIComponent(contact))) return null
    if (contact.startsWith('mailto:')) {
      return /^[A-Za-z0-9.!#$&'*+/=^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/.test(contact.slice(7)) ? contact : null
    }
    const url = new URL(contact)
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null
  } catch { return null }
}

const configuredName = (import.meta.env.VITE_PRIVACY_OPERATOR || '').trim()
export const privacyOperator = configuredName.length <= 128 && !/[\u0000-\u001f\u007f]/.test(configuredName) ? configuredName : ''
export const privacyContact = privacyContactUrl(import.meta.env.VITE_PRIVACY_CONTACT_URL)
export const operatorPublished = Boolean(privacyOperator && privacyContact)
export const PRIVACY_VERSION = '2026-10-31'
