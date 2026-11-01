import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { api, ApiError } from '../api'
import type { Complaint, Tracking } from '../api'
import { I18nProvider } from '../i18n'
import PublicPage from '../pages/PublicPage'
import TrackingPage from '../pages/TrackingPage'

const ID = 'e736efb8-5ad0-4640-9bca-627d9398bc05'
const receipt: Complaint = {
  id: ID, lang: 'banglish', category: null, severity: null, priority_score: 2,
  priority_label: 'low', status: 'in_review', area_id: 'dhaka_mirpur_10', group_id: null,
  version: 1, group_version: null, created_at: '2026-10-30T08:00:00Z', updated_at: '2026-10-30T08:00:00Z',
}
const tracking: Tracking = { id: ID, lang: 'en', category: 'Water', severity: 'High', priority_score: 9,
  priority_label: 'high', status: 'in_progress', review_required: false, area_id: 'dhaka_mirpur_10',
  created_at: receipt.created_at, updated_at: receipt.updated_at }

function publicPage() { return render(<MemoryRouter><I18nProvider><PublicPage /></I18nProvider></MemoryRouter>) }
function ResetTrackingRoute() { const navigate = useNavigate(); return <button onClick={() => navigate('/track')}>Look up another ID</button> }
function trackingPage(path: string) {
  return render(<MemoryRouter initialEntries={[path]}><I18nProvider><ResetTrackingRoute /><Routes><Route path="/track/:id?" element={<TrackingPage />} /></Routes></I18nProvider></MemoryRouter>)
}

describe('public complaint submission', () => {
  it('shows the demo disclaimer, limits complaints to 1,000 characters and leaves contact optional', async () => {
    const submit = vi.spyOn(api, 'submit').mockResolvedValue(receipt)
    publicPage()
    expect(screen.getByText('This is an AI decision-support demo, not an official government system.')).toBeVisible()
    const text = screen.getByLabelText(/Your complaint/)
    expect(text).toHaveAttribute('maxlength', '1000')
    expect(screen.getByLabelText(/Phone or email/)).not.toBeRequired()
    fireEvent.change(text, { target: { value: 'Water unavailable. '.repeat(60).slice(0, 1001) } })
    fireEvent.click(screen.getByRole('button', { name: 'Send complaint' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Describe your complaint in 1–1,000 characters.')
    expect(submit).not.toHaveBeenCalled()
    fireEvent.change(text, { target: { value: 'No water for three days' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send complaint' }))
    await screen.findByRole('heading', { name: 'Your complaint was received' })
    expect(submit).toHaveBeenCalledWith({ text: 'No water for three days' })
  })

  it('detects the language locally and waits for a saved response before showing the receipt', async () => {
    let complete!: (value: Complaint) => void
    const submit = vi.spyOn(api, 'submit').mockReturnValue(new Promise(resolve => { complete = resolve }))
    publicPage()
    fireEvent.change(screen.getByLabelText(/Your complaint/), { target: { value: '  pani nai 3 din dhore  ' } })
    expect(screen.getByText('Detected: Banglish')).toBeVisible()
    expect(submit).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText(/Area or landmark/), { target: { value: '  Mirpur 10  ' } })
    fireEvent.change(screen.getByLabelText(/Phone or email/), { target: { value: '  volunteer@example.org  ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send complaint' }))
    expect(screen.getByRole('button', { name: 'Sending your complaint…' })).toBeDisabled()
    expect(screen.queryByRole('heading', { name: 'Your complaint was received' })).not.toBeInTheDocument()
    expect(submit).toHaveBeenCalledWith({ text: 'pani nai 3 din dhore', location: 'Mirpur 10', contact: 'volunteer@example.org' })
    await act(async () => { complete(receipt) })
    expect(await screen.findByRole('heading', { name: 'Your complaint was received' })).toHaveFocus()
    expect(screen.getByText(ID)).toBeVisible()
    expect(screen.getByRole('link', { name: 'View progress' })).toHaveAttribute('href', `/track/${ID}`)
    expect(localStorage.getItem('nagarik:locale')).toBe('en')
    expect(Object.keys(localStorage)).toEqual(['nagarik:locale'])
  })

  it('preserves the draft and offers an error rather than a false success after a failed write', async () => {
    const submit = vi.spyOn(api, 'submit').mockRejectedValue(new ApiError(0, 'Unable to reach the service', 'network'))
    publicPage()
    fireEvent.change(screen.getByLabelText(/Your complaint/), { target: { value: 'No water for three days' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send complaint' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to reach the service')
    expect(screen.getByLabelText(/Your complaint/)).toHaveValue('No water for three days')
    expect(screen.getByRole('button', { name: 'Send complaint' })).toBeEnabled()
    expect(screen.queryByRole('heading', { name: 'Your complaint was received' })).not.toBeInTheDocument()
    expect(submit).toHaveBeenCalledTimes(1)
  })

  it('allows a human correction to the estimated complaint language', async () => {
    const submit = vi.spyOn(api, 'submit').mockResolvedValue({ ...receipt, lang: 'en' })
    publicPage()
    fireEvent.change(screen.getByLabelText(/Your complaint/), { target: { value: 'pani nai' } })
    fireEvent.change(screen.getByLabelText('Complaint language'), { target: { value: 'en' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send complaint' }))
    await screen.findByRole('heading', { name: 'Your complaint was received' })
    expect(submit).toHaveBeenCalledWith({ text: 'pani nai', lang: 'en' })
  })
})

describe('public tracking', () => {
  it('renders current progress but never exposes incidental private fields from a response', async () => {
    vi.spyOn(api, 'track').mockResolvedValue({ ...tracking, text: 'private complaint wording', contact: 'private@example.org' } as Tracking)
    trackingPage('/track/' + ID)
    expect(await screen.findByRole('heading', { name: 'In progress' })).toBeVisible()
    expect(screen.getByText('Water')).toBeVisible()
    expect(screen.queryByText('private complaint wording')).not.toBeInTheDocument()
    expect(screen.queryByText('private@example.org')).not.toBeInTheDocument()
    expect(api.track).toHaveBeenCalledWith(ID, expect.any(AbortSignal))
  })

  it('rejects a malformed ID before making a tracking request', async () => {
    const track = vi.spyOn(api, 'track')
    trackingPage('/track/not-an-id')
    expect(await screen.findByRole('alert')).toHaveTextContent('Enter a valid tracking ID')
    expect(track).not.toHaveBeenCalled()
  })

  it('displays a helpful not-found state without inventing complaint progress', async () => {
    vi.spyOn(api, 'track').mockRejectedValue(new ApiError(404, 'Record not found'))
    trackingPage('/track/' + ID)
    expect(await screen.findByRole('alert')).toHaveTextContent('No complaint was found with this ID')
    expect(screen.queryByRole('heading', { name: 'In progress' })).not.toBeInTheDocument()
  })

  it('releases the loading state when a pending tracking route is cleared', async () => {
    vi.spyOn(api, 'track').mockReturnValue(new Promise<Tracking>(() => {}))
    trackingPage('/track/' + ID)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find my complaint' })).toBeDisabled())
    fireEvent.click(screen.getByRole('button', { name: 'Look up another ID' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find my complaint' })).toBeEnabled())
    expect(screen.getByLabelText('Tracking ID')).toHaveValue('')
  })
})
