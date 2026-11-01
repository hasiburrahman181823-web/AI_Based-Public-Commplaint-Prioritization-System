import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from '../api'
import type { Analytics, Complaint, Group } from '../api'
import { I18nProvider } from '../i18n'
import AdminDashboard from '../pages/AdminDashboard'

vi.mock('../auth', () => ({ useAuth: () => ({ token: 'synthetic-admin-token', logout: vi.fn() }) }))

const ID = 'e736efb8-5ad0-4640-9bca-627d9398bc05'
const GROUP = '3addb6f4-e1c4-48a4-bcd6-fe564a7c40ab'
const complaint: Complaint = {
  id: ID, text: 'No water for three days', lang: 'en', category: 'Water', severity: 'High',
  priority_score: 9, priority_label: 'high', status: 'in_review', area_id: 'dhaka_mirpur_10',
  group_id: GROUP, version: 7, group_version: 3, location: 'Mirpur 10', confidence: 0.91,
  contact: 'private-test-contact@example.org', review_required: true,
  created_at: '2026-10-30T08:00:00Z', updated_at: '2026-10-30T08:00:00Z',
  analysis: { priority_score: 9, explanation: { rule_breakdown: [{ name: 'duration', contribution: 2, evidence: [{ hours: 72 }] }] }, review_reasons: ['low_confidence'] },
}
const analytics: Analytics = {
  total_complaints: 1, total_groups: 1, needs_human_review: 1, by_category: { Water: 1 },
  by_priority: { high: 1 }, by_status: { in_review: 1 }, by_area: { dhaka_mirpur_10: 1 }, by_language: { en: 1 },
  priority_over_time: [{ date: '2026-10-30', priority: 'high', count: 1 }],
  review_workload: [{ status: 'in_review', review_required: true, count: 1 }],
}
const group: Group = { id: GROUP, version: 3, size: 3, status: 'proposed', created_at: complaint.created_at, updated_at: complaint.updated_at }

function mount() { return render(<I18nProvider><AdminDashboard /></I18nProvider>) }

beforeEach(() => {
  vi.spyOn(api, 'analytics').mockResolvedValue(analytics)
  vi.spyOn(api, 'listComplaints').mockResolvedValue({ items: [complaint], total: 1, page: 1, page_size: 15 })
  vi.spyOn(api, 'getComplaint').mockResolvedValue(complaint)
  vi.spyOn(api, 'listGroups').mockResolvedValue({ items: [group], total: 1, page: 1, page_size: 10 })
  vi.spyOn(api, 'getGroup').mockResolvedValue(group)
})

describe('administrator review decisions', () => {
  it('renders stored complaint, location and contact markup as literal text without creating executable elements', async () => {
    const text = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    const location = '<svg onload="alert(3)">Mirpur 10</svg>'
    const contact = '<a href="javascript:alert(4)">private@example.org</a>'
    const stored = { ...complaint, text, location, contact }
    vi.mocked(api.listComplaints).mockResolvedValue({ items: [stored], total: 1, page: 1, page_size: 15 })
    vi.mocked(api.getComplaint).mockResolvedValue(stored)
    const { container } = mount()
    fireEvent.click((await screen.findAllByRole('button', { name: `Open complaint ${ID}` }))[0])
    const dialog = await screen.findByRole('dialog', { name: 'Complaint review' })
    expect(await within(dialog).findByText(text)).toBeVisible()
    expect(within(dialog).getByText(location)).toBeVisible()
    expect(within(dialog).getByText(contact)).toBeVisible()
    expect(container.querySelector('img, script, [onerror], [onload], a[href^="javascript:"]')).toBeNull()
  })

  it('requests the persisted review flag for the Needs review tab', async () => {
    mount()
    await screen.findAllByRole('button', { name: `Open complaint ${ID}` })
    fireEvent.click(screen.getByRole('tab', { name: /Needs review/ }))
    await waitFor(() => expect(api.listComplaints).toHaveBeenCalledWith(expect.objectContaining({ review_required: true }), 'synthetic-admin-token', expect.any(AbortSignal)))
  })

  it('saves a reason and expected version with a priority override', async () => {
    const patch = vi.spyOn(api, 'patchComplaint').mockResolvedValue({ ...complaint, priority_score: 15, priority_label: 'urgent', version: 8 })
    mount()
    fireEvent.click((await screen.findAllByRole('button', { name: `Open complaint ${ID}` }))[0])
    const dialog = await screen.findByRole('dialog', { name: 'Complaint review' })
    expect(await within(dialog).findByText('Why this priority?')).toBeVisible()
    expect(within(dialog).getByText(complaint.contact!)).toBeVisible()
    fireEvent.change(within(dialog).getByLabelText('Priority score'), { target: { value: '15' } })
    fireEvent.change(within(dialog).getByLabelText('Priority label'), { target: { value: 'urgent' } })
    fireEvent.change(within(dialog).getByLabelText('Reason for review'), { target: { value: 'Expert reviewed essential-service impact' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save review' }))
    await waitFor(() => expect(patch).toHaveBeenCalledWith(ID, expect.objectContaining({ expected_version: 7, reason: 'Expert reviewed essential-service impact', priority_score: 15, priority_label: 'urgent' }), 'synthetic-admin-token'))
    expect(patch).toHaveBeenCalledTimes(1)
  })

  it('retains the decision after a stale-version response and asks for a refresh', async () => {
    const patch = vi.spyOn(api, 'patchComplaint').mockRejectedValue(new ApiError(409, 'Stale version'))
    mount()
    fireEvent.click((await screen.findAllByRole('button', { name: `Open complaint ${ID}` }))[0])
    const dialog = await screen.findByRole('dialog', { name: 'Complaint review' })
    const reason = await within(dialog).findByLabelText('Reason for review')
    fireEvent.change(reason, { target: { value: 'Review evidence before committing' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save review' }))
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('Refresh and check the latest version')
    expect(reason).toHaveValue('Review evidence before committing')
    expect(patch).toHaveBeenCalledTimes(1)
    expect(within(dialog).queryByText('Review saved. The queue has been updated.')).not.toBeInTheDocument()
  })

  it('compares actual language versions and splits only selected reports with a reviewed reason', async () => {
    const members: Complaint[] = [
      { ...complaint, id: 'first-en', lang: 'en', text: 'No water for three days' },
      { ...complaint, id: 'second-bn', lang: 'bn', text: 'তিন দিন ধরে পানি নেই' },
      { ...complaint, id: 'third-banglish', lang: 'banglish', text: 'pani nai 3 din dhore' },
    ]
    vi.mocked(api.listComplaints).mockImplementation(async params => ({ items: params.group_id ? members : [complaint], total: params.group_id ? 3 : 1, page: 1, page_size: 30 }))
    const split = vi.spyOn(api, 'splitGroup').mockResolvedValue({ original_group: { ...group, size: 2, version: 4, status: 'reviewed_split' }, new_group: { ...group, id: 'new-group', size: 1, version: 1, status: 'reviewed_split' } })
    mount()
    await screen.findAllByRole('button', { name: `Open complaint ${ID}` })
    fireEvent.click(screen.getByRole('tab', { name: 'Duplicate groups' }))
    const groupButton = await screen.findByRole('button', { name: /3 reports/ })
    fireEvent.click(groupButton)
    const checkbox = await screen.findByRole('checkbox', { name: 'Select report second-bn' })
    expect(screen.getByText('তিন দিন ধরে পানি নেই')).toBeVisible()
    expect(screen.getByText('pani nai 3 din dhore')).toBeVisible()
    fireEvent.click(checkbox)
    fireEvent.change(screen.getByLabelText('Reason for group decision'), { target: { value: 'Reviewed evidence identifies a distinct incident' } })
    fireEvent.click(screen.getByRole('button', { name: /Split selected \(1\)/ }))
    await waitFor(() => expect(split).toHaveBeenCalledWith(GROUP, { expected_version: 3, reason: 'Reviewed evidence identifies a distinct incident', complaint_ids: ['second-bn'] }, 'synthetic-admin-token'))
    expect(split).toHaveBeenCalledTimes(1)
  })
})
