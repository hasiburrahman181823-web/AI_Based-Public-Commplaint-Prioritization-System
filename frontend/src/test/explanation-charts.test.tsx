import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { I18nProvider } from '../i18n'
import ExplanationPanel from '../components/ExplanationPanel'
import Charts from '../components/Charts'
import type { Analytics } from '../api'

describe('decision evidence', () => {
  it('shows a label-only human override without attributing it to the original model', () => {
    render(<I18nProvider><ExplanationPanel currentScore={9} currentLabel="urgent" analysis={{
      priority_score: 9, priority_label: 'high', explanation: {
        rule_breakdown: [{ name: 'duration', contribution: 2, evidence: [{ duration_hours: 72 }] }],
        category: { top_terms: [{ term: 'water', contribution: 0.7 }] },
      },
    }} /></I18nProvider>)
    expect(screen.getByText(/An administrator changed this priority/)).toBeVisible()
    expect(screen.getByText('Duration')).toBeVisible()
    expect(screen.getByText('+2')).toBeVisible()
    expect(screen.getByText('water')).toBeVisible()
    expect(screen.getByText(/not calibrated probability points or causal effects/)).toBeVisible()
  })

  it('does not fabricate contributions when no automatic explanation is available', () => {
    render(<I18nProvider><ExplanationPanel currentScore={0} analysis={null} /></I18nProvider>)
    expect(screen.getByText('No automatic rule breakdown was saved for this report.')).toBeVisible()
    expect(screen.queryByText('Category evidence')).not.toBeInTheDocument()
  })
})

describe('accessible operational charts', () => {
  it('exposes the actual time-series values as an accessible table with current-state semantics', () => {
    const analytics: Analytics = {
      total_complaints: 3, total_groups: 2, needs_human_review: 2, by_category: { Water: 3 },
      by_area: { dhaka_mirpur_10: 3 }, by_priority: { high: 2, low: 1 }, by_status: { in_review: 2, resolved: 1 }, by_language: { bn: 1, en: 1, banglish: 1 },
      priority_over_time: [{ date: '2026-10-30', priority: 'high', count: 2 }, { date: '2026-10-30', priority: 'low', count: 1 }],
      review_workload: [{ status: 'in_review', review_required: true, count: 2 }, { status: 'resolved', review_required: false, count: 1 }],
    }
    render(<I18nProvider><Charts data={analytics} /></I18nProvider>)
    expect(screen.getByText(/do not reconstruct past priority changes/)).toBeVisible()
    expect(screen.getByRole('img', { name: /Complaint counts by priority and submission date/ })).toBeVisible()
    fireEvent.click(screen.getByText('View chart data'))
    const table = screen.getByRole('table', { name: 'Priority counts by UTC submission date' })
    const row = within(table).getByRole('row', { name: '2026-10-30 2 1' })
    expect(within(row).getAllByRole('cell').map(cell => cell.textContent)).toEqual(['2', '1'])
    expect(screen.getByRole('table', { name: 'Review workload by status' })).toBeVisible()
  })
})
