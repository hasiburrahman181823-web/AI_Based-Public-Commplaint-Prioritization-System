import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { I18nProvider } from '../i18n'
import ExplanationPanel from '../components/ExplanationPanel'
import Charts from '../components/Charts'
import type { Analytics } from '../api'

describe('readable saved analysis', () => {
  it('localizes actual rule evidence and review codes while preserving the reported duration and phrases', () => {
    localStorage.setItem('nagarik:locale', 'bn')
    render(<I18nProvider><ExplanationPanel currentScore={2} analysis={{
      priority_score: 2,
      review_reasons: ['classification_models_unavailable:category,severity', 'validated_review_policy_unavailable', 'priority_label_mapping_unknown', 'custom_review_signal'],
      explanation: {
        rule_breakdown: [
          { name: 'severity', contribution: 0, evidence: [{ value: null, known: false }] },
          { name: 'duration', contribution: 2, evidence: [{ duration_hours: 72, duration_days: 3, band: 'duration > 48h', matched_expressions: ['3 din dhore'] }] },
          { name: 'duplicate_reports', contribution: 0, evidence: [{ report_count: 1, band: null, interpretation: 'report volume, not unique people affected' }] },
        ],
        notes: ['Unknown or missing category; its rule contribution is zero and needs label review.'],
      },
    }} /></I18nProvider>)
    expect(screen.getByText(/স্বয়ংক্রিয় শ্রেণিবিভাগ পাওয়া যাচ্ছে না \(বিভাগ, তীব্রতা\)/)).toBeVisible()
    expect(screen.getByText(/যাচাইকৃত আস্থার সীমা পাওয়া যায়নি/)).toBeVisible()
    expect(screen.getByText('লেবেল পাওয়া যায়নি বা অগ্রাধিকারের নিয়মের সাথে মেলেনি')).toBeVisible()
    const duration = screen.getByText(/ঘণ্টা: ৭২/)
    expect(duration).toHaveTextContent('দিন: ৩')
    expect(duration).toHaveTextContent('স্থায়িত্ব > ৪৮ ঘণ্টা')
    expect(duration).toHaveTextContent('3 din dhore')
    expect(screen.getByText(/অভিযোগের সংখ্যা: ১/)).toHaveTextContent('স্বতন্ত্র ক্ষতিগ্রস্ত মানুষের সংখ্যা নয়')
    expect(screen.getByText('অন্য সংরক্ষিত কারণ: custom review signal')).toBeVisible()
    expect(screen.queryByText(/known: false|classification models unavailable|duration hours:/)).not.toBeInTheDocument()
  })

  it('uses the real MiniLM token-deletion schema without presenting embedding dimensions as words', () => {
    render(<I18nProvider><ExplanationPanel currentScore={5} analysis={{ explanation: {
      category: {
        method: 'linear_base_margin', feature_kind: 'embedding_dimension',
        top_terms: [{ term: 'pani', probability_drop_on_deletion: 0.125 }],
        top_features: [{ term: 'embedding_dimension_17', contribution: 2.5 }],
      },
    } }} /></I18nProvider>)
    expect(screen.getByText('pani')).toBeVisible()
    expect(screen.getByText('+0.13')).toBeVisible()
    expect(screen.getByText(/Approximate token-deletion sensitivity/)).toBeVisible()
    expect(screen.queryByText('embedding_dimension_17')).not.toBeInTheDocument()
  })

  it('states when no word-level evidence exists instead of inventing terms or contributions', () => {
    render(<I18nProvider><ExplanationPanel currentScore={0} analysis={{ explanation: {
      category: { feature_kind: 'embedding_dimension', top_terms: [], top_features: [{ term: 'embedding_dimension_2', contribution: 0.8 }] },
      severity: { top_terms: [{ term: 'invalid-with-no-weight' }, { contribution: 1 }] },
    } }} /></I18nProvider>)
    expect(screen.getAllByText('No readable word-level evidence was saved for this prediction.')).toHaveLength(2)
    expect(screen.queryByText('embedding_dimension_2')).not.toBeInTheDocument()
    expect(screen.queryByText('invalid-with-no-weight')).not.toBeInTheDocument()
    expect(screen.queryByText('+0')).not.toBeInTheDocument()
  })
})

describe('configured priority labels', () => {
  it('renders the actual moderate label in Bangla and keeps custom medium labels supported', () => {
    localStorage.setItem('nagarik:locale', 'bn')
    const data: Analytics = {
      total_complaints: 2, total_groups: 2, needs_human_review: 0,
      by_category: {}, by_area: {}, by_priority: { moderate: 1, medium: 1 }, by_status: {}, by_language: {},
      priority_over_time: [{ date: '2026-10-31', priority: 'moderate', count: 1 }, { date: '2026-10-31', priority: 'medium', count: 1 }],
      review_workload: [],
    }
    render(<I18nProvider><Charts data={data} /></I18nProvider>)
    expect(screen.getAllByText('মাঝারি').length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByText('moderate')).not.toBeInTheDocument()
    expect(screen.queryByText('medium')).not.toBeInTheDocument()
  })
})
