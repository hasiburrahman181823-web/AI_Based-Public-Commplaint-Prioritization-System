import { useI18n } from '../i18n'

type JsonObject = Record<string, unknown>
const object = (value: unknown): JsonObject => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {}
const rows = (value: unknown): JsonObject[] => Array.isArray(value) ? value.map(object) : []
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []

export default function ExplanationPanel({ analysis, currentScore, currentLabel }: { analysis?: JsonObject | null; currentScore: number; currentLabel?: string }) {
  const { locale } = useI18n()
  const tr = (en: string, bn: string) => locale === 'bn' ? bn : en
  const explanation = object(analysis?.explanation)
  const breakdown = rows(explanation.rule_breakdown)
  const names: Record<string, string> = {
    severity: tr('Severity', 'তীব্রতা'), category: tr('Essential service', 'অপরিহার্য সেবা'),
    duration: tr('Duration', 'স্থায়িত্ব'), duplicate_reports: tr('Related reports', 'সম্পর্কিত অভিযোগ'),
    vulnerable_group: tr('Vulnerable people', 'ঝুঁকিতে থাকা মানুষ'), vulnerable: tr('Vulnerable people', 'ঝুঁকিতে থাকা মানুষ'),
    life_risk: tr('Life-risk evidence', 'জীবনের ঝুঁকির প্রমাণ'), score_cap: tr('Configured score limit', 'নির্ধারিত স্কোরসীমা'),
  }
  const number = (value: number) => new Intl.NumberFormat(locale === 'bn' ? 'bn-BD' : 'en-GB', { maximumFractionDigits: 2 }).format(value)
  const labels: Record<string, string> = {
    category: tr('category', 'বিভাগ'), severity: tr('severity', 'তীব্রতা'),
    Low: tr('Low', 'নিম্ন'), Medium: tr('Medium', 'মাঝারি'), High: tr('High', 'উচ্চ'),
    Water: tr('Water', 'পানি'), Electricity: tr('Electricity', 'বিদ্যুৎ'), Healthcare: tr('Healthcare', 'স্বাস্থ্যসেবা'),
    Food: tr('Food', 'খাদ্য'), Waste: tr('Waste', 'বর্জ্য'), 'Road & Transport': tr('Road & Transport', 'রাস্তা ও পরিবহন'),
    hospital: tr('Hospital', 'হাসপাতাল'), school: tr('School', 'বিদ্যালয়'), child: tr('Children', 'শিশু'), elderly: tr('Older people', 'বয়স্ক মানুষ'),
    fire: tr('Fire', 'আগুন'), electric_shock: tr('Electric shock', 'বিদ্যুৎস্পৃষ্ট হওয়ার ঝুঁকি'), flood: tr('Flood', 'বন্যা'), gas_leak: tr('Gas leak', 'গ্যাস লিক'),
  }
  const readable = (value: string) => value.replaceAll('_', ' ')
  const bandText = (value: string): string => {
    const duration = /^duration\s*(>=|>|<=|<)\s*(\d+(?:\.\d+)?)h$/i.exec(value)
    if (duration) return `${tr('Duration', 'স্থায়িত্ব')} ${duration[1]} ${number(Number(duration[2]))} ${tr('hours', 'ঘণ্টা')}`
    const reports = /^duplicate report group\s*(>=|>|<=|<)\s*(\d+)$/i.exec(value)
    if (reports) return `${tr('Related reports', 'সম্পর্কিত অভিযোগ')} ${reports[1]} ${number(Number(reports[2]))}`
    return readable(value)
  }
  const reasons: Record<string, string> = {
    validated_review_policy_unavailable: tr('A validated confidence threshold is not available; a person needs to review the prediction.', 'যাচাইকৃত আস্থার সীমা পাওয়া যায়নি; একজন কর্মকর্তার পূর্বাভাস পর্যালোচনা করা প্রয়োজন।'),
    validation_policy_reject_all: tr('The validated review policy requires a person to check every prediction.', 'যাচাইকৃত পর্যালোচনা নীতি অনুযায়ী প্রতিটি পূর্বাভাস একজন কর্মকর্তার যাচাই করা প্রয়োজন।'),
    low_classification_confidence: tr('Classification confidence is below the approved review threshold.', 'শ্রেণিবিভাগের আস্থা অনুমোদিত সীমার নিচে।'),
    low_confidence: tr('Prediction confidence needs human review.', 'পূর্বাভাসের আস্থা একজন কর্মকর্তার যাচাই করা প্রয়োজন।'),
    language_inference_ambiguous: tr('The complaint language could not be identified confidently.', 'অভিযোগের ভাষা নিশ্চিতভাবে শনাক্ত করা যায়নি।'),
    location_evidence_conflicting_or_ambiguous: tr('The area information conflicts or is ambiguous; confirm the incident location.', 'এলাকার তথ্য পরস্পরবিরোধী বা অস্পষ্ট; ঘটনার স্থান নিশ্চিত করুন।'),
    location_evidence_conflict: tr('The supplied area and the location evidence conflict.', 'দেওয়া এলাকা ও লেখার অবস্থানসংক্রান্ত তথ্য পরস্পরবিরোধী।'),
    location_evidence_ambiguous: tr('More than one incident area is possible.', 'ঘটনার একাধিক এলাকা হতে পারে।'),
    canonical_area_unavailable: tr('The incident area could not be matched reliably.', 'ঘটনার এলাকা নির্ভরযোগ্যভাবে মেলানো যায়নি।'),
    location_granularity_differs: tr('The reports name areas at different levels of detail.', 'অভিযোগগুলোতে এলাকার নাম ভিন্ন মাত্রার বিস্তারিতভাবে দেওয়া হয়েছে।'),
    different_canonical_areas: tr('The reports refer to different known areas.', 'অভিযোগগুলো আলাদা পরিচিত এলাকার।'),
    supported_ongoing_intervals_disjoint: tr('The supported incident time intervals do not overlap.', 'প্রমাণসমর্থিত ঘটনার সময়সীমাগুলো মেলে না।'),
    duplicate_report_lineage_incomplete: tr('It is unclear whether related versions come from distinct original reports.', 'সম্পর্কিত সংস্করণগুলো আলাদা মূল অভিযোগ থেকে এসেছে কি না তা স্পষ্ট নয়।'),
    priority_label_mapping_unknown: tr('Category or severity is missing, or has no matching priority rule.', 'বিভাগ বা তীব্রতা পাওয়া যায়নি, অথবা এর সাথে অগ্রাধিকারের কোনো নিয়ম মেলেনি।'),
    life_risk_keyword_requires_review: tr('Potential life-risk language needs human verification.', 'জীবনের সম্ভাব্য ঝুঁকিসংক্রান্ত ভাষা একজন কর্মকর্তার যাচাই করা প্রয়োজন।'),
    life_risk_keyword_context_conflict: tr('Risk-related wording and a nearby negation conflict; check the actual danger.', 'ঝুঁকিসংক্রান্ত শব্দ ও কাছাকাছি অস্বীকারসূচক বক্তব্য পরস্পরবিরোধী; প্রকৃত বিপদ যাচাই করুন।'),
    duplicate_auto_group_guard_requires_review: tr('Some possible duplicates have incomplete or conflicting area or time evidence.', 'সম্ভাব্য সদৃশ অভিযোগের কিছুতে এলাকা বা সময়ের প্রমাণ অসম্পূর্ণ বা পরস্পরবিরোধী।'),
    duplicate_group_ambiguity_or_admin_partition: tr('Related reports span different groups or a previously reviewed split. Check before merging.', 'সম্পর্কিত অভিযোগগুলো ভিন্ন দলে আছে অথবা আগে পর্যালোচনা করে আলাদা করা হয়েছে। একত্র করার আগে যাচাই করুন।'),
    below_duplicate_threshold: tr('The similarity score is below the approved duplicate threshold.', 'সাদৃশ্যের স্কোর সদৃশ অভিযোগ শনাক্তের অনুমোদিত সীমার নিচে।'),
    candidate_not_in_current_store: tr('A related report is no longer available in the current complaint list.', 'একটি সম্পর্কিত অভিযোগ বর্তমান তালিকায় আর পাওয়া যাচ্ছে না।'),
  }
  const reviewReason = (value: string) => {
    if (value.startsWith('classification_models_unavailable:')) {
      const targets = value.slice('classification_models_unavailable:'.length).split(',').map(target => labels[target.trim()] || readable(target.trim())).filter(Boolean)
      return `${tr('Automatic classification is unavailable', 'স্বয়ংক্রিয় শ্রেণিবিভাগ পাওয়া যাচ্ছে না')}${targets.length ? ` (${targets.join(', ')})` : ''}. ${tr('Please review these labels manually.', 'অনুগ্রহ করে এই লেবেলগুলো নিজে যাচাই করুন।')}`
    }
    return reasons[value] || `${tr('Other saved reason', 'অন্য সংরক্ষিত কারণ')}: ${readable(value)}`
  }
  const warningText = (value: string) => {
    const missing = /^Unknown or missing (category|severity); its rule contribution is zero and needs label review\.$/.exec(value)
    if (missing) return `${labels[missing[1]]}: ${tr('the label is unavailable or not mapped to a rule, so it adds no points and needs review.', 'লেবেল পাওয়া যায়নি বা নিয়মের সাথে মেলেনি; তাই কোনো পয়েন্ট যোগ হয়নি এবং পর্যালোচনা প্রয়োজন।')}`
    const warnings: Record<string, string> = {
      'Duration unavailable or ambiguous; no duration boost was imputed.': tr('Duration is missing or unclear. No duration points were assumed.', 'স্থায়িত্বের তথ্য পাওয়া যায়নি বা অস্পষ্ট। ধরে নিয়ে কোনো স্থায়িত্বের পয়েন্ট যোগ করা হয়নি।'),
      'A duration range is ambiguous; no point estimate was imputed.': tr('The text gives a range of durations; no single duration was assumed.', 'লেখায় স্থায়িত্বের একটি পরিসর দেওয়া হয়েছে; একটিমাত্র স্থায়িত্ব ধরে নেওয়া হয়নি।'),
      'Unsupported compound number words in duration.': tr('The written duration number could not be interpreted reliably.', 'কথায় লেখা স্থায়িত্বের সংখ্যাটি নির্ভরযোগ্যভাবে বোঝা যায়নি।'),
      'Duration is outside the supported finite numeric range.': tr('The stated duration is outside the supported numeric range.', 'দেওয়া স্থায়িত্ব সমর্থিত সংখ্যাসীমার বাইরে।'),
      'Multiple distinct durations found; no duration was selected.': tr('Several different durations were found; none was chosen automatically.', 'একাধিক ভিন্ন স্থায়িত্ব পাওয়া গেছে; স্বয়ংক্রিয়ভাবে কোনোটি বেছে নেওয়া হয়নি।'),
      'Duration exceeds supported datetime range; no interval was inferred.': tr('The stated duration is too large to establish a supported incident time interval.', 'দেওয়া স্থায়িত্ব এত বেশি যে নির্ভরযোগ্য ঘটনার সময়সীমা নির্ধারণ করা যায়নি।'),
    }
    return warnings[value] || `${tr('Saved note', 'সংরক্ষিত নোট')}: ${value}`
  }
  const evidenceText = (value: unknown): string => {
    if (typeof value === 'number') return number(value)
    if (typeof value === 'string') return value
    if (typeof value === 'boolean') return value ? tr('Yes', 'হ্যাঁ') : tr('No', 'না')
    if (Array.isArray(value)) return value.map(evidenceText).filter(Boolean).join(' · ')
    const data = object(value)
    if (typeof data.known === 'boolean') {
      const label = typeof data.value === 'string' ? (labels[data.value] || data.value) : typeof data.value === 'number' ? number(data.value) : ''
      return data.known ? `${tr('Matched label', 'মিলযুক্ত লেবেল')}: ${label || tr('Not recorded', 'সংরক্ষিত নেই')}`
        : `${tr('Label unavailable or not mapped to a priority rule', 'লেবেল পাওয়া যায়নি বা অগ্রাধিকারের নিয়মের সাথে মেলেনি')}${label ? `: ${label}` : ''}`
    }
    const keys: Record<string, string> = {
      value: tr('Label', 'লেবেল'), label: tr('Label', 'লেবেল'), duration_hours: tr('Hours', 'ঘণ্টা'), duration_days: tr('Days', 'দিন'), hours: tr('Hours', 'ঘণ্টা'), days: tr('Days', 'দিন'),
      band: tr('Applied rule', 'প্রযোজ্য নিয়ম'), matched_expressions: tr('Matched expressions', 'মিলযুক্ত প্রকাশ'), report_count: tr('Report count', 'অভিযোগের সংখ্যা'),
      interpretation: tr('Meaning', 'অর্থ'), concept: tr('Signal', 'সংকেত'), alias: tr('Matching phrase', 'মিলযুক্ত বাক্যাংশ'), matched_text: tr('Text evidence', 'লেখার প্রমাণ'),
      start: tr('Text start', 'লেখার শুরু'), end: tr('Text end', 'লেখার শেষ'), max_score: tr('Score limit', 'স্কোরসীমা'),
    }
    return Object.entries(data).filter(([, item]) => item !== null && item !== '' && item !== undefined)
      .map(([key, item]) => {
        const content = key === 'band' && typeof item === 'string' ? bandText(item)
          : key === 'interpretation' && item === 'report volume, not unique people affected' ? tr('Number of reports, not unique people affected', 'অভিযোগের সংখ্যা, স্বতন্ত্র ক্ষতিগ্রস্ত মানুষের সংখ্যা নয়')
          : key === 'concept' && typeof item === 'string' ? labels[item] || readable(item) : evidenceText(item)
        return `${keys[key] || readable(key)}: ${content}`
      }).join(' · ')
  }
  const originalScore = typeof analysis?.priority_score === 'number' ? analysis.priority_score : null

  return <section aria-labelledby="priority-explanation-title" className="space-y-5">
    <div>
      <p className="text-xs font-semibold uppercase tracking-[.14em] text-brand">{tr('Decision support', 'সিদ্ধান্ত সহায়তা')}</p>
      <h3 id="priority-explanation-title" className="mt-1 text-lg font-semibold text-brand">{tr('Why this priority?', 'এই অগ্রাধিকার কেন?')}</h3>
      <p className="mt-1 text-sm leading-6 text-brand">{tr('Rule points explain the automatic score. Review the complaint and its context before making a decision.', 'নিয়মভিত্তিক পয়েন্ট স্বয়ংক্রিয় স্কোর ব্যাখ্যা করে। সিদ্ধান্তের আগে অভিযোগ ও প্রেক্ষাপট পর্যালোচনা করুন।')}</p>
    </div>
    {originalScore !== null && (originalScore !== currentScore || Boolean(currentLabel && analysis?.priority_label && currentLabel !== analysis.priority_label)) && <div className="rounded-xl border border-teal-200 bg-teal-50 p-3 text-sm text-brand">
      {tr('An administrator changed this priority. The evidence below preserves the original automatic analysis:', 'প্রশাসক এই অগ্রাধিকার পরিবর্তন করেছেন। নিচে মূল স্বয়ংক্রিয় বিশ্লেষণের প্রমাণ রাখা হয়েছে:')} {number(originalScore)}
    </div>}
    {breakdown.length > 0 ? <div className="overflow-hidden rounded-xl border border-teal-200 bg-white">
      {breakdown.map((row, index) => {
        const contribution = typeof row.contribution === 'number' ? row.contribution : 0
        return <div key={`${String(row.name)}-${index}`} className="flex items-start gap-3 border-b border-teal-100 p-3 last:border-0">
          <span className={`min-w-10 rounded-lg px-2 py-1 text-center text-sm font-bold tabular-nums ${contribution > 0 ? 'bg-teal-50 text-brand' : contribution < 0 ? 'bg-teal-50 text-brand' : 'bg-teal-100 text-brand'}`}>
            {contribution > 0 ? '+' : ''}{number(contribution)}
          </span>
          <div className="min-w-0"><p className="text-sm font-semibold text-brand">{names[String(row.name)] || String(row.name || tr('Rule', 'নিয়ম')).replaceAll('_', ' ')}</p>
            <p className="mt-1 break-words text-xs leading-5 text-brand">{evidenceText(row.evidence) || tr('No matching evidence', 'মিলযুক্ত প্রমাণ নেই')}</p>
          </div>
        </div>
      })}
    </div> : <div className="rounded-xl border border-dashed border-teal-300 p-4 text-sm leading-6 text-brand">{tr('No automatic rule breakdown was saved for this report.', 'এই অভিযোগের জন্য স্বয়ংক্রিয় নিয়মের ব্যাখ্যা সংরক্ষিত নেই।')}</div>}
    {(['category', 'severity'] as const).map(target => {
      const model = object(explanation[target])
      // ClassifierRuntime.predict emits top_terms for TF-IDF margins and
      // optional embedding token deletion. Its top_features may be embedding
      // dimensions; those must never be presented as words or invented terms.
      const terms = rows(model.top_terms).filter(term => typeof term.term === 'string' && term.term.length > 0 &&
        ((typeof term.contribution === 'number' && Number.isFinite(term.contribution)) ||
          (typeof term.probability_drop_on_deletion === 'number' && Number.isFinite(term.probability_drop_on_deletion)))).slice(0, 8)
      if (!Object.keys(model).length) return null
      const approximate = terms.some(term => typeof term.probability_drop_on_deletion === 'number')
      return <div key={target}>
        <h4 className="text-sm font-semibold text-brand">{target === 'category' ? tr('Category evidence', 'বিভাগের প্রমাণ') : tr('Severity evidence', 'তীব্রতার প্রমাণ')}</h4>
        {terms.length ? <><div className="mt-2 flex flex-wrap gap-2">{terms.map((term, index) => {
          const contribution = typeof term.contribution === 'number' ? term.contribution : Number(term.probability_drop_on_deletion || 0)
          return <span key={index} className="inline-flex max-w-full items-center gap-2 rounded-lg border border-teal-200 bg-white px-2.5 py-1.5 text-xs text-brand">
            <span className="whitespace-pre-wrap break-all">{String(term.term || '')}</span>
            <span className="shrink-0 font-mono text-brand">{contribution >= 0 ? '+' : ''}{number(contribution)}</span>
          </span>
        })}</div>
        <p className="mt-2 text-xs leading-5 text-brand">{approximate
          ? tr('Approximate token-deletion sensitivity; context dependent, not an additive probability explanation.', 'শব্দ বাদ দেওয়ার আনুমানিক প্রভাব; এটি প্রেক্ষাপটনির্ভর, যোগযোগ্য সম্ভাবনার ব্যাখ্যা নয়।')
          : tr('Linear-model margin contributions; these are not calibrated probability points or causal effects.', 'রৈখিক মডেলের মার্জিনে অবদান; এগুলো ক্যালিব্রেটেড সম্ভাবনার পয়েন্ট বা কার্যকারণ নয়।')}</p></>
          : <p className="mt-2 text-xs leading-5 text-brand">{tr('No readable word-level evidence was saved for this prediction.', 'এই পূর্বাভাসের জন্য পাঠযোগ্য শব্দভিত্তিক প্রমাণ সংরক্ষিত নেই।')}</p>}
      </div>
    })}
    {strings(analysis?.review_reasons).length > 0 && <div className="rounded-xl bg-teal-50 p-3">
      <h4 className="text-sm font-semibold text-brand">{tr('Reasons for human review', 'মানব পর্যালোচনার কারণ')}</h4>
      <ul className="mt-2 list-inside list-disc space-y-1 text-xs leading-5 text-brand">{strings(analysis?.review_reasons).map(reason => <li key={reason}>{reviewReason(reason)}</li>)}</ul>
    </div>}
    {strings(explanation.notes).length > 0 && <details className="rounded-xl border border-teal-200 bg-white p-3 text-sm text-brand">
      <summary className="cursor-pointer font-medium text-brand">{tr('Evidence limitations', 'প্রমাণের সীমাবদ্ধতা')}</summary>
      <ul className="mt-2 space-y-2 text-xs leading-5">{strings(explanation.notes).map((note, index) => <li key={index}>{warningText(note)}</li>)}</ul>
    </details>}
    <p className="text-xs leading-5 text-brand">{tr('Related-report counts estimate submitted reports, not the number of people affected. Confidence describes classification; it does not certify priority or incident identity.', 'সম্পর্কিত অভিযোগের সংখ্যা ক্ষতিগ্রস্ত মানুষের সংখ্যা নয়। আস্থার মান শ্রেণিবিভাগের জন্য; এটি অগ্রাধিকার বা একই ঘটনার নিশ্চয়তা দেয় না।')}</p>
  </section>
}
