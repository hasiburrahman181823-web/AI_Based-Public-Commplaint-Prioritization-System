import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { ApiError } from './api'

export type Locale = 'en' | 'bn'
const translations = {
  brand: ['Nagarik', 'নাগরিক'], portal: ['Community grievance portal', 'জনসেবা অভিযোগ পোর্টাল'],
  report: ['Report an issue', 'অভিযোগ জানান'], track: ['Track a complaint', 'অভিযোগের অগ্রগতি'],
  staff: ['Staff sign in', 'কর্মকর্তা প্রবেশ'], workspace: ['Admin workspace', 'প্রশাসনিক কর্মক্ষেত্র'],
  skip: ['Skip to content', 'মূল বিষয়বস্তুতে যান'], language: ['Choose interface language', 'ইন্টারফেসের ভাষা বেছে নিন'],
  local: ['Your neighbourhood', 'আপনার এলাকার জন্য'],
  heroStart: ['Report a local problem.', 'এলাকার সমস্যা'], heroEnd: ['Track it in one place.', 'জানান সহজে।'],
  intro: ['Tell us about water, roads, electricity or another public-service issue. Include the area and how long it has been happening.', 'পানি, রাস্তা, বিদ্যুৎ বা অন্য জনসেবার সমস্যা লিখুন। কোন এলাকায় এবং কত দিন ধরে সমস্যা হচ্ছে, তা জানান।'],
  noAccount: ['No account needed', 'কোনো অ্যাকাউন্ট লাগে না'], privateContact: ['Contact stays private', 'যোগাযোগের তথ্য গোপন থাকে'],
  how: ['How to submit', 'যেভাবে অভিযোগ জানাবেন'],
  step1: ['Describe the problem', 'সমস্যাটি লিখুন'], step1Hint: ['Say what happened, when it started and who is affected.', 'কী সমস্যা, কত দিন ধরে চলছে এবং কারা ক্ষতিগ্রস্ত তা জানান।'],
  step2: ['Add the area, if known', 'এলাকা জানা থাকলে দিন'], step2Hint: ['An area or nearby landmark helps identify the location.', 'এলাকা বা কাছের পরিচিত জায়গার নাম দিন।'],
  step3: ['Save your tracking ID', 'ট্র্যাকিং আইডি রাখুন'], step3Hint: ['Use it to check the saved status of your complaint.', 'অভিযোগের সর্বশেষ অবস্থা দেখতে এই আইডি লাগবে।'],
  formTitle: ['Write your complaint', 'আপনার অভিযোগ লিখুন'], formHint: ['You can write in Bangla, English or Banglish.', 'বাংলা, ইংরেজি বা বাংলিশে লিখতে পারেন।'],
  complaint: ['Your complaint', 'আপনার অভিযোগ'], required: ['required', 'আবশ্যক'],
  complaintHint: ['For example: There has been no water in Mirpur 10 for three days…', 'যেমন: মিরপুর ১০-এ তিন দিন ধরে পানি নেই…'],
  chars: ['characters', 'অক্ষর'], detected: ['Detected', 'শনাক্ত'], auto: ['Auto-detect', 'স্বয়ংক্রিয় শনাক্ত'],
  detectionHint: ['Language detection is an estimate. You can correct it.', 'ভাষা শনাক্তকরণ আনুমানিক। চাইলে সংশোধন করুন।'],
  textLanguage: ['Complaint language', 'অভিযোগের ভাষা'], area: ['Area or landmark', 'এলাকা বা পরিচিত স্থান'],
  optional: ['optional', 'ঐচ্ছিক'], areaHint: ['e.g. Mirpur 10, Dhaka', 'যেমন: মিরপুর ১০, ঢাকা'],
  contact: ['Phone or email', 'ফোন বা ইমেইল'], contactHint: ['Only authorised staff can see this. No automatic SMS or email is sent.', 'শুধু অনুমোদিত কর্মকর্তা এটি দেখতে পাবেন। স্বয়ংক্রিয় এসএমএস বা ইমেইল পাঠানো হয় না।'],
  privacy: ['Your complaint is visible to authorised staff. Tracking shows a limited status summary. Keep your tracking ID private.', 'আপনার অভিযোগ অনুমোদিত কর্মকর্তারা দেখতে পাবেন। ট্র্যাকিংয়ে সংক্ষিপ্ত অগ্রগতি দেখা যায়। ট্র্যাকিং আইডি গোপন রাখুন।'],
  privacyNotice: ['Privacy notice', 'গোপনীয়তার বিজ্ঞপ্তি'],
  demoPrivacy: ['Demo: use synthetic complaints until operator privacy details are published.', 'ডেমো: পরিচালকের গোপনীয়তার তথ্য প্রকাশ না হওয়া পর্যন্ত কাল্পনিক অভিযোগ ব্যবহার করুন।'],
  submit: ['Send complaint', 'অভিযোগ পাঠান'], sending: ['Sending your complaint…', 'অভিযোগ পাঠানো হচ্ছে…'],
  sendHint: ['Keep the tracking ID shown after you submit.', 'জমা দেওয়ার পর পাওয়া ট্র্যাকিং আইডিটি রেখে দিন।'],
  received: ['Your complaint was received', 'আপনার অভিযোগ পাওয়া গেছে'],
  receivedHint: ['Save this tracking ID. You will need it to check progress.', 'অগ্রগতি দেখতে এই ট্র্যাকিং আইডি লাগবে। এটি সংরক্ষণ করুন।'],
  trackingId: ['Tracking ID', 'ট্র্যাকিং আইডি'], copy: ['Copy ID', 'আইডি কপি করুন'], copied: ['Copied', 'কপি হয়েছে'], copyFailed: ['Select and copy the ID below.', 'নিচের আইডিটি নির্বাচন করে কপি করুন।'],
  another: ['Report another issue', 'আরেকটি অভিযোগ জানান'], viewProgress: ['View progress', 'অগ্রগতি দেখুন'],
  trackTitle: ['A clear view of your complaint.', 'আপনার অভিযোগের অগ্রগতি জানুন।'],
  trackHint: ['Enter the tracking ID from your receipt. You do not need an account.', 'প্রাপ্তিস্বীকারে থাকা ট্র্যাকিং আইডি লিখুন। কোনো অ্যাকাউন্ট লাগবে না।'],
  trackAction: ['Find my complaint', 'আমার অভিযোগ খুঁজুন'], tracking: ['Looking up your complaint…', 'আপনার অভিযোগ খোঁজা হচ্ছে…'],
  invalidId: ['Enter a valid tracking ID from your receipt.', 'প্রাপ্তিস্বীকার থেকে সঠিক ট্র্যাকিং আইডি লিখুন।'],
  status: ['Current status', 'বর্তমান অবস্থা'], submittedAt: ['Received', 'পাওয়া গেছে'], updatedAt: ['Last updated', 'সর্বশেষ হালনাগাদ'],
  priority: ['Priority', 'অগ্রাধিকার'], category: ['Category', 'বিভাগ'], reviewNeeded: ['Awaiting human review', 'কর্মকর্তার পর্যালোচনার অপেক্ষায়'],
  trackingPrivacy: ['For your privacy, tracking does not display complaint text or contact information. Anyone with this ID can see this status summary.', 'গোপনীয়তার জন্য ট্র্যাকিংয়ে অভিযোগের লেখা বা যোগাযোগের তথ্য দেখানো হয় না। এই আইডি যার কাছে আছে তিনি সংক্ষিপ্ত অগ্রগতি দেখতে পারবেন।'],
  next: ['What happens next?', 'এরপর কী হবে?'], nextHint: ['Authorised staff review the issue and update its status. This page shows the latest saved update; refresh to check again.', 'অনুমোদিত কর্মকর্তা অভিযোগ পর্যালোচনা করে অবস্থা হালনাগাদ করবেন। এখানে সর্বশেষ সংরক্ষিত তথ্য দেখা যায়; আবার দেখতে রিফ্রেশ করুন।'],
  refresh: ['Refresh status', 'অগ্রগতি রিফ্রেশ করুন'],
  signIn: ['Sign in to your workspace', 'আপনার কর্মক্ষেত্রে প্রবেশ করুন'],
  signInHint: ['Review reports, connect related incidents and make informed decisions.', 'অভিযোগ পর্যালোচনা, সম্পর্কিত ঘটনা শনাক্ত এবং তথ্যভিত্তিক সিদ্ধান্ত নিন।'],
  username: ['Username', 'ব্যবহারকারীর নাম'], password: ['Password', 'পাসওয়ার্ড'], login: ['Sign in', 'প্রবেশ করুন'], loggingIn: ['Signing in…', 'প্রবেশ করা হচ্ছে…'],
  sessionHint: ['Authorised staff only. Your session ends when you refresh or close this page.', 'শুধু অনুমোদিত কর্মকর্তাদের জন্য। পৃষ্ঠা রিফ্রেশ বা বন্ধ করলে সেশন শেষ হবে।'],
  back: ['Back to the public portal', 'জনসেবা পোর্টালে ফিরুন'], notFound: ['This page could not be found.', 'এই পৃষ্ঠাটি পাওয়া যায়নি।'],
  retry: ['Try again', 'আবার চেষ্টা করুন'], unknown: ['Not available', 'পাওয়া যায়নি'], loading: ['Loading…', 'লোড হচ্ছে…'],
} satisfies Record<string, readonly [string, string]>
export type TranslationKey = keyof typeof translations
const Context = createContext<{ locale: Locale; setLocale: (locale: Locale) => void; t: (key: TranslationKey) => string } | null>(null)
export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(() => { try { return localStorage.getItem('nagarik:locale') === 'bn' ? 'bn' : 'en' } catch { return 'en' } })
  useEffect(() => { document.documentElement.lang = locale; try { localStorage.setItem('nagarik:locale', locale) } catch { /* Preferences are optional. */ } }, [locale])
  return <Context.Provider value={{ locale, setLocale, t: key => translations[key][locale === 'bn' ? 1 : 0] }}>{children}</Context.Provider>
}
export function useI18n() { const value = useContext(Context); if (!value) throw new Error('I18nProvider missing'); return value }
const labels: Record<string, [string, string]> = {
  submitted: ['Received', 'গৃহীত'], in_review: ['Under review', 'পর্যালোচনাধীন'], confirmed: ['Confirmed', 'নিশ্চিত'], in_progress: ['In progress', 'কাজ চলছে'], resolved: ['Resolved', 'সমাধান হয়েছে'], rejected: ['Closed — not accepted', 'বন্ধ — গৃহীত হয়নি'],
  low: ['Low', 'নিম্ন'], moderate: ['Moderate', 'মাঝারি'], medium: ['Medium', 'মাঝারি'], high: ['High', 'উচ্চ'], urgent: ['Urgent', 'জরুরি'], unknown: ['Unknown', 'অজানা'],
  Water: ['Water', 'পানি'], Electricity: ['Electricity', 'বিদ্যুৎ'], 'Road & Transport': ['Road & Transport', 'রাস্তা ও পরিবহন'], 'Waste Management': ['Waste Management', 'বর্জ্য ব্যবস্থাপনা'], Health: ['Health', 'স্বাস্থ্য'],
  dhaka_mirpur_10: ['Mirpur 10', 'মিরপুর ১০'], dhaka_mohammadpur: ['Mohammadpur', 'মোহাম্মদপুর'], dhaka_uttara_sector_4: ['Uttara sector 4', 'উত্তরা সেক্টর ৪'],
  bn: ['Bangla', 'বাংলা'], en: ['English', 'ইংরেজি'], banglish: ['Banglish', 'বাংলিশ'],
}
export function label(value: string | null | undefined, locale: Locale) { return value ? (labels[value]?.[locale === 'bn' ? 1 : 0] || value.replaceAll('_', ' ')) : (locale === 'bn' ? 'পাওয়া যায়নি' : 'Not available') }
export function dateLabel(value: string, locale: Locale) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(locale === 'bn' ? 'bn-BD' : 'en-GB', { dateStyle: 'medium', timeStyle: 'short' }).format(date) }
export function errorMessage(error: unknown, locale: Locale): string {
  const bn = locale === 'bn'
  if (!(error instanceof ApiError)) return bn ? 'কাজটি সম্পন্ন হয়নি। আবার চেষ্টা করুন।' : 'Something went wrong. Please try again.'
  if (error.status === 401) return bn ? 'সঠিক প্রবেশতথ্য দিন অথবা আবার প্রবেশ করুন।' : 'Check your sign-in details or sign in again.'
  if (error.status === 404) return bn ? 'এই আইডিতে কোনো অভিযোগ পাওয়া যায়নি। আইডিটি পরীক্ষা করুন।' : 'No complaint was found with this ID. Check the ID and try again.'
  if (error.status === 409) return bn ? 'অন্য একটি পরিবর্তন সংরক্ষিত হয়েছে। সর্বশেষ তথ্য নিয়ে আবার পর্যালোচনা করুন।' : 'This record changed. Refresh it and review the latest version before saving.'
  if (error.status === 429) return bn ? 'অল্প সময়ে অনেক অনুরোধ হয়েছে। কিছুক্ষণ পর আবার চেষ্টা করুন।' : 'Too many requests. Please wait a little before trying again.'
  if (error.status === 422 || error.status === 413) return bn ? 'তথ্য গ্রহণ করা হয়নি। লেখার দৈর্ঘ্য, পুনরাবৃত্তি এবং ফর্মের তথ্য পরীক্ষা করুন।' : 'The input was not accepted. Check the form, text length and repeated content.'
  if (error.code === 'timeout') return bn ? 'সময় শেষ হয়েছে; প্রাপ্তি নিশ্চিত করা যায়নি। পুনরায় পাঠানোর আগে সংযোগ পরীক্ষা করুন।' : 'The request timed out before receipt was confirmed. Check your connection before resending.'
  if (error.status === 0) return bn ? 'সেবার সাথে সংযোগ হয়নি। ইন্টারনেট সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।' : 'Unable to reach the service. Check your connection and try again.'
  return bn ? 'সেবা সাময়িকভাবে পাওয়া যাচ্ছে না। একটু পর চেষ্টা করুন।' : 'The service is temporarily unavailable. Please try again shortly.'
}
