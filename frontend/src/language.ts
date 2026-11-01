import type { Lang } from './api'
const words = new Set(['pani', 'pany', 'nai', 'nei', 'bidyut', 'biddut', 'rasta', 'ache', 'achhe', 'dhore', 'onek', 'ekhane', 'amader', 'agun', 'baccha', 'haspatal', 'bonna'])
/** Matches the server's deliberately small heuristic; this is not a language model. */
export function detectLanguage(text: string): Lang | null {
  if (!text.trim()) return null
  const clean = text.normalize('NFKC').toLowerCase().replace(/(?:https?:\/\/|www\.)\S+/g, ' ')
  if (/[\u0980-\u09FF]/u.test(clean.replace(/[০-৯]/g, ''))) return 'bn'
  return new Set((clean.match(/[a-z]+/g) || []).filter(word => words.has(word))).size >= 2 ? 'banglish' : 'en'
}
