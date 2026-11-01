/** All network access is centralized here. Never retry a mutation automatically. */
export type Lang = 'bn' | 'en' | 'banglish'
export type Status = 'submitted' | 'in_review' | 'confirmed' | 'in_progress' | 'resolved' | 'rejected'
export interface Complaint {
  id: string; lang: Lang; category: string | null; severity: string | null
  priority_score: number; priority_label: string; status: Status; area_id: string | null
  group_id: string | null; version: number; group_version: number | null
  created_at: string; updated_at: string; text?: string; location?: string | null
  timestamp?: string | null; contact?: string | null; confidence?: number | null
  review_required?: boolean; analysis?: Record<string, unknown>
}
export interface Group {
  id: string; status: string; version: number; size: number; created_at: string; updated_at: string
  complaint_ids?: string[]; confirmed_by?: string | null; members_truncated?: boolean
}
export interface Page<T> { items: T[]; total: number; page: number; page_size: number }
export interface Tracking {
  id: string; lang: Lang; category: string | null; severity: string | null; status: Status
  priority_score: number; priority_label: string; review_required: boolean; area_id: string | null
  created_at: string; updated_at: string
}
export interface Analytics {
  total_complaints: number; total_groups: number; needs_human_review: number
  by_category: Record<string, number>; by_priority: Record<string, number>; by_status: Record<string, number>
  by_area: Record<string, number>; by_language: Record<string, number>
  priority_over_time: { date: string; priority: string; count: number }[]
  review_workload: { status: string; review_required: boolean; count: number }[]
}
export type Params = Record<string, string | number | boolean | undefined | null>
export interface CreateComplaint { text: string; lang?: Lang; location?: string; contact?: string }
export interface ComplaintPatch {
  expected_version: number; reason: string; status?: Status; priority_score?: number
  priority_label?: string; category?: string; severity?: string
}
export interface GroupReview { expected_version: number; reason: string; complaint_ids?: string[] }
export class ApiError extends Error {
  constructor(public status: number, message: string, public code = 'request_failed') { super(message); this.name = 'ApiError' }
}

export function validateBaseUrl(value: string): string {
  const result = value.trim().replace(/\/+$/, '')
  if (result.startsWith('/') && !result.startsWith('//') && !/[?#\\]/.test(result)) return result
  try {
    const url = new URL(result)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error()
    return result
  } catch { throw new Error('VITE_API_BASE_URL must be an HTTP(S) URL or a path such as /api') }
}
export const API_BASE_URL = validateBaseUrl(import.meta.env.VITE_API_BASE_URL || '/api')

export async function request<T>(path: string, options: { method?: string; body?: unknown; token?: string | null; signal?: AbortSignal } = {}): Promise<T> {
  const controller = new AbortController()
  const abort = () => controller.abort()
  options.signal?.addEventListener('abort', abort, { once: true })
  if (options.signal?.aborted) controller.abort()
  const timer = setTimeout(abort, 60000)
  try {
    const response = await fetch(API_BASE_URL + path, {
      method: options.method || 'GET', credentials: 'omit', cache: 'no-store', signal: controller.signal,
      headers: { Accept: 'application/json', ...(options.body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(options.token ? { Authorization: `Bearer ${options.token}` } : {}) },
      ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),
    })
    const body = await response.json().catch(() => null)
    if (!response.ok) {
      if (response.status === 401 && options.token) window.dispatchEvent(new Event('nagarik:unauthorized'))
      const detail = typeof body?.detail === 'string' ? body.detail : 'The request could not be completed.'
      throw new ApiError(response.status, detail)
    }
    if (body === null) throw new ApiError(502, 'The server returned an invalid response.', 'invalid_response')
    return body as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (options.signal?.aborted) throw error
    if (controller.signal.aborted) throw new ApiError(0, 'The request timed out before confirmation.', 'timeout')
    throw new ApiError(0, 'Unable to reach the service.', 'network')
  } finally { clearTimeout(timer); options.signal?.removeEventListener('abort', abort) }
}
function query(params: Params = {}) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  return search.size ? '?' + search.toString() : ''
}
export const api = {
  submit: async (body: CreateComplaint) => {
    const result = await request<Complaint>('/complaints', { method: 'POST', body })
    // A proxy's unrelated JSON success is not proof that a report was saved.
    if (!result || typeof result.id !== 'string' || !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(result.id)
        || !Number.isInteger(result.version) || result.version < 1 || typeof result.status !== 'string') {
      throw new ApiError(502, 'The server did not return a valid submission receipt.', 'invalid_response')
    }
    return result
  },
  track: (id: string, signal?: AbortSignal) => request<Tracking>(`/tracking/${encodeURIComponent(id)}`, { signal }),
  login: (username: string, password: string) => request<{ access_token: string; token_type: string; expires_in: number }>('/auth/token', { method: 'POST', body: { username, password } }),
  listComplaints: (params: Params, token: string, signal?: AbortSignal) => request<Page<Complaint>>('/complaints' + query(params), { token, signal }),
  getComplaint: (id: string, token: string, signal?: AbortSignal) => request<Complaint>(`/complaints/${encodeURIComponent(id)}`, { token, signal }),
  patchComplaint: (id: string, body: ComplaintPatch, token: string) => request<Complaint>(`/complaints/${encodeURIComponent(id)}`, { method: 'PATCH', body, token }),
  listGroups: (params: Params, token: string, signal?: AbortSignal) => request<Page<Group>>('/duplicate-groups' + query(params), { token, signal }),
  getGroup: (id: string, token: string, signal?: AbortSignal) => request<Group>(`/duplicate-groups/${encodeURIComponent(id)}`, { token, signal }),
  confirmGroup: (id: string, body: GroupReview, token: string) => request<Group>(`/duplicate-groups/${encodeURIComponent(id)}/confirm`, { method: 'POST', body, token }),
  splitGroup: (id: string, body: GroupReview & { complaint_ids: string[] }, token: string) => request<{ original_group: Group; new_group: Group }>(`/duplicate-groups/${encodeURIComponent(id)}/split`, { method: 'POST', body, token }),
  analytics: (token: string, signal?: AbortSignal) => request<Analytics>('/analytics', { token, signal }),
}
