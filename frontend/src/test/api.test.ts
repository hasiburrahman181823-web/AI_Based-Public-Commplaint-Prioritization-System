import { describe, expect, it, vi } from 'vitest'
import { api, ApiError, request, validateBaseUrl } from '../api'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})

describe('configured API boundary', () => {
  it('accepts proxy/HTTP URLs and rejects credentials, protocol-relative and script URLs', () => {
    expect(validateBaseUrl('/api/')).toBe('/api')
    expect(validateBaseUrl('https://example.org/service///')).toBe('https://example.org/service')
    for (const value of ['//example.org', 'javascript:alert(1)', 'https://name:password@example.org', '/api?token=secret', '/api\\elsewhere']) {
      expect(() => validateBaseUrl(value)).toThrow(/VITE_API_BASE_URL/)
    }
  })

  it('submits optional contact anonymously without cookies or automatically retrying', async () => {
    const fetcher = vi.fn().mockResolvedValue(json({ id: '25fce77e-0fc7-4b31-b2f7-d39f1f340062', version: 1, status: 'submitted' }, 201))
    vi.stubGlobal('fetch', fetcher)
    const payload = { text: 'No water for three days', location: 'Mirpur 10', contact: 'volunteer@example.org' }
    await api.submit(payload)
    expect(fetcher).toHaveBeenCalledTimes(1)
    const [url, options] = fetcher.mock.calls[0]
    expect(url).toMatch(/\/complaints$/)
    expect(options.method).toBe('POST')
    expect(options.credentials).toBe('omit')
    expect(options.headers.Authorization).toBeUndefined()
    expect(JSON.parse(options.body)).toEqual(payload)
  })

  it('links a complaint with a citizen bearer token when signed in', async () => {
    const fetcher = vi.fn().mockResolvedValue(json({ id: '25fce77e-0fc7-4b31-b2f7-d39f1f340062', version: 1, status: 'submitted' }, 201))
    vi.stubGlobal('fetch', fetcher)
    await api.submit({ text: 'No water for three days' }, 'synthetic-citizen-token')
    const [, options] = fetcher.mock.calls[0]
    expect(options.headers.Authorization).toBe('Bearer synthetic-citizen-token')
    expect(options.credentials).toBe('omit')
    expect(JSON.parse(options.body)).toEqual({ text: 'No water for three days' })
  })

  it('sends explicit admin credentials and retains false-valued review filters', async () => {
    const fetcher = vi.fn().mockResolvedValue(json({ items: [], total: 0, page: 1, page_size: 25 }))
    vi.stubGlobal('fetch', fetcher)
    await api.listComplaints({ review_required: false, area: 'Mirpur 10', status: '', category: undefined }, 'synthetic-token')
    const [url, options] = fetcher.mock.calls[0]
    expect(url).toContain('review_required=false')
    expect(url).toContain('area=Mirpur+10')
    expect(url).not.toContain('status=')
    expect(url).not.toContain('undefined')
    expect(options.headers.Authorization).toBe('Bearer synthetic-token')
  })

  it('uses only the redacted tracking endpoint without an administrator bearer token', async () => {
    const fetcher = vi.fn().mockResolvedValue(json({ id: 'example' }))
    vi.stubGlobal('fetch', fetcher)
    await api.track('a/b')
    expect(fetcher.mock.calls[0][0]).toMatch(/\/tracking\/a%2Fb$/)
    expect(fetcher.mock.calls[0][1].headers.Authorization).toBeUndefined()
  })

  it('returns version conflicts to the caller instead of replaying a stale mutation', async () => {
    const fetcher = vi.fn().mockResolvedValue(json({ detail: 'Stale version' }, 409))
    vi.stubGlobal('fetch', fetcher)
    await expect(api.patchComplaint('complaint-1', { expected_version: 2, reason: 'Reviewed evidence', status: 'confirmed' }, 'synthetic-token'))
      .rejects.toMatchObject({ status: 409, message: 'Stale version' })
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('expires an admin session after unauthorized API access but not a failed public login', async () => {
    const dispatch = vi.spyOn(window, 'dispatchEvent')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'Unauthorized' }, 401)))
    await expect(api.login('admin', 'wrong-password')).rejects.toBeInstanceOf(ApiError)
    expect(dispatch).not.toHaveBeenCalled()
    await expect(api.analytics('expired-token')).rejects.toMatchObject({ status: 401 })
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'nagarik:unauthorized' }))
  })

  it('rejects an HTML proxy response rather than treating it as a saved receipt', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Missing API proxy</html>', { status: 200 })))
    await expect(api.submit({ text: 'Water unavailable' })).rejects.toMatchObject({ status: 502, code: 'invalid_response' })
  })

  it('rejects unrelated JSON success instead of showing an unconfirmed tracking receipt', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ status: 'ready', database: 'ok' })))
    await expect(api.submit({ text: 'Water unavailable' })).rejects.toMatchObject({ status: 502, code: 'invalid_response' })
  })

  it('reports an uncertain network failure once without duplicate writes', async () => {
    const fetcher = vi.fn().mockRejectedValue(new TypeError('Network disconnected'))
    vi.stubGlobal('fetch', fetcher)
    await expect(request('/complaints', { method: 'POST', body: { text: 'Water unavailable' } }))
      .rejects.toMatchObject({ status: 0, code: 'network' })
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
