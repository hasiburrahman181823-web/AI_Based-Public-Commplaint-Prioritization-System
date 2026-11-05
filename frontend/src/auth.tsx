import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from './api'
import type { CitizenAccount, CitizenAuthResponse } from './api'

type CitizenSession = { token: string; expires: number; account: CitizenAccount | null }
const CITIZEN_SESSION_KEY = 'nagarik:citizen-session'

function storedCitizenSession(): CitizenSession | null {
  try {
    const value = sessionStorage.getItem(CITIZEN_SESSION_KEY)
    if (!value) return null
    const parsed = JSON.parse(value) as Pick<CitizenSession, 'token' | 'expires'>
    if (typeof parsed.token !== 'string' || typeof parsed.expires !== 'number' || parsed.expires <= Date.now()
        || parsed.token.length > 8192) {
      sessionStorage.removeItem(CITIZEN_SESSION_KEY)
      return null
    }
    return { ...parsed, account: null }
  } catch { return null }
}

function saveCitizenSession(response: CitizenAuthResponse): CitizenSession {
  const value = { token: response.access_token, expires: Date.now() + response.expires_in * 1000, account: response.account }
  try { sessionStorage.setItem(CITIZEN_SESSION_KEY, JSON.stringify({ token: value.token, expires: value.expires })) } catch { /* Keep the session in memory when storage is blocked. */ }
  return value
}

function clearCitizenStorage() {
  try { sessionStorage.removeItem(CITIZEN_SESSION_KEY) } catch { /* Storage may be disabled. */ }
}

type AuthContextValue = {
  token: string | null
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  citizenToken: string | null
  citizen: CitizenAccount | null
  citizenLogin: (email: string, password: string) => Promise<void>
  citizenRegister: (body: { email: string; password: string; full_name?: string; phone?: string }) => Promise<void>
  citizenLogout: () => void
}
const Context = createContext<AuthContextValue | null>(null)
export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<{ token: string; expires: number } | null>(null)
  const [citizenSession, setCitizenSession] = useState<CitizenSession | null>(storedCitizenSession)
  useEffect(() => {
    const logoutAdmin = () => setSession(null)
    const logoutCitizen = () => { setCitizenSession(null); clearCitizenStorage() }
    window.addEventListener('nagarik:unauthorized', logoutAdmin)
    window.addEventListener('nagarik:citizen-unauthorized', logoutCitizen)
    const adminTimer = session ? setTimeout(logoutAdmin, Math.max(0, session.expires - Date.now())) : undefined
    const citizenTimer = citizenSession ? setTimeout(logoutCitizen, Math.max(0, citizenSession.expires - Date.now())) : undefined
    return () => {
      window.removeEventListener('nagarik:unauthorized', logoutAdmin)
      window.removeEventListener('nagarik:citizen-unauthorized', logoutCitizen)
      clearTimeout(adminTimer)
      clearTimeout(citizenTimer)
    }
  }, [session, citizenSession])
  useEffect(() => {
    if (!citizenSession?.token || citizenSession.account) return
    let active = true
    api.citizenMe(citizenSession.token).then(account => {
      if (active) setCitizenSession(current => current?.token === citizenSession.token ? { ...current, account } : current)
    }).catch(() => {
      if (active) { setCitizenSession(null); clearCitizenStorage() }
    })
    return () => { active = false }
  }, [citizenSession?.token, citizenSession?.account])
  async function login(username: string, password: string) { const response = await api.login(username, password); setSession({ token: response.access_token, expires: Date.now() + response.expires_in * 1000 }) }
  async function citizenLogin(email: string, password: string) { setCitizenSession(saveCitizenSession(await api.citizenLogin(email, password))) }
  async function citizenRegister(body: { email: string; password: string; full_name?: string; phone?: string }) {
    setCitizenSession(saveCitizenSession(await api.citizenRegister(body)))
  }
  function citizenLogout() { setCitizenSession(null); clearCitizenStorage() }
  return <Context.Provider value={{
    token: session?.token || null, login, logout: () => setSession(null),
    citizenToken: citizenSession?.token || null, citizen: citizenSession?.account || null,
    citizenLogin, citizenRegister, citizenLogout,
  }}>{children}</Context.Provider>
}
export function useAuth() { const value = useContext(Context); if (!value) throw new Error('AuthProvider missing'); return value }
