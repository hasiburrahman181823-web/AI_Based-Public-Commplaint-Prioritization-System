import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from './api'

const Context = createContext<{ token: string | null; login: (username: string, password: string) => Promise<void>; logout: () => void } | null>(null)
export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<{ token: string; expires: number } | null>(null)
  useEffect(() => {
    const logout = () => setSession(null)
    window.addEventListener('nagarik:unauthorized', logout)
    const timer = session ? setTimeout(logout, Math.max(0, session.expires - Date.now())) : undefined
    return () => { window.removeEventListener('nagarik:unauthorized', logout); clearTimeout(timer) }
  }, [session])
  async function login(username: string, password: string) { const response = await api.login(username, password); setSession({ token: response.access_token, expires: Date.now() + response.expires_in * 1000 }) }
  return <Context.Provider value={{ token: session?.token || null, login, logout: () => setSession(null) }}>{children}</Context.Provider>
}
export function useAuth() { const value = useContext(Context); if (!value) throw new Error('AuthProvider missing'); return value }
