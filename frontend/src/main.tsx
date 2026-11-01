import { Component, StrictMode } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { AuthProvider } from './auth'
import { I18nProvider } from './i18n'
import './styles.css'

class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch(_error: Error, _info: ErrorInfo) { /* Do not log complaint or credential content. */ }
  render() { return this.state.failed ? <main className="tracking-layout"><h1>Something went wrong / একটি সমস্যা হয়েছে</h1><p>Please reload the page. পৃষ্ঠাটি আবার লোড করুন।</p><button className="button-primary mt-5" onClick={() => window.location.reload()}>Reload / আবার লোড করুন</button></main> : this.props.children }
}
createRoot(document.getElementById('root')!).render(<StrictMode><ErrorBoundary><I18nProvider><AuthProvider><BrowserRouter><App/></BrowserRouter></AuthProvider></I18nProvider></ErrorBoundary></StrictMode>)
