import { lazy, Suspense, useEffect } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useAuth } from './auth'
import { useI18n } from './i18n'
import { Icon, Loading } from './ui'
import PublicPage from './pages/PublicPage'
import TrackingPage from './pages/TrackingPage'
import LoginPage from './pages/LoginPage'
import PrivacyPage from './pages/PrivacyPage'

const AdminDashboard = lazy(() => import('./pages/AdminDashboard'))
const AccountPage = lazy(() => import('./pages/AccountPage'))
const CitizenComplaintsPage = lazy(() => import('./pages/CitizenComplaintsPage'))
export default function App() {
  const { locale, setLocale, t } = useI18n(), { token, logout, citizenToken } = useAuth(), location = useLocation()
  useEffect(() => { window.scrollTo(0, 0); document.title = `${t('brand')} — ${location.pathname === '/privacy' ? t('privacyNotice') : location.pathname.startsWith('/admin') ? t('workspace') : location.pathname.startsWith('/track') ? t('track') : t('report')}` }, [location.pathname, locale])
  return <><a href="#main" className="skip-link">{t('skip')}</a><header className="site-header"><div className="header-inner"><Link to="/" className="brand" aria-label={t('brand')}><span className="brand-icon"><Icon name="building"/></span><span><strong>{t('brand')}</strong><small>{t('portal')}</small></span></Link><nav aria-label={locale === 'bn' ? 'প্রধান মেনু' : 'Main navigation'}><NavLink to="/" end>{t('report')}</NavLink><NavLink to="/track">{t('track')}</NavLink></nav><div className="header-actions"><div className="language-toggle" role="group" aria-label={t('language')}><button aria-pressed={locale === 'en'} onClick={() => setLocale('en')}>EN</button><button lang="bn" aria-pressed={locale === 'bn'} onClick={() => setLocale('bn')}>বাংলা</button></div>{citizenToken ? <Link className="staff-link" to="/account">{locale === 'bn' ? 'আমার অ্যাকাউন্ট' : 'My account'}<Icon name="arrow"/></Link> : <Link className="staff-link" to="/account">{locale === 'bn' ? 'লগইন / নিবন্ধন' : 'Sign in'}<Icon name="arrow"/></Link>}{token && location.pathname.startsWith('/admin') ? <button className="staff-link" onClick={logout}>{locale === 'bn' ? 'প্রস্থান' : 'Sign out'}</button> : <Link className="staff-link" to={token ? '/admin' : '/admin/login'}>{token ? t('workspace') : t('staff')}<Icon name="arrow"/></Link>}</div></div></header>
    <main id="main" tabIndex={-1} className={location.pathname === '/admin' ? 'admin-main' : 'public-main'}><Suspense fallback={<Loading/>}><Routes><Route path="/" element={<PublicPage/>}/><Route path="/track/:id?" element={<TrackingPage/>}/><Route path="/privacy" element={<PrivacyPage/>}/><Route path="/account" element={<AccountPage/>}/><Route path="/account/complaints" element={<CitizenComplaintsPage/>}/><Route path="/admin/login" element={<LoginPage/>}/><Route path="/admin" element={token ? <AdminDashboard/> : <Navigate to="/admin/login" replace/>}/><Route path="*" element={<div className="tracking-layout"><h1>{t('notFound')}</h1><Link className="button-primary mt-6" to="/">{t('back')}</Link></div>}/></Routes></Suspense></main>
    <footer className="site-footer"><span>© {new Date().getFullYear()} {t('brand')}</span><Link to="/privacy">{t('privacyNotice')}</Link><Link to="/track">{t('track')} →</Link></footer>
  </>
}
