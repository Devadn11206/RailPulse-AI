import { useState } from 'react'
import { Lock, User, Shield, AlertCircle, Key, CheckCircle } from 'lucide-react'
import { api } from '../api'

export default function LoginModal({ isOpen, onClose, onLoginSuccess }) {
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('admin123')
  const [role, setRole] = useState('admin')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  if (!isOpen) return null

  const handleRoleQuickSelect = (selectedRole) => {
    setRole(selectedRole)
    if (selectedRole === 'admin') {
      setUsername('admin')
      setPassword('admin123')
    } else {
      setUsername('analyst')
      setPassword('analyst123')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const res = await api.login({ username, password })
      if (res && res.access_token) {
        localStorage.setItem('railpulse_token', res.access_token)
        localStorage.setItem('railpulse_user', JSON.stringify({
          username,
          role: res.role || role,
        }))
        if (onLoginSuccess) {
          onLoginSuccess({ username, role: res.role || role, token: res.access_token })
        }
        onClose()
      } else {
        setError(res?.detail || 'Authentication failed. Please check credentials.')
      }
    } catch (err) {
      setError(err?.message || 'Server connection error during login.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-md rounded-xl border border-cobalt bg-midnight p-6 shadow-2xl shadow-cobalt/20">
        
        {/* Header */}
        <div className="mb-5 flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cobalt/20 text-lime border border-lime/30">
              <Lock className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 flex items-center gap-2">
                RailPulse AI RBAC Login
                <span className="rounded bg-lime/10 px-2 py-0.5 text-[11px] font-mono text-lime border border-lime/30">
                  JWT Auth
                </span>
              </h3>
              <p className="text-xs text-slate-400">Authenticate for analytical & simulation controls</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-100 text-lg leading-none"
          >
            ✕
          </button>
        </div>

        {/* Quick Role Fill Hint */}
        <div className="mb-4 rounded-lg border border-slate-800 bg-slate2/60 p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
            <span className="flex items-center gap-1 font-semibold text-slate-300">
              <Key className="h-3.5 w-3.5 text-lime" /> Quick Select Role:
            </span>
            <span className="font-mono text-[10px] text-slate-500">FastAPI PRD Module 08</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleRoleQuickSelect('admin')}
              className={`flex items-center justify-center gap-1.5 rounded py-1.5 text-xs font-semibold border transition-all ${
                role === 'admin'
                  ? 'border-lime bg-lime/15 text-lime'
                  : 'border-slate-800 bg-midnight text-slate-400 hover:border-slate-700'
              }`}
            >
              <Shield className="h-3.5 w-3.5" />
              Admin (admin / admin123)
            </button>
            <button
              type="button"
              onClick={() => handleRoleQuickSelect('analyst')}
              className={`flex items-center justify-center gap-1.5 rounded py-1.5 text-xs font-semibold border transition-all ${
                role === 'analyst'
                  ? 'border-lime bg-lime/15 text-lime'
                  : 'border-slate-800 bg-midnight text-slate-400 hover:border-slate-700'
              }`}
            >
              <User className="h-3.5 w-3.5" />
              Analyst (analyst / analyst123)
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-500/50 bg-red-950/30 p-2.5 text-xs text-red-300">
            <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Username
            </label>
            <div className="relative">
              <User className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full rounded-lg border border-slate-800 bg-slate2 py-2 pl-9 pr-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                className="w-full rounded-lg border border-slate-800 bg-slate2 py-2 pl-9 pr-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Role Authority Scope
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate2 py-2 px-3 text-sm text-slate-100 outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt"
            >
              <option value="admin">Admin (Full Control: Simulation + Audit + All Predictions)</option>
              <option value="analyst">Analyst (Monitoring + XAI Uncertainty Bounds)</option>
            </select>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-lime py-2.5 text-sm font-bold text-midnight shadow-md shadow-lime/20 transition-all hover:brightness-110 active:scale-[0.99] disabled:opacity-50"
            >
              {loading ? (
                <>Authenticating...</>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4" /> Authenticate & Issue JWT Token
                </>
              )}
            </button>
          </div>
        </form>

      </div>
    </div>
  )
}

