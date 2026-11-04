import { Activity, Search, Cpu, Zap, User, LogOut, Shield, Key, Network } from 'lucide-react'
import { shortName } from '../api'

export default function CorridorHeader({
  health,
  worst,
  query,
  setQuery,
  stations = [],
  onOpenSimulator,
  isSimActive = false,
  currentUser,
  onOpenLogin,
  onLogout,
  scale = 'core',
  onScaleChange,
}) {
  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-cobalt bg-slate2 px-5 py-3 shadow-md">
      <h1 className="flex items-center gap-2 text-xl font-bold">
        <Activity className="h-5 w-5 text-lime" aria-hidden /> RailPulse AI — Spatio-Temporal Corridor Monitor
      </h1>

      {/* Interactive Top Bar Status Indicator */}
      <span className="flex items-center gap-2 rounded-full border border-lime/50 bg-midnight px-3 py-1 text-xs font-semibold text-lime shadow-sm">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-lime opacity-80" />
          <span className="relative h-2.5 w-2.5 rounded-full bg-lime" />
        </span>
        ● GNN Engine Live | AMP Active (RTX 3050)
      </span>

      {/* Topology Scale Toggle: Core 10 vs Regional 52 */}
      <div className="flex items-center rounded-lg border border-cobalt/80 bg-midnight p-0.5 text-xs">
        <button
          onClick={() => onScaleChange && onScaleChange('core')}
          className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-semibold transition-all ${
            scale === 'core'
              ? 'bg-cobalt text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Core Corridor (10)
        </button>
        <button
          onClick={() => onScaleChange && onScaleChange('regional')}
          className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-semibold transition-all ${
            scale === 'regional'
              ? 'bg-lime text-midnight font-bold shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Network className="h-3 w-3" />
          Regional Network (52)
        </button>
      </div>

      {/* Simulator Action Button */}
      <button
        onClick={onOpenSimulator}
        className={`flex items-center gap-2 rounded-full px-3.5 py-1 text-xs font-bold transition-all shadow-sm ${
          isSimActive
            ? 'bg-lime text-midnight border border-lime animate-pulse'
            : 'border border-lime text-lime bg-lime/10 hover:bg-lime hover:text-midnight'
        }`}
      >
        <Zap className="h-3.5 w-3.5" />
        {isSimActive ? 'Shock Simulation Active' : '⚡ Simulate Network Shock'}
      </button>

      {/* User RBAC Authentication Badge & Controls */}
      {currentUser ? (
        <div className="flex items-center gap-2 rounded-full border border-cobalt bg-midnight/90 py-1 pl-3 pr-1 text-xs">
          <div className="flex items-center gap-1.5 text-slate-200">
            {currentUser.role === 'admin' ? (
              <Shield className="h-3.5 w-3.5 text-lime" />
            ) : (
              <User className="h-3.5 w-3.5 text-blue-400" />
            )}
            <span className="font-semibold text-slate-300 capitalize">{currentUser.role}:</span>
            <span className="font-mono text-lime">{currentUser.username}</span>
          </div>
          <button
            onClick={onLogout}
            title="Sign out & clear JWT token"
            className="flex items-center gap-1 rounded-full bg-red-950/40 border border-red-500/40 px-2 py-0.5 text-[11px] text-red-300 hover:bg-red-900/60 transition-colors"
          >
            <LogOut className="h-3 w-3" /> Logout
          </button>
        </div>
      ) : (
        <button
          onClick={onOpenLogin}
          className="flex items-center gap-1.5 rounded-full border border-cobalt bg-cobalt/20 px-3.5 py-1 text-xs font-semibold text-slate-100 hover:bg-cobalt hover:text-white transition-all shadow-sm"
        >
          <Key className="h-3.5 w-3.5 text-lime" /> Sign In (JWT)
        </button>
      )}

      {worst && !isSimActive && (
        <span className="rounded-full border border-lime bg-lime/10 px-3 py-1 text-xs font-semibold text-lime">
          ⚠️ Highest Delay: {shortName(worst.name)} (+{worst.delay_30m.toFixed(2)}m)
        </span>
      )}

      <label className="relative ml-auto w-full min-w-[220px] sm:w-64">
        <Search className="absolute left-3 top-2 h-4 w-4 text-slate-500" aria-hidden />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          list="stations"
          placeholder="Search station and center map"
          className="w-full rounded-lg border border-slate-800 bg-midnight py-1.5 pl-9 pr-3 text-xs text-slate-100 outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt"
        />
        <datalist id="stations">
          {stations.map((s) => (
            <option key={s.eva_id} value={s.name} />
          ))}
        </datalist>
      </label>
    </header>
  )
}
