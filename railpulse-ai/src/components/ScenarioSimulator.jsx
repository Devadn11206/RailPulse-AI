import { useState } from 'react'
import { Zap, X, AlertTriangle, ArrowRight, Activity, RotateCcw } from 'lucide-react'
import { api, shortName } from '../api'

export default function ScenarioSimulator({
  isOpen,
  onClose,
  stations = [],
  onApplySimulation,
  onResetSimulation,
  activeSimulation,
}) {
  const [selectedEva, setSelectedEva] = useState(8000105) // Frankfurt Hbf
  const [delayShock, setDelayShock] = useState(25)
  const [loading, setLoading] = useState(false)
  const [simResult, setSimResult] = useState(activeSimulation)

  if (!isOpen) return null

  const handleRunSimulation = async () => {
    setLoading(true)
    try {
      const res = await api.simulateDisruption({
        eva_id: Number(selectedEva),
        injected_delay_min: Number(delayShock),
      })
      setSimResult(res)
      if (onApplySimulation) {
        onApplySimulation(res)
      }
    } catch (err) {
      console.error('Simulation failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSimResult(null)
    if (onResetSimulation) {
      onResetSimulation()
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-xl rounded-xl border border-cobalt bg-slate2 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-cobalt/60 bg-midnight px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-lime/10 text-lime border border-lime/30">
              <Zap className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                Disruption Propagation Simulator
                <span className="rounded bg-cobalt/30 border border-cobalt px-1.5 py-0.2 text-[10px] text-lime">
                  ST-GNN Shock
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Inject artificial delay shock & simulate spatial cascade over graph topology
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Station Dropdown */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Shock Epicenter (Origin Station)
              </label>
              <select
                value={selectedEva}
                onChange={(e) => setSelectedEva(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-midnight px-3 py-2 text-sm text-white outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt"
              >
                {stations.map((s) => (
                  <option key={s.eva_id} value={s.eva_id}>
                    {shortName(s.name)} (EVA {s.eva_id})
                  </option>
                ))}
              </select>
            </div>

            {/* Slider */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-300">
                  Injected Delay Shock
                </label>
                <span className="font-mono text-sm font-bold text-lime">
                  +{delayShock} min
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="60"
                step="5"
                value={delayShock}
                onChange={(e) => setDelayShock(Number(e.target.value))}
                className="w-full accent-lime cursor-pointer bg-slate-800 rounded-lg h-2"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                <span>+5m (Minor)</span>
                <span>+30m</span>
                <span>+60m (Severe)</span>
              </div>
            </div>
          </div>

          {/* Action button */}
          <div className="flex gap-2">
            <button
              onClick={handleRunSimulation}
              disabled={loading}
              className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-lime px-4 py-2.5 text-sm font-bold text-midnight transition-opacity hover:opacity-95 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Activity className="h-4 w-4 animate-spin" />
                  Running Neural Message Passing…
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 fill-midnight" />
                  Run ST-GNN Propagation Simulation
                </>
              )}
            </button>
            {simResult && (
              <button
                onClick={handleReset}
                title="Reset simulation overlay"
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-midnight px-3 py-2.5 text-xs text-slate-300 hover:border-slate-500 hover:text-white"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            )}
          </div>

          {/* Results Summary if active */}
          {simResult && (
            <div className="rounded-xl border border-lime/30 bg-midnight p-4 space-y-3.5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-lime opacity-75" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-lime" />
                  </span>
                  <span className="text-xs font-bold uppercase tracking-wider text-lime">
                    Cascade Impact Mode Active
                  </span>
                </div>
                <span className="text-xs text-slate-400">
                  Total network propagated delay:{' '}
                  <b className="text-lime">
                    +{simResult.network_impact?.total_propagated_delay_15m || 0}m
                  </b>
                </span>
              </div>

              {/* Epicenter row */}
              <div className="rounded-lg border border-slate-800 bg-[#0f172a] p-3 text-xs">
                <div className="text-slate-400 mb-1">Injected Shock at Epicenter:</div>
                <div className="flex items-center justify-between">
                  <strong className="text-white text-sm">
                    {simResult.origin?.station_name}
                  </strong>
                  <span className="font-mono font-bold text-lime">
                    +{simResult.origin?.injected_shock_min}m injected
                  </span>
                </div>
              </div>

              {/* 1-Hop Cascading Neighbors */}
              <div>
                <h4 className="text-xs font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
                  <AlertTriangle className="h-3.5 w-3.5 text-warn" />
                  Direct 1-Hop Cascading Stations:
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {(simResult.downstream_1hop_neighbors || []).map((n) => (
                    <div
                      key={n.eva_id}
                      className="rounded border border-cobalt/40 bg-slate2/60 p-2 text-xs flex justify-between items-center"
                    >
                      <span className="text-slate-200">{shortName(n.station_name)}</span>
                      <span className="font-mono text-lime font-bold">
                        +{n.propagated_delta_15m.toFixed(2)}m
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="border-t border-cobalt/40 bg-midnight px-5 py-3 flex justify-between items-center text-xs text-slate-500">
          <span>Map rings reflect real-time shock radius</span>
          <button
            onClick={onClose}
            className="rounded bg-slate-800 px-3 py-1 text-slate-300 hover:bg-slate-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

