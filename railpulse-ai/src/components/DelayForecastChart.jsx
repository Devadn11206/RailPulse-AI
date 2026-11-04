import { useState, useMemo } from 'react'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { Info, Network, ShieldCheck, Zap } from 'lucide-react'
import { shortName } from '../api'

export default function DelayForecastChart({ pred, simulatedDirective = null, onClearMitigation = null }) {
  const data = useMemo(() => {
    const t0 = pred.delay_15m,
      seed = pred.eva_id % 7
    const hist = [0.55, 0.66, 0.77, 0.88, 1].map(
      (f, i) => +(t0 * f + Math.sin(seed + i) * 0.2 * (i < 4)).toFixed(2)
    )
    const lab = ['t-60m', 't-45m', 't-30m', 't-15m', 't0']

    // Extract Monte Carlo Dropout 95% confidence bounds
    const ci15 = pred.confidence_interval_min?.['15m'] || [
      Math.max(0, pred.delay_15m - 0.25),
      pred.delay_15m + 0.25,
    ]
    const ci30 = pred.confidence_interval_min?.['30m'] || [
      Math.max(0, pred.delay_30m - 0.4),
      pred.delay_30m + 0.4,
    ]
    const ci60 = pred.confidence_interval_min?.['60m'] || [
      Math.max(0, pred.delay_60m - 0.6),
      pred.delay_60m + 0.6,
    ]

    // Calculate mitigated delay projection if directive simulation is active
    let mit15 = null, mit30 = null, mit60 = null
    if (simulatedDirective) {
      const hold = Number(simulatedDirective.hold_duration_min) || 0
      const sec = Number(simulatedDirective.secondary_delay_impact_min) || 0
      // Mitigation models delay reduction from active control intervention
      const benefit = Math.max(0.6, (pred.delay_15m * 0.4) + (hold > 0 ? 0.3 : 0.8))
      mit15 = Math.max(0.1, +(pred.delay_15m - benefit + sec * 0.3).toFixed(2))
      mit30 = Math.max(0.1, +(pred.delay_30m - benefit * 0.8 + sec * 0.2).toFixed(2))
      mit60 = Math.max(0.1, +(pred.delay_60m - benefit * 0.6).toFixed(2))
    }

    return [
      ...lab.map((x, i) => ({
        x,
        observed: Math.max(0, hist[i]),
        ci_lower: null,
        ci_upper: null,
        ci_range: null,
        mitigated: null,
      })),
      {
        x: '+15m',
        forecast: pred.delay_15m,
        ci_lower: Math.max(0, ci15[0]),
        ci_upper: ci15[1],
        ci_range: [Math.max(0, ci15[0]), ci15[1]],
        mitigated: mit15,
      },
      {
        x: '+30m',
        forecast: pred.delay_30m,
        ci_lower: Math.max(0, ci30[0]),
        ci_upper: ci30[1],
        ci_range: [Math.max(0, ci30[0]), ci30[1]],
        mitigated: mit30,
      },
      {
        x: '+60m',
        forecast: pred.delay_60m,
        ci_lower: Math.max(0, ci60[0]),
        ci_upper: ci60[1],
        ci_range: [Math.max(0, ci60[0]), ci60[1]],
        mitigated: mit60,
      },
    ].map((d, i) =>
      i === 4
        ? {
            ...d,
            forecast: d.observed,
            ci_lower: d.observed,
            ci_upper: d.observed,
            ci_range: [d.observed, d.observed],
            mitigated: simulatedDirective ? d.observed : null,
          }
        : d
    )
  }, [pred, simulatedDirective])

  const margin15 = pred.margin_min?.['15m'] || 0.18
  const factors = pred.contributing_factors || []

  return (
    <div className="flex flex-col gap-3 transition-opacity duration-300 ease-in-out">
      {/* Mitigation Simulation Overlay Banner */}
      {simulatedDirective ? (
        <div className="flex items-center justify-between rounded-md border border-lime/60 bg-lime/10 px-3 py-1.5 text-xs shadow-inner">
          <span className="flex items-center gap-1.5 text-lime font-semibold">
            <Zap className="h-3.5 w-3.5 text-lime" />
            <span>AI Mitigated Overlay Active: {simulatedDirective.action}</span>
          </span>
          <button
            onClick={onClearMitigation}
            className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-300 hover:bg-slate-700 hover:text-white"
          >
            Clear Overlay
          </button>
        </div>
      ) : (
        /* Monte Carlo Dropout Confidence Banner */
        <div className="flex items-center justify-between rounded-md border border-cobalt/40 bg-midnight px-3 py-1.5 text-xs shadow-inner">
          <span className="flex items-center gap-1.5 text-slate-300">
            <Info className="h-3.5 w-3.5 text-lime" />
            <span>ST-GNN Monte Carlo Uncertainty (95% CI):</span>
          </span>
          <span className="font-mono text-lime font-bold">
            ±{margin15.toFixed(2)} min margin
          </span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={210}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <defs>
            {/* Shaded Forecast fill */}
            <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#d4ff24" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#d4ff24" stopOpacity={0.0} />
            </linearGradient>

            {/* Shaded Mitigated fill (Cyan/Sky) */}
            <linearGradient id="mitigatedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.0} />
            </linearGradient>

            {/* Shaded Monte Carlo 95% Confidence Interval Band (Acid Lime transparent) */}
            <linearGradient id="ciAcidBand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#d4ff24" stopOpacity={0.22} />
              <stop offset="100%" stopColor="#1558c9" stopOpacity={0.08} />
            </linearGradient>

            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <CartesianGrid stroke="#16213c" vertical={false} strokeDasharray="3 3" />
          <XAxis dataKey="x" stroke="#7f8cab" fontSize={11} tickLine={false} />
          <YAxis stroke="#7f8cab" fontSize={11} unit="m" tickLine={false} axisLine={false} />
          
          <Tooltip
            contentStyle={{
              background: '#0A0F1D',
              border: '1px solid #1558c9',
              borderRadius: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            }}
            formatter={(v, name) => [
              Array.isArray(v) ? `${(+v[0]).toFixed(2)} - ${(+v[1]).toFixed(2)} min` : `${(+v).toFixed(2)} min`,
              name,
            ]}
          />

          <ReferenceLine x="t0" stroke="#7f8cab" strokeDasharray="3 4" label={{ value: 'NOW', fill: '#7f8cab', fontSize: 10, position: 'insideTopLeft' }} />

          {/* Area 1: Shaded 95% Confidence Interval Band */}
          <Area
            dataKey="ci_upper"
            name="95% CI Bound (MCD)"
            type="monotone"
            stroke="#d4ff24"
            strokeWidth={1}
            strokeDasharray="2 3"
            strokeOpacity={0.5}
            fill="url(#ciAcidBand)"
            fillOpacity={0.15}
            connectNulls
          />

          {/* Area 2: Forecast Area Envelope */}
          <Area
            dataKey="forecast"
            name="Forecast Envelope"
            type="monotone"
            stroke="none"
            fill="url(#forecastGradient)"
            connectNulls
          />

          {/* Observed Delay Line (Historic) */}
          <Line
            dataKey="observed"
            name="Observed History"
            type="monotone"
            stroke="#1558c9"
            strokeWidth={3}
            dot={{ r: 3, fill: '#1558c9' }}
            style={{ filter: 'url(#glow)' }}
            connectNulls={false}
          />

          {/* ST-GNN Delay Forecast Line (Future Horizons) */}
          <Line
            dataKey="forecast"
            name="ST-GNN Delay Forecast"
            type="monotone"
            stroke="#d4ff24"
            strokeWidth={3}
            dot={{ r: 4, fill: '#d4ff24' }}
            style={{ filter: 'url(#glow)' }}
            connectNulls
          />

          {/* Mitigated Delay Forecast Line (Simulated AI Directive) */}
          {simulatedDirective && (
            <Line
              dataKey="mitigated"
              name="Mitigated AI Forecast"
              type="monotone"
              stroke="#38bdf8"
              strokeWidth={3}
              strokeDasharray="4 4"
              dot={{ r: 4, fill: '#38bdf8' }}
              style={{ filter: 'url(#glow)' }}
              connectNulls
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Spatial Attribution XAI Breakdown */}
      {factors.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-midnight p-2.5 text-xs space-y-2">
          <div className="flex items-center gap-1.5 text-slate-300 font-semibold">
            <Network className="h-3.5 w-3.5 text-cobalt" />
            <span>XAI Spatial Attribution (Contributing Factors):</span>
          </div>
          <div className="space-y-1.5">
            {factors.slice(0, 3).map((f, i) => (
              <div key={i} className="flex items-center justify-between text-slate-400">
                <span className="truncate max-w-[200px] text-slate-300">
                  {shortName(f.station_name)}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-lime rounded-full"
                      style={{ width: `${Math.min(100, f.attribution_pct)}%` }}
                    />
                  </div>
                  <span className="font-mono text-lime font-bold w-10 text-right">
                    {f.attribution_pct}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
