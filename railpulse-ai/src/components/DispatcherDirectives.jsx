import { ShieldAlert, Users, ArrowRight, ShieldCheck, Clock, Zap } from 'lucide-react'
import { shortName } from '../api'

export default function DispatcherDirectives({
  directives = [],
  connections = [],
  onSelectStation,
  onSimulateDirective,
  activeSimulatedDirective = null,
}) {
  // Extract high-visibility recommendation (prioritize HOLD or REROUTE, else first directive)
  const topDirective =
    directives.find((d) => d.action === 'HOLD' || d.action === 'REROUTE') ||
    directives[0] ||
    null

  return (
    <div className="flex flex-col gap-3">
      {/* 2. High-Visibility Operational Banner */}
      {topDirective ? (
        <div className="relative overflow-hidden rounded-lg border border-lime bg-slate2 p-3.5 shadow-[0_0_15px_rgba(212,255,36,0.15)]">
          {/* Top row: Glowing shield icon & Title */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="relative flex h-5 w-5 items-center justify-center">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-lime/40 opacity-75" />
                <ShieldCheck className="relative h-4 w-4 text-lime drop-shadow-[0_0_6px_#d4ff24]" />
              </span>
              <span className="text-xs font-bold uppercase tracking-wider text-lime">
                AI DISPATCHER RECOMMENDATION
              </span>
            </div>

            {/* Action Badge */}
            <span
              className={`px-2.5 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase ${
                topDirective.action === 'HOLD'
                  ? 'bg-lime/20 text-lime border border-lime/50 shadow-[0_0_8px_rgba(212,255,36,0.3)]'
                  : topDirective.action === 'REROUTE'
                  ? 'bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/50'
                  : 'bg-cobalt/20 text-slate-200 border border-cobalt/50'
              }`}
            >
              {topDirective.action}
            </span>
          </div>

          {/* Target Train & Directive details */}
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between text-slate-200">
              <span className="text-slate-400">Target Service:</span>
              <span className="font-semibold text-white truncate max-w-[210px]" title={topDirective.target_train}>
                {topDirective.target_train}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800">
              <div className="rounded bg-midnight p-2 border border-slate-800">
                <span className="text-[10px] text-slate-400 block mb-0.5">Hold Duration</span>
                <span className="font-mono text-sm font-bold text-lime flex items-center gap-1">
                  <Clock className="h-3 w-3 text-lime" />
                  +{topDirective.hold_duration_min.toFixed(1)}m
                </span>
              </div>
              <div className="rounded bg-midnight p-2 border border-slate-800">
                <span className="text-[10px] text-slate-400 block mb-0.5">Connection Saved</span>
                <span className="font-mono text-sm font-bold text-slate-100">
                  {topDirective.connection_preservation_pct.toFixed(1)}%
                </span>
              </div>
            </div>

            {topDirective.rationale && (
              <p className="mt-1 text-[11px] text-slate-400 italic leading-snug">
                "{topDirective.rationale}"
              </p>
            )}
          </div>

          {/* 3. Simulate Directive Action Button */}
          <div className="mt-3 flex items-center justify-between gap-2 border-t border-slate-800/80 pt-2.5">
            <span className="text-[10px] text-slate-400">
              Secondary cascade: <span className="text-slate-300 font-mono">+{topDirective.secondary_delay_impact_min.toFixed(1)}m</span>
            </span>
            <button
              onClick={() => onSimulateDirective && onSimulateDirective(topDirective)}
              className={`flex items-center gap-1.5 rounded px-3 py-1 text-xs font-bold transition-all shadow-sm ${
                activeSimulatedDirective === topDirective
                  ? 'bg-lime text-midnight border border-lime shadow-[0_0_10px_#d4ff24]'
                  : 'bg-lime/10 text-lime border border-lime/60 hover:bg-lime hover:text-midnight'
              }`}
            >
              <Zap className="h-3 w-3" />
              {activeSimulatedDirective === topDirective ? 'Mitigation Active' : 'Simulate Directive'}
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-midnight p-3 text-xs text-slate-400">
          <ShieldCheck className="h-4 w-4 text-slate-500" />
          <span>Listening for sub-second dispatcher mitigation directives...</span>
        </div>
      )}

      {/* Monitored Passenger Transfers List */}
      <div className="flex max-h-56 flex-col gap-2 overflow-y-auto pr-1">
        {connections.length === 0 ? (
          <div className="rounded-lg border border-slate-800 bg-midnight p-3 text-xs text-slate-400">
            No vulnerable transfer connections detected.
          </div>
        ) : (
          connections.map((c, idx) => {
            const isMissed = c.connection_status === 'MISSED'
            const riskPct = (c.missed_connection_probability * 100).toFixed(1)
            const bufferText =
              c.effective_transfer_buffer_min >= 0
                ? `+${c.effective_transfer_buffer_min.toFixed(2)}m buffer`
                : `${c.effective_transfer_buffer_min.toFixed(2)}m buffer`

            return (
              <div
                key={`${c.eva_id}-${idx}`}
                onClick={() => onSelectStation && onSelectStation(c.eva_id)}
                className="group flex flex-col gap-1.5 rounded-lg border border-cobalt/40 bg-midnight p-2.5 transition-colors hover:border-lime/60 hover:bg-[#0f172a] cursor-pointer"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-200 text-xs flex items-center gap-1.5">
                    <Users className="h-3 w-3 text-cobalt flex-none group-hover:text-lime" />
                    {shortName(c.station_name)}
                  </span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider uppercase ${
                      isMissed
                        ? 'bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/40'
                        : 'bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/40'
                    }`}
                  >
                    {c.connection_status}
                  </span>
                </div>

                <div className="flex items-center gap-1 text-[11px] text-slate-400">
                  <span className="truncate max-w-[120px] text-slate-300" title={c.inbound_service}>
                    {c.inbound_service.split('(')[0].trim()}
                  </span>
                  <ArrowRight className="h-2.5 w-2.5 text-slate-500 flex-none" />
                  <span className="truncate max-w-[120px] text-slate-300" title={c.connecting_service}>
                    {c.connecting_service.split('(')[0].trim()}
                  </span>
                </div>

                <div className="mt-0.5 flex items-center justify-between border-t border-slate-800/80 pt-1 text-[11px]">
                  <span className="font-mono text-lime font-bold">{bufferText}</span>
                  <span
                    className="font-mono font-bold"
                    style={{ color: isMissed ? '#EF4444' : '#F59E0B' }}
                  >
                    {riskPct}% risk
                  </span>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

