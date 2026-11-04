import { ShieldAlert, Users, ArrowRight } from 'lucide-react'
import { shortName } from '../api'

export default function PassengerImpactCard({ connections = [], onSelectStation }) {
  if (!connections || connections.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-midnight p-3 text-xs text-slate-400">
        No vulnerable transfer connections detected.
      </div>
    )
  }

  return (
    <div className="flex max-h-72 flex-col gap-2.5 overflow-y-auto pr-1">
      {connections.map((c, idx) => {
        const isMissed = c.connection_status === 'MISSED'
        const riskPct = (c.missed_connection_probability * 100).toFixed(1)
        const bufferText = c.effective_transfer_buffer_min >= 0
          ? `+${c.effective_transfer_buffer_min.toFixed(2)}m buffer`
          : `${c.effective_transfer_buffer_min.toFixed(2)}m buffer`

        return (
          <div
            key={`${c.eva_id}-${idx}`}
            onClick={() => onSelectStation && onSelectStation(c.eva_id)}
            className="group flex flex-col gap-1.5 rounded-lg border border-cobalt/40 bg-midnight p-3 transition-colors hover:border-lime/60 hover:bg-[#0f172a] cursor-pointer"
          >
            {/* Top row: Station & Status pill */}
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-slate-200 text-sm flex items-center gap-1.5">
                <Users className="h-3.5 w-3.5 text-cobalt flex-none group-hover:text-lime" />
                {shortName(c.station_name)}
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase ${
                  isMissed
                    ? 'bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/40'
                    : 'bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/40'
                }`}
              >
                {c.connection_status}
              </span>
            </div>

            {/* Inbound -> Connecting route */}
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className="truncate max-w-[140px] text-slate-300" title={c.inbound_service}>
                {c.inbound_service.split('(')[0].trim()}
              </span>
              <ArrowRight className="h-3 w-3 text-slate-500 flex-none" />
              <span className="truncate max-w-[140px] text-slate-300" title={c.connecting_service}>
                {c.connecting_service.split('(')[0].trim()}
              </span>
            </div>

            {/* Bottom row: Effective buffer and Risk Gauge */}
            <div className="mt-1 flex items-center justify-between border-t border-slate-800/80 pt-1.5 text-xs">
              <span className="font-mono text-lime font-bold">
                {bufferText}
              </span>
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-slate-400">Missed Risk:</span>
                <span
                  className="font-mono font-bold"
                  style={{ color: isMissed ? '#EF4444' : '#F59E0B' }}
                >
                  {riskPct}%
                </span>
              </div>
            </div>

            {/* Linear Risk Meter */}
            <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, Math.max(5, c.missed_connection_probability * 100))}%`,
                  backgroundColor: isMissed ? '#EF4444' : '#F59E0B',
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

