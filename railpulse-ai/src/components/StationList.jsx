import { delayColor, shortName } from '../api'

export default function StationList({ preds, selectedId, onSelect }) {
  const sorted = [...preds].sort((a, b) => b.delay_15m - a.delay_15m)
  return (
    <div className="flex max-h-72 flex-col gap-2 overflow-y-auto">
      {sorted.map((p) => (
        <button key={p.eva_id} onClick={() => onSelect(p.eva_id)}
          className={`flex w-full items-center gap-3 rounded-lg border bg-slate2 px-3 py-2 text-left transition-colors hover:bg-[#0f1830] ${p.eva_id === selectedId ? 'border-lime' : 'border-cobalt'}`}>
          <i className="h-2.5 w-2.5 flex-none rounded-full" style={{ background: delayColor(p.delay_15m) }} />
          <span className="flex-1">{shortName(p.name)}</span>
          <b className="text-lime">+{p.delay_15m.toFixed(2)}m</b>
        </button>))}
    </div>)
}
