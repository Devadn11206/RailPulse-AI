import { useEffect, useState, useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, Tooltip, useMap } from 'react-leaflet'
import { delayColor, shortName, centrality } from '../api'

function FlyTo({ target }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo([target.lat, target.lon], 9, { duration: 0.8 })
  }, [target, map])
  return null
}

export default function CorridorMap({
  topo,
  preds,
  selectedId,
  onSelect,
  flyTarget,
  activeSimulation = null,
}) {
  const by = Object.fromEntries(topo.stations.map((s) => [s.eva_id, s]))
  const pr = Object.fromEntries(preds.map((p) => [p.eva_id, p]))
  const shockOriginEva = activeSimulation?.origin?.eva_id
  const downstreamMap = Object.fromEntries(
    (activeSimulation?.downstream_1hop_neighbors || []).map((n) => [n.eva_id, n])
  )

  // Extract edge attributions from the corridor predictions
  const attentionMap = useMemo(() => {
    const map = new Map()
    for (const p of preds) {
      if (p.edge_attributions && Array.isArray(p.edge_attributions)) {
        for (const attr of p.edge_attributions) {
          const key1 = `${attr.source_eva}-${attr.target_eva}`
          const key2 = `${attr.target_eva}-${attr.source_eva}`
          map.set(key1, attr.attention_weight)
          map.set(key2, attr.attention_weight)
        }
      }
    }
    return map
  }, [preds])

  // Dynamic animation loop for moving pulse dots along track edges
  const [pulsePhase, setPulsePhase] = useState(0)
  useEffect(() => {
    let animId
    let start = performance.now()
    const loop = (now) => {
      const elapsed = (now - start) / 3000 // 3.0s loop cycle
      setPulsePhase(elapsed % 1)
      animId = requestAnimationFrame(loop)
    }
    animId = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animId)
  }, [])

  return (
    <div className="relative h-[70vh] min-h-[460px] overflow-hidden rounded-lg border border-cobalt bg-midnight lg:h-full">
      <MapContainer center={[50.1109, 8.6821]} zoom={8} className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FlyTo target={flyTarget} />

        {/* Directed Track Topology Lines with GATv2 Attention Weights */}
        {topo.edges.map(([a, b], i) => {
          const A = by[a]
          const B = by[b]
          if (!A || !B) return null

          const predA = pr[a]
          const isHighDelay = (predA?.delay_15m || 0) > 5.0
          const isShockOriginEdge = a === shockOriginEva || b === shockOriginEva
          const isDownstreamEdge = Boolean(downstreamMap[a] || downstreamMap[b])

          // Retrieve GATv2 dynamic attention weight alpha_ij
          const rawAlpha = attentionMap.get(`${a}-${b}`)
          // Default synthetic attention if not explicitly returned: 0.55
          const alpha = typeof rawAlpha === 'number' ? rawAlpha : 0.55
          const isHighAttention = alpha > 0.75

          const dx = B.lon - A.lon
          const dy = B.lat - A.lat
          const l = Math.hypot(dx, dy) * 40
          const off = [(-dy / l) * 0.008, (dx / l) * 0.008]

          // Dynamic styling based on GATv2 attention
          const strokeOpacity = Math.min(1.0, Math.max(0.35, 0.3 + alpha * 0.7))
          let weight = isHighAttention ? 4.0 : 2.0
          let edgeColor = '#1558c9'
          let edgeClass = 'rail-edge'

          if (isShockOriginEdge) {
            edgeClass = 'rail-edge-shocked'
            edgeColor = '#d4ff24'
            weight = 4.0
          } else if (isHighDelay) {
            edgeClass = 'rail-edge-severe'
            edgeColor = '#EF4444'
            weight = 3.0
          } else if (isHighAttention) {
            edgeClass = 'rail-edge-high-attention'
            edgeColor = '#d4ff24'
          } else if (isDownstreamEdge) {
            edgeColor = '#F59E0B'
            weight = 2.5
          }

          return (
            <Polyline
              key={`edge-${i}`}
              positions={[
                [A.lat + off[0], A.lon + off[1]],
                [B.lat + off[0], B.lon + off[1]],
              ]}
              pathOptions={{
                color: edgeColor,
                weight: weight,
                opacity: strokeOpacity,
                className: edgeClass,
              }}
            >
              <Tooltip className="track-tooltip" sticky>
                <span>
                  Track: {shortName(A.name)} → {shortName(B.name)}
                  <br />
                  <strong className="text-lime">
                    GATv2 Influence (α = {alpha.toFixed(2)})
                  </strong>
                  {isHighAttention && ' — Critical Bottleneck'}
                </span>
              </Tooltip>
            </Polyline>
          )
        })}

        {/* Live Moving Pulse Dots Along Edges (Simulating Real-Time Train Flow) */}
        {topo.edges.map(([a, b], i) => {
          const A = by[a]
          const B = by[b]
          if (!A || !B) return null

          const predA = pr[a]
          const isHighDelay = (predA?.delay_15m || 0) > 5.0
          const isShockEdge = a === shockOriginEva || b === shockOriginEva

          // Compute moving position between Station A and Station B
          const edgePhase = (pulsePhase + i * 0.04) % 1
          const currLat = A.lat + (B.lat - A.lat) * edgePhase
          const currLon = A.lon + (B.lon - A.lon) * edgePhase

          const dotColor = isShockEdge ? '#d4ff24' : isHighDelay ? '#EF4444' : '#d4ff24'
          const dotClass = isHighDelay ? 'pulse-telemetry-dot-warn' : 'pulse-telemetry-dot'

          return (
            <CircleMarker
              key={`pulse-${i}`}
              center={[currLat, currLon]}
              radius={isShockEdge ? 3.5 : 2.5}
              pathOptions={{
                color: dotColor,
                fillColor: dotColor,
                fillOpacity: 0.9,
                weight: 1,
                className: dotClass,
              }}
            />
          )
        })}

        {/* Disruption Shock Ripple Rings on Epicenter Station */}
        {shockOriginEva && by[shockOriginEva] && (
          <>
            <CircleMarker
              center={[by[shockOriginEva].lat, by[shockOriginEva].lon]}
              radius={24}
              pathOptions={{
                color: '#d4ff24',
                weight: 2.5,
                fillColor: '#d4ff24',
                fillOpacity: 0.2,
                className: 'shock-wave-circle',
              }}
            />
            <CircleMarker
              center={[by[shockOriginEva].lat, by[shockOriginEva].lon]}
              radius={48}
              pathOptions={{
                color: '#EF4444',
                weight: 2,
                fillColor: '#EF4444',
                fillOpacity: 0.1,
                className: 'shock-wave-circle-outer',
              }}
            />
          </>
        )}

        {/* Station Node Markers */}
        {topo.stations.map((s) => {
          const p = pr[s.eva_id] || { delay_15m: 0.5, delay_30m: 0.7, delay_60m: 0.9 }
          const isShockOrigin = s.eva_id === shockOriginEva
          const downstreamInfo = downstreamMap[s.eva_id]
          const isDownstream = Boolean(downstreamInfo)

          let strokeColor = s.eva_id === selectedId ? '#d4ff24' : '#030407'
          let markerRadius = s.eva_id === selectedId ? 10 : topo.stations.length > 20 ? 6.5 : 9

          if (isShockOrigin) {
            strokeColor = '#d4ff24'
            markerRadius = 14
          } else if (isDownstream) {
            strokeColor = '#F59E0B'
            markerRadius = 11
          }

          return (
            <CircleMarker
              key={s.eva_id}
              center={[s.lat, s.lon]}
              radius={markerRadius}
              pathOptions={{
                color: strokeColor,
                weight: isShockOrigin || isDownstream ? 3.5 : 2.5,
                fillColor: isShockOrigin ? '#EF4444' : delayColor(p.delay_15m),
                fillOpacity: 1,
              }}
              eventHandlers={{ click: () => onSelect(s.eva_id) }}
            >
              <Popup className="rp-popup">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-base font-bold text-white">{shortName(s.name)}</h3>
                  {isShockOrigin && (
                    <span className="rounded bg-red-950/60 border border-red-500 px-1.5 py-0.5 text-[9px] font-bold text-red-400 uppercase">
                      Shock Epicenter
                    </span>
                  )}
                  {isDownstream && (
                    <span className="rounded bg-amber-950/60 border border-amber-500 px-1.5 py-0.5 text-[9px] font-bold text-amber-400 uppercase">
                      Cascading 1-Hop
                    </span>
                  )}
                </div>
                <p className="mb-2 text-xs text-slate-400">EVA ID {s.eva_id}</p>

                {downstreamInfo && (
                  <div className="mb-2 rounded bg-midnight/90 border border-amber-500/40 p-2 text-xs">
                    <span className="text-slate-400 block mb-0.5">Propagated Shock Impact:</span>
                    <b className="text-lime text-sm">
                      +{downstreamInfo.propagated_delta_15m.toFixed(2)}m (15m cascade)
                    </b>
                  </div>
                )}

                <Row
                  k="Node centrality score"
                  v={centrality(s.eva_id, topo.edges, topo.stations.length)}
                />
                <Row
                  k="Active track edges"
                  v={
                    topo.edges.filter((e) => e[0] === s.eva_id || e[1] === s.eva_id).length
                  }
                />
                {[
                  ['15m', 'delay_15m'],
                  ['30m', 'delay_30m'],
                  ['60m', 'delay_60m'],
                ].map(([l, k]) => (
                  <div
                    key={k}
                    className="flex justify-between border-b border-slate-800 py-1 text-sm"
                  >
                    <span>Forecast +{l}</span>
                    <b className="text-base text-lime">+{p[k].toFixed(2)}m</b>
                  </div>
                ))}
              </Popup>
            </CircleMarker>
          )
        })}
      </MapContainer>

      {/* Floating Map Legend & Control Center Overlay */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-[500] flex flex-wrap items-center gap-3 rounded-lg border border-cobalt/40 bg-midnight/90 px-3 py-1.5 text-xs text-slate-300 backdrop-blur-sm shadow-lg">
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-lime opacity-75" />
            <span className="relative h-2 w-2 rounded-full bg-lime" />
          </span>
          <span className="font-semibold text-slate-200">GATv2 Track Attention:</span>
        </div>
        <span className="flex items-center gap-1">
          <span className="h-0.5 w-4 bg-[#1558c9] opacity-50" /> α &lt; 0.5 (Low)
        </span>
        <span className="flex items-center gap-1">
          <span className="h-1 w-4 bg-lime shadow-[0_0_6px_#d4ff24]" /> α &gt; 0.75 (Critical Bottleneck)
        </span>
      </div>
    </div>
  )
}

const Row = ({ k, v }) => (
  <div className="flex justify-between border-b border-slate-800 py-1 text-sm">
    <span>{k}</span>
    <b>{v}</b>
  </div>
)
