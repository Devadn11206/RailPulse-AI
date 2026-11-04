import { useEffect, useMemo, useState, useRef } from 'react'
import { api, delayColor, shortName } from './api'
import CorridorHeader from './components/CorridorHeader'
import CorridorMap from './components/CorridorMap'
import StationList from './components/StationList'
import DelayForecastChart from './components/DelayForecastChart'
import DispatcherDirectives from './components/DispatcherDirectives'
import ScenarioSimulator from './components/ScenarioSimulator'
import LoginModal from './components/LoginModal'

const Panel = ({ title, badge, children }) => (
  <section className="rounded-lg border border-cobalt bg-slate2 p-4">
    <div className="mb-3 flex items-center justify-between">
      <h2 className="font-semibold text-slate-100">{title}</h2>
      {badge && <span className="text-xs text-slate-400">{badge}</span>}
    </div>
    {children}
  </section>
)

export default function App() {
  const [health, setHealth] = useState(null)
  const [topo, setTopo] = useState(null)
  const [preds, setPreds] = useState([])
  const [connections, setConnections] = useState([])
  const [directives, setDirectives] = useState([])
  const [simulatedDirective, setSimulatedDirective] = useState(null)
  const [selectedId, setSelectedId] = useState(8000240)
  const [scale, setScale] = useState('core') // 'core' (10) or 'regional' (52)
  const [query, setQuery] = useState('')
  const [flyTarget, setFlyTarget] = useState(null)

  // Disruption Simulator State
  const [isSimulatorOpen, setIsSimulatorOpen] = useState(false)
  const [activeSimulation, setActiveSimulation] = useState(null)

  // Auth & RBAC State
  const [isLoginOpen, setIsLoginOpen] = useState(false)
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const stored = localStorage.getItem('railpulse_user')
      return stored ? JSON.parse(stored) : null
    } catch {
      return null
    }
  })

  const handleLogout = () => {
    localStorage.removeItem('railpulse_token')
    localStorage.removeItem('railpulse_user')
    setCurrentUser(null)
    setIsLoginOpen(true)
  }

  const handleLoginSuccess = (userObj) => {
    setCurrentUser(userObj)
  }

  const handleOpenSimulator = () => {
    if (!currentUser) {
      setIsLoginOpen(true)
      return
    }
    setIsSimulatorOpen(true)
  }

  // 1. WebSocket Telemetry Feed Listener (ws://localhost:8001/ws/telemetry)
  useEffect(() => {
    let ws = null
    let reconnectTimer = null

    const connectWs = () => {
      try {
        ws = new WebSocket('ws://localhost:8001/ws/telemetry')

        ws.onopen = () => {
          console.log('[WebSocket] Connected to /ws/telemetry feed')
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'TELEMETRY_SNAPSHOT' && Array.isArray(data.mitigation_directives)) {
              setDirectives(data.mitigation_directives)
            }
          } catch (err) {
            console.error('[WebSocket] Failed parsing telemetry snapshot:', err)
          }
        }

        ws.onerror = (err) => {
          console.warn('[WebSocket] Feed connection warning:', err)
        }

        ws.onclose = () => {
          // Reconnect after 3 seconds
          reconnectTimer = setTimeout(connectWs, 3000)
        }
      } catch (e) {
        reconnectTimer = setTimeout(connectWs, 3000)
      }
    }

    connectWs()

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws) ws.close()
    }
  }, [])

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [p, c] = await Promise.all([
          api.predict(),
          api.passengerConnections ? api.passengerConnections() : Promise.resolve([]),
        ])
        if (alive) {
          setPreds(p)
          setConnections(c)
          // Fallback directives extraction from passenger connections if WebSocket hasn't delivered yet
          if (c.length > 0 && directives.length === 0) {
            const extracted = []
            for (const item of c) {
              if (item.mitigation_directives && Array.isArray(item.mitigation_directives)) {
                extracted.push(...item.mitigation_directives)
              }
            }
            if (extracted.length > 0) {
              setDirectives(extracted)
            }
          }
        }
      } catch (e) {
        console.error(e)
      }
    }
    Promise.all([api.health(), api.topology(scale)])
      .then(([h, t]) => {
        if (alive) {
          setHealth(h)
          setTopo(t)
        }
      })
      .catch(console.error)
    load()
    const id = setInterval(load, 6000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [scale])

  useEffect(() => {
    const q = query.trim().toLowerCase()
    const hit = q && topo?.stations.find((s) => s.name.toLowerCase().includes(q))
    if (hit) {
      setSelectedId(hit.eva_id)
      setFlyTarget({ ...hit })
    }
  }, [query, topo])

  const [selectedStationDetail, setSelectedStationDetail] = useState(null)

  // Fetch full XAI prediction details when selectedId changes
  useEffect(() => {
    let alive = true
    if (!selectedId) return
    if (api.predictStation) {
      api.predictStation(selectedId).then((res) => {
        if (alive && res) {
          setSelectedStationDetail(res)
          if (res.mitigation_directives && Array.isArray(res.mitigation_directives) && res.mitigation_directives.length > 0) {
            setDirectives((prev) => [...res.mitigation_directives, ...prev.filter(p => p.target_train !== res.mitigation_directives[0].target_train)])
          }
        }
      })
    }
    return () => {
      alive = false
    }
  }, [selectedId])

  const stats = useMemo(() => {
    if (!preds.length) return null
    const avg = preds.reduce((a, p) => a + p.delay_15m, 0) / preds.length
    const max = preds.reduce((a, b) => (a.delay_15m > b.delay_15m ? a : b))
    return { avg, max, stability: Math.max(0, 100 - avg * 9 - (max.delay_15m > 5 ? 10 : 0)) }
  }, [preds])

  const worst = useMemo(() => [...preds].sort((a, b) => b.delay_60m - a.delay_60m)[0], [preds])
  const sel = preds.find((p) => p.eva_id === selectedId)

  if (!topo || !stats) return <div className="p-8 text-slate-400">Loading corridor…</div>

  const kpis = [
    ['Average corridor delay', `+${stats.avg.toFixed(2)}m`, delayColor(stats.avg)],
    ['Max station delay', `+${stats.max.delay_15m.toFixed(2)}m`, '#d4ff24'],
    ['Network stability index', `${stats.stability.toFixed(1)}%`, '#e6ebf7'],
    ['ST-GNN vs baseline R²', '+0.1983', '#d4ff24'],
  ]

  return (
    <div className="flex min-h-screen flex-col bg-midnight">
      <CorridorHeader
        health={health}
        worst={worst}
        query={query}
        setQuery={setQuery}
        stations={topo.stations}
        onOpenSimulator={handleOpenSimulator}
        isSimActive={Boolean(activeSimulation)}
        currentUser={currentUser}
        onOpenLogin={() => setIsLoginOpen(true)}
        onLogout={handleLogout}
        scale={scale}
        onScaleChange={(newScale) => setScale(newScale)}
      />

      <main className="grid flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(360px,1fr)]">
        <CorridorMap
          topo={topo}
          preds={preds}
          selectedId={selectedId}
          onSelect={setSelectedId}
          flyTarget={flyTarget}
          activeSimulation={activeSimulation}
        />

        <aside className="flex min-w-0 flex-col gap-4 overflow-y-auto max-h-[calc(100vh-80px)] pr-0.5">
          <Panel title="Corridor health">
            <div className="grid grid-cols-2 gap-3">
              {kpis.map(([k, v, c]) => (
                <div key={k} className="rounded-lg border border-slate-800 bg-midnight p-3">
                  <small className="block text-slate-500">{k}</small>
                  <strong className="text-2xl" style={{ color: c }}>
                    {v}
                  </strong>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title={`Delay forecast & XAI: ${sel ? shortName(sel.name) : ''}`}>
            {sel && (
              <DelayForecastChart
                pred={selectedStationDetail || sel}
                simulatedDirective={simulatedDirective}
                onClearMitigation={() => setSimulatedDirective(null)}
              />
            )}
          </Panel>

          <Panel
            title="Dispatcher DSS & Passenger Risk"
            badge={
              <span className="font-mono text-xs text-lime">
                {directives.length > 0 ? `${directives.length} active directives` : `${connections.length} monitored`}
              </span>
            }
          >
            <DispatcherDirectives
              directives={directives}
              connections={connections}
              activeSimulatedDirective={simulatedDirective}
              onSimulateDirective={(d) => {
                if (simulatedDirective === d) {
                  setSimulatedDirective(null)
                } else {
                  setSimulatedDirective(d)
                }
              }}
              onSelectStation={(evaId) => {
                setSelectedId(evaId)
                const station = topo.stations.find((s) => s.eva_id === evaId)
                if (station) setFlyTarget({ ...station })
              }}
            />
          </Panel>

          <Panel title="Stations by predicted delay">
            <StationList
              preds={preds}
              selectedId={selectedId}
              onSelect={(id) => {
                setSelectedId(id)
                setFlyTarget({ ...topo.stations.find((s) => s.eva_id === id) })
              }}
            />
          </Panel>
        </aside>
      </main>

      {/* Disruption Simulator Modal */}
      <ScenarioSimulator
        isOpen={isSimulatorOpen}
        onClose={() => setIsSimulatorOpen(false)}
        stations={topo.stations}
        activeSimulation={activeSimulation}
        onApplySimulation={(res) => setActiveSimulation(res)}
        onResetSimulation={() => setActiveSimulation(null)}
      />

      {/* JWT Authentication & RBAC Login Modal */}
      <LoginModal
        isOpen={isLoginOpen}
        onClose={() => setIsLoginOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />
    </div>
  )
}
