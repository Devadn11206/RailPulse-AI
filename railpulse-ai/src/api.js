// Toggle: set VITE_USE_REAL_API=true (e.g. in .env) to call the FastAPI backend.
export const USE_REAL_API = import.meta.env.VITE_USE_REAL_API === 'true'
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001'

export const getAuthHeaders = () => {
  const token = localStorage.getItem('railpulse_token')
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

export const STATIONS = [
  { eva_id: 8000105, name: 'Frankfurt(Main)Hbf', lat: 50.1071, lon: 8.6638 },
  { eva_id: 8000240, name: 'Mainz Hbf', lat: 50.0013, lon: 8.2589 },
  { eva_id: 8000244, name: 'Mannheim Hbf', lat: 49.4797, lon: 8.4698 },
  { eva_id: 8000150, name: 'Hanau Hbf', lat: 50.1211, lon: 8.9296 },
  { eva_id: 8000068, name: 'Darmstadt Hbf', lat: 49.8725, lon: 8.6313 },
  { eva_id: 8000250, name: 'Wiesbaden Hbf', lat: 50.0712, lon: 8.2435 },
  { eva_id: 8000013, name: 'Aschaffenburg Hbf', lat: 49.9803, lon: 9.1434 },
  { eva_id: 8070003, name: 'Frankfurt Flughafen Fernbf', lat: 50.0531, lon: 8.5707 },
  { eva_id: 8000124, name: 'Gießen Hbf', lat: 50.5793, lon: 8.6644 },
  { eva_id: 8000257, name: 'Worms Hbf', lat: 49.6347, lon: 8.3568 },
]
const PAIRS = [[0,7],[7,1],[1,5],[5,7],[7,4],[0,4],[0,3],[3,6],[0,8],[4,2],[4,9],[9,2],[1,9],[4,6],[3,8],[1,4],[0,6]]
const MOCK_DELAYS = [[.005,.006,.006],[3.46,3.65,3.55],[2.4,2.7,2.5],[1.1,1.3,1.2],[1.9,2.2,2.1],[2.1,2.3,2.2],[.8,.9,.9],[1.4,1.6,1.5],[.4,.5,.5],[2.8,3.1,2.9]]
const j = (v) => Math.max(0, +(v + (Math.random() - 0.5) * 0.3).toFixed(3))

const MOCK_CONNECTIONS = [
  {
    eva_id: 8000240,
    station_name: 'Mainz Hbf',
    inbound_service: 'RE 3 (Saarbrücken -> Frankfurt)',
    connecting_service: 'ICE 26 (Hamburg -> Wien)',
    platform_change: 'Track 2 -> Track 5',
    incoming_delay_min: 3.46,
    scheduled_transfer_margin_min: 7.0,
    effective_transfer_buffer_min: 3.54,
    connection_status: 'VULNERABLE',
    missed_connection_probability: 0.3035,
  },
  {
    eva_id: 8000105,
    station_name: 'Frankfurt Hbf',
    inbound_service: 'ICE 1557 (Dresden -> Frankfurt)',
    connecting_service: 'ICE 27 (Frankfurt -> Passau)',
    platform_change: 'Track 7 -> Track 12',
    incoming_delay_min: 5.5,
    scheduled_transfer_margin_min: 9.0,
    effective_transfer_buffer_min: 3.5,
    connection_status: 'VULNERABLE',
    missed_connection_probability: 0.31,
  },
  {
    eva_id: 8000105,
    station_name: 'Frankfurt Hbf',
    inbound_service: 'RE 50 (Fulda -> Frankfurt)',
    connecting_service: 'S-Bahn S8 (Airport)',
    platform_change: 'Track 102 -> Track 104',
    incoming_delay_min: 8.2,
    scheduled_transfer_margin_min: 6.0,
    effective_transfer_buffer_min: -2.2,
    connection_status: 'MISSED',
    missed_connection_probability: 0.9772,
  },
]

const mock = {
  health: async () => ({ status: 'ok', model_loaded: true, gpu: true, vram: '0.21 MB' }),
  topology: async () => ({
    stations: STATIONS,
    edges: PAIRS.flatMap(([a, b]) => [[STATIONS[a].eva_id, STATIONS[b].eva_id], [STATIONS[b].eva_id, STATIONS[a].eva_id]]),
  }),
  predict: async () => STATIONS.map((s, i) => ({ eva_id: s.eva_id, name: s.name,
    delay_15m: j(MOCK_DELAYS[i][0]), delay_30m: j(MOCK_DELAYS[i][1]), delay_60m: j(MOCK_DELAYS[i][2]) })),
  passengerConnections: async () => MOCK_CONNECTIONS,
  simulateDisruption: async ({ eva_id, injected_delay_min }) => {
    const shock = Number(injected_delay_min) || 25.0
    return {
      origin: {
        eva_id: Number(eva_id),
        station_name: STATIONS.find((s) => s.eva_id === Number(eva_id))?.name || 'Frankfurt Hbf',
        injected_shock_min: shock,
        baseline_forecast: [0.01, 0.01, 0.01],
        shocked_forecast: [+(shock * 0.28).toFixed(2), +(shock * 0.27).toFixed(2), +(shock * 0.25).toFixed(2)],
        delta_forecast: [+(shock * 0.28).toFixed(2), +(shock * 0.27).toFixed(2), +(shock * 0.25).toFixed(2)],
      },
      downstream_1hop_neighbors: [
        {
          eva_id: 8000244,
          station_name: 'Mannheim Hbf',
          hop_distance: 1,
          baseline_15m: 0.01,
          shocked_15m: 2.11,
          propagated_delta_15m: 2.10,
          propagated_delta_30m: 1.95,
          propagated_delta_60m: 1.80,
        },
        {
          eva_id: 8000240,
          station_name: 'Mainz Hbf',
          hop_distance: 1,
          baseline_15m: 3.46,
          shocked_15m: 5.12,
          propagated_delta_15m: 1.66,
          propagated_delta_30m: 1.54,
          propagated_delta_60m: 1.40,
        },
      ],
      network_impact: {
        total_propagated_delay_15m: +(shock * 0.32).toFixed(2),
        most_impacted_downstream: {
          eva_id: 8000244,
          station_name: 'Mannheim Hbf',
          propagated_delta_15m: 2.10,
        },
      },
    }
  },
  login: async ({ username, password }) => {
    if (username === 'admin' && password === 'admin123') {
      return { access_token: 'mock-admin-token-xyz', role: 'admin' }
    } else if (username === 'analyst' && password === 'analyst123') {
      return { access_token: 'mock-analyst-token-xyz', role: 'analyst' }
    }
    throw new Error('Invalid credentials')
  },
}

const real = {
  health: () => fetch(`${BASE}/health`, { headers: getAuthHeaders() }).then((r) => r.json()).catch(() => mock.health()),

  topology: (scale = 'core') =>
    fetch(`${BASE}/api/v1/topology?scale=${scale}`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((t) => ({
        stations: t.stations,
        edges: t.edges.map((e) => [e.source, e.target]),
      }))
      .catch(() => mock.topology()),

  predict: () =>
    fetch(`${BASE}/api/v1/predict/corridor`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((arr) =>
        arr.map((p) => ({
          eva_id: p.eva_id,
          name: p.station_name,
          delay_15m: p.delays[0],
          delay_30m: p.delays[1],
          delay_60m: p.delays[2],
          confidence_interval_min: p.confidence_interval_min,
          margin_min: p.margin_min,
          contributing_factors: p.contributing_factors,
          edge_attributions: p.edge_attributions,
        }))
      )
      .catch(() => mock.predict()),

  predictStation: (eva_id) =>
    fetch(`${BASE}/api/v1/predict/station`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ eva_id: Number(eva_id) }),
    })
      .then((r) => r.json())
      .then((p) => ({
        eva_id: p.eva_id,
        name: p.station_name,
        delay_15m: p.delays[0],
        delay_30m: p.delays[1],
        delay_60m: p.delays[2],
        confidence_interval_min: p.confidence_interval_min,
        margin_min: p.margin_min,
        contributing_factors: p.contributing_factors,
        edge_attributions: p.edge_attributions,
      }))
      .catch(() => null),

  passengerConnections: () =>
    fetch(`${BASE}/api/v1/passenger/connections`, { headers: getAuthHeaders() })
      .then((r) => {
        if (!r.ok) throw new Error('API error')
        return r.json()
      })
      .catch(() => mock.passengerConnections()),

  simulateDisruption: (payload) =>
    fetch(`${BASE}/api/v1/simulate/disruption`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(payload),
    })
      .then((r) => {
        if (!r.ok) throw new Error('API error')
        return r.json()
      })
      .catch(() => mock.simulateDisruption(payload)),

  login: (credentials) =>
    fetch(`${BASE}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    }).then(async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || 'Login failed')
      }
      return r.json()
    }),
}

export const api = USE_REAL_API ? real : mock

export const delayColor = (d) => (d < 2 ? '#10B981' : d <= 5 ? '#F59E0B' : '#EF4444')
export const shortName = (n) => n.replace('(Main)', '').replace(' Fernbf', ' Airport')
export const centrality = (id, edges, n) => (edges.filter((e) => e[0] === id).length / (n - 1)).toFixed(3)
