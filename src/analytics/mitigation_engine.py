"""Dispatcher Decision Support System (DSS) & Mitigation Engine.

Implements PRD Module 07:
- Evaluates real-time passenger interchange risks across corridor hub stations.
- Solves quantitative hold-delay trade-offs (holding connecting trains delta_t in [2, 8] min).
- Formulates actionable dispatching directives:
  - HOLD: Minor delay where a short hold preserves transfer without cascading network failure.
  - DISPATCH_ON_TIME: Transfer is either safe or holding causes unacceptable secondary delay.
  - REROUTE: Massive delay (>15m) where holding is unviable; redirect passengers to alternate lines.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.analytics.passenger_impact import DEFAULT_CORRIDOR_SCHEDULED_TRANSFERS


class DispatcherDecisionSupportEngine:
    """Evaluates connection preservation and secondary delay impacts to recommend mitigation actions."""

    def __init__(self, max_allowable_hold_min: float = 8.0, min_hold_threshold_min: float = 2.0):
        self.max_hold_min = float(max_allowable_hold_min)
        self.min_hold_min = float(min_hold_threshold_min)

    def evaluate_transfer_mitigation(
        self,
        station_id: int,
        predicted_delays: Dict[str, float] | List[float],
        connection_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate optimal mitigation directives for a station's connecting services.

        Parameters
        ----------
        station_id:
            Target station EVA number.
        predicted_delays:
            Dictionary (delay_15m, delay_30m, delay_60m) or list of predicted delays.
        connection_data:
            Optional itemized passenger connections for the station. Defaults to standard corridor schedule.

        Returns
        -------
        List of structured mitigation directives:
            [
              {
                "action": "HOLD" | "DISPATCH_ON_TIME" | "REROUTE",
                "target_train": str,
                "inbound_train": str,
                "hold_duration_min": float,
                "connection_preservation_pct": float,
                "secondary_delay_impact_min": float,
                "rationale": str,
              }, ...
            ]
        """
        if isinstance(predicted_delays, dict):
            current_delay = float(predicted_delays.get("delay_15m", 0.0))
        elif isinstance(predicted_delays, (list, tuple)) and len(predicted_delays) > 0:
            current_delay = float(predicted_delays[0])
        else:
            current_delay = 0.0

        if connection_data is None:
            connection_data = [
                c for c in DEFAULT_CORRIDOR_SCHEDULED_TRANSFERS if c.get("eva_id") == station_id
            ]
            if not connection_data:
                # If no predefined transfer exists for this station, synthesize a representative interchange
                connection_data = [
                    {
                        "eva_id": station_id,
                        "station_name": f"Station {station_id}",
                        "inbound_service": "Regional Feeder",
                        "connecting_service": "InterCity Express",
                        "scheduled_transfer_margin_min": 7.0,
                        "platform_change": "Track 2 -> Track 3",
                    }
                ]

        directives: List[Dict[str, Any]] = []

        for conn in connection_data:
            sched_margin = float(conn.get("scheduled_transfer_margin_min", 7.0))
            effective_buffer = sched_margin - current_delay
            connecting_train = conn.get("connecting_service", "Connecting Service")
            inbound_train = conn.get("inbound_service", "Inbound Service")

            # Scenario 1: Safe Buffer (>= 5 min) -> Dispatch on schedule
            if effective_buffer >= 5.0:
                directives.append({
                    "action": "DISPATCH_ON_TIME",
                    "target_train": connecting_train,
                    "inbound_train": inbound_train,
                    "hold_duration_min": 0.0,
                    "connection_preservation_pct": 99.5,
                    "secondary_delay_impact_min": 0.0,
                    "rationale": f"Interchange buffer (+{effective_buffer:.1f}m) exceeds safe 5m threshold. Normal departure approved.",
                })

            # Scenario 2: Moderate Delay / Vulnerable (deficit between -1m and -8m) -> Hold Trade-off
            elif effective_buffer >= -self.max_hold_min:
                # Needed hold duration to restore minimum 2m passenger walking buffer
                deficit = -effective_buffer + 2.0
                hold_min = round(min(self.max_hold_min, max(self.min_hold_min, math.ceil(deficit))), 1)

                # Adjusted buffer with hold
                adjusted_buffer = effective_buffer + hold_min
                # Sigmoid preservation probability
                preservation_prob = 1.0 / (1.0 + math.exp(-0.75 * adjusted_buffer))
                preservation_pct = round(preservation_prob * 100.0, 1)

                # Secondary delay propagation onto downstream block sections (~60% attenuation)
                secondary_delay = round(hold_min * 0.65, 2)

                directives.append({
                    "action": "HOLD",
                    "target_train": connecting_train,
                    "inbound_train": inbound_train,
                    "hold_duration_min": hold_min,
                    "connection_preservation_pct": preservation_pct,
                    "secondary_delay_impact_min": secondary_delay,
                    "rationale": (
                        f"Hold {connecting_train} by +{hold_min}m to secure {preservation_pct}% connection probability "
                        f"with bounded secondary corridor impact (+{secondary_delay}m)."
                    ),
                })

            # Scenario 3: Severe Delay (>15 min) -> Reroute
            else:
                directives.append({
                    "action": "REROUTE",
                    "target_train": connecting_train,
                    "inbound_train": inbound_train,
                    "hold_duration_min": 0.0,
                    "connection_preservation_pct": 5.0,
                    "secondary_delay_impact_min": 0.0,
                    "rationale": (
                        f"Inbound delay (+{current_delay:.1f}m) renders holding {connecting_train} unviable. "
                        f"Instruct station staff to reroute passengers to subsequent regional hourly departures."
                    ),
                })

        return directives

