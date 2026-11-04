"""Passenger Transfer Impact & Connection Risk Intelligence Engine.

Implements PRD Module 05: Evaluates transfer connection vulnerability
across railway network stations based on predicted incoming delays and
scheduled interchange buffers using sigmoid probability calibration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Standard typical high-volume interchange connections across the Frankfurt corridor
# with realistic scheduled transfer margins (in minutes) and connecting train IDs.
DEFAULT_CORRIDOR_SCHEDULED_TRANSFERS: List[Dict[str, Any]] = [
    {
        "eva_id": 8000105,
        "station_name": "Frankfurt Hbf",
        "inbound_service": "ICE 1557 (Dresden -> Frankfurt)",
        "connecting_service": "ICE 27 (Frankfurt -> Passau)",
        "scheduled_transfer_margin_min": 9.0,
        "platform_change": "Track 7 -> Track 12",
    },
    {
        "eva_id": 8000105,
        "station_name": "Frankfurt Hbf",
        "inbound_service": "RE 50 (Fulda -> Frankfurt)",
        "connecting_service": "S-Bahn S8 (Airport)",
        "scheduled_transfer_margin_min": 6.0,
        "platform_change": "Track 102 -> Track 104",
    },
    {
        "eva_id": 8000244,
        "station_name": "Mannheim Hbf",
        "inbound_service": "ICE 612 (München -> Dortmund)",
        "connecting_service": "TGV 9556 (Frankfurt -> Paris)",
        "scheduled_transfer_margin_min": 8.0,
        "platform_change": "Track 4 -> Track 5",
    },
    {
        "eva_id": 8000240,
        "station_name": "Mainz Hbf",
        "inbound_service": "RE 3 (Saarbrücken -> Frankfurt)",
        "connecting_service": "ICE 26 (Hamburg -> Wien)",
        "scheduled_transfer_margin_min": 7.0,
        "platform_change": "Track 2 -> Track 5",
    },
    {
        "eva_id": 8000207,
        "station_name": "Köln Hbf",
        "inbound_service": "ICE 105 (Amsterdam -> Basel)",
        "connecting_service": "ICE 517 (Münster -> München)",
        "scheduled_transfer_margin_min": 8.0,
        "platform_change": "Track 6 -> Track 7",
    },
    {
        "eva_id": 8000370,
        "station_name": "Wiesbaden Hbf",
        "inbound_service": "S1 (Rödermark -> Wiesbaden)",
        "connecting_service": "RB 10 (Rheingau Linx)",
        "scheduled_transfer_margin_min": 6.0,
        "platform_change": "Track 3 -> Track 1",
    },
    {
        "eva_id": 8000130,
        "station_name": "Fulda",
        "inbound_service": "RE 5 (Kassel -> Bebra -> Fulda)",
        "connecting_service": "ICE 774 (Frankfurt -> Hamburg)",
        "scheduled_transfer_margin_min": 7.0,
        "platform_change": "Track 1 -> Track 3",
    },
    {
        "eva_id": 8000191,
        "station_name": "Karlsruhe Hbf",
        "inbound_service": "RE 2 (Konstanz -> Karlsruhe)",
        "connecting_service": "ICE 100 (Basel -> Hamburg)",
        "scheduled_transfer_margin_min": 10.0,
        "platform_change": "Track 1 -> Track 4",
    },
    {
        "eva_id": 8000096,
        "station_name": "Düsseldorf Hbf",
        "inbound_service": "RE 1 (Aachen -> Hamm)",
        "connecting_service": "ICE 628 (München -> Essen)",
        "scheduled_transfer_margin_min": 7.0,
        "platform_change": "Track 16 -> Track 15",
    },
    {
        "eva_id": 8000390,
        "station_name": "Würzburg Hbf",
        "inbound_service": "RE 54 (Frankfurt -> Bamberg)",
        "connecting_service": "ICE 582 (München -> Hamburg)",
        "scheduled_transfer_margin_min": 9.0,
        "platform_change": "Track 5 -> Track 6",
    },
]


class PassengerImpactEngine:
    """Evaluates transfer vulnerability and missed connection risks."""

    def __init__(self, minimum_transfer_window_min: float = 5.0, calibration_steepness: float = 0.8):
        self.minimum_transfer_window_min = float(minimum_transfer_window_min)
        self.k = float(calibration_steepness)

    def evaluate_connection_risk(
        self,
        incoming_delay_min: float,
        scheduled_transfer_margin_min: float,
    ) -> Dict[str, Any]:
        """Compute effective transfer buffer and connection risk probability.

        Effective transfer buffer = scheduled_transfer_margin_min - incoming_delay_min.
        Status:
          - "SAFE": effective_buffer > minimum_transfer_window_min (e.g. > 5m)
          - "VULNERABLE": 0 <= effective_buffer <= minimum_transfer_window_min
          - "MISSED": effective_buffer < 0

        Missed connection probability uses sigmoid calibration:
          P = 1.0 / (1.0 + exp(k * (effective_buffer - 2.5)))
          where effective_buffer < 0 yields P ~ 0.85-1.0,
          effective_buffer == 2.5 yields P = 0.50,
          effective_buffer > 5.0 yields P < 0.12.
        """
        effective_buffer = float(scheduled_transfer_margin_min - incoming_delay_min)

        if effective_buffer > self.minimum_transfer_window_min:
            status = "SAFE"
        elif effective_buffer >= 0.0:
            status = "VULNERABLE"
        else:
            status = "MISSED"

        # Center sigmoid midpoint around 2.5 min buffer (half of the 5.0 min window)
        exponent = -self.k * (effective_buffer - (self.minimum_transfer_window_min / 2.0))
        # Numerical guard against overflow
        exponent = max(-50.0, min(50.0, exponent))
        missed_prob = 1.0 / (1.0 + math.exp(-exponent))

        return {
            "incoming_delay_min": round(float(incoming_delay_min), 2),
            "scheduled_transfer_margin_min": round(float(scheduled_transfer_margin_min), 2),
            "effective_transfer_buffer_min": round(effective_buffer, 2),
            "connection_status": status,
            "missed_connection_probability": round(float(missed_prob), 4),
        }

    def analyze_corridor_connections(
        self,
        station_predictions: List[Dict[str, Any]],
        scheduled_connections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Scans corridor hub stations and returns evaluated passenger transfers.

        Parameters
        ----------
        station_predictions:
            List of station prediction dicts containing ``eva_id`` and either
            ``delay_15m`` or ``delays`` list.
        scheduled_connections:
            Optional list of connections. If None, uses DEFAULT_CORRIDOR_SCHEDULED_TRANSFERS.

        Returns
        -------
        List of connection evaluations sorted by highest missed connection probability.
        """
        connections = scheduled_connections or DEFAULT_CORRIDOR_SCHEDULED_TRANSFERS

        # Build map eva_id -> delay
        delay_map: Dict[int, float] = {}
        for p in station_predictions:
            eva = p.get("eva_id")
            if eva is None:
                continue
            if "delays" in p and isinstance(p["delays"], (list, tuple)) and len(p["delays"]) > 0:
                delay_map[eva] = float(p["delays"][0])
            elif "delay_15m" in p:
                delay_map[eva] = float(p["delay_15m"])
            else:
                delay_map[eva] = 0.0

        evaluations: List[Dict[str, Any]] = []
        for conn in connections:
            eva = conn["eva_id"]
            incoming_delay = delay_map.get(eva, 0.0)
            margin = conn["scheduled_transfer_margin_min"]

            risk = self.evaluate_connection_risk(incoming_delay, margin)
            evaluations.append({
                "eva_id": eva,
                "station_name": conn.get("station_name", f"Station {eva}"),
                "inbound_service": conn.get("inbound_service", "Unknown"),
                "connecting_service": conn.get("connecting_service", "Unknown"),
                "platform_change": conn.get("platform_change", "N/A"),
                **risk,
            })

        # Sort by highest missed connection probability first
        evaluations.sort(key=lambda x: x["missed_connection_probability"], reverse=True)
        return evaluations

