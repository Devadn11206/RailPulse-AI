"""Enhanced tabular feature engineering for Phase 3 baseline improvement.

Takes flat X_flat [N, 84] (delay, sin_h, cos_h, sin_d, cos_d, w_mean, temp per timestep)
and engineers spatial lag differentials, network rolling momentum, cyclical interactions,
and station identity features.
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter1d

logger = logging.getLogger(__name__)


class EnhancedTabularFeatureBuilder:
    """Builds enhanced tabular features from flat temporal windows."""

    def __init__(self, num_stations: int = 10, lookback_steps: int = 12, n_features: int = 7):
        self.num_stations = int(num_stations)
        self.lookback_steps = int(lookback_steps)
        self.n_features = int(n_features)
        self.flat_dim = self.lookback_steps * self.n_features  # 84

    def _delay_column(self, X_flat: np.ndarray, t: int) -> np.ndarray:
        """Extract delay at timestep t (0 = most recent, lookback-1 = oldest)."""
        return X_flat[:, t * self.n_features + 0]

    def _cyclical_columns(self, X_flat: np.ndarray, t: int) -> dict:
        """Extract cyclical features at timestep t."""
        return {
            "sin_h": X_flat[:, t * self.n_features + 1],
            "cos_h": X_flat[:, t * self.n_features + 2],
            "sin_d": X_flat[:, t * self.n_features + 3],
            "cos_d": X_flat[:, t * self.n_features + 4],
            "temp": X_flat[:, t * self.n_features + 6],
        }

    def build(self, X_flat: np.ndarray) -> np.ndarray:
        """Engineer enhanced features from flat X.

        Args:
            X_flat: [N, 84] float32 array.

        Returns:
            X_enhanced: [N, 84 + n_engineered] float32 array with original
            features preserved plus engineered features appended.
        """
        X_flat = np.asarray(X_flat, dtype=np.float32)
        N = X_flat.shape[0]

        # Station identity inferred from row order (data grouped by station cyclically)
        station_ids = (np.arange(N) % self.num_stations).astype(np.float32)

        # Delay series per sample: [N, 12] where col 0 = most recent (t0)
        delays = X_flat[:, 0::7]  # columns 0, 7, 14, ..., 77

        d0 = delays[:, 0]
        d1 = delays[:, 1]
        d2 = delays[:, 2]
        d5 = delays[:, 5]

        engineered: list = []

        # ------------------------------------------------------------------
        # 1. Spatial delay lags & differentials
        # ------------------------------------------------------------------
        # Velocity: change in delay from t0 to t1, t0 to t5
        velocity_01 = (d0 - d1).astype(np.float32)
        velocity_05 = (d0 - d5).astype(np.float32)
        # Acceleration: (t0-t1) - (t1-t2)
        acceleration = ((d0 - d1) - (d1 - d2)).astype(np.float32)

        engineered.extend([velocity_01, velocity_05, acceleration])

        # ------------------------------------------------------------------
        # 2. Network-wide rolling delay momentum (station-grouped)
        # Approximate corridor network context using per-station rolling
        # mean/max of delay_t0 across that station's own recent windows.
        # ------------------------------------------------------------------
        for agg_name, agg_fn in (("mean", np.mean), ("max", np.max)):
            station_agg = np.zeros(N, dtype=np.float32)
            for s in range(self.num_stations):
                mask = np.arange(N) % self.num_stations == s
                if mask.sum() == 0:
                    continue
                station_delays = d0[mask]
                # Rolling mean/max over last 5 windows of this station
                roll = np.convolve(
                    station_delays, np.ones(5) / 5.0, mode="same"
                ) if agg_name == "mean" else np.convolve(
                    station_delays, np.ones(5), mode="same"
                )
                if agg_name == "max":
                    roll = maximum_filter1d(station_delays, size=5, mode="nearest")
                station_agg[mask] = roll.astype(np.float32)
            engineered.append(station_agg)

        # Ratio of current station delay to its station-group rolling mean
        network_mean = engineered[-2]  # last mean appended
        delay_to_network_ratio = (d0 / (network_mean + 1e-6)).astype(np.float32)
        engineered.append(delay_to_network_ratio)

        # ------------------------------------------------------------------
        # 3. Cyclical & interaction features (t0)
        # ------------------------------------------------------------------
        cyc0 = self._cyclical_columns(X_flat, 0)
        sin_h0 = cyc0["sin_h"]
        cos_h0 = cyc0["cos_h"]
        sin_d0 = cyc0["sin_d"]
        cos_d0 = cyc0["cos_d"]
        temp0 = cyc0["temp"]

        # Hour-of-day x day-of-week interaction
        hour_dow_sin = (sin_h0 * sin_d0).astype(np.float32)
        hour_dow_cos = (cos_h0 * cos_d0).astype(np.float32)
        # Temperature x recent delay trend interaction
        temp_delay_interaction = (temp0 * velocity_01).astype(np.float32)

        engineered.extend([hour_dow_sin, hour_dow_cos, temp_delay_interaction])

        # ------------------------------------------------------------------
        # 4. Station identity (as feature for tree models)
        # ------------------------------------------------------------------
        engineered.append(station_ids)

        # Stack engineered features
        engineered_arr = np.column_stack(engineered).astype(np.float32)

        # Concatenate original flat features + engineered
        X_enhanced = np.column_stack([X_flat, engineered_arr]).astype(np.float32)

        logger.info(
            f"Enhanced features: {X_flat.shape[1]} -> {X_enhanced.shape[1]} "
            f"(+{engineered_arr.shape[1]} engineered)"
        )
        return X_enhanced

    @property
    def n_engineered(self) -> int:
        # velocity_01, velocity_05, acceleration (3)
        # network mean, network max, delay_to_network_ratio (3)
        # hour_dow_sin, hour_dow_cos, temp_delay_interaction (3)
        # station_ids (1)
        return 10


def build_enhanced_features(
    X_flat: np.ndarray, num_stations: int = 10
) -> np.ndarray:
    """Convenience function to build enhanced features."""
    builder = EnhancedTabularFeatureBuilder(num_stations=num_stations)
    return builder.build(X_flat)
