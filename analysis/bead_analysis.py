#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
from pathlib import Path


class BeadDataAnalyzer:
    def __init__(self):
        """Initialize the analyzer with default parameters."""
        self.time_series = None
        self.positions = None
        self.angles_deg = None
        self.bead_positions = {}
        self.num_beads = 0
        self.dt = None

    def parse_filename(self, filename: str) -> dict:
        params = {}

        distances_match = re.search(r"\[(.*?)\]D", filename)
        if distances_match:
            params["distances"] = [
                float(x) for x in distances_match.group(1).split(",")
            ]

        params["alpha"] = float(
            re.search(
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)alpha", filename
            ).group(1)
        )
        params["amplitude"] = float(re.search(r"(\d+\.\d+)A", filename).group(1))

        return params

    def read_data_file(self, filepath: str) -> None:
        # ndmin keeps the result two-dimensional for a one-row file.
        data = np.loadtxt(filepath, ndmin=2)

        # Reject incomplete files before indexing their rows or columns.
        if data.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two rows.")

        if data.shape[1] < 4 or (data.shape[1] - 2) % 2:
            raise ValueError("Expected columns: frame, time, x0, y0, x1, y1, ...")

        self.time_series = data[:, 1]
        self.dt = self.time_series[1] - self.time_series[0]

        num_cols = data.shape[1]
        self.num_beads = (num_cols - 2) // 2

        self.positions = np.zeros((len(self.time_series), self.num_beads, 2))

        for i in range(self.num_beads):
            x_col = 2 + 2 * i
            y_col = 3 + 2 * i

            self.positions[:, i, 0] = data[:, x_col]
            self.positions[:, i, 1] = data[:, y_col]

        # New runs write a same-stem .axes sidecar. Missing metadata means a
        # legacy horizontal trajectory. Keep .positions in laboratory space.
        sidecar = Path(filepath).with_suffix(".axes")
        self.angles_deg = np.zeros(self.num_beads)
        if sidecar.exists():
            axes = np.loadtxt(sidecar, ndmin=2)
            if axes.shape != (self.num_beads, 3) or not np.all(np.isfinite(axes)):
                raise ValueError("Invalid .axes metadata; expected N rows of angle, X, Y.")
            self.angles_deg = axes[:, 0]

    @property
    def analysis_positions(self):
        """Local longitudinal/transverse coordinates for ALL phase/spectrum paths.

        Rotate absolute positions: at zero angle these are exactly the previous
        x/y coordinates. Subtracting their means gives the same displacements
        as projecting r-R and subtracting its mean. Lab data remain available
        in .positions for physical trajectory plots and geometry checks.
        """
        theta = np.deg2rad(self.angles_deg)
        c, s = np.cos(theta), np.sin(theta)
        x, y = self.positions[..., 0], self.positions[..., 1]
        return np.stack((x*c + y*s, -x*s + y*c), axis=-1)


class MultiBeadCoherenceAnalyzer:
    def __init__(
        self,
        time_series,
        positions,
        reference_positions=None,
        hysteresis=None,
    ):
        """
        reference_positions:
            Optional array shaped (num_beads, num_dimensions).
            If absent, use each coordinate's full-run mean.

        hysteresis:
            Nonnegative half-width in position units.
            If None, estimate it separately for each bead/dimension.

        Optional arguments preserve existing two-argument calls from
        the sweep and optimisation scripts.
        """
        self.time_series = np.asarray(time_series, dtype=float)
        self.positions = np.asarray(positions, dtype=float)

        if self.time_series.ndim != 1 or len(self.time_series) < 2:
            raise ValueError("At least two recorded times are required.")

        if not np.all(np.isfinite(self.time_series)) or np.any(
            np.diff(self.time_series) <= 0
        ):
            raise ValueError("Recorded times must be finite and increasing.")

        if (
            self.positions.ndim != 3
            or self.positions.shape[0] != len(self.time_series)
            or not np.all(np.isfinite(self.positions))
        ):
            raise ValueError(
                "Positions must be finite with shape " "(samples, beads, dimensions)."
            )

        self.num_beads = self.positions.shape[1]
        self.dt = self.time_series[1] - self.time_series[0]

        # These levels define the repeatable event used to measure phase.
        # They may be physical references or means supplied by the caller.
        if reference_positions is None:
            self.reference_positions = np.mean(self.positions, axis=0)
        else:
            self.reference_positions = np.asarray(reference_positions, dtype=float)

            if self.reference_positions.shape != self.positions.shape[1:] or not np.all(
                np.isfinite(self.reference_positions)
            ):
                raise ValueError(
                    "reference_positions must be finite with shape "
                    "(num_beads, num_dimensions)."
                )

        if hysteresis is not None:
            if not np.isfinite(hysteresis) or hysteresis < 0:
                raise ValueError("hysteresis must be finite and nonnegative.")

        self.hysteresis = hysteresis
        # Trajectories are treated as immutable after construction.
        self._crossing_cache = {}
        self._phase_cache = {}

    def _get_crossing_settings(self, bead_index, dimension=0):
        """
        Use identical crossing levels and bands for detection and plots.
        """
        level = self.reference_positions[bead_index, dimension]

        if self.hysteresis is not None:
            half_width = float(self.hysteresis)
        else:
            signal = self.positions[:, bead_index, dimension]

            # Quantiles reduce sensitivity to isolated extreme excursions.
            low, high = np.percentile(signal, [5, 95])
            amplitude_estimate = 0.5 * (high - low)

            # This is a tunable detector setting, not a physical constant.
            half_width = 0.05 * amplitude_estimate

        return float(level), float(half_width)

    def _get_zero_crossing_times(self, bead_index, dimension=0):
        key = (bead_index, dimension)
        if key not in self._crossing_cache:
            self._crossing_cache[key] = self._detect_crossings(bead_index, dimension)
        return self._crossing_cache[key]

    def _detect_crossings(self, bead_index, dimension=0):
        """
        Detect confirmed upward crossings using hysteresis.

        1. Arm after reaching level - half_width.
        2. Record a candidate upward crossing of level.
        3. Accept it after reaching level + half_width.
        4. Require a return below the lower boundary before another event.

        Save the interpolated reference-crossing time, not the later
        time at which the upper boundary confirms the crossing.
        """
        signal = self.positions[:, bead_index, dimension]
        tvals = self.time_series

        level, half_width = self._get_crossing_settings(bead_index, dimension)

        lower = level - half_width
        upper = level + half_width

        crossing_times = []

        # Ignore a partial initial stroke unless the bead has already
        # reached the lower side of the confirmation band.
        armed = signal[0] <= lower
        candidate_time = None

        for i in range(1, len(signal)):
            previous = signal[i - 1]
            current = signal[i]

            if previous <= lower:
                armed = True
                candidate_time = None

            if armed and candidate_time is None and previous <= level < current:
                frac = (level - previous) / (current - previous)
                candidate_time = tvals[i - 1] + frac * (tvals[i] - tvals[i - 1])

            # Small recrossings of the centre cannot independently
            # create additional accepted events.
            if armed and candidate_time is not None and current >= upper:
                crossing_times.append(candidate_time)
                armed = False
                candidate_time = None

            if current <= lower:
                armed = True
                candidate_time = None

        return crossing_times

    def count_crossings(self, bead_index, dimension=0):
        """
        Raw crossings of the selected level in both directions.

        This diagnostic count is not hysteresis-filtered and is not
        used for phase reconstruction.
        """
        signal = self.positions[:, bead_index, dimension]
        level, _ = self._get_crossing_settings(bead_index, dimension)

        above = (signal > level).astype(int)
        return np.sum(np.abs(np.diff(above)))

    def _get_mean_period(self, bead_index=0, dimension=0):
        crossing_times = self._get_zero_crossing_times(bead_index, dimension)

        if len(crossing_times) < 2:
            return np.nan

        return np.mean(np.diff(crossing_times))

    def _get_phase(self, bead_index, dimension=0):
        key = (bead_index, dimension)
        if key not in self._phase_cache:
            self._phase_cache[key] = self._reconstruct_phase(bead_index, dimension)
        return self._phase_cache[key]

    def _reconstruct_phase(self, bead_index, dimension=0):
        """
        Reconstruct unwrapped phase at every recorded time.

        Confirmed upward crossings receive phases 0, 2*pi, 4*pi, ...
        Interpolate between crossings without extrapolating beyond them.
        """
        crossing_times = np.asarray(
            self._get_zero_crossing_times(bead_index, dimension),
            dtype=float,
        )

        if len(crossing_times) < 2:
            return np.full(len(self.time_series), np.nan)

        crossing_phases = 2 * np.pi * np.arange(len(crossing_times))

        return np.interp(
            self.time_series,
            crossing_times,
            crossing_phases,
            left=np.nan,
            right=np.nan,
        )

    def _get_phase_differences(self, ref_bead=0, bead_j=1, dimension=0):
        """
        Return common times and phi_j(t) - phi_ref(t), in radians.

        Independent cycle-number origins can add constant multiples
        of 2*pi. This does not affect drift or circular coherence.
        """
        ref_phase = self._get_phase(ref_bead, dimension)
        j_phase = self._get_phase(bead_j, dimension)

        valid = np.isfinite(ref_phase) & np.isfinite(j_phase)

        return (
            self.time_series[valid],
            j_phase[valid] - ref_phase[valid],
        )

    def coherence_windowed_real_sum(
        self, ref_bead=0, dimension=0, periods_per_window=10
    ):
        """
        Existing bead-to-bead coherence in reference-cycle windows.

        Retains C = 1 - std(exp(i * delta_phi)) and the existing
        rule of dropping the first 10% of usable windows.

        This method does not measure locking to the external signal.
        """
        if periods_per_window < 1 or int(periods_per_window) != periods_per_window:
            raise ValueError("periods_per_window must be a positive integer.")

        periods_per_window = int(periods_per_window)
        coherence_array = np.full(self.num_beads, np.nan)

        ref_times = np.asarray(
            self._get_zero_crossing_times(ref_bead, dimension),
            dtype=float,
        )

        if len(ref_times) < periods_per_window + 1:
            return coherence_array

        coherence_array[ref_bead] = 1.0

        for j in range(self.num_beads):
            if j == ref_bead:
                continue

            times, delta_phis = self._get_phase_differences(ref_bead, j, dimension)
            j_times = np.asarray(
                self._get_zero_crossing_times(j, dimension),
                dtype=float,
            )

            if len(j_times) < 2:
                continue

            window_cohs = []

            for start in range(
                0,
                len(ref_times) - periods_per_window,
                periods_per_window,
            ):
                left = ref_times[start]
                right = ref_times[start + periods_per_window]

                # Require defined phase over the complete window.
                if j_times[0] > left or j_times[-1] < right:
                    continue

                mask = (times >= left) & (times < right)
                window = delta_phis[mask]

                if len(window) < 2:
                    continue

                z = np.exp(1j * window)
                mu = np.mean(z)
                std_complex = np.sqrt(np.mean(np.abs(z - mu) ** 2))

                window_cohs.append(1.0 - std_complex)

            if window_cohs:
                drop = int(len(window_cohs) * 0.1)
                coherence_array[j] = np.mean(window_cohs[drop:])

        return coherence_array

    # SECTION 2:
    # Compare bead phases with the external forcing clock.
    # Existing optimisation calls to bead-to-bead coherence still work.

    def get_signal_phase_difference(
        self,
        bead_index,
        signal_period,
        dimension=0,
        signal_phase=0.0,
    ):
        """
        Return recorded times and phi_bead - phi_signal, in radians.

        The simulator applies sign(cos(2*pi*time/T)) after time=5.
        Switching the force on does not restart that cosine clock.

        A square wave still has an underlying cycle phase:
            phi_signal = 2*pi*time/T + signal_phase
        """
        if not np.isfinite(signal_period) or signal_period <= 0:
            raise ValueError("signal_period must be finite and positive.")

        if not np.isfinite(signal_phase):
            raise ValueError("signal_phase must be finite.")

        bead_phase = self._get_phase(bead_index, dimension)
        omega = 2.0 * np.pi / signal_period

        # Do not wrap modulo 2*pi: we need to see accumulated drift.
        # Undefined bead phases remain NaN.
        difference = bead_phase - (omega * self.time_series + signal_phase)

        return self.time_series.copy(), difference

    @staticmethod
    def _time_weights(times):
        """
        Trapezoidal weights for averages over time.

        This prevents densely sampled parts of a nonuniform recording
        from receiving disproportionate weight.
        """
        dt = np.diff(times)
        weights = np.empty(len(times))

        weights[0] = dt[0] / 2.0
        weights[-1] = dt[-1] / 2.0
        weights[1:-1] = (dt[:-1] + dt[1:]) / 2.0

        return weights / np.sum(weights)

    @staticmethod
    def _count_signal_windings(times, differences, anchor, hysteresis):
        """
        Count confirmed transitions between relative-phase branches.

        Branch centres are:
            anchor + 2*pi*k

        A forward transition requires passing the upper midpoint plus
        a confirmation margin. Returning requires passing the same
        midpoint minus that margin.

        This suppresses repeated counts from small boundary fluctuations.

        IMPORTANT:
        These are slip candidates, not proof of isolated phase slips.
        Continuous frequency drift also produces branch transitions.
        Counts depend on the anchor, margin and observation interval.
        Event times are the first samples confirming each transition.
        """
        branch = 0
        events = []

        for time, phase in zip(times, differences):
            displacement = phase - anchor - 2.0 * np.pi * branch

            while displacement > np.pi + hysteresis:
                branch += 1
                events.append(
                    {
                        "time": float(time),
                        "direction": 1,
                    }
                )
                displacement -= 2.0 * np.pi

            while displacement < -np.pi - hysteresis:
                branch -= 1
                events.append(
                    {
                        "time": float(time),
                        "direction": -1,
                    }
                )
                displacement += 2.0 * np.pi

        return events

    def measure_signal_locking(
        self,
        bead_index,
        signal_period,
        start_time,
        end_time=None,
        dimension=0,
        signal_phase=0.0,
        slip_hysteresis=np.pi / 4.0,
    ):
        """
        Measure 1:1 signal locking over an explicit settled interval.

        Frequency mismatch:
            Slope of the unwrapped relative phase.

        Phase consistency:
            R = abs(time-average of exp(i*relative_phase)).

        Winding events:
            Confirmed transitions between relative-phase branches.

        We do not detrend before calculating R, because actual phase
        drift must remain part of the measurement.

        No automatic locked/unlocked label is assigned.
        """
        if end_time is None:
            end_time = self.time_series[-1]

        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("Analysis interval boundaries must be finite.")

        if start_time >= end_time:
            raise ValueError("Analysis start must precede analysis end.")

        if not np.isfinite(slip_hysteresis) or not 0 < slip_hysteresis < np.pi:
            raise ValueError("slip_hysteresis must lie strictly between 0 and pi.")

        times, differences = self.get_signal_phase_difference(
            bead_index,
            signal_period,
            dimension,
            signal_phase,
        )

        # Restrict statistics to the requested interval and to times
        # where the bead's phase is actually defined.
        valid = (times >= start_time) & (times <= end_time) & np.isfinite(differences)

        times = times[valid]
        differences = differences[valid]

        # Missing measurements remain None, becoming null in JSON.
        # A nonoscillating bead must not be reported as perfectly locked.
        result = {
            "status": "insufficient_phase_coverage",
            "num_samples": int(len(times)),
            "actual_start_time": (float(times[0]) if len(times) else None),
            "actual_end_time": (float(times[-1]) if len(times) else None),
            "duration": (float(times[-1] - times[0]) if len(times) else 0.0),
            "frequency_mismatch_rad_per_time": None,
            "frequency_mismatch_cycles_per_time": None,
            "relative_frequency_mismatch": None,
            "mean_frequency_cycles_per_time": None,
            "phase_consistency_R": None,
            "mean_phase_offset_rad": None,
            "net_relative_cycles": None,
            "slip_candidates_forward": None,
            "slip_candidates_backward": None,
            "slip_candidates_total": None,
            "slip_candidate_rate_per_time": None,
            "slip_anchor_rad": None,
            "slip_events": [],
        }

        # Two forcing periods is a minimum for returning measurements.
        # It is not a guarantee of enough data to establish locking.
        if len(times) < 3 or times[-1] - times[0] < 2.0 * signal_period:
            return result

        weights = self._time_weights(times)

        # Subtract only a constant for numerical stability.
        # This does not remove physical phase drift.
        local_phase = differences - differences[0]
        centred_time = times - np.sum(weights * times)

        # Weighted straight-line fit:
        # slope = omega_bead - omega_signal.
        slope = np.sum(weights * centred_time * local_phase) / np.sum(
            weights * centred_time**2
        )

        omega = 2.0 * np.pi / signal_period

        # Standard circular phase consistency.
        # A constant phase lag can have R=1.
        mean_phasor = np.sum(weights * np.exp(1j * differences))
        consistency = float(np.clip(abs(mean_phasor), 0.0, 1.0))

        # Use the first usable signal period to establish a branch anchor.
        # A median reduces sensitivity to one noisy initial sample.
        initial = times <= times[0] + signal_period
        anchor = float(np.median(differences[initial]))

        events = self._count_signal_windings(
            times,
            differences,
            anchor,
            slip_hysteresis,
        )

        forward = sum(event["direction"] == 1 for event in events)
        backward = sum(event["direction"] == -1 for event in events)

        duration = times[-1] - times[0]

        result.update(
            {
                "status": "ok",
                "frequency_mismatch_rad_per_time": float(slope),
                "frequency_mismatch_cycles_per_time": float(slope / (2.0 * np.pi)),
                "relative_frequency_mismatch": float(slope / omega),
                "mean_frequency_cycles_per_time": float(
                    (omega + slope) / (2.0 * np.pi)
                ),
                "phase_consistency_R": consistency,
                # A circular mean angle is undefined when the resultant vanishes.
                "mean_phase_offset_rad": (
                    float(np.angle(mean_phasor)) if consistency > 1e-12 else None
                ),
                "net_relative_cycles": float(
                    (differences[-1] - differences[0]) / (2.0 * np.pi)
                ),
                "slip_candidates_forward": forward,
                "slip_candidates_backward": backward,
                "slip_candidates_total": forward + backward,
                "slip_candidate_rate_per_time": float((forward + backward) / duration),
                "slip_anchor_rad": anchor,
                "slip_events": events,
            }
        )

        return result


def frequency_spectra(reader, start=5.0, frequency_range=(0.1, 20.0), nperseg=4096):
    """Welch spectra after settling; peaks describe the unforced coupled chain.

    These are not automatically isolated-oscillator natural frequencies.
    Keep the frequency range explicit so a peak at its edge is recognisable.
    """
    from scipy.signal import welch

    mask = reader.time_series > start
    times = reader.time_series[mask]
    if len(times) < 8:
        raise ValueError("Need at least eight samples after frequency-analysis start.")
    spacing = np.diff(times)
    if not np.allclose(spacing, spacing[0], rtol=1e-4, atol=1e-8):
        raise ValueError("Welch spectra require uniformly sampled data.")
    frequencies, power = welch(
        reader.analysis_positions[mask, :, 0],
        fs=1 / spacing[0],
        axis=0,
        nperseg=min(nperseg, len(times)),
    )
    band = (frequencies > frequency_range[0]) & (frequencies < frequency_range[1])
    if not np.any(band):
        raise ValueError("No spectral bins inside the requested frequency range.")
    peaks = frequencies[band][np.argmax(power[band], axis=0)]
    mean_power = power.mean(axis=1)
    return {
        "frequencies": frequencies,
        "power": power,
        "peaks": peaks,
        "mean_power": mean_power,
        "mean_spectrum_peak": float(frequencies[band][np.argmax(mean_power[band])]),
    }


def measure_experiments(
    readers, period, start, all_beads=False, hysteresis=None, slip_hysteresis=np.pi / 4
):
    """Measure a matched triple on a common endpoint observation interval.

    Preserve the original evaluator's full-record mean crossing levels and
    endpoint score. Interior beads use that interval but can have missing phase.
    """
    analysers = {
        key: MultiBeadCoherenceAnalyzer(
            r.time_series, r.analysis_positions, hysteresis=hysteresis
        )
        for key, r in readers.items()
    }
    clock = readers["forward"].time_series
    n = readers["forward"].num_beads
    common = clock >= start
    for analyser in analysers.values():
        if len(analyser.time_series) != len(clock) or not np.allclose(
            analyser.time_series, clock, rtol=0, atol=1e-9
        ):
            raise ValueError("Experiments must have identical recording times.")
        for bead in (0, n - 1):
            common &= np.isfinite(analyser._get_phase(bead))
    times = clock[common]
    if len(times) < 3 or times[-1] - times[0] < 2 * period:
        raise ValueError(
            "Insufficient common endpoint phase coverage; no score assigned."
        )
    rows = []
    for direction, analyser in analysers.items():
        for bead in (range(n) if all_beads else (0, n - 1)):
            metric = analyser.measure_signal_locking(
                bead, period, times[0], times[-1], slip_hysteresis=slip_hysteresis
            )
            if bead in (0, n - 1) and metric["status"] != "ok":
                raise ValueError("Endpoint locking measurement unavailable.")
            rows.append({"run": direction, "bead": bead, **metric})
    return rows, [float(times[0]), float(times[-1])], analysers


def directional_score(rows, n):
    """Current objective: receiving-bead R forward minus receiving-bead R backward.

    Controls, slips and mismatch are diagnostics, not hidden score penalties.
    """
    forward = next(
        r["phase_consistency_R"]
        for r in rows
        if r["run"] == "forward" and r["bead"] == n - 1
    )
    backward = next(
        r["phase_consistency_R"]
        for r in rows
        if r["run"] == "backward" and r["bead"] == 0
    )
    return float(forward), float(backward), float(forward - backward)


def classify_locking(metric, r_min=0.9, mismatch_max=0.01, slip_rate_max=0.0):
    """Explicit finite-window classification; None means unavailable, not unlocked."""
    if metric["status"] != "ok":
        return None
    return bool(
        metric["phase_consistency_R"] >= r_min
        and abs(metric["relative_frequency_mismatch"]) <= mismatch_max
        and metric["slip_candidate_rate_per_time"] <= slip_rate_max
    )
