"""
mean(receiver R forward - receiver R backward), NOT a binary
locking probability. Frequency mismatch, windings, driven endpoints and
unforced controls are retained for interpreting every objective measurement.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer
from diode_analysis import matched_inputs

MIN_DISTANCE = 3.1
OBJECTIVE_VERSION = "matched_endpoint_R_difference_v1"


class DirectionalEvaluator:
    def __init__(self, executable="./src/rowers", template_input="./templates/6beads.input",
                 analysis_start=7.0, slip_hysteresis=np.pi / 4, hysteresis=None):
        self.executable = str(Path(executable).resolve())
        self.template_input = str(Path(template_input).resolve())
        if not Path(self.executable).is_file():
            raise FileNotFoundError(self.executable)
        if not np.isfinite(analysis_start) or analysis_start <= 5:
            raise ValueError("analysis_start must be finite and later than onset t=5.")
        if not np.isfinite(slip_hysteresis) or not 0 < slip_hysteresis < np.pi:
            raise ValueError("slip_hysteresis must be between zero and pi.")
        if hysteresis is not None and (not np.isfinite(hysteresis) or hysteresis < 0):
            raise ValueError("hysteresis must be finite and nonnegative.")
        self.analysis_start = float(analysis_start)
        self.slip_hysteresis = float(slip_hysteresis)
        self.hysteresis = hysteresis
        self.N, _, experiments = matched_inputs(Path(self.template_input))
        self._base = experiments["forward"][:-2]
        self.num_inner = self.N - 2
        self.chain_length = sum(self._base[22:22 + self.N - 1])
        if self._base[4] <= 0 or self._base[13] < 1 or self._base[13] != int(self._base[13]):
            raise ValueError("Positive dt and integer write interval required.")
        if self._base[3] <= 0 or MIN_DISTANCE <= 2 * self._base[3]:
            raise ValueError("MIN_DISTANCE must exceed the bead diameter.")
        print(f"[Template] N={self.N}, chain length={self.chain_length:g}")

    def _positions_to_distances(self, inner_positions):
        return np.diff([0.0, *inner_positions, self.chain_length]).tolist()

    def _is_valid(self, inner_positions):
        x = np.asarray(inner_positions, dtype=float)
        return (x.shape == (self.num_inner,) and np.all(np.isfinite(x))
                and np.all(np.asarray(self._positions_to_distances(x)) >= MIN_DISTANCE - 1e-12))

    def _prepare_search(self, T_range, F0_range, n_runs, total_time, num_cores,
                        n_iterations, output_dir):
        for bounds in (T_range, F0_range):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
                raise ValueError("Bounds must be finite, positive and strictly ordered.")
        for label, value in [("n_runs", n_runs), ("num_cores", num_cores)]:
            if value != int(value) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")
        if n_iterations != int(n_iterations) or n_iterations < 0:
            raise ValueError("n_iterations must be a nonnegative integer.")
        if not np.isfinite(self.chain_length) or self.chain_length < (self.N - 1) * MIN_DISTANCE:
            raise ValueError("Chain length cannot accommodate the minimum reference gaps.")
        if not np.isfinite(total_time) or total_time - self.analysis_start < 2 * T_range[1] + 2 * self._base[4]:
            raise ValueError("Duration must leave at least two periods at T_max after settling.")
        if T_range[0] < 20 * self._base[4]:
            raise ValueError("T_min needs at least 20 integration steps per forcing period.")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = {
            "objective": OBJECTIVE_VERSION, "N": self.N, "chain_length": self.chain_length,
            "analysis_start": self.analysis_start, "slip_hysteresis": self.slip_hysteresis,
            "hysteresis": self.hysteresis, "total_time": total_time, "n_runs": n_runs,
            "sampling_policy": "min(template_interval, floor(T/(20*dt)))",
            "template_sha256": hashlib.sha256(Path(self.template_input).read_bytes()).hexdigest(),
            "executable_sha256": hashlib.sha256(Path(self.executable).read_bytes()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_writer_sha256": hashlib.sha256(Path(__import__('diode_analysis').__file__).read_bytes()).hexdigest(),
            "analyser_sha256": hashlib.sha256(Path(__import__('bead_analysis').__file__).read_bytes()).hexdigest(),
        }
        protocol_path = self.output_dir / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != self.protocol:
            raise ValueError("Output folder belongs to a different protocol; use a new folder.")
        protocol_path.write_text(json.dumps(self.protocol, indent=2) + "\n")

    def _build_params(self, inner_positions, T_signal, F0, total_time, direction="forward"):
        if not self._is_valid(inner_positions) or T_signal <= 0 or F0 <= 0:
            raise ValueError("Invalid configuration.")
        p = self._base.copy()  # Preserve per-bead lambdas AND both stiffness blocks.
        p[5] = total_time
        # Increase recording resolution when a candidate has a short signal period.
        p[13] = min(int(p[13]), max(1, int(np.floor(T_signal / (20 * p[4])))))
        p[19] = 0.0 if direction == "control" else F0
        p[20] = T_signal
        p[22:22 + self.N - 1] = self._positions_to_distances(inner_positions)
        bead = self.N - 1 if direction == "backward" else 0
        return p + [1, bead]

    def _simulate(self, inner_positions, T_signal, F0, total_time, direction, folder):
        p = self._build_params(inner_positions, T_signal, F0, total_time, direction)
        work = folder / direction
        work.mkdir()
        inp = work / f"{direction}.input"
        inp.write_text(" ".join(format(v, ".17g") for v in p) + "\n")
        with (work / "simulation.log").open("w") as log:
            process = subprocess.run([self.executable, "-C", "-R", str(inp)],
                                     cwd=work, stdout=log, stderr=subprocess.STDOUT)
        files = list((work / direction).glob("*.dat"))
        if process.returncode or len(files) != 1 or not files[0].stat().st_size:
            raise RuntimeError(f"Simulator failed; see {work / 'simulation.log'}")
        reader = BeadDataAnalyzer()
        reader.read_data_file(str(files[0]))
        spacing = p[4] * p[13]
        if reader.num_beads != self.N or not np.allclose(
            np.diff(reader.time_series), spacing, rtol=1e-4, atol=1e-8
        ) or abs(reader.time_series[-1] - total_time) > 2 * spacing:
            raise RuntimeError("Trajectory dimensions, duration or physical timestamps are invalid.")
        return MultiBeadCoherenceAnalyzer(reader.time_series, reader.positions,
                                          hysteresis=self.hysteresis)

    def _replicate(self, positions, T, F0, duration, work):
        # One matched triple per replicate; independent C++ randomness in each run.
        work.mkdir()
        analysers = {name: self._simulate(positions, T, F0, duration, name, work)
                     for name in ("forward", "backward", "control")}
        clock = analysers["forward"].time_series
        common = clock >= self.analysis_start
        for analyser in analysers.values():
            if len(analyser.time_series) != len(clock) or not np.allclose(
                analyser.time_series, clock, rtol=0, atol=1e-9
            ):
                raise RuntimeError("Matched experiments have different recorded times.")
            for bead in (0, self.N - 1):
                common &= np.isfinite(analyser._get_phase(bead))
        times = clock[common]
        if len(times) < 3 or times[-1] - times[0] < 2 * T:
            raise RuntimeError("Insufficient common endpoint phase coverage; no score assigned.")
        rows = []
        for name, analyser in analysers.items():
            for bead in (0, self.N - 1):
                metric = analyser.measure_signal_locking(bead, T, times[0], times[-1],
                                                        slip_hysteresis=self.slip_hysteresis)
                if metric["status"] != "ok":
                    raise RuntimeError("Unavailable signal-locking measurement; no score assigned.")
                rows.append({"run": name, "bead": bead, **metric})
        f = next(r["phase_consistency_R"] for r in rows if r["run"] == "forward" and r["bead"] == self.N-1)
        b = next(r["phase_consistency_R"] for r in rows if r["run"] == "backward" and r["bead"] == 0)
        return {"forward_R": f, "backward_R": b, "score": f-b,
                "common_interval": [float(times[0]), float(times[-1])], "metrics": rows}

    def evaluate_state(self, inner_positions, T_signal, F0, n_runs=3, total_time=100, num_cores=4):
        if not hasattr(self, "protocol"):
            raise RuntimeError("Call optimise() before evaluating configurations.")
        key = uuid.uuid4().hex
        folder = self.output_dir / "simulation_work" / key
        folder.mkdir(parents=True)
        record = {"evaluation_id": key, "positions": list(map(float, inner_positions)),
                  "T_signal": float(T_signal), "F0": float(F0), "protocol": self.protocol}
        try:
            with ThreadPoolExecutor(max_workers=min(num_cores, n_runs)) as pool:
                futures = [pool.submit(self._replicate, inner_positions, T_signal, F0,
                                       total_time, folder / f"replicate_{i}") for i in range(n_runs)]
                replicates = [future.result() for future in futures]
            scores = np.array([r["score"] for r in replicates])
            f = float(np.mean([r["forward_R"] for r in replicates]))
            b = float(np.mean([r["backward_R"] for r in replicates]))
            record.update(status="ok", forward_R=f, backward_R=b, score=float(np.mean(scores)),
                          score_sem=float(np.std(scores, ddof=1)/np.sqrt(n_runs)) if n_runs > 1 else None,
                          replicates=replicates)
            self._audit(record)
            shutil.rmtree(folder)  # Numerical diagnostics persist; discard successful raw trajectories.
            return f, b, record["score"]
        except Exception as error:
            record.update(status="failed", error=str(error), retained_work=str(folder))
            self._audit(record)
            raise RuntimeError(f"Evaluation failed; no fabricated score. Work retained at {folder}: {error}") from error

    def _audit(self, record):
        with (self.output_dir / "objective_evaluations.jsonl").open("a") as fh:
            fh.write(json.dumps(record, allow_nan=False) + "\n")

    def _evaluate_batch(self, configs, n_runs, total_time, num_cores):
        # Replicates run in parallel; avoid nested multiprocessing pools.
        return [self.evaluate_state(pos, T, F0, n_runs, total_time, num_cores)
                for pos, T, F0 in configs]

    def _load_warm_start(self, json_path, T_range, F0_range):
        # Old coherence histories and repeated Metropolis residence states cannot
        # be used as fresh observations of the new objective.
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records_path = path if path.suffix == ".jsonl" else path.parent / "objective_evaluations.jsonl"
        if not records_path.is_file():
            raise ValueError("Warm start requires the new objective_evaluations.jsonl audit.")
        X, y = [], []
        for line in records_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("protocol") != self.protocol:
                raise ValueError("Warm-start protocol differs from the current objective or simulation settings.")
            if r["status"] != "ok":
                continue
            pos, T, F0 = r["positions"], r["T_signal"], r["F0"]
            if self._is_valid(pos) and T_range[0] <= T <= T_range[1] and F0_range[0] <= F0 <= F0_range[1]:
                X.append(pos + [T, F0]); y.append(r["score"])
        print(f"[Warm start] {len(X)} actual evaluations with matching protocol")
        return np.asarray(X).reshape(-1, self.num_inner + 2), np.asarray(y)
