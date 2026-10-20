"""Shared controlled-junction, logic and memory experiments.

The simulator and optimisers remain in their existing modules. This module only
constructs cases and interprets responses; it never imposes a logical output on
bead dynamics. Run this file for characterisation, or pass --experiment to the
existing optimisation.py. Templates are exploratory, not demonstrated devices.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import load_configuration, run_simulation, save_json, fingerprint, drive_coordinate
from optimisation import Evaluator
from bead_analysis import MultiBeadCoherenceAnalyzer


def validate_experiment(spec):
    """Small explicit schema, so typos cannot silently change an objective."""
    allowed = {"kind", "cases", "output", "reference", "period", "threshold", "ring"}
    if set(spec) - allowed or spec.get("kind") not in ("control", "logic", "memory"):
        raise ValueError("Experiment kind must be control, logic or memory; unknown keys are rejected")
    if not np.isfinite(spec.get("period", 0)) or spec["period"] <= 0:
        raise ValueError("Specify a positive diagnostic period")
    if not 0 < spec.get("threshold", .8) < 1:
        raise ValueError("threshold must lie between zero and one")
    cases = spec.get("cases", [])
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Cases need unique names")
    for c in cases:
        if set(c) - {"name", "scales", "window", "windows", "expected", "overrides"}:
            raise ValueError("Unknown case setting")
        if not c["name"].replace("_", "").isalnum():
            raise ValueError("Case names must contain only letters, numbers and underscores")
        for window in [c.get("window", [])] + c.get("windows", []):
            if len(window) != 2 or not np.all(np.isfinite(window)) or window[0] < 0 or window[1]-window[0] < 2*spec["period"]:
                raise ValueError("Each window must span at least two diagnostic periods")
        if any(k != "drives" and not drive_coordinate(k) for k in c.get("overrides", {})):
            raise ValueError("Case overrides may change drives only; geometry/traps stay matched")
        if spec["kind"] == "logic" and c.get("expected") not in (0, 1):
            raise ValueError("Logic cases need expected=0 or 1")
        if not all(np.isfinite(x) and x >= 0 for x in c.get("scales", [])):
            raise ValueError("Drive scales must be finite and nonnegative")
    if spec["kind"] == "control" and {c["name"] for c in cases} != {"neither", "input", "control", "both"}:
        raise ValueError("Control cases must be neither, input, control, both")
    if spec["kind"] == "logic" and len(cases) != 4:
        raise ValueError("The initial two-input logic protocol needs four truth-table cases")
    if spec["kind"] == "memory" and len(cases) != 2:
        raise ValueError("Memory comparison needs exactly two writing protocols")


def pair_metrics(analyser, bead, reference, start, stop, period, slip_hysteresis):
    """Time-weighted pair locking on a complete, declared observation window.

    Reuse the existing crossing phase and slip detector. These are offline
    interpolated phases; no causal switching latency is inferred from them.
    """
    t = analyser.time_series
    mask = (t >= start) & (t <= stop)
    t = t[mask]
    delta = (analyser._get_phase(bead) - analyser._get_phase(reference))[mask]
    if len(t) < 3 or not np.all(np.isfinite(delta)):
        raise ValueError("Insufficient phase coverage in the requested pair window")
    w = analyser._time_weights(t)
    z = np.sum(w*np.exp(1j*delta))
    centred = t-np.sum(w*t)
    slope = np.sum(w*centred*(delta-delta[0]))/np.sum(w*centred**2)
    anchor = float(np.median(delta[t <= t[0]+period]))
    slips = analyser._count_signal_windings(t, delta, anchor, slip_hysteresis)
    return {"R": float(abs(z)), "mean_phase": float(np.angle(z)),
            "frequency_mismatch": float(slope/(2*np.pi)), "slip_count": len(slips),
            "interval": [float(t[0]), float(t[-1])]}, z


def measure_window(analyser, config, spec, output, reference, start, stop, slip_hysteresis):
    """One response calculation shared by score windows and sequence diagnostics."""
    pairs, signals = [], []
    # Report the actual first drive's clock (including its phase); the protocol
    # period separately defines the minimum diagnostic observation duration.
    drive = config.values["drives"][0] if config.values["drives"] else None
    for bead in output:
        metric, _ = pair_metrics(analyser, bead, reference, start, stop,
                                 spec["period"], slip_hysteresis)
        pairs.append({"bead": bead, **metric})
        if drive and drive["waveform"] != "constant":
            signal = analyser.measure_signal_locking(bead, drive["period"], start, stop,
                signal_phase=drive["phase"], slip_hysteresis=slip_hysteresis)
            signals.append({"bead": bead, **signal})
    row = {"R": float(np.mean([m["R"] for m in pairs])), "pairs": pairs,
           "signal_metrics": signals, "window": [start, stop]}
    pattern = None
    if spec["kind"] == "memory":
        ring = config.targets(spec["ring"])
        if len(ring) < 3:
            raise ValueError("A ring needs at least three beads in cyclic order")
        mask = (analyser.time_series >= start) & (analyser.time_series <= stop)
        phases = np.column_stack([analyser._get_phase(b)[mask] for b in ring])
        if not np.all(np.isfinite(phases)):
            raise ValueError("Incomplete ring phase coverage")
        differences = np.roll(phases, -1, axis=1)-phases
        weights = analyser._time_weights(analyser.time_series[mask])
        pattern = np.sum(weights[:,None]*np.exp(1j*differences), axis=0)
        winding = np.sum(np.angle(np.exp(1j*differences)), axis=1)/(2*np.pi)
        row.update(neighbour_phase=np.angle(pattern).tolist(),
                   neighbour_R=np.abs(pattern).tolist(),
                   winding_values=np.unique(np.rint(winding).astype(int)).tolist())
    return row, pattern


class ExperimentEvaluator(Evaluator):
    """Only protocol construction/scoring differ from the legacy evaluator."""
    def __init__(self, *args, experiment, **kwargs):
        super().__init__(*args, **kwargs)
        validate_experiment(experiment)
        self.experiment = deepcopy(experiment)
        self.protocol.update(objective="computing_experiment_v1", experiment=self.experiment)
        self.protocol["code_sha256"]["experiments.py"] = fingerprint(__file__)

    def summarise(self, replicates):
        # The parent already supplies mean score and its SEM for every protocol.
        return {}

    def _replicate(self, config, folder):
        spec = self.experiment
        if config.values["drives"] is None:
            raise ValueError("Computing experiments require an explicit drives list")
        if config.values["seed"] < 0:
            # Match cases within a replicate, while independent repeats still
            # receive independent seeds. Record the realised seed for replay.
            config = config.updated(seed=int(np.random.SeedSequence().generate_state(1)[0]))
        output = config.targets(spec.get("output", "output"))
        reference = config.targets(spec.get("reference", "input"))[0]
        if reference in output:
            raise ValueError("Output must not include its own reference bead")
        rows, patterns = {}, []
        for case in spec["cases"]:
            current = config.updated(**case.get("overrides", {}))
            drives = deepcopy(current.values["drives"])
            scales = case.get("scales", [1.0]*len(drives))
            if len(scales) != len(drives):
                raise ValueError("Each case scale list must match the number of drives")
            for d, scale in zip(drives, scales):
                d["amplitude"] *= scale
            start, stop = case["window"]
            if stop > current.values["total_time"] or start < 0:
                raise ValueError("Measurement window lies outside simulation")
            if spec["kind"] == "memory" and any(d["amplitude"] and d["stop"] > start for d in drives):
                raise ValueError("Memory readout must follow the end of every writing drive")
            # Resolve all relevant clocks, not just the old shared T_signal.
            period = min([spec["period"]] + [d["period"] for d in drives if d["waveform"] != "constant"])
            dt = current.values["dt"]
            if period < 20*dt:
                raise ValueError("Diagnostic/drive periods need >=20 integration steps")
            interval = min(int(current.values["sampling_write"]), max(1, int(period/(20*dt))))
            current = current.updated(drives=drives, sampling_write=interval)
            reader = run_simulation(current, self.executable, folder/case["name"])
            # Fixed trap references give a consistent crossing convention when
            # different inputs change the mean displacement of an oscillator.
            theta = np.deg2rad(current.values["angles_deg"])
            xy = current.reference_positions
            references = np.column_stack([xy[:,0]*np.cos(theta)+xy[:,1]*np.sin(theta),
                                          -xy[:,0]*np.sin(theta)+xy[:,1]*np.cos(theta)])
            analyser = MultiBeadCoherenceAnalyzer(reader.time_series, reader.analysis_positions,
                                                  references, self.hysteresis)
            row, pattern = measure_window(analyser, current, spec, output, reference,
                                          start, stop, self.slip_hysteresis)
            row["expected"] = case.get("expected")
            if pattern is not None:
                patterns.append(pattern)
            row["windows"] = []
            for begin, end in case.get("windows", []):
                if end > current.values["total_time"]:
                    raise ValueError("Diagnostic window lies outside simulation")
                if spec["kind"] == "memory" and any(d["amplitude"] and d["stop"] > begin for d in drives):
                    raise ValueError("Memory diagnostic windows must follow writing")
                diagnostic, _ = measure_window(analyser, current, spec, output, reference,
                                               begin, end, self.slip_hysteresis)
                row["windows"].append(diagnostic)
            rows[case["name"]] = row
            save_json(folder/case["name"]/"metrics.json", row)
        if spec["kind"] == "control":
            # Reward opening the input path; penalise control-only entrainment.
            score = rows["both"]["R"]-rows["input"]["R"]-max(0, rows["control"]["R"]-rows["neither"]["R"])
        elif spec["kind"] == "logic":
            score = 1-np.mean([abs(r["R"]-r["expected"]) for r in rows.values()])
            for r in rows.values():
                r["correct"] = int(r["R"] >= spec.get("threshold", .8)) == r["expected"]
        else:
            # Separation of the two mean neighbour-phase patterns (0..1).
            # Unstable patterns have small mean phasors and cannot score highly.
            score = float(np.mean(np.abs(patterns[0]-patterns[1]))/2)
        return {"score": float(score), "cases": rows, "seed": config.values["seed"]}


def plot_results(record, config, output):
    """Small common report: physical layout and measured case responses."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    xy = config.reference_positions
    theta = np.deg2rad(config.values["angles_deg"])
    axes[0].scatter(xy[:,0], xy[:,1])
    axes[0].quiver(xy[:,0], xy[:,1], np.cos(theta), np.sin(theta), angles="xy")
    for i, point in enumerate(xy):
        axes[0].annotate(str(i), point)
    axes[0].set(xlabel="Reference x", ylabel="Reference y", title="Beads and stroke directions", aspect="equal")
    names = list(record["replicates"][0]["cases"])
    values = np.array([[r["cases"][n]["R"] for n in names] for r in record["replicates"]])
    sem = values.std(axis=0, ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.zeros(len(names))
    axes[1].bar(names, values.mean(axis=0), yerr=sem)
    axes[1].set(ylabel="Mean output-to-reference R", ylim=(0, 1.05), title="Cases (error bars: repeat SEM)")
    fig.savefig(Path(output)/"response.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--exe", default="src/rowers")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--cores", type=int, default=3)
    p.add_argument("--sweep", help="One numerical coordinate, e.g. drives[1].amplitude")
    p.add_argument("--values", help="JSON list of sweep values; requires --sweep")
    p.add_argument("--keep", action="store_true", help="Retain raw trajectories and per-case metrics")
    args = p.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    config = load_configuration(args.config)
    spec = json.loads(Path(args.experiment).read_text())
    evaluator = ExperimentEvaluator(args.exe, output, args.runs, args.cores,
                                    keep=args.keep, experiment=spec)
    if bool(args.sweep) != bool(args.values):
        raise ValueError("Supply both --sweep and --values")
    if args.sweep:
        # Reuse ParameterSpace to reject ignored/structural coordinates and to
        # enforce exactly the same parameter rules as optimisation.
        from simulation import ParameterSpace
        values = np.asarray(json.loads(args.values), dtype=float)
        if values.ndim != 1 or len(values)<2 or not np.all(np.isfinite(values)) or values.min()==values.max():
            raise ValueError("A sweep needs at least two distinct finite values")
        space = ParameterSpace(config, {args.sweep: {"bounds": [float(values.min()), float(values.max())],
                                                     "initial": float(values[0])}})
        if len(space.initial) != 1:
            raise ValueError("Sweep one scalar or indexed coordinate")
        records = []
        for value in values:
            record = evaluator.evaluate(space.configuration([value]))
            records.append({"value": float(value), **record})
            save_json(output/"sweep.json", {"parameter": args.sweep, "records": records})
        fig, axes = plt.subplots(2,1,figsize=(8,7),constrained_layout=True)
        axes[0].plot(values,[r["score"] for r in records],"o-",label="Objective")
        axes[0].set_ylabel("Mean score")
        for name in records[0]["replicates"][0]["cases"]:
            response = [np.mean([rep["cases"][name]["R"] for rep in r["replicates"]]) for r in records]
            axes[1].plot(values,response,"o-",label=name)
        axes[1].set(xlabel=args.sweep,ylabel="Mean output-to-reference R",ylim=(0,1.05))
        axes[1].legend()
        fig.savefig(output/"sweep.png",dpi=160)
        plt.close(fig)
        print(f"Sweep results in {output}")
    else:
        record = evaluator.evaluate(config)
        save_json(output/"result.json", record)
        plot_results(record, config, output)
        print(f"Score {record['score']:.6g}; results in {output}")


if __name__ == "__main__":
    main()
