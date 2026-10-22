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
from signal_switch import signal_switch_score
from bead_analysis import MultiBeadCoherenceAnalyzer


def validate_experiment(spec):
    """Small explicit schema, so typos cannot silently change an objective."""
    allowed = {"kind", "cases", "output", "reference", "period", "threshold", "ring", "objective", "off_max", "on_min", "frequency_tolerance", "max_slips"}
    if set(spec) - allowed or spec.get("kind") not in ("control", "logic", "memory"):
        raise ValueError("Experiment kind must be control, logic or memory; unknown keys are rejected")
    if not np.isfinite(spec.get("period", 0)) or spec["period"] <= 0:
        raise ValueError("Specify a positive diagnostic period")
    if not 0 < spec.get("threshold", .8) < 1:
        raise ValueError("threshold must lie between zero and one")
    # New scoring is opt-in: existing protocols retain their original objective.
    objective = spec.get("objective", "contrast")
    if objective not in ("contrast", "switch", "signal_switch"):
        raise ValueError("objective must be contrast, switch or signal_switch")
    if spec["kind"] != "control" and any(k in spec for k in ("objective", "off_max", "on_min")):
        raise ValueError("objective/off_max/on_min apply only to control experiments")
    if objective not in ("switch", "signal_switch") and any(k in spec for k in ("off_max", "on_min")):
        raise ValueError("off_max/on_min require objective=switch")
    if objective in ("switch", "signal_switch"):
        off, on = spec.get("off_max", .3), spec.get("on_min", .9)
        if not np.all(np.isfinite([off, on])) or not 0 <= off < on <= 1:
            raise ValueError("Require 0 <= off_max < on_min <= 1")
    if objective != "signal_switch" and any(k in spec for k in ("frequency_tolerance","max_slips")):
        raise ValueError("frequency_tolerance/max_slips require signal_switch")
    if objective == "signal_switch":
        tolerance, slips = spec.get("frequency_tolerance",.01), spec.get("max_slips",0)
        if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(slips,int) or slips < 0:
            raise ValueError("Positive frequency_tolerance and nonnegative integer max_slips required")
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


def control_score(rows, spec):
    """Score each matched repeat before averaging; never threshold mean bars.

    A switch must have low input-only R AND high both-drives R. The hinge
    penalties stop rewarding a target once reached. Retain the existing
    control-only entrainment penalty; zero is the best possible switch score.
    These are design targets for phase coherence, not proof of information flow.
    """
    if spec.get("objective") == "signal_switch":
        return signal_switch_score(rows,spec)
    off_r, on_r = rows["input"]["R"], rows["both"]["R"]
    penalty = max(0.0, rows["control"]["R"] - rows["neither"]["R"])
    if spec.get("objective", "contrast") == "contrast":
        return on_r - off_r - penalty, {}
    off, on = spec.get("off_max", .3), spec.get("on_min", .9)
    off_loss, on_loss = max(0.0, off_r-off), max(0.0, on-on_r)
    return -(off_loss + on_loss + penalty), {
        "off_loss": off_loss, "on_loss": on_loss,
        "control_penalty": penalty,
        # Success refers to the two switch targets, independently of penalty.
        "switch_success": bool(off_r <= off and on_r >= on),
    }


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
    if spec.get("objective") == "signal_switch":
        if not drive or drive["waveform"] == "constant":
            raise ValueError("Signal-switch objective needs a periodic input clock")
        # Reuse the same fixed-reference phase convention for input and output.
        pair_metrics(analyser, reference, reference, start, stop, spec["period"], slip_hysteresis)
        row["input_signal"] = analyser.measure_signal_locking(reference, drive["period"],start,stop,
            signal_phase=drive["phase"],slip_hysteresis=slip_hysteresis)
        if row["input_signal"]["status"] != "ok" or any(m["status"] != "ok" for m in signals):
            raise ValueError("Incomplete signal phase coverage")
        row["pair_R"] = row["R"]  # Preserve mutual coherence as a diagnostic.
        row["R"] = float(np.mean([m["phase_consistency_R"] for m in signals]))
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
        if experiment.get("objective") == "signal_switch":
            self.protocol["code_sha256"]["signal_switch.py"] = fingerprint(Path(__file__).with_name("signal_switch.py"))
        self.protocol.update(objective="computing_experiment_v1", experiment=self.experiment)
        self.protocol["code_sha256"]["experiments.py"] = fingerprint(__file__)

    def summarise(self, replicates):
        # Report individual-repeat reliability separately from the mean score.
        if self.experiment.get("objective") in ("switch", "signal_switch"):
            successes = sum(r["switch_success"] for r in replicates)
            return {"switch_successes": successes, "switch_trials": len(replicates),
                    "switch_success_fraction": successes / len(replicates),
                    **({"input_success_fraction":float(np.mean([r["input_success"] for r in replicates])),
                        "input_loss":float(np.mean([r["input_loss"] for r in replicates]))}
                       if self.experiment.get("objective")=="signal_switch" else {}),
                    **{key: float(np.mean([r[key] for r in replicates]))
                       for key in ("off_loss", "on_loss", "control_penalty")}}
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
        if spec.get("objective") == "signal_switch":
            drives = config.values["drives"]
            if len(drives)!=2 or config.targets(drives[0]["targets"]) != [reference]:
                raise ValueError("Signal-switch requires drive 0 on the single reference/input bead and drive 1 on control")
            if config.targets(drives[1]["targets"]) != config.targets("control"):
                raise ValueError("Drive 1 must target the control group")
            if drives[0]["waveform"] == "constant" or any(d["amplitude"] <= 0 for d in drives):
                raise ValueError("Signal-switch requires a periodic input and positive drive amplitudes")
            for case in spec["cases"]:
                for start,stop in [case["window"]]+case.get("windows",[]):
                    if any(d["start"] > start or d["stop"] < stop for d in drives):
                        raise ValueError("Both drive schedules must cover all stationary measurement windows")
            expected = {"neither":[0,0],"input":[1,0],"control":[0,1],"both":[1,1]}
            if any(c.get("scales")!=expected[c["name"]] or c.get("overrides") for c in spec["cases"]):
                raise ValueError("Signal-switch requires standard four cases without overrides")
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
        diagnostics = {}
        if spec["kind"] == "control":
            score, diagnostics = control_score(rows, spec)
        elif spec["kind"] == "logic":
            score = 1-np.mean([abs(r["R"]-r["expected"]) for r in rows.values()])
            for r in rows.values():
                r["correct"] = int(r["R"] >= spec.get("threshold", .8)) == r["expected"]
        else:
            # Separation of the two mean neighbour-phase patterns (0..1).
            # Unstable patterns have small mean phasors and cannot score highly.
            score = float(np.mean(np.abs(patterns[0]-patterns[1]))/2)
        return {"score": float(score), "cases": rows, "seed": config.values["seed"], **diagnostics}


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
    spec = record.get("protocol", {}).get("experiment", {})
    if spec.get("objective") in ("switch", "signal_switch"):
        axes[1].axhline(spec.get("off_max", .3), color="tab:red", ls="--", label="Off-case maximum" if spec.get("objective")=="signal_switch" else "Input-only maximum")
        axes[1].axhline(spec.get("on_min", .9), color="tab:green", ls="--", label="Both-drives minimum")
        axes[1].legend(fontsize=8)
        axes[1].set_title(f"Switch targets met: {record['switch_successes']}/{record['switch_trials']} repeats")
    if spec.get("objective") == "signal_switch":
        axes[1].set_ylabel("Mean output-to-input-drive R")
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
