#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer


def matched_inputs(template):
    # The C++ parser reads numbers from the first line, in positional blocks.
    p = [float(x) for x in template.read_text().splitlines()[0].split()]
    if len(p) < 22 or p[0] != int(p[0]) or p[0] < 2:
        raise ValueError("Expected a chain template with at least two beads.")
    if not np.all(np.isfinite(p)) or p[20] <= 0 or p[19] <= 0:
        raise ValueError("Template needs finite parameters, positive force and period.")
    n = int(p[0])
    cursor = 22 + n - 1
    if len(p) < cursor:
        raise ValueError("Missing chain distances.")
    # Reproduce the simulator's fallback for omitted lambdas.
    lambdas = p[cursor:cursor + n]
    cursor += len(lambdas)
    lambdas += [p[21]] * (n - len(lambdas))
    tail = p[cursor:]
    # Explicit stiffness blocks prevent signal indices being read as stiffnesses.
    kx = tail[:n] if len(tail) >= n else [p[8]] * n
    if len(tail) >= n:
        tail = tail[n:]
    ky = tail[:n] if len(tail) >= n else [p[10]] * n
    if len(tail) >= n:
        tail = tail[n:]
    if tail:
        count = int(tail[0])
        if tail[0] != count or count < 0 or len(tail) != count + 1:
            raise ValueError("Malformed optional signal-bead block.")
    base = p[:22 + n - 1] + lambdas + kx + ky
    experiments = {}
    for name, bead in [("forward", 0), ("backward", n - 1), ("control", 0)]:
        values = base.copy()
        if name == "control":
            values[19] = 0.0  # Disable physical forcing, retaining its reference clock.
        experiments[name] = values + [1, bead]
    return n, p[20], experiments


def run_experiment(exe, name, values, folder):
    # Every invocation has a new folder: stale data cannot masquerade as success.
    work = folder / name
    work.mkdir()
    inp = work / f"{name}.input"
    inp.write_text(" ".join(format(x, ".17g") for x in values) + "\n")
    result = subprocess.run([str(exe), "-C", "-R", str(inp)], cwd=work,
                            capture_output=True, text=True)
    (work / "simulation.log").write_text(result.stdout + result.stderr)
    files = list((work / name).glob("*.dat"))
    # main.cpp can catch failures and still return zero; require an output too.
    if result.returncode or len(files) != 1 or not files[0].stat().st_size:
        raise RuntimeError(f"Simulation failed; inspect {work / 'simulation.log'}")
    reader = BeadDataAnalyzer()
    reader.read_data_file(str(files[0]))
    return reader, files[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("src/rowers"))
    parser.add_argument("--template", type=Path, default=Path("templates/6beads.input"))
    parser.add_argument("--outdir", type=Path, default=Path("diode_results"))
    parser.add_argument("--analysis-start", type=float, required=True)
    parser.add_argument("--analysis-end", type=float)
    parser.add_argument("--hysteresis", type=float)
    parser.add_argument("--slip-hysteresis", type=float, default=np.pi / 4)
    args = parser.parse_args()
    if not np.isfinite(args.analysis_start) or args.analysis_start <= 5:
        parser.error("--analysis-start must be after the simulator's onset at t=5.")
    if args.analysis_end is not None and (
        not np.isfinite(args.analysis_end) or args.analysis_end <= args.analysis_start
    ):
        parser.error("--analysis-end must be finite and later than analysis start.")
    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f"Executable not found: {exe}")
    n, period, inputs = matched_inputs(args.template)
    folder = args.outdir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    analysers, trajectories = {}, {}
    for name, values in inputs.items():
        print(f"Running {name}...", flush=True)
        reader, trajectory = run_experiment(exe, name, values, folder)
        if reader.num_beads != n:
            raise ValueError("Trajectory bead count does not match template.")
        # Check physical-time output: old incorrect timestamps invalidate locking.
        expected_dt = values[4] * values[13]
        if not np.allclose(np.diff(reader.time_series), expected_dt, rtol=1e-4, atol=1e-8):
            raise ValueError("Unexpected recording interval; use the Section 1 timestamp fix.")
        if abs(reader.time_series[-1] - values[5]) > 2 * expected_dt:
            raise ValueError("Trajectory ended too early or uses incorrect timestamps.")
        analysers[name] = MultiBeadCoherenceAnalyzer(
            reader.time_series, reader.positions, hysteresis=args.hysteresis)
        trajectories[name] = str(trajectory)

    # Use the SAME recorded times with defined endpoint phases in every run.
    # This prevents different phase coverage giving unequal observation windows.
    clock = analysers["forward"].time_series
    common = (clock >= args.analysis_start)
    if args.analysis_end is not None:
        common &= clock <= args.analysis_end
    for analyser in analysers.values():
        if len(analyser.time_series) != len(clock) or not np.allclose(
            analyser.time_series, clock, rtol=0, atol=1e-9
        ):
            raise ValueError("The three runs must use identical recording times.")
        for bead in (0, n - 1):
            common &= np.isfinite(analyser._get_phase(bead))
    usable = clock[common]
    enough = len(usable) >= 3 and usable[-1] - usable[0] >= 2 * period
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if enough:
        start, end = float(usable[0]), float(usable[-1])
        for name, analyser in analysers.items():
            for bead in (0, n - 1):
                role = ("unforced" if name == "control" else
                        "driven" if bead == (0 if name == "forward" else n - 1)
                        else "receiver")
                metric = analyser.measure_signal_locking(
                    bead, period, start, end, slip_hysteresis=args.slip_hysteresis)
                rows.append({"run": name, "bead": bead, "role": role, **metric})
            receiver = n - 1 if name == "forward" else 0
            beads = (0, n - 1) if name == "control" else (receiver,)
            for bead in beads:
                times, delta = analyser.get_signal_phase_difference(bead, period)
                mask = (times >= start) & (times <= end)
                # Remove only a constant for readability; physical drift is retained.
                ax = axes[0 if bead == n - 1 else 1]
                ax.plot(times[mask], delta[mask] - delta[mask][0],
                        label=f"{name}: bead {bead}")
    else:
        start = end = None
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient common endpoint phase coverage",
                    ha="center", transform=ax.transAxes)
    for ax, title in zip(axes, ["Forward receiver and its control", "Backward receiver and its control"]):
        ax.set_title(title)
        ax.set_ylabel("Change in relative phase [rad]")
        ax.grid(alpha=0.2)
        if enough:
            ax.legend()
    axes[-1].set_xlabel("Recorded time")
    fig.tight_layout()
    fig.savefig(folder / "diode_phase.png", dpi=200)
    plt.close(fig)
    receivers = {row["run"]: row for row in rows if row["role"] == "receiver"}
    difference = None
    if receivers and all(row["status"] == "ok" for row in receivers.values()):
        difference = (receivers["forward"]["phase_consistency_R"]
                      - receivers["backward"]["phase_consistency_R"])
    summary = {"status": "ok" if enough else "insufficient_common_phase_coverage",
               "signal_period": period, "signal_onset": 5.0,
               "inputs": inputs, "trajectories": trajectories,
               "requested_interval": [args.analysis_start, args.analysis_end],
               "common_interval": [start, end], "metrics": rows,
               "receiver_R_forward_minus_backward": difference,
               "notes": ["Independent initial conditions and thermal noise in each run.",
                         "Control uses a mathematical clock, with zero physical forcing.",
                         "Winding transitions are slip candidates, including steady drift.",
                         "One three-run comparison is not a robustness estimate."]}
    (folder / "diode_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("\nRun        Bead Role       delta_f [cycles/time]       R   Windings")
    for row in rows:
        if row["status"] == "ok":
            print(f"{row['run']:10} {row['bead']:4} {row['role']:10} "
                  f"{row['frequency_mismatch_cycles_per_time']:20.6g} "
                  f"{row['phase_consistency_R']:7.4f} {row['slip_candidates_total']:10}")
        else:
            print(f"{row['run']} bead {row['bead']}: N/A")
    print(f"Receiver R difference: {difference}; saved results in {folder}")


if __name__ == "__main__":
    main()