"""Two optimisers sharing the same experiments and explicit objective.

Run from the project root: python analysis/optimisation.py --help
A search JSON selects free parameters; all remaining inputs stay fixed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import uuid
import numpy as np
import matplotlib

import sys

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulation import (
    ParameterSpace,
    load_configuration,
    run_simulation,
    save_json,
    json_value,
    fingerprint,
)
from bead_analysis import measure_experiments, directional_score

OBJECTIVE = "matched_endpoint_R_difference_v1"


# Per-bead arrays and indexed search coordinates are resolved by simulation.py.
# The evaluator preserves those values and wall settings in all matched runs.
class Evaluator:
    """Independent matched triples, reused by optimisation and characterisation."""

    def __init__(
        self,
        executable,
        output,
        runs=3,
        cores=3,
        start=7.0,
        keep=False,
        all_beads=False,
        hysteresis=None,
        slip_hysteresis=np.pi / 4,
    ):
        if runs < 1 or cores < 1 or not np.isfinite(start) or start <= 5:
            raise ValueError(
                "Positive runs/cores and an analysis start after t=5 required."
            )
        if hysteresis is not None and (not np.isfinite(hysteresis) or hysteresis < 0):
            raise ValueError("Position hysteresis must be nonnegative.")
        if not np.isfinite(slip_hysteresis) or not 0 < slip_hysteresis < np.pi:
            raise ValueError("Phase-slip hysteresis must lie between zero and pi.")
        self.executable = Path(executable).resolve()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.runs, self.cores, self.start = runs, cores, start
        self.keep, self.all_beads = keep, all_beads
        self.hysteresis, self.slip_hysteresis = hysteresis, slip_hysteresis
        self.protocol = {
            "objective": OBJECTIVE,
            "runs": runs,
            "analysis_start": start,
            "hysteresis": hysteresis,
            "slip_hysteresis": slip_hysteresis,
            "executable_sha256": fingerprint(self.executable),
            "code_sha256": {
                f: fingerprint(Path(__file__).parent / f)
                for f in ("simulation.py", "bead_analysis.py", "optimisation.py")
            },
        }

    def _replicate(self, config, folder):
        p = config.values
        if p["drives"] is not None:
            raise ValueError("Legacy diode evaluation uses F0/T_signal; use --experiment for independent drives")
        if p["T_signal"] < 20 * p["dt"]:
            raise ValueError("Need at least 20 integration steps per signal period.")
        if p["total_time"] - self.start < 2 * p["T_signal"] + 2 * p["dt"]:
            raise ValueError("Simulation too short for two settled signal periods.")
        interval = min(
            int(p["sampling_write"]),
            max(1, int(np.floor(p["T_signal"] / (20 * p["dt"])))),
        )
        readers = {}
        for direction, bead in [
            ("forward", 0),
            ("backward", config.n - 1),
            ("control", 0),
        ]:
            experiment = config.updated(
                sampling_write=interval,
                signal_beads=[bead],
                F0=0.0 if direction == "control" else p["F0"],
            )
            readers[direction] = run_simulation(
                experiment, self.executable, folder / direction
            )
        rows, times, _ = measure_experiments(
            readers,
            p["T_signal"],
            self.start,
            self.all_beads,
            self.hysteresis,
            self.slip_hysteresis,
        )
        f, b, score = directional_score(rows, config.n)
        return {
            "forward_R": f,
            "backward_R": b,
            "score": score,
            "common_interval": times,
            "metrics": rows,
        }

    def evaluate(self, config):
        key = uuid.uuid4().hex
        folder = self.output / "simulation_work" / key
        # A fixed base seed produces reproducible but distinct replicates. Each
        # replicate uses matched random initial conditions across its cases.
        record = {
            "evaluation_id": key,
            "configuration": config.values,
            "protocol": self.protocol,
            "work_directory": str(folder.resolve()),
        }
        try:
            with ThreadPoolExecutor(max_workers=min(self.cores, self.runs)) as pool:
                futures = [
                    pool.submit(self._replicate, config if config.values["seed"] < 0 else config.updated(seed=(config.values["seed"] + i) % 2**32), folder / f"replicate_{i}")
                    for i in range(self.runs)
                ]
                replicates = [future.result() for future in futures]
            scores = [r["score"] for r in replicates]
            record.update(
                status="ok",
                replicates=replicates,
                **self.summarise(replicates),
                score=float(np.mean(scores)),
                score_sem=(
                    float(np.std(scores, ddof=1) / np.sqrt(self.runs))
                    if self.runs > 1
                    else None
                ),
            )
            if not self.keep:
                shutil.rmtree(folder)
                record["work_directory"] = None
        except Exception as error:
            record.update(status="failed", error=str(error))
            self._audit(record)
            raise RuntimeError(
                f"Evaluation failed; work retained at {folder}: {error}"
            ) from error
        self._audit(record)
        return record

    def summarise(self, replicates):
        """Protocol hook; execution, audit, repeats and cleanup stay shared."""
        return {key: float(np.mean([r[key] for r in replicates]))
                for key in ("forward_R", "backward_R")}

    def _audit(self, record):
        with (self.output / "objective_evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(json_value(record), allow_nan=False) + "\n")


def plot_history(history, space, output):
    """The same plots automatically accommodate any selected coordinates."""
    records = history["history"]
    if not records:
        return
    scores = np.array([r["score"] for r in records])
    x = np.array([r["vector"] for r in records])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(scores, label="Current/evaluated score")
    axes[0].plot(np.maximum.accumulate(scores), label="Best in this history")
    if history["method"] == "metropolis":
        accepted = [i for i, r in enumerate(records) if r.get("accepted")]
        axes[0].scatter(accepted, scores[accepted], s=15, label="Accepted states")
    axes[0].set_ylabel("Objective score")
    if "forward_R" in records[0]:
        axes[1].plot([r["forward_R"] for r in records], label="Forward receiver")
        axes[1].plot([r["backward_R"] for r in records], label="Backward receiver")
        axes[1].set_ylabel("Phase consistency R")
    else:
        axes[1].plot([r.get("score_sem") or 0 for r in records], label="Repeat SEM")
        axes[1].set_ylabel("Score uncertainty")
    for i, name in enumerate(space.labels):
        axes[2].plot(space.normalise(x)[:, i], label=name)
    axes[2].set_ylabel("Parameter (fraction of bounds)")
    axes[2].set_xlabel("History entry")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
    fig.savefig(output / "progress.png", dpi=180)
    plt.close(fig)
    # Keep geometry and raw driving coordinates available as separate panels.
    fig, axes = plt.subplots(
        len(space.labels) + 1,
        1,
        figsize=(9, 2.3 * (len(space.labels) + 1)),
        squeeze=False,
    )
    for i, name in enumerate(space.labels):
        axes[i, 0].plot(x[:, i])
        axes[i, 0].set_ylabel(name)
    if records[0]["configuration"].get("reference_x"):
        from simulation import Configuration
        distances = np.array([Configuration(r["configuration"]).pair_distances() for r in records])
        axes[-1, 0].plot(distances.min(axis=1), label="Closest pair")
        axes[-1, 0].set_ylabel("Minimum reference separation")
    else:
        gaps = np.array([r["configuration"]["distances"] for r in records])
        for i in range(gaps.shape[1]):
            axes[-1, 0].plot(gaps[:, i], label=f"gap {i}")
        axes[-1, 0].set_ylabel("Reference gap")
    axes[-1, 0].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output / "parameters.png", dpi=150)
    plt.close(fig)


def build_gp(dimension):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        np.ones(dimension), (0.05, 10), nu=2.5
    )
    return GaussianProcessRegressor(
        kernel=kernel + WhiteKernel(0.01, (1e-5, 0.5)),
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )


def acquisition_values(gp, x, method, best, kappa):
    from scipy.stats import norm

    mean, std = gp.predict(x, return_std=True)
    if method == "ucb":
        return mean + kappa * std
    improvement = mean - best - 0.01
    z = improvement / np.maximum(std, 1e-12)
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def optimise(args):
    base = load_configuration(args.config)
    if args.set:
        base = base.updated(**{key: json.loads(value) for key, value in args.set})
    specification = json.loads(Path(args.search).read_text())
    modes = [getattr(args, k, None) for k in ("experiment", "transistor", "device", "fast_device")]
    if sum(bool(v) for v in modes) > 1:
        raise ValueError("Choose only one experiment/objective mode")
    if getattr(args, "fast_device", None):
        if not getattr(args, "fast_spec", None):
            raise ValueError("--fast-device requires --fast-spec")
        if any(not k.startswith(("reference_x[", "reference_y[")) for k in specification):
            raise ValueError("Fast device search accepts indexed geometry coordinates only")
        if base.values["seed"] < 0:
            raise ValueError("Set a fixed simulator seed for fast search")
    elif getattr(args, "fast_spec", None):
        raise ValueError("--fast-spec requires --fast-device")
    if getattr(args, "transistor", None):
        if args.experiment:
            raise ValueError("Use either --transistor or --experiment")
        # The control sweep is fixed by the protocol; optimising this coordinate
        # would silently have no effect. Keep the first version geometry-only.
        if any(not k.startswith(("reference_x[", "reference_y[")) for k in specification):
            raise ValueError("Transistor search currently accepts indexed geometry coordinates only")
        if base.values["seed"] < 0:
            raise ValueError("Set a fixed simulator seed for matched transistor search")
    if getattr(args, "device", None):
        if args.experiment or getattr(args, "transistor", None):
            raise ValueError("Use only one of --device, --transistor or --experiment")
        if any(not k.startswith(("reference_x[", "reference_y[")) for k in specification):
            raise ValueError("Device search accepts indexed geometry coordinates only")
        if base.values["seed"] < 0:
            raise ValueError("Set a fixed simulator seed for device search")
    space = ParameterSpace(base, specification)
    if (
        args.iterations < 0
        or args.initial < 1
        or args.candidates < 1
        or args.search_temperature <= 0
        or args.kappa < 0
    ):
        raise ValueError("Invalid search settings.")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    evaluator_class = Evaluator
    extra = {}
    if args.experiment:
        from experiments import ExperimentEvaluator
        evaluator_class = ExperimentEvaluator
        extra["experiment"] = json.loads(Path(args.experiment).read_text())
    if getattr(args, "transistor", None):
        from transistor import TransistorEvaluator
        evaluator_class = TransistorEvaluator
        extra["transistor"] = json.loads(Path(args.transistor).read_text())
    if getattr(args, "device", None):
        from transistor_device import DeviceEvaluator, check_source
        evaluator_class = DeviceEvaluator
        extra["device"] = json.loads(Path(args.device).read_text())
        check_source(base, extra["device"])
    if getattr(args, "fast_device", None):
        from transistor_fast import FastDeviceEvaluator
        from transistor_device import check_source
        evaluator_class = FastDeviceEvaluator
        extra["device"] = json.loads(Path(args.fast_device).read_text())
        extra["fast"] = json.loads(Path(args.fast_spec).read_text())
        check_source(base, extra["device"])
    evaluator = evaluator_class(
        args.exe,
        output,
        args.runs,
        args.cores,
        args.start,
        args.keep,
        hysteresis=args.hysteresis,
        slip_hysteresis=args.slip_hysteresis,
        **extra,
    )
    rng = np.random.default_rng(args.seed)
    history = {
        "method": args.method,
        "base_configuration": base.values,
        "free_parameters": specification,
        "coordinate_labels": space.labels,
        "protocol": evaluator.protocol,
        "settings": vars(args),
        "history": [],
    }
    observations = []
    # Only actual evaluations are training data, never repeated residence states.
    if args.warm_start:
        if args.method != "bayesian":
            raise ValueError("Warm start is supported for Bayesian search only.")
        for line in Path(args.warm_start).read_text().splitlines():
            record = json.loads(line)
            if record.get("protocol") != evaluator.protocol:
                raise ValueError(
                    "Warm-start protocol differs; do not mix measurements."
                )
            if record.get("status") != "ok":
                continue
            config = record["configuration"]
            from simulation import Configuration

            c = Configuration(config)
            vector = np.concatenate([np.atleast_1d(c.get(k)) for k in specification])
            rebuilt = space.configuration(vector)
            if json_value(rebuilt.values) != json_value(c.values):
                raise ValueError("Warm-start fixed parameters differ.")
            observations.append({**record, "vector": vector.tolist()})
    best = max(observations, key=lambda r: r["score"]) if observations else None

    def evaluate(vector):
        nonlocal best
        result = evaluator.evaluate(space.configuration(vector))
        result["vector"] = vector.tolist()
        observations.append(result)
        if best is None or result["score"] > best["score"]:
            best = result
        return result

    def checkpoint():
        history["best"] = best
        save_json(output / "history.json", history)
        save_json(output / "best.json", best)

    if args.method == "metropolis":
        current = evaluate(space.initial)
        history["history"].append({**current, "accepted": True})
        checkpoint()
        for iteration in range(args.iterations):
            candidate = (
                np.asarray(current["vector"])
                + rng.normal(size=len(space.lo)) * space.steps
            )
            try:
                space.configuration(candidate)
            except ValueError:
                accepted, reason = False, "outside valid parameter space"
            else:
                trial = evaluate(candidate)
                delta = trial["score"] - current["score"]
                accepted = delta >= 0 or rng.random() < np.exp(
                    delta / args.search_temperature
                )
                reason = "evaluated"
                if accepted:
                    current = trial
            history["history"].append(
                {**current, "accepted": bool(accepted), "proposal_status": reason}
            )
            checkpoint()
            print(
                f'Metropolis {iteration+1}: current={current["score"]:.4f}, best={best["score"]:.4f}',
                flush=True,
            )
    else:
        initial = (
            [space.initial] + list(space.sample(rng, args.initial - 1))
            if args.initial > 1
            else [space.initial]
        )
        for vector in initial:
            history["history"].append({**evaluate(vector), "phase": "initial"})
            checkpoint()
        gp = build_gp(len(space.lo))
        for iteration in range(args.iterations):
            gp.fit(
                space.normalise([r["vector"] for r in observations]),
                [r["score"] for r in observations],
            )
            candidates = space.sample(rng, args.candidates)
            acquisition = acquisition_values(
                gp,
                space.normalise(candidates),
                args.acquisition,
                best["score"],
                args.kappa,
            )
            index = int(np.argmax(acquisition))
            history["history"].append(
                {
                    **evaluate(candidates[index]),
                    "phase": "bayesian",
                    "acquisition": float(acquisition[index]),
                }
            )
            checkpoint()
            print(f'Bayesian {iteration+1}: best={best["score"]:.4f}', flush=True)
        gp.fit(
            space.normalise([r["vector"] for r in observations]),
            [r["score"] for r in observations],
        )
        if len(space.lo) >= 2:
            # First two scalar coordinates are preferred; fallback to two positions.
            scalar = [s.start for k, s in space.slices.items() if s.stop - s.start == 1]
            a, b = scalar[:2] if len(scalar) >= 2 else (0, 1)
            xx, yy = np.meshgrid(
                np.linspace(space.lo[a], space.hi[a], 50),
                np.linspace(space.lo[b], space.hi[b], 50),
            )
            vectors = np.tile(best["vector"], (xx.size, 1))
            vectors[:, a] = xx.ravel()
            vectors[:, b] = yy.ravel()
            mean, std = gp.predict(space.normalise(vectors), return_std=True)
            for i, v in enumerate(vectors):
                try:
                    space.configuration(v)
                except ValueError:
                    mean[i] = std[i] = np.nan
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for ax, z, title in zip(
                axes, (mean, std), ("GP predicted score", "GP uncertainty")
            ):
                im = ax.pcolormesh(xx, yy, z.reshape(xx.shape), shading="auto")
                fig.colorbar(im, ax=ax)
                ax.set(xlabel=space.labels[a], ylabel=space.labels[b], title=title)
            fig.savefig(output / "gp_slice.png", dpi=170)
            plt.close(fig)
    checkpoint()
    plot_history(history, space, output)
    print(f'Best observed score: {best["score"]:.5f}; saved to {output}/best.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["metropolis", "bayesian"], default="metropolis")
    p.add_argument("--config", default="templates/6beads.input")
    p.add_argument("--search", default="search.json")
    p.add_argument("--fast-device", help="Device protocol for the cheaper transfer-shape search")
    p.add_argument("--fast-spec", help="Fast search grid and weights JSON")
    p.add_argument("--device", help="Weak-control device protocol with input-phase intervention and cycle objective")
    p.add_argument("--transistor", help="Opt-in multi-amplitude coherence transistor protocol JSON")
    p.add_argument("--experiment", help="Experiment JSON; omitted preserves the existing diode objective")
    p.add_argument(
        "--set",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON_VALUE"),
        help="Override fixed inputs, e.g. --set total_time 100.",
    )
    p.add_argument("--exe", default="src/rowers")
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--cores", type=int, default=3)
    p.add_argument("--start", type=float, default=7.0)
    p.add_argument(
        "--hysteresis",
        type=float,
        help="Crossing confirmation half-width in position units.",
    )
    p.add_argument("--slip-hysteresis", type=float, default=np.pi / 4)
    p.add_argument("--search-temperature", type=float, default=0.02)
    p.add_argument("--initial", type=int, default=12)
    p.add_argument("--candidates", type=int, default=50000)
    p.add_argument("--acquisition", choices=["ucb", "ei"], default="ucb")
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Python proposal seed. Use --set seed INTEGER to seed simulator repeats.",
    )
    p.add_argument(
        "--warm-start",
        help="New-pipeline objective_evaluations.jsonl with matching protocol/fixed inputs.",
    )
    p.add_argument(
        "--keep", action="store_true", help="Keep successful raw simulations too."
    )
    optimise(p.parse_args())


if __name__ == "__main__":
    main()
