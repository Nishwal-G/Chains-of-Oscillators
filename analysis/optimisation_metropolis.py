#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import datetime
from optimisation_common import DirectionalEvaluator


MIN_DISTANCE = 3.1   # hard lower bound between any two adjacent beads 


class BeadMetropolisoptimiser(DirectionalEvaluator):
    def optimise(self,
                 chain_length=None,
                 init_positions=None,
                 init_T=0.3,
                 init_F0=2.5,
                 # proposal steps
                 pos_step=0.5,
                 T_step=0.02,
                 F0_step=0.2,
                 # parameter bounds
                 T_range=(0.01, 2.0),
                 F0_range=(0.5, 10.0),
                 # optimisation settings
                 n_iterations=50,
                 temperature=0.05,
                 n_runs=3,
                 total_time=100,
                 num_cores=4,
                 output_dir="metro_opt"):

        os.makedirs(output_dir, exist_ok=True)

        if chain_length is not None:
            self.chain_length = float(chain_length)

        self._prepare_search(T_range, F0_range, n_runs, total_time, num_cores,
                             n_iterations, output_dir)

        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Metropolis temperature must be positive.")
        if not np.all(np.isfinite([pos_step, T_step, F0_step])) or min(pos_step, T_step, F0_step) < 0:
            raise ValueError("Proposal standard deviations must be finite and nonnegative.")
        if not (T_range[0] <= init_T <= T_range[1] and F0_range[0] <= init_F0 <= F0_range[1]):
            raise ValueError("Initial period and force must lie within bounds.")
        if init_positions is not None:
            current_pos = np.array(init_positions, dtype=float)
        else:
            # equal spacing
            spacing = self.chain_length / (self.N - 1)
            current_pos = np.array(
                [spacing * i for i in range(1, self.N - 1)], dtype=float
            )
        current_T   = float(np.clip(init_T,   T_range[0],  T_range[1]))
        current_F0  = float(np.clip(init_F0, F0_range[0], F0_range[1]))

        if len(current_pos) != self.num_inner:
            raise ValueError(
                f"Expected {self.num_inner} inner positions for N={self.N}, "
                f"got {len(current_pos)}."
            )
        if not self._is_valid(current_pos):
            raise ValueError(
                f"Initial positions violate MIN_DISTANCE={MIN_DISTANCE} µm. "
                f"Gaps: {self._positions_to_distances(current_pos)}"
            )

        ndim = self.num_inner + 2   # positions + T + F0
        print(f"\n{'='*62}")
        print(f"Metropolis optimisation  —  {ndim}-D parameter space")
        print(f"  Beads: N={self.N}  |  inner (free): {self.num_inner}  "
              f"|  chain length: {self.chain_length:.2f} µm")
        print(f"  Initial positions (µm): {[f'{p:.3f}' for p in current_pos]}")
        print(f"  Initial distances (µm): {[f'{d:.3f}' for d in self._positions_to_distances(current_pos)]}")
        print(f"  Initial T={current_T:.4f} s,  F0={current_F0:.4f}")
        print(f"  Iterations: {n_iterations}  |  n_runs: {n_runs}  |  cores: {num_cores}")
        print(f"  Step sizes:  pos={pos_step} µm,  T={T_step} s,  F0={F0_step}")
        print(f"{'='*62}\n")

        f_curr, r_curr, asym_curr = self.evaluate_state(
            current_pos, current_T, current_F0,
            n_runs, total_time, num_cores,
        )
        current_energy = -asym_curr
        print(f"[Iter 0]  Fwd={f_curr:.4f}  Bwd={r_curr:.4f}  Asym={asym_curr:.4f}")

        history = {
            "iteration": [0],
            "positions": [current_pos.tolist()],
            "distances": [self._positions_to_distances(current_pos)],
            "T_signal":  [current_T],
            "F0":        [current_F0],
            "forward_R":   [f_curr],
            "backward_R":   [r_curr],
            "asymmetry": [asym_curr],
            "accepted":  [True],
            "config": {
                "N":            self.N,
                "chain_length": self.chain_length,
                "T_range":      list(T_range),
                "F0_range":     list(F0_range),
                "temperature":  temperature,
                "n_runs":       n_runs,
                "pos_step":     pos_step,
                "T_step":       T_step,
                "F0_step":      F0_step,
                "timestamp":    datetime.datetime.now().isoformat(),
                "protocol": self.protocol,
            },
        }

        best_asym  = asym_curr
        best_state = (current_pos.copy(), current_T, current_F0)
        auto_rejected = 0

        for i in range(1, n_iterations + 1):
            # proposal
            next_pos = current_pos + np.random.normal(0, pos_step, size=self.num_inner)
            # Reject out-of-bounds Gaussian proposals instead of clipping them.
            # Clipping creates asymmetric proposal mass at the bounds.
            next_T = float(current_T + np.random.normal(0, T_step))
            next_F0 = float(current_F0 + np.random.normal(0, F0_step))

            print(f"\n--- Iteration {i}/{n_iterations} ---")

            # auto rejections
            if not (self._is_valid(next_pos) and T_range[0] <= next_T <= T_range[1]
                    and F0_range[0] <= next_F0 <= F0_range[1]):
                auto_rejected += 1
                min_gap = min(self._positions_to_distances(next_pos))
                print(f"AUTO-REJECTED (geometry or parameter bounds; min gap {min_gap:.3f} µm, "
                      f"{auto_rejected} total)")
                history["iteration"].append(i)
                history["positions"].append(current_pos.tolist())
                history["distances"].append(self._positions_to_distances(current_pos))
                history["T_signal"].append(current_T)
                history["F0"].append(current_F0)
                history["forward_R"].append(f_curr)
                history["backward_R"].append(r_curr)
                history["asymmetry"].append(asym_curr)
                history["accepted"].append(False)
                if i % 5 == 0:
                    self.save_history(history, output_dir)
                    self.plot_history(history, output_dir)
                continue

            print(f"Proposing:  pos={[f'{p:.2f}' for p in next_pos]}  "
                  f"T={next_T:.4f}  F0={next_F0:.4f}")

            f_next, r_next, asym_next = self.evaluate_state(
                next_pos, next_T, next_F0,
                n_runs, total_time, num_cores,
            )
            next_energy = -asym_next
            delta_E     = next_energy - current_energy

            # acceptance criterion
            if delta_E < 0:
                accepted = True
            else:
                p_acc    = np.exp(-delta_E / temperature)
                accepted = np.random.random() < p_acc

            if accepted:
                print(f"ACCEPTED   Asym {asym_curr:.4f} → {asym_next:.4f}  "
                      f"(ΔE={delta_E:+.4f})")
                current_pos, current_T, current_F0 = next_pos.copy(), next_T, next_F0
                current_energy = next_energy
                asym_curr, f_curr, r_curr = asym_next, f_next, r_next
                if asym_curr > best_asym:
                    best_asym  = asym_curr
                    best_state = (current_pos.copy(), current_T, current_F0)
            else:
                p_acc_str = f"{np.exp(-delta_E / temperature):.4f}"
                print(f"REJECTED   Asym stays {asym_curr:.4f}  "
                      f"(ΔE={delta_E:+.4f}, p_acc={p_acc_str})")

            history["iteration"].append(i)
            history["positions"].append(current_pos.tolist())
            history["distances"].append(self._positions_to_distances(current_pos))
            history["T_signal"].append(current_T)
            history["F0"].append(current_F0)
            history["forward_R"].append(f_curr)
            history["backward_R"].append(r_curr)
            history["asymmetry"].append(asym_curr)
            history["accepted"].append(accepted)

            if i % 5 == 0:
                self.save_history(history, output_dir)
                self.plot_history(history, output_dir)

        # print summary
        best_pos, best_T, best_F0 = best_state
        print(f"\n{'='*62}")
        print(f"Optimisation finished!  Best asymmetry: {best_asym:.4f}")
        print(f"  positions (µm): {[f'{p:.3f}' for p in best_pos]}")
        print(f"  distances (µm): {[f'{d:.3f}' for d in self._positions_to_distances(best_pos)]}")
        print(f"  T_signal={best_T:.4f} s,  F0={best_F0:.4f}")
        print(f"  Auto-rejected proposals: {auto_rejected}/{n_iterations}")
        print(f"{'='*62}")

        history["best"] = {"positions": best_pos.tolist(), "T_signal": best_T,
                           "F0": best_F0, "score": best_asym}
        self.save_history(history, output_dir)
        self.plot_history(history, output_dir)
        return history


    # json dump
    def save_history(self, history, output_dir):
        with open(os.path.join(output_dir, "metropolis_history.json"), "w") as fh:
            json.dump(history, fh, indent=4, cls=_NumpyEncoder, allow_nan=False)

    # plots
    def plot_history(self, history, output_dir):
        iters     = history["iteration"]
        positions = np.array(history["positions"])
        distances = np.array(history["distances"])
        T_vals    = history["T_signal"]
        F0_vals   = history["F0"]
        asym_vals = history["asymmetry"]
        fwd_vals  = history["forward_R"]
        rev_vals  = history["backward_R"]
        accepted  = np.array(history["accepted"])
        cfg       = history.get("config", {})

        num_inner = positions.shape[1]
        num_gaps  = distances.shape[1]

        plt.rcParams.update({
            "font.family":     "serif",
            "font.size":       10,
            "axes.labelsize":  12,
            "axes.titlesize":  12,
            "text.usetex":     False,
        })

        fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

        # (0, 0)
        ax = axes[0, 0]
        sc = ax.scatter(T_vals, F0_vals, c=asym_vals, cmap="viridis", s=20, alpha=0.7, zorder=3)
        ax.plot(T_vals, F0_vals, "k-", alpha=0.25, linewidth=0.8)
        plt.colorbar(sc, ax=ax, label=r"Asymmetry $\Delta R$")
        ax.scatter(T_vals[0],  F0_vals[0],  c="yellow", s=100, marker="*",
                   edgecolors="black", zorder=5, label="Start")
        ax.scatter(T_vals[-1], F0_vals[-1], c="blue",   s=100, marker="X",
                   edgecolors="white",  zorder=5, label="End")
        ax.set_xlabel(r"$T_\mathrm{signal}$ (s)")
        ax.set_ylabel(r"$F_0$")
        ax.set_title(r"Path in ($T_\mathrm{signal}$, $F_0$) space", fontweight="bold")
        ax.legend(fontsize=8)

        # (0, 1)
        ax = axes[0, 1]
        ax.plot(iters, asym_vals, color="green", linewidth=1.5)
        acc_idx = [k for k in range(len(iters)) if accepted[k]]
        ax.scatter([iters[k] for k in acc_idx],
                   [asym_vals[k] for k in acc_idx],
                   c="green", s=18, zorder=3, label="Accepted")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Asymmetry $\Delta R$")
        ax.set_title("Asymmetry history", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1, 0)
        ax = axes[1, 0]
        colors_pos = plt.cm.plasma(np.linspace(0.1, 0.9, num_inner))
        for k in range(num_inner):
            ax.plot(iters, positions[:, k], color=colors_pos[k],
                    linewidth=1.3, label=fr"$x_{{{k+1}}}$")
        L = cfg.get("chain_length", self.chain_length)
        ax.axhline(0, color="black", linestyle=":",  linewidth=0.9, label=r"$x_0$ (fixed)")
        ax.axhline(L, color="black", linestyle="--", linewidth=0.9, label=fr"$x_{{N-1}}$ = {L:.1f} µm")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Position (µm)")
        ax.set_title("Inner bead positions", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (1, 1)
        ax = axes[1, 1]
        colors_gap = plt.cm.cool(np.linspace(0.1, 0.9, num_gaps))
        for k in range(num_gaps):
            ax.plot(iters, distances[:, k], color=colors_gap[k],
                    linewidth=1.3, label=fr"$d_{k}$")
        ax.axhline(MIN_DISTANCE, color="red", linestyle="--", linewidth=0.9,
                   label=f"min = {MIN_DISTANCE} µm")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distance (µm)")
        ax.set_title("Inter-bead distances", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (2, 0)
        ax = axes[2, 0]
        ax.plot(iters, fwd_vals, "b-", linewidth=1.3, label="Forward")
        ax.plot(iters, rev_vals, "r-", linewidth=1.3, label="Backward")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Receiver phase consistency R")
        ax.set_title("Receiver signal-locking components", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2, 1)
        ax   = axes[2, 1]
        ax2  = ax.twinx()
        p1,  = ax.plot( iters, T_vals,  "m-", linewidth=1.3, label=r"$T_\mathrm{signal}$")
        p2,  = ax2.plot(iters, F0_vals, "c-", linewidth=1.3, label=r"$F_0$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$T_\mathrm{signal}$ (s)", color="m")
        ax2.set_ylabel(r"$F_0$", color="c")
        ax.set_title("Driving parameters", fontweight="bold")
        ax.legend(handles=[p1, p2], fontsize=8)

        plt.savefig(os.path.join(output_dir, "metropolis_plots.png"), dpi=300)
        plt.close()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Metropolis optimisation of bead asymmetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # optimisation settings
    parser.add_argument("--iterations",    type=int,   default=50)
    parser.add_argument("--temp",          type=float, default=0.02,
                        help="Metropolis temperature (dimensionless, ~0.01–0.1)")
    parser.add_argument("--runs",          type=int,   default=3,
                        help="Ensemble runs per evaluation point")
    parser.add_argument("--cores",         type=int,   default=6)
    parser.add_argument("--time",          type=int,   default=100,
                        help="Simulation duration (s)")
    # chain geometry
    parser.add_argument("--chain_length",   type=float, required=True,
                        help="Total chain length (µm) — distance from the first to the last bead.")
    # initial state
    parser.add_argument("--init_positions", type=float, nargs="+", default=None,
                        help="Initial inner bead positions in µm "
                             "(must be N-2 values). Defaults to equally spaced.")
    parser.add_argument("--T_init",        type=float, default=0.3)
    parser.add_argument("--F0_init",       type=float, default=2.5)
    # step sizes
    parser.add_argument("--pos_step",      type=float, default=0.5,
                        help="Std of Gaussian position proposal (µm)")
    parser.add_argument("--T_step",        type=float, default=0.02)
    parser.add_argument("--F0_step",       type=float, default=0.2)
    # parameter bounds
    parser.add_argument("--T_min",         type=float, default=0.01)
    parser.add_argument("--T_max",         type=float, default=2.0)
    parser.add_argument("--F0_min",        type=float, default=0.5)
    parser.add_argument("--F0_max",        type=float, default=10.0)
    # engine / template
    parser.add_argument("--exe",           type=str,   default="./src/rowers")
    parser.add_argument("--template",      type=str,
                        default="./templates/6beads.input")
    parser.add_argument("--analysis-start", type=float, default=7.0,
                        help="Settled analysis start after signal onset at t=5.")
    parser.add_argument("--slip-hysteresis", type=float, default=np.pi / 4)
    parser.add_argument("--hysteresis", type=float, default=None)
    parser.add_argument("--name",          type=str,   default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name  = f"metro_results_{timestamp}"
    if args.name:
        dir_name += f"_{args.name}"

    optimiser = BeadMetropolisoptimiser(
        executable=args.exe, template_input=args.template,
        analysis_start=args.analysis_start, slip_hysteresis=args.slip_hysteresis,
        hysteresis=args.hysteresis
    )
    optimiser.optimise(
        chain_length   = args.chain_length,
        init_positions = args.init_positions,
        init_T         = args.T_init,
        init_F0        = args.F0_init,
        pos_step       = args.pos_step,
        T_step         = args.T_step,
        F0_step        = args.F0_step,
        T_range        = (args.T_min,  args.T_max),
        F0_range       = (args.F0_min, args.F0_max),
        n_iterations   = args.iterations,
        temperature    = args.temp,
        n_runs         = args.runs,
        total_time     = args.time,
        num_cores      = args.cores,
        output_dir     = dir_name,
    )

