#!/usr/bin/env python3
"""Plot and summarise an existing rowers trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a completed rowers simulation."
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to a .dat trajectory, e.g. 6beads/6beads.dat",
    )
    parser.add_argument(
        "--ref-bead",
        type=int,
        default=0,
        help="Reference bead for coherence; numbering starts at 0.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Paired crossings per coherence window (default: 10).",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Duration of the final trajectory segment to plot.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory; default: analysis_output beside the trajectory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figures in plot windows.",
    )
    args = parser.parse_args()

    if not args.trajectory.is_file():
        parser.error(f"Trajectory not found: {args.trajectory}")
    if args.window < 2:
        parser.error("--window must be at least 2.")
    if args.zoom <= 0:
        parser.error("--zoom must be positive.")

    # Load times and positions using the existing project reader.
    data = BeadDataAnalyzer()
    data.read_data_file(str(args.trajectory))

    t = data.time_series
    positions = data.positions
    n_beads = data.num_beads

    if len(t) < 2 or not np.all(np.isfinite(positions)):
        parser.error("Trajectory needs at least two rows and finite positions.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        parser.error("Recorded times must be finite and strictly increasing.")
    if not 0 <= args.ref_bead < n_beads:
        parser.error(f"--ref-bead must be between 0 and {n_beads - 1}.")

    # Reuse the supplied coherence calculation without changing its definition.
    analyzer = MultiBeadCoherenceAnalyzer(t, positions)
    coherence = analyzer.coherence_windowed_real_sum(
        ref_bead=args.ref_bead,
        dimension=0,
        periods_per_window=args.window,
    )

    # Reuse the existing positive-direction crossing detector.
    # This helper has a leading underscore but is available in the supplied class.
    crossing_times = [
        np.asarray(analyzer._get_zero_crossing_times(i, dimension=0))
        for i in range(n_beads)
    ]
    periods = np.array([
        np.mean(np.diff(times)) if len(times) >= 2 else np.nan
        for times in crossing_times
    ])

    outdir = (
        args.outdir
        if args.outdir is not None
        else args.trajectory.parent / "analysis_output"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.trajectory.stem

    # Subtract mean positions to make different beads' strokes comparable.
    # These are mean-centred coordinates, not displacements from known references.
    x = positions[:, :, 0]
    x_centred = x - np.mean(x, axis=0)
    colours = plt.get_cmap("tab10")

    # Figure 1: full trajectory and a close-up of its final segment.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    zoom_mask = t >= t[-1] - args.zoom

    for i in range(n_beads):
        colour = colours(i % 10)
        axes[0].plot(
            t, x_centred[:, i], color=colour,
            linewidth=0.8, label=f"Bead {i}",
        )
        axes[1].plot(
            t[zoom_mask], x_centred[zoom_mask, i],
            color=colour, linewidth=1.2,
        )

    axes[0].set_title(f"{prefix}: full trajectory")
    axes[0].legend(ncol=min(n_beads, 4), fontsize=9)
    axes[1].set_title(f"Final {args.zoom:g} recorded time units")

    for ax in axes:
        ax.set_xlabel("Recorded time")
        ax.set_ylabel("x − mean(x)")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    trajectory_plot = outdir / f"{prefix}_trajectories.png"
    fig.savefig(trajectory_plot, dpi=200)

    # Figure 2: coherence of each bead relative to the selected reference.
    fig_coh, ax = plt.subplots(figsize=(8, 4))
    indices = np.arange(n_beads)
    finite = np.isfinite(coherence)

    ax.bar(
        indices[finite],
        coherence[finite],
        color=[colours(i % 10) for i in indices[finite]],
    )

    # Mark missing measurements explicitly rather than plotting them as zero.
    for i in indices[~finite]:
        ax.text(i, 0.05, "N/A", ha="center", fontsize=9)

    ax.set(
        xticks=indices,
        xlabel="Bead index",
        ylabel="Coherence",
        ylim=(0, 1.05),
        title=f"Coherence relative to bead {args.ref_bead}",
    )
    ax.grid(axis="y", alpha=0.2)
    fig_coh.tight_layout()

    coherence_plot = outdir / f"{prefix}_coherence.png"
    fig_coh.savefig(coherence_plot, dpi=200)

    # Convert missing numerical results to JSON null rather than NaN.
    def json_number(value):
        return float(value) if np.isfinite(value) else None

    notes = [
        "Coherence uses the unchanged bead_analysis.py implementation.",
        "That implementation uses exp(i * lag_in_cycles), without the "
        "conventional factor of 2*pi.",
        "Times are read from the trajectory; the simulator's frame-derived "
        "timestamp convention is not corrected here.",
        "Periods use all positive-direction mean crossings, including transients.",
        "A single trajectory does not provide a forward-minus-reverse diode score.",
    ]

    summary = {
        "trajectory": str(args.trajectory),
        "num_beads": n_beads,
        "num_samples": len(t),
        "recorded_start_time": float(t[0]),
        "recorded_end_time": float(t[-1]),
        "reference_bead": args.ref_bead,
        "crossings_per_window": args.window,
        "beads": [
            {
                "index": i,
                "positive_crossings": len(crossing_times[i]),
                "mean_period": json_number(periods[i]),
                "coherence": json_number(coherence[i]),
            }
            for i in range(n_beads)
        ],
        "endpoint_coherence_0_to_last": (
            json_number(coherence[-1]) if args.ref_bead == 0 else None
        ),
        "notes": notes,
    }

    summary_path = outdir / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Print a compact report.
    print(f"\nLoaded {args.trajectory}")
    print(f"{n_beads} beads; {len(t)} recorded samples")
    print(f"\n{'Bead':>6} {'Crossings':>10} {'Mean period':>14} {'Coherence':>12}")
    for i in range(n_beads):
        print(
            f"{i:6d} {len(crossing_times[i]):10d} "
            f"{periods[i]:14.6g} {coherence[i]:12.6f}"
        )

    if args.ref_bead == 0:
        print(f"\nEndpoint coherence: {coherence[-1]:.6f}")

    print("\nCoherence uses the existing implementation, including its phase convention.")
    print("\nSaved:")
    for path in (trajectory_plot, coherence_plot, summary_path):
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_coh)


if __name__ == "__main__":
    main()