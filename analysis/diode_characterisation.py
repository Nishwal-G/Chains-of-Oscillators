"""Characterise a chain or analyse saved trajectories.

Modes: trajectory, frequency, mechanism and diagram. Information-transfer
experiments require a future C++ signal extension and are not implemented here.
The trajectory mode retains the previous plots and JSON summary.
"""

import matplotlib

import sys

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bead_analysis import (
    BeadDataAnalyzer,
    MultiBeadCoherenceAnalyzer,
)


def trajectory_main(argv):
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
        help="Reference cycles per coherence window (default: 10).",
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
        help="Default: analysis_output beside the trajectory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figures in plot windows.",
    )
    parser.add_argument(
        "--reference-x", "--reference-longitudinal",
        type=float,
        nargs="+",
        default=None,
        help="Fixed local longitudinal reference coordinates, one per bead (x for zero angles).",
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=None,
        help=(
            "Position confirmation-band half-width. "
            "Default: 5%% of estimated oscillation amplitude."
        ),
    )

    # SECTION 2:
    # Supply the ACTUAL forcing period from the simulation input.
    # Without this argument, the previous analysis command still works.
    parser.add_argument(
        "--signal-period",
        type=float,
        default=None,
        help="Actual forcing period; enables signal-locking analysis.",
    )

    # Explicitly exclude startup and the response to switching on the signal.
    # Choosing this cutoff remains a physical judgement checked with plots.
    parser.add_argument(
        "--analysis-start",
        type=float,
        default=None,
        help="Settled interval start; required with --signal-period.",
    )
    parser.add_argument(
        "--analysis-end",
        type=float,
        default=None,
        help="Settled interval end; default: final recorded time.",
    )

    # Current C++ force switches on after time=5, using its original clock.
    # This argument describes the simulation; it does not change the force.
    parser.add_argument(
        "--signal-onset",
        type=float,
        default=5.0,
        help="Force switch-on time; current simulator uses 5.",
    )
    parser.add_argument(
        "--signal-phase",
        type=float,
        default=0.0,
        help="Offset in phi_signal=2*pi*time/T+offset, in radians.",
    )

    # This is phase hysteresis, distinct from position hysteresis above.
    parser.add_argument(
        "--slip-hysteresis",
        type=float,
        default=np.pi / 4.0,
        help="Phase-branch confirmation margin; default: pi/4 radians.",
    )

    args = parser.parse_args(argv)

    if not args.trajectory.is_file():
        parser.error(f"Trajectory not found: {args.trajectory}")

    if args.window < 2:
        parser.error("--window must be at least 2.")

    if not np.isfinite(args.zoom) or args.zoom <= 0:
        parser.error("--zoom must be finite and positive.")

    if args.hysteresis is not None:
        if not np.isfinite(args.hysteresis) or args.hysteresis < 0:
            parser.error("--hysteresis must be finite and nonnegative.")

    data = BeadDataAnalyzer()

    try:
        data.read_data_file(str(args.trajectory))
    except (ValueError, OSError) as error:
        parser.error(str(error))

    t = data.time_series
    # All crossing, motion and locking plots use the bead-local coordinates.
    # The reader retains laboratory x/y in data.positions.
    positions = data.analysis_positions
    n_beads = data.num_beads

    if not np.all(np.isfinite(positions)):
        parser.error("Trajectory positions must be finite.")

    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        parser.error("Recorded times must be finite and strictly increasing.")

    if not 0 <= args.ref_bead < n_beads:
        parser.error(f"--ref-bead must be between 0 and {n_beads - 1}.")

    # SECTION 2: validate the requested settled interval.
    signal_enabled = args.signal_period is not None
    analysis_mask = None

    if signal_enabled:
        if not np.isfinite(args.signal_period) or args.signal_period <= 0:
            parser.error("--signal-period must be finite and positive.")

        if args.analysis_start is None or not np.isfinite(args.analysis_start):
            parser.error("Supply a finite --analysis-start with --signal-period.")

        if not np.isfinite(args.signal_onset) or not np.isfinite(args.signal_phase):
            parser.error("Signal onset and phase must be finite.")

        if (
            not np.isfinite(args.slip_hysteresis)
            or not 0 < args.slip_hysteresis < np.pi
        ):
            parser.error("--slip-hysteresis must lie strictly between 0 and pi.")

        if args.analysis_end is None:
            args.analysis_end = float(t[-1])

        if (
            not np.isfinite(args.analysis_end)
            or args.analysis_start < t[0]
            or args.analysis_end > t[-1]
            or args.analysis_start >= args.analysis_end
        ):
            parser.error(
                "Analysis interval must lie within recorded times, " "with start < end."
            )

        if args.analysis_start <= args.signal_onset:
            parser.error(
                "--analysis-start must be after signal onset "
                "and should exclude settling."
            )

        analysis_mask = (t >= args.analysis_start) & (t <= args.analysis_end)

        if np.count_nonzero(analysis_mask) < 3:
            parser.error("Analysis interval needs at least three recorded samples.")

    elif args.analysis_start is not None or args.analysis_end is not None:
        parser.error("Analysis interval options require --signal-period.")

    reference_positions = None

    if args.reference_x is not None:
        if len(args.reference_x) != n_beads or not np.all(
            np.isfinite(args.reference_x)
        ):
            parser.error("--reference-x needs one finite coordinate per bead.")

        reference_positions = positions.mean(axis=0)
        reference_positions[:, 0] = args.reference_x

    elif signal_enabled:
        # SECTION 2:
        # Define the crossing centre using the settled interval's mean.
        # This avoids mixing the startup transient into the later centre.
        #
        # This remains a CONSTANT mean, not a moving-centre estimator.
        #
        # Reconstruct phase on the full record so crossings around the
        # interval boundaries remain available. Restrict statistics later.
        reference_positions = positions[analysis_mask].mean(axis=0)

    level_source = (
        "supplied_reference_x"
        if args.reference_x is not None
        else "settled_interval_mean" if signal_enabled else "full_run_mean"
    )

    analyzer = MultiBeadCoherenceAnalyzer(
        t,
        positions,
        reference_positions=reference_positions,
        hysteresis=args.hysteresis,
    )

    # Retain the existing bead-to-bead summary.
    # Its windows still cover the full record, with the first 10% dropped.
    coherence = analyzer.coherence_windowed_real_sum(
        ref_bead=args.ref_bead,
        dimension=0,
        periods_per_window=args.window,
    )

    # SECTION 2:
    # New signal metrics use only the requested settled interval.
    # Each bead's actual phase coverage is included in its result.
    locking = None

    if signal_enabled:
        locking = [
            analyzer.measure_signal_locking(
                i,
                args.signal_period,
                args.analysis_start,
                args.analysis_end,
                dimension=0,
                signal_phase=args.signal_phase,
                slip_hysteresis=args.slip_hysteresis,
            )
            for i in range(n_beads)
        ]

    crossing_times = [
        np.asarray(analyzer._get_zero_crossing_times(i, dimension=0))
        for i in range(n_beads)
    ]

    # These legacy periods include the full record, including transients.
    periods = np.array(
        [
            np.mean(np.diff(times)) if len(times) >= 2 else np.nan
            for times in crossing_times
        ]
    )

    settings = [analyzer._get_crossing_settings(i, dimension=0) for i in range(n_beads)]
    levels = np.array([level for level, width in settings])
    widths = np.array([width for level, width in settings])

    outdir = (
        args.outdir
        if args.outdir is not None
        else args.trajectory.parent / "analysis_output"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.trajectory.stem
    colours = plt.get_cmap("tab10")
    figures = []
    indices = np.arange(n_beads)
    saved_paths = []

    x = positions[:, :, 0]

    # Centre plots on the same levels used by the crossing detector.
    x_centred = x - levels
    zoom_mask = t >= t[-1] - args.zoom

    centre_label = (
        "Longitudinal coordinate − reference"
        if args.reference_x is not None
        else "Longitudinal displacement from settled mean" if signal_enabled else "Longitudinal displacement from mean"
    )

    plot_trajectory_motion(
        args,
        centre_label,
        colours,
        figures,
        n_beads,
        outdir,
        prefix,
        saved_paths,
        t,
        x_centred,
        zoom_mask,
    )
    plot_trajectory_coherence(
        args, coherence, colours, figures, n_beads, outdir, prefix, saved_paths
    )
    plot_trajectory_crossings(
        centre_label,
        colours,
        crossing_times,
        figures,
        n_beads,
        outdir,
        prefix,
        saved_paths,
        t,
        widths,
        x_centred,
        zoom_mask,
    )
    plot_trajectory_phases(
        analyzer, args, colours, figures, n_beads, outdir, prefix, saved_paths
    )
    plot_trajectory_signal(
        analyzer,
        args,
        colours,
        figures,
        indices,
        locking,
        n_beads,
        outdir,
        prefix,
        saved_paths,
        signal_enabled,
    )
    write_trajectory_summary(
        args,
        coherence,
        crossing_times,
        figures,
        level_source,
        levels,
        locking,
        n_beads,
        outdir,
        periods,
        prefix,
        saved_paths,
        signal_enabled,
        t,
        widths,
    )


def plot_trajectory_motion(
    args,
    centre_label,
    colours,
    figures,
    n_beads,
    outdir,
    prefix,
    saved_paths,
    t,
    x_centred,
    zoom_mask,
):
    # Figure 1: full trajectory and final segment.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    for i in range(n_beads):
        colour = colours(i % 10)

        axes[0].plot(
            t,
            x_centred[:, i],
            color=colour,
            linewidth=0.8,
            label=f"Bead {i}",
        )
        axes[1].plot(
            t[zoom_mask],
            x_centred[zoom_mask, i],
            color=colour,
            linewidth=1.2,
        )

    axes[0].set_title(f"{prefix}: full trajectory")
    axes[0].legend(ncol=min(n_beads, 4), fontsize=9)
    axes[1].set_title(f"Final {args.zoom:g} recorded time units")

    for ax in axes:
        ax.set_xlabel("Recorded time")
        ax.set_ylabel(centre_label)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    path = outdir / f"{prefix}_trajectories.png"
    fig.savefig(path, dpi=200)
    figures.append(fig)
    saved_paths.append(path)


def plot_trajectory_coherence(
    args, coherence, colours, figures, n_beads, outdir, prefix, saved_paths
):
    # Figure 2: existing bead-to-bead coherence.
    fig_coh, ax = plt.subplots(figsize=(8, 4))
    indices = np.arange(n_beads)
    finite = np.isfinite(coherence)

    ax.bar(
        indices[finite],
        coherence[finite],
        color=[colours(i % 10) for i in indices[finite]],
    )

    for i in indices[~finite]:
        ax.text(i, 0.05, "N/A", ha="center", fontsize=9)

    ax.set(
        xticks=indices,
        xlabel="Bead index",
        ylabel="Coherence",
        ylim=(0, 1.05),
        title=("Full-record windowed coherence " f"relative to bead {args.ref_bead}"),
    )
    ax.grid(axis="y", alpha=0.2)

    fig_coh.tight_layout()
    path = outdir / f"{prefix}_coherence.png"
    fig_coh.savefig(path, dpi=200)
    figures.append(fig_coh)
    saved_paths.append(path)


def plot_trajectory_crossings(
    centre_label,
    colours,
    crossing_times,
    figures,
    n_beads,
    outdir,
    prefix,
    saved_paths,
    t,
    widths,
    x_centred,
    zoom_mask,
):
    # Figure 3: inspect accepted upward crossings.
    fig_cross, axes = plt.subplots(
        n_beads,
        1,
        figsize=(11, max(3, 1.8 * n_beads)),
        sharex=True,
        squeeze=False,
    )

    for i, ax in enumerate(axes[:, 0]):
        ax.plot(
            t[zoom_mask],
            x_centred[zoom_mask, i],
            color=colours(i % 10),
            linewidth=1.1,
        )

        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(
            -widths[i],
            color="grey",
            linestyle="--",
            linewidth=0.7,
        )
        ax.axhline(
            widths[i],
            color="grey",
            linestyle="--",
            linewidth=0.7,
        )

        events = crossing_times[i]
        events = events[(events >= t[zoom_mask][0]) & (events <= t[-1])]

        # Dots mark reference crossings, not later confirmation times.
        ax.scatter(
            events,
            np.zeros(len(events)),
            color="black",
            marker="o",
            s=22,
            zorder=3,
        )
        ax.set_ylabel(f"Bead {i}\n{centre_label}")
        ax.grid(alpha=0.2)

    axes[0, 0].set_title(
        "Crossing check: dots = accepted upward crossings; " "dashed = band"
    )
    axes[-1, 0].set_xlabel("Recorded time")

    fig_cross.tight_layout()
    path = outdir / f"{prefix}_crossings.png"
    fig_cross.savefig(path, dpi=200)
    figures.append(fig_cross)
    saved_paths.append(path)


def plot_trajectory_phases(
    analyzer, args, colours, figures, n_beads, outdir, prefix, saved_paths
):
    # Figure 4: existing common-time bead-to-bead phase differences.
    fig_phase, ax = plt.subplots(figsize=(11, 4))
    plotted = False

    for i in range(n_beads):
        if i == args.ref_bead:
            continue

        phase_times, differences = analyzer._get_phase_differences(
            ref_bead=args.ref_bead,
            bead_j=i,
            dimension=0,
        )

        if len(phase_times):
            ax.plot(
                phase_times,
                differences,
                color=colours(i % 10),
                label=f"Bead {i}",
            )
            plotted = True

    ax.set(
        xlabel="Recorded time",
        ylabel="Unwrapped phase difference (rad)",
        title=(f"φᵢ(t) − φ{args.ref_bead}(t): common-time phases"),
    )

    if plotted:
        ax.legend(
            ncol=min(max(n_beads - 1, 1), 4),
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient confirmed crossings",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.grid(alpha=0.2)
    fig_phase.tight_layout()

    path = outdir / f"{prefix}_phase_differences.png"
    fig_phase.savefig(path, dpi=200)
    figures.append(fig_phase)
    saved_paths.append(path)


def plot_trajectory_signal(
    analyzer,
    args,
    colours,
    figures,
    indices,
    locking,
    n_beads,
    outdir,
    prefix,
    saved_paths,
    signal_enabled,
):
    # SECTION 2 — Figure 5:
    # Plot phi_i(t) - phi_signal(t) without wrapping or detrending.
    # Shading identifies the interval used for the numerical measurements.
    if signal_enabled:
        fig_signal, axes_signal = plt.subplots(2, 1, figsize=(11, 8))

        for i in range(n_beads):
            signal_times, differences = analyzer.get_signal_phase_difference(
                i,
                args.signal_period,
                signal_phase=args.signal_phase,
            )

            colour = colours(i % 10)

            axes_signal[0].plot(
                signal_times,
                differences,
                color=colour,
                label=f"Bead {i}",
                linewidth=1,
            )

            selected = (
                (signal_times >= args.analysis_start)
                & (signal_times <= args.analysis_end)
                & np.isfinite(differences)
            )

            axes_signal[1].plot(
                signal_times[selected],
                differences[selected],
                color=colour,
                linewidth=1,
            )

            # Triangles mark confirmed phase-branch transitions.
            # Continuous drift can also produce these slip candidates.
            for event in locking[i]["slip_events"]:
                phase_at_event = np.interp(
                    event["time"],
                    signal_times[selected],
                    differences[selected],
                )

                axes_signal[1].scatter(
                    event["time"],
                    phase_at_event,
                    color=colour,
                    marker=("^" if event["direction"] > 0 else "v"),
                    s=30,
                    zorder=3,
                )

        axes_signal[0].axvline(
            args.signal_onset,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="Signal on",
        )
        axes_signal[0].axvspan(
            args.analysis_start,
            args.analysis_end,
            color="grey",
            alpha=0.12,
            label="Analysis interval",
        )
        axes_signal[0].set_title("Unwrapped phase relative to the forcing clock")
        axes_signal[0].legend(ncol=4, fontsize=8)

        axes_signal[1].set_title(
            "Settled interval; triangles = confirmed winding "
            "transitions (slip candidates)"
        )

        for ax in axes_signal:
            ax.set_xlabel("Recorded time")
            ax.set_ylabel("phi_i(t) − (Omega*t + offset) [rad]")
            ax.grid(alpha=0.2)

        fig_signal.tight_layout()
        path = outdir / f"{prefix}_signal_phase.png"
        fig_signal.savefig(path, dpi=200)
        figures.append(fig_signal)
        saved_paths.append(path)

        # SECTION 2 — Figure 6:
        # Show frequency mismatch, circular consistency and winding counts.
        # R is distinct from the older bead-to-bead coherence formula C.
        fig_lock, axes_lock = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

        metrics = [
            (
                "frequency_mismatch_cycles_per_time",
                "Frequency mismatch [cycles/time]",
            ),
            (
                "phase_consistency_R",
                "Phase consistency R",
            ),
            (
                "slip_candidates_total",
                "Winding transitions [count]",
            ),
        ]

        for ax, (key, label) in zip(axes_lock, metrics):
            values = np.array(
                [row[key] if row[key] is not None else np.nan for row in locking]
            )
            finite_values = np.isfinite(values)

            ax.bar(
                indices[finite_values],
                values[finite_values],
                color=[colours(i % 10) for i in indices[finite_values]],
            )

            for i in indices[~finite_values]:
                ax.text(
                    i,
                    0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)

        axes_lock[0].axhline(0, color="black", linewidth=0.7)
        axes_lock[1].set_ylim(0, 1.05)
        axes_lock[-1].set_xticks(indices)
        axes_lock[-1].set_xlabel("Bead index")

        axes_lock[0].set_title(
            f"Signal locking: T={args.signal_period:g}, "
            f"interval [{args.analysis_start:g}, "
            f"{args.analysis_end:g}]"
        )

        fig_lock.tight_layout()
        path = outdir / f"{prefix}_signal_locking.png"
        fig_lock.savefig(path, dpi=200)
        figures.append(fig_lock)
        saved_paths.append(path)


def write_trajectory_summary(
    args,
    coherence,
    crossing_times,
    figures,
    level_source,
    levels,
    locking,
    n_beads,
    outdir,
    periods,
    prefix,
    saved_paths,
    signal_enabled,
    t,
    widths,
):
    def json_number(value):
        # Use JSON null for unavailable results, rather than invalid NaN.
        return float(value) if np.isfinite(value) else None

    notes = [
        "Phase is reconstructed in radians between confirmed upward crossings.",
        "Coherence uses common-time phases and complete reference-cycle windows.",
        "The first 10% of usable coherence windows are discarded.",
        (
            "Legacy bead-to-bead coherence uses full-record windows; "
            "signal metrics use the explicit settled interval."
        ),
        "Legacy mean periods include transients.",
        "Times are used as stored; old timestamps are not repaired.",
        "Independent phase origins may add constant multiples of 2*pi.",
        "A single trajectory does not establish diode behaviour.",
    ]

    summary = {
        "trajectory": str(args.trajectory),
        "num_beads": n_beads,
        "num_samples": len(t),
        "recorded_start_time": float(t[0]),
        "recorded_end_time": float(t[-1]),
        "reference_bead": args.ref_bead,
        "reference_cycles_per_window": args.window,
        "crossing_level_source": level_source,
        "hysteresis_setting": (
            "automatic_5_percent_of_robust_amplitude"
            if args.hysteresis is None
            else float(args.hysteresis)
        ),
        "beads": [
            {
                "index": i,
                # Keep the legacy JSON key, but document its local-coordinate meaning.
                "crossing_level_x": float(levels[i]),
                "crossing_coordinate": "longitudinal",
                "hysteresis_half_width": float(widths[i]),
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

    if signal_enabled:
        # SECTION 2:
        # Keep signal metrics in their own JSON section so they cannot
        # be confused with the older bead-to-bead coherence values.
        summary["signal_locking"] = {
            "signal_period": float(args.signal_period),
            "signal_omega": float(2.0 * np.pi / args.signal_period),
            "signal_onset": float(args.signal_onset),
            "signal_phase_offset_rad": float(args.signal_phase),
            "requested_start_time": float(args.analysis_start),
            "requested_end_time": float(args.analysis_end),
            "slip_hysteresis_rad": float(args.slip_hysteresis),
            "beads": [{"index": i, **row} for i, row in enumerate(locking)],
        }

        notes.extend(
            [
                (
                    "Signal phase uses the original time clock; "
                    "switch-on does not reset it."
                ),
                (
                    "Signal R is abs(time-averaged exp(i*relative_phase)); "
                    "phase is not detrended."
                ),
                (
                    "Slip candidates are hysteretic branch transitions; "
                    "continuous drift also produces them."
                ),
                (
                    "Winding branches use the median relative phase "
                    "in the first usable forcing period as their anchor."
                ),
                (
                    "At least two usable forcing periods are required; "
                    "longer observations are needed for reliable conclusions."
                ),
                (
                    "Mean phase offset depends on the crossing convention "
                    "and signal phase origin."
                ),
            ]
        )

    summary_path = outdir / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    saved_paths.append(summary_path)

    print(f"\nLoaded {args.trajectory}")
    print(f"{n_beads} beads; {len(t)} recorded samples")
    print(f"Crossing levels: {level_source}")

    print(f"\n{'Bead':>6} {'Crossings':>10} " f"{'Mean period':>14} {'Coherence':>12}")

    for i in range(n_beads):
        print(
            f"{i:6d} {len(crossing_times[i]):10d} "
            f"{periods[i]:14.6g} {coherence[i]:12.6f}"
        )

    if args.ref_bead == 0:
        print(f"\nEndpoint coherence: {coherence[-1]:.6f}")

    print("\nPhases use confirmed crossings and common recorded times.")

    if signal_enabled:
        print(
            f"\nSignal T={args.signal_period:g}; "
            f"settled analysis interval "
            f"[{args.analysis_start:g}, {args.analysis_end:g}]"
        )
        print(
            f"{'Bead':>6} {'delta_f':>14} {'R':>10} "
            f"{'Windings':>10} {'Coverage':>18}"
        )

        for i, row in enumerate(locking):
            if row["status"] != "ok":
                print(f"{i:6d}  N/A: fewer than two usable " "forcing periods")
                continue

            print(
                f"{i:6d} "
                f"{row['frequency_mismatch_cycles_per_time']:14.6g} "
                f"{row['phase_consistency_R']:10.6f} "
                f"{row['slip_candidates_total']:10d} "
                f"{row['actual_start_time']:.4g} to "
                f"{row['actual_end_time']:.4g}"
            )

        print(
            "delta_f is bead minus signal frequency, "
            "in cycles per recorded time unit."
        )
        print(
            "Windings are slip candidates; inspect the "
            "unwrapped curve for steady drift."
        )

    else:
        print(
            "Signal analysis disabled: supply --signal-period " "and --analysis-start."
        )

    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


# The following modes use the same runner and measurements as optimisation.
# Importing this module does not run an experiment.
import sys
import shutil
from simulation import load_configuration, run_simulation, save_json, fingerprint
from optimisation import Evaluator
from bead_analysis import frequency_spectra, classify_locking


def override_axis(config, name, value):
    """Scalar inputs plus four explicit geometry controls for diagram axes.

    Geometry changes use original config as their reference. Invalid gaps are
    rejected, never clipped. 'asymmetry' interpolates from uniform to supplied
    geometry and preserves length. d_grad/g_factor preserve mean spacing.
    """
    n = config.n - 1
    if name == "asymmetry":
        mean = config.length / n
        gaps = mean + value * (np.asarray(config.values["distances"]) - mean)
        return config.updated(distances=gaps.tolist())
    if name == "d_avg":
        gaps = np.asarray(config.values["distances"]) * value / (config.length / n)
        return config.updated(distances=gaps.tolist())
    if name == "d_grad":
        gaps = config.length / n + (np.arange(n) - (n - 1) / 2) * value
        return config.updated(distances=gaps.tolist())
    if name == "g_factor":
        if value <= 0:
            raise ValueError("Geometric gap ratio must be positive.")
        weights = value ** np.arange(n)
        return config.updated(
            distances=(weights / weights.sum() * config.length).tolist()
        )
    # Individual vector entries are addressed as lambdas[2], kx_per_bead[0], ...
    if "[" in name and name.endswith("]"):
        key, index = name[:-1].split("[")
        values = config.get(key)
        values[int(index)] = float(value)
        return config.updated(**{key: values})
    if name not in config.values or np.ndim(config.values[name]) != 0:
        raise ValueError(f"{name} is not a scalar axis; use e.g. kx_per_bead[0].")
    from simulation import SEARCHABLE

    if name not in SEARCHABLE:
        raise ValueError(f"{name} is not an exposed physical sweep parameter.")
    return config.updated(**{name: float(value)})


def frequency_measurement(config, args, folder):
    """Unforced ensemble, with per-bead and mean spectra retained numerically."""
    results = []
    for repeat in range(args.runs):
        reader = run_simulation(
            config.updated(F0=0.0), args.exe, folder / f"replicate_{repeat}"
        )
        results.append(
            frequency_spectra(
                reader, args.start, (args.frequency_min, args.frequency_max)
            )
        )
    peaks = np.array([r["peaks"] for r in results])
    record = {
        "configuration": config.values,
        "start": args.start,
        "runs": args.runs,
        "frequency_range": [args.frequency_min, args.frequency_max],
        "replicates": results,
        "mean_peaks": peaks.mean(axis=0),
        "peak_sem": (
            peaks.std(axis=0, ddof=1) / np.sqrt(args.runs) if args.runs > 1 else None
        ),
        "mean_spectrum_peak": float(
            np.mean([r["mean_spectrum_peak"] for r in results])
        ),
    }
    if not args.keep:
        shutil.rmtree(folder)
    return record


def plot_frequency(record, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    spectra = record["replicates"]
    frequency = np.array(spectra[0]["frequencies"])
    power = np.mean([r["power"] for r in spectra], axis=0)
    for bead in range(power.shape[1]):
        axes[0].semilogy(frequency, power[:, bead], label=f"Bead {bead}")
    axes[0].set(
        xlim=record["frequency_range"],
        xlabel="Frequency",
        ylabel="Power spectral density",
    )
    axes[0].legend(fontsize=8)
    axes[1].errorbar(
        np.arange(power.shape[1]),
        record["mean_peaks"],
        yerr=record["peak_sem"],
        fmt="o-",
    )
    axes[1].set(xlabel="Bead", ylabel="Dominant unforced frequency")
    fig.savefig(output / "frequency.png", dpi=180)
    plt.close(fig)


def mechanism(config, args, output):
    evaluator = Evaluator(
        args.exe,
        output,
        args.runs,
        args.cores,
        args.start,
        keep=True,
        all_beads=True,
        hysteresis=args.hysteresis,
        slip_hysteresis=args.slip_hysteresis,
    )
    record = evaluator.evaluate(config)
    # Plot a representative trajectory, while numerical summaries include all repeats.
    folder = Path(record["work_directory"])
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, constrained_layout=True
    )
    for ax, direction in zip(axes, ("forward", "backward", "control")):
        reader = BeadDataAnalyzer()
        reader.read_data_file(
            str(next((folder / "replicate_0" / direction / "run").glob("*.dat")))
        )
        analyser = MultiBeadCoherenceAnalyzer(
            reader.time_series, reader.analysis_positions, hysteresis=args.hysteresis
        )
        for bead in range(config.n):
            times, delta = analyser.get_signal_phase_difference(
                bead, config.values["T_signal"]
            )
            mask = (times >= args.start) & np.isfinite(delta)
            # Retain drift; no detrending. Integer-cycle phase origins are arbitrary.
            ax.plot(times[mask], delta[mask] / (2 * np.pi), label=f"Bead {bead}")
        ax.set_ylabel(f"{direction}\nRelative phase / 2π")
        ax.legend(ncol=min(config.n, 6), fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.savefig(output / "relative_phase.png", dpi=180)
    plt.close(fig)
    metrics = (
        "phase_consistency_R",
        "relative_frequency_mismatch",
        "slip_candidate_rate_per_time",
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, constrained_layout=True
    )
    for direction in ("forward", "backward", "control"):
        for ax, key in zip(axes, metrics):
            data = np.array(
                [
                    [
                        next(
                            r[key] if r[key] is not None else np.nan
                            for r in rep["metrics"]
                            if r["run"] == direction and r["bead"] == bead
                        )
                        for bead in range(config.n)
                    ]
                    for rep in record["replicates"]
                ]
            )
            # Missing interior measurements remain gaps rather than zeros.
            mean = np.array(
                [
                    (
                        np.mean(col[np.isfinite(col)])
                        if np.any(np.isfinite(col))
                        else np.nan
                    )
                    for col in data.T
                ]
            )
            ax.plot(range(config.n), mean, "o-", label=direction)
            ax.set_ylabel(key.replace("_", " "))
            ax.legend()
    axes[-1].set_xlabel("Bead index")
    fig.savefig(output / "mechanism.png", dpi=180)
    plt.close(fig)
    if not args.keep:
        shutil.rmtree(folder)
        record["work_directory"] = None
    save_json(output / "mechanism.json", record)
    # Keep the audit consistent if raw trajectories were removed after plotting.
    (output / "objective_evaluations.jsonl").write_text(
        json.dumps(record, allow_nan=False) + "\n"
    )


def legacy_measurement(config, args, folder):
    """Historical protocol: reverse distances, drive original signal beads,
    and calculate real-sum bead-to-bead coherence after t>=5.

    This is deliberately NOT called signal transmission. Per-bead parameters
    are not reversed, matching the previous sweep. Missing results raise.
    """
    values = {}
    for direction in ("forward", "backward"):
        scores = []
        for repeat in range(args.runs):
            c = (
                config
                if direction == "forward"
                else config.updated(
                    distances=list(reversed(config.values["distances"]))
                )
            )
            reader = run_simulation(c, args.exe, folder / f"{direction}_{repeat}")
            mask = reader.time_series >= 5.0
            analyser = MultiBeadCoherenceAnalyzer(
                reader.time_series[mask], reader.analysis_positions[mask]
            )
            score = float(analyser.coherence_windowed_real_sum(ref_bead=0)[-1])
            if not np.isfinite(score):
                raise ValueError(
                    "Legacy coherence unavailable; use a longer simulation."
                )
            scores.append(score)
        values[direction] = scores
    values["score"] = float(np.mean(values["forward"]) - np.mean(values["backward"]))
    if not args.keep:
        shutil.rmtree(folder)
    return values


def response_class(record, n, args):
    """Fraction of repeats classified as locked, plus the corresponding control.

    Controls are shown separately; locking does not establish causal transfer.
    A regime uses the explicit minimum fraction selected by the user.
    """
    fractions = {}
    for name, direction, bead in [
        ("forward", "forward", n - 1),
        ("backward", "backward", 0),
        ("control_last", "control", n - 1),
        ("control_first", "control", 0),
    ]:
        flags = []
        for rep in record["replicates"]:
            row = next(
                r for r in rep["metrics"] if r["run"] == direction and r["bead"] == bead
            )
            flags.append(
                classify_locking(row, args.r_min, args.mismatch_max, args.slip_rate_max)
            )
        fractions[name] = (
            None if any(x is None for x in flags) else float(np.mean(flags))
        )
    if fractions["forward"] is None or fractions["backward"] is None:
        regime = None
    else:
        regime = int(fractions["forward"] >= args.lock_fraction) + 2 * int(
            fractions["backward"] >= args.lock_fraction
        )
    return {"locking_fractions": fractions, "regime": regime}


def diagram(config, args, output):
    x_name, x_lo, x_hi = args.x
    y_name, y_lo, y_hi = args.y if args.y else ("fixed", 0, 0)
    if x_name == y_name:
        raise ValueError("Choose two distinct axes.")
    # These combinations otherwise overwrite the same geometry or coefficients.
    conflicts = [
        {"asymmetry", "d_grad"},
        {"asymmetry", "g_factor"},
        {"d_grad", "g_factor"},
        {"kx", "kx_per_bead"},
        {"ky", "ky_per_bead"},
        {"default_lambda", "lambdas"},
    ]
    roots = {x_name.split("[")[0], y_name.split("[")[0]}
    if any(c <= roots for c in conflicts):
        raise ValueError(
            "Axes overwrite the same parameter; choose independent controls."
        )
    xs = np.linspace(float(x_lo), float(x_hi), args.points)
    ys = (
        np.linspace(float(y_lo), float(y_hi), args.points)
        if args.y
        else np.array([0.0])
    )
    if (
        not np.all(np.isfinite(xs))
        or not np.all(np.isfinite(ys))
        or xs[0] >= xs[-1]
        or (args.y and ys[0] >= ys[-1])
    ):
        raise ValueError("Axis ranges must be finite and increasing.")
    records = []
    score = np.full((len(ys), len(xs)), np.nan)
    regimes = score.copy()
    forward = score.copy()
    backward = score.copy()
    cf = score.copy()
    cb = score.copy()
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            row = {"x": float(x), "y": float(y), "status": "ok"}
            try:
                # Set mean spacing first, so a gradient is constructed at that mean.
                pairs = [(x_name, x)] + ([(y_name, y)] if args.y else [])
                pairs.sort(key=lambda pair: pair[0] != "d_avg")
                c = config
                for name, value in pairs:
                    c = override_axis(c, name, value)
                row["configuration"] = c.values
            except (ValueError, IndexError) as error:
                row.update(status="invalid_configuration", error=str(error))
            else:
                work = output / f"point_{j}_{i}"
                try:
                    if args.metric == "signal":
                        evaluator = Evaluator(
                            args.exe,
                            work,
                            args.runs,
                            args.cores,
                            args.start,
                            args.keep,
                            hysteresis=args.hysteresis,
                            slip_hysteresis=args.slip_hysteresis,
                        )
                        result = evaluator.evaluate(c)
                        row.update(result=result, **response_class(result, c.n, args))
                        score[j, i] = result["score"]
                        forward[j, i] = result["forward_R"]
                        backward[j, i] = result["backward_R"]
                        regimes[j, i] = (
                            row["regime"] if row["regime"] is not None else np.nan
                        )
                        cf[j, i] = row["locking_fractions"]["control_last"]
                        cb[j, i] = row["locking_fractions"]["control_first"]
                    elif args.metric == "frequency":
                        result = frequency_measurement(c, args, work)
                        row["result"] = result
                        score[j, i] = result["mean_spectrum_peak"]
                    else:
                        result = legacy_measurement(c, args, work)
                        row["result"] = result
                        score[j, i] = result["score"]
                except (ValueError, RuntimeError) as error:
                    row.update(status="measurement_failed", error=str(error))
            records.append(row)
            save_json(
                output / "diagram.json",
                {
                    "configuration": config.values,
                    "settings": vars(args),
                    "x_values": xs,
                    "y_values": ys,
                    "records": records,
                },
            )
            print(
                f'Point {len(records)}/{len(xs)*len(ys)}: {row["status"]}', flush=True
            )
    if args.metric == "signal":
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        panels = [
            (score, "Directional R difference"),
            (forward, "Forward receiver R"),
            (backward, "Backward receiver R"),
            (regimes, "Locking regime"),
            (cf, "Unforced last: locked fraction"),
            (cb, "Unforced first: locked fraction"),
        ]
    else:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axes = np.array([ax])
        panels = [
            (
                score,
                (
                    "Unforced mean-spectrum peak"
                    if args.metric == "frequency"
                    else "Legacy coherence difference"
                ),
            )
        ]
    for ax, (data, title) in zip(axes.ravel(), panels):
        if not args.y:
            ax.plot(xs, data[0], "o-")
            ax.set(xlabel=x_name, ylabel=title, title=title)
            if title == "Locking regime":
                ax.set_yticks([0, 1, 2, 3], ["Neither", "Forward", "Backward", "Both"])
            continue
        if title == "Locking regime":
            from matplotlib.colors import ListedColormap, BoundaryNorm

            cmap = ListedColormap(["#dddddd", "#3182bd", "#de2d26", "#756bb1"])
            im = ax.pcolormesh(
                xs,
                ys,
                np.ma.masked_invalid(data),
                cmap=cmap,
                norm=BoundaryNorm(np.arange(-0.5, 4), 4),
                shading="auto",
            )
            bar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
            bar.ax.set_yticklabels(["Neither", "Forward", "Backward", "Both"])
        else:
            limits = {}
            if args.metric == "signal":
                limits = (
                    {"vmin": -1, "vmax": 1, "cmap": "coolwarm"}
                    if title == "Directional R difference"
                    else {"vmin": 0, "vmax": 1}
                )
            im = ax.pcolormesh(
                xs, ys, np.ma.masked_invalid(data), shading="auto", **limits
            )
            fig.colorbar(im, ax=ax)
        ax.set(xlabel=x_name, ylabel=y_name, xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        "Blank cells indicate unavailable measurements or invalid configurations",
        fontsize=11,
    )
    fig.savefig(output / "diagram.png", dpi=180)
    plt.close(fig)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "trajectory":
        trajectory_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["mechanism", "frequency", "diagram"])
    parser.add_argument("--config", default="templates/6beads.input")
    parser.add_argument("--exe", default="src/rowers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--start", type=float, default=7.0)
    parser.add_argument("--time", type=float, help="Override simulation duration.")
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON_VALUE"),
        help="Override fixed input values.",
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--hysteresis", type=float)
    parser.add_argument("--slip-hysteresis", type=float, default=np.pi / 4)
    parser.add_argument("--frequency-min", type=float, default=0.1)
    parser.add_argument("--frequency-max", type=float, default=20.0)
    parser.add_argument("--x", nargs=3, metavar=("NAME", "MIN", "MAX"))
    parser.add_argument("--y", nargs=3, metavar=("NAME", "MIN", "MAX"))
    parser.add_argument(
        "--points", type=int, default=15, help="Number of points on each axis."
    )
    parser.add_argument(
        "--metric", choices=["signal", "legacy", "frequency"], default="signal"
    )
    parser.add_argument("--r-min", type=float, default=0.9)
    parser.add_argument("--mismatch-max", type=float, default=0.01)
    parser.add_argument("--slip-rate-max", type=float, default=0.0)
    parser.add_argument(
        "--lock-fraction",
        type=float,
        default=1.0,
        help="Fraction of repeats required to classify a direction as locked.",
    )
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.cores < 1
        or args.points < 2
        or not 0 <= args.r_min <= 1
        or not 0 < args.lock_fraction <= 1
        or args.mismatch_max < 0
        or args.slip_rate_max < 0
    ):
        parser.error("Invalid repeat count, grid size or classification thresholds.")
    if args.mode == "diagram" and not args.x:
        parser.error(
            "Diagram requires --x NAME MIN MAX; add --y for a two-dimensional map."
        )
    config = load_configuration(args.config)
    if args.set:
        config = config.updated(**{key: json.loads(value) for key, value in args.set})
    if args.time is not None:
        config = config.updated(total_time=args.time)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "configuration": config.values,
            "settings": vars(args),
            "executable_sha256": fingerprint(args.exe),
            "code_sha256": {
                f: fingerprint(Path(__file__).parent / f)
                for f in (
                    "simulation.py",
                    "bead_analysis.py",
                    "optimisation.py",
                    "diode_characterisation.py",
                )
            },
        },
    )
    if args.mode == "mechanism":
        mechanism(config, args, output)
    elif args.mode == "frequency":
        record = frequency_measurement(config, args, output / "simulation_work")
        save_json(output / "frequency.json", record)
        plot_frequency(record, output)
    else:
        diagram(config, args, output)


if __name__ == "__main__":
    main()
