import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bead_analysis import (
    BeadDataAnalyzer,
    MultiBeadCoherenceAnalyzer,
)


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
        "--reference-x",
        type=float,
        nargs="+",
        default=None,
        help="Fixed reference x coordinates, one per bead.",
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

    args = parser.parse_args()

    if not args.trajectory.is_file():
        parser.error(f"Trajectory not found: {args.trajectory}")

    if args.window < 2:
        parser.error("--window must be at least 2.")

    if not np.isfinite(args.zoom) or args.zoom <= 0:
        parser.error("--zoom must be finite and positive.")

    if args.hysteresis is not None:
        if not np.isfinite(args.hysteresis) or args.hysteresis < 0:
            parser.error(
                "--hysteresis must be finite and nonnegative."
            )

    data = BeadDataAnalyzer()

    try:
        data.read_data_file(str(args.trajectory))
    except (ValueError, OSError) as error:
        parser.error(str(error))

    t = data.time_series
    positions = data.positions
    n_beads = data.num_beads

    if not np.all(np.isfinite(positions)):
        parser.error("Trajectory positions must be finite.")

    if (
        not np.all(np.isfinite(t))
        or np.any(np.diff(t) <= 0)
    ):
        parser.error(
            "Recorded times must be finite and strictly increasing."
        )

    if not 0 <= args.ref_bead < n_beads:
        parser.error(
            f"--ref-bead must be between 0 and {n_beads - 1}."
        )

    # SECTION 2: validate the requested settled interval.
    signal_enabled = args.signal_period is not None
    analysis_mask = None

    if signal_enabled:
        if (
            not np.isfinite(args.signal_period)
            or args.signal_period <= 0
        ):
            parser.error(
                "--signal-period must be finite and positive."
            )

        if (
            args.analysis_start is None
            or not np.isfinite(args.analysis_start)
        ):
            parser.error(
                "Supply a finite --analysis-start with --signal-period."
            )

        if (
            not np.isfinite(args.signal_onset)
            or not np.isfinite(args.signal_phase)
        ):
            parser.error("Signal onset and phase must be finite.")

        if (
            not np.isfinite(args.slip_hysteresis)
            or not 0 < args.slip_hysteresis < np.pi
        ):
            parser.error(
                "--slip-hysteresis must lie strictly between 0 and pi."
            )

        if args.analysis_end is None:
            args.analysis_end = float(t[-1])

        if (
            not np.isfinite(args.analysis_end)
            or args.analysis_start < t[0]
            or args.analysis_end > t[-1]
            or args.analysis_start >= args.analysis_end
        ):
            parser.error(
                "Analysis interval must lie within recorded times, "
                "with start < end."
            )

        if args.analysis_start <= args.signal_onset:
            parser.error(
                "--analysis-start must be after signal onset "
                "and should exclude settling."
            )

        analysis_mask = (
            (t >= args.analysis_start)
            & (t <= args.analysis_end)
        )

        if np.count_nonzero(analysis_mask) < 3:
            parser.error(
                "Analysis interval needs at least three recorded samples."
            )

    elif (
        args.analysis_start is not None
        or args.analysis_end is not None
    ):
        parser.error(
            "Analysis interval options require --signal-period."
        )

    reference_positions = None

    if args.reference_x is not None:
        if (
            len(args.reference_x) != n_beads
            or not np.all(np.isfinite(args.reference_x))
        ):
            parser.error(
                "--reference-x needs one finite coordinate per bead."
            )

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
        else "settled_interval_mean"
        if signal_enabled
        else "full_run_mean"
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
        np.asarray(
            analyzer._get_zero_crossing_times(i, dimension=0)
        )
        for i in range(n_beads)
    ]

    # These legacy periods include the full record, including transients.
    periods = np.array([
        np.mean(np.diff(times)) if len(times) >= 2 else np.nan
        for times in crossing_times
    ])

    settings = [
        analyzer._get_crossing_settings(i, dimension=0)
        for i in range(n_beads)
    ]
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
    saved_paths = []

    x = positions[:, :, 0]

    # Centre plots on the same levels used by the crossing detector.
    x_centred = x - levels
    zoom_mask = t >= t[-1] - args.zoom

    centre_label = (
        "x − reference x"
        if args.reference_x is not None
        else "x − settled mean(x)"
        if signal_enabled
        else "x − mean(x)"
    )

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
    axes[1].set_title(
        f"Final {args.zoom:g} recorded time units"
    )

    for ax in axes:
        ax.set_xlabel("Recorded time")
        ax.set_ylabel(centre_label)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    path = outdir / f"{prefix}_trajectories.png"
    fig.savefig(path, dpi=200)
    figures.append(fig)
    saved_paths.append(path)

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
        title=(
            "Full-record windowed coherence "
            f"relative to bead {args.ref_bead}"
        ),
    )
    ax.grid(axis="y", alpha=0.2)

    fig_coh.tight_layout()
    path = outdir / f"{prefix}_coherence.png"
    fig_coh.savefig(path, dpi=200)
    figures.append(fig_coh)
    saved_paths.append(path)

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
        events = events[
            (events >= t[zoom_mask][0])
            & (events <= t[-1])
        ]

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
        "Crossing check: dots = accepted upward crossings; "
        "dashed = band"
    )
    axes[-1, 0].set_xlabel("Recorded time")

    fig_cross.tight_layout()
    path = outdir / f"{prefix}_crossings.png"
    fig_cross.savefig(path, dpi=200)
    figures.append(fig_cross)
    saved_paths.append(path)

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
        title=(
            f"φᵢ(t) − φ{args.ref_bead}(t): common-time phases"
        ),
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

    # SECTION 2 — Figure 5:
    # Plot phi_i(t) - phi_signal(t) without wrapping or detrending.
    # Shading identifies the interval used for the numerical measurements.
    if signal_enabled:
        fig_signal, axes_signal = plt.subplots(
            2, 1, figsize=(11, 8)
        )

        for i in range(n_beads):
            signal_times, differences = (
                analyzer.get_signal_phase_difference(
                    i,
                    args.signal_period,
                    signal_phase=args.signal_phase,
                )
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
                    marker=(
                        "^" if event["direction"] > 0 else "v"
                    ),
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
        axes_signal[0].set_title(
            "Unwrapped phase relative to the forcing clock"
        )
        axes_signal[0].legend(ncol=4, fontsize=8)

        axes_signal[1].set_title(
            "Settled interval; triangles = confirmed winding "
            "transitions (slip candidates)"
        )

        for ax in axes_signal:
            ax.set_xlabel("Recorded time")
            ax.set_ylabel(
                "phi_i(t) − (Omega*t + offset) [rad]"
            )
            ax.grid(alpha=0.2)

        fig_signal.tight_layout()
        path = outdir / f"{prefix}_signal_phase.png"
        fig_signal.savefig(path, dpi=200)
        figures.append(fig_signal)
        saved_paths.append(path)

        # SECTION 2 — Figure 6:
        # Show frequency mismatch, circular consistency and winding counts.
        # R is distinct from the older bead-to-bead coherence formula C.
        fig_lock, axes_lock = plt.subplots(
            3, 1, figsize=(9, 9), sharex=True
        )

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
            values = np.array([
                row[key] if row[key] is not None else np.nan
                for row in locking
            ])
            finite_values = np.isfinite(values)

            ax.bar(
                indices[finite_values],
                values[finite_values],
                color=[
                    colours(i % 10)
                    for i in indices[finite_values]
                ],
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

        axes_lock[0].axhline(
            0, color="black", linewidth=0.7
        )
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
                "crossing_level_x": float(levels[i]),
                "hysteresis_half_width": float(widths[i]),
                "positive_crossings": len(crossing_times[i]),
                "mean_period": json_number(periods[i]),
                "coherence": json_number(coherence[i]),
            }
            for i in range(n_beads)
        ],
        "endpoint_coherence_0_to_last": (
            json_number(coherence[-1])
            if args.ref_bead == 0
            else None
        ),
        "notes": notes,
    }

    if signal_enabled:
        # SECTION 2:
        # Keep signal metrics in their own JSON section so they cannot
        # be confused with the older bead-to-bead coherence values.
        summary["signal_locking"] = {
            "signal_period": float(args.signal_period),
            "signal_omega": float(
                2.0 * np.pi / args.signal_period
            ),
            "signal_onset": float(args.signal_onset),
            "signal_phase_offset_rad": float(args.signal_phase),
            "requested_start_time": float(args.analysis_start),
            "requested_end_time": float(args.analysis_end),
            "slip_hysteresis_rad": float(args.slip_hysteresis),
            "beads": [
                {"index": i, **row}
                for i, row in enumerate(locking)
            ],
        }

        notes.extend([
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
        ])

    summary_path = outdir / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    saved_paths.append(summary_path)

    print(f"\nLoaded {args.trajectory}")
    print(f"{n_beads} beads; {len(t)} recorded samples")
    print(f"Crossing levels: {level_source}")

    print(
        f"\n{'Bead':>6} {'Crossings':>10} "
        f"{'Mean period':>14} {'Coherence':>12}"
    )

    for i in range(n_beads):
        print(
            f"{i:6d} {len(crossing_times[i]):10d} "
            f"{periods[i]:14.6g} {coherence[i]:12.6f}"
        )

    if args.ref_bead == 0:
        print(
            f"\nEndpoint coherence: {coherence[-1]:.6f}"
        )

    print(
        "\nPhases use confirmed crossings and common recorded times."
    )

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
                print(
                    f"{i:6d}  N/A: fewer than two usable "
                    "forcing periods"
                )
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
            "Signal analysis disabled: supply --signal-period "
            "and --analysis-start."
        )

    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()