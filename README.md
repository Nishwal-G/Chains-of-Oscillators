# Rowers

Simulation and optimisation of hydrodynamically coupled, cilia-like bead
oscillators. A C++ engine evolves the overdamped dynamics; four Python modules
configure experiments, search for directional responses and analyse the results.

The usual workflow is **choose a template → optimise → load the best configuration
→ characterise its behaviour**. Characterisation can also start directly from a
template or an existing trajectory.

Run all commands below from the **repository root**, which contains `src/`,
`analysis/` and `templates/`. The Python experiment runner currently uses chain
geometry and Rotne–Prager mobility. Other geometries remain available through
the C++ executable directly.

## 1. Build the C++ simulator

You need a C++17 compiler, `make`, and Eigen headers. On macOS, install the
Xcode command-line tools if a compiler is not already available. Build with:

```bash
make -C src
```

This produces `src/rowers`. The Makefile attempts to download Eigen 3.4.0 if the
`src/eigen-3.4.0` directory does not exist; that step requires network access and
`curl`. An existing but incomplete directory does not trigger the download.
Verify that `src/eigen-3.4.0/Eigen/Dense` exists if compilation reports missing
Eigen headers. Reinstall the complete Eigen directory if necessary.

Rebuild after changing C++ source. The supplied Makefile does not track header
dependencies, so after changing a `.h` file use:

```bash
make -C src clean
make -C src
```

For the uploaded pre-refactor source, `simulator_fixes.patch` fixes a missing
`time` argument in `saveData`, the minimum input-length check and stale comments.
If these corrections are not already present, place the patch at the root and run:

```bash
git apply --check simulator_fixes.patch
git apply simulator_fixes.patch
make -C src
```

The patch does not change the force law. Do not apply it twice.

## 2. Set up Python

Use Python 3.10 or newer and install the requirements in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependencies are NumPy, SciPy, Matplotlib and scikit-learn. Activate the
virtual environment at the start of each terminal session. `search.json` belongs
at the repository root alongside `requirements.txt`.

## 3. Run the simulator directly

```bash
./src/rowers -C -R templates/6beads.input
```

Both flag positions are required. The first selects geometry and the second
selects the hydrodynamic approximation:

| Position | Flag | Meaning |
| --- | --- | --- |
| Geometry | `-C` | Chain with reference positions along x. |
| Geometry | `-R` | Ring of reference positions. |
| Geometry | `-A` | Arbitrary reference positions supplied in the input. |
| Mobility | `-R` | Rotne–Prager finite-size correction; used by the Python pipeline. |
| Mobility | `-O` | Oseen point-particle approximation. |

The ring and arbitrary layouts still switch active traps along global x. They
are not exposed by the new Python search and have not been validated here.
Oseen is a far-field approximation and can become unsuitable at close separation.
Do not infer a guarantee of positive-definite mobility for every configuration
from the presence of the `-R` option; overlapping/coincident beads and numerical
factorisation require particular care in this implementation.

The command writes a `6beads/` directory under the current working directory:

| Output | Contents |
| --- | --- |
| `*.dat` | Frame index, physical time, then x/y bead coordinates. |
| `*.trap` | Trap-centre records in the simulator's own output format. |
| `*.forces` | Frame, physical time and force components after the timestamp fix. |

The analyser reads `.dat` trajectories. Do not assume that `.trap` uses the
same columns. Direct runs can overwrite existing files with the same names;
Python-managed experiments use fresh work directories.

**The current parser reads only the first line of each input file.** Use one
configuration per file. Multiple files can be processed sequentially:

```bash
./src/rowers -C -R run1.input run2.input run3.input
```

## 4. Python pipeline

| File | Responsibility |
| --- | --- |
| `simulation.py` | Named inputs, parameter space, constraints and C++ execution. |
| `bead_analysis.py` | Crossing detection, phase reconstruction, coherence, locking and spectra. |
| `optimisation.py` | Matched experiments, Metropolis/Bayesian search, audit and progress plots. |
| `diode_characterisation.py` | Trajectory plots, unforced frequencies, mechanism and parameter diagrams. |

Only the last two are command-line entry points. Both use the same simulation
and analysis functions. Characterisation imports `Evaluator` from optimisation;
this import never starts a search.

## 5. Optimisation

```bash
python analysis/optimisation.py --method metropolis --config templates/6beads.input --search search.json --set total_time 100 --iterations 50 --runs 3 --cores 3 --out results/metro
python analysis/optimisation.py --method bayesian --config templates/6beads.input --search search.json --set total_time 100 --initial 12 --iterations 50 --runs 3 --cores 3 --out results/bayes
```

The example search includes inner reference positions, period, force and global
alpha. Bounds are starting choices, not a scientifically validated parameter
range. Remove the `alpha` entry to keep the template's exponent fixed. Commands
are examples, not requests to perform a long search automatically.

Each evaluated candidate costs `3 * runs` simulations: forward forcing, backward
forcing, and an unforced control per repeat. Repeats run concurrently; candidate
configurations are evaluated sequentially. `--cores` controls concurrent repeats.
`--initial` counts the specified initial point plus random initial points.
Invalid Metropolis proposals are rejected before simulation.

### Select free and fixed parameters

`search.json` is the sole selection of free variables:

```json
{
  "positions": {"step": 0.5},
  "T_signal": {"bounds": [0.01, 2.0], "step": 0.02},
  "F0": {"bounds": [0.5, 10.0], "step": 0.2},
  "alpha": {"bounds": [0.5, 2.0], "step": 0.05}
}
```

Parameters omitted here stay fixed. Initial values come from the input unless
an entry supplies `initial`. For example, add:

```json
"temperature": {"bounds": [0, 100], "initial": 70, "step": 5}
```

Global `kx`, `ky`, and `default_lambda` intentionally set all corresponding
per-bead values. To vary beads independently, use `kx_per_bead`, `ky_per_bead`,
or `lambdas` instead. Do not select a global value and its per-bead array together.
Scalar bounds/steps broadcast across an array; separate bounds can be given as
`[[lower_0, lower_1, ...], [upper_0, upper_1, ...]]`. The array length is determined
by the number of beads. Positions have N-2 entries; endpoints remain fixed.

Use `--set NAME JSON_VALUE` for fixed overrides, for example:

```bash
python analysis/optimisation.py --search search.json --set temperature 0 --set total_time 100 --out results/zero_temperature
```

For an array, quote the JSON value: `--set distances '[6, 7, 8, 9, 10]'`.
This is also how to choose a different total chain length. All gaps must satisfy
the constraints. Updating a free parameter with `--set` sets its baseline/initial
value, not a constraint keeping it fixed.

`simulation.py` contains `SCALARS`, the positional input mapping, and `SEARCHABLE`,
the supported continuous coordinates. Adding a currently supported parameter to
`search.json` updates both optimisers and their plots automatically. Adding a
new physical parameter requires a real C++ input mapping and validation; adding
an arbitrary catalogue name cannot change the simulator. Numerical settings,
bead count and unused C++ fields are not free optimisation variables.

### What the search maximises

The score remains the original signal-based objective:

`mean_over_repeats(R_last_forward - R_first_backward)`.

R is the magnitude of the time-averaged signal-relative phase phasor. Frequency
mismatch, winding/slip candidates and controls are saved but do not secretly alter
this score. A high score is evidence of directional phase consistency over the
observation window, not proof of causal information transfer.

The signal is the simulator's square wave starting after t=5. Default analysis
starts at t=7. At least two signal periods of common endpoint phase coverage are
required; substantially longer measurements are appropriate for scientific
claims. The recording interval is refined for short signal periods. Integration
still uses the input dt; output refinement does not establish timestep convergence.

Metropolis accepts improvements and accepts worse scores with probability
`exp((new_score-current_score)/search_temperature)`. This temperature is separate
from physical temperature. Bayesian optimisation uses a Matern Gaussian process
with learned white noise and UCB (default) or expected improvement; it searches
50,000 random valid candidates by default. SEM is recorded but is not supplied
as a separate heteroscedastic GP noise model.

### Outputs and reproducibility

- `best.json`: complete physical configuration of the best measured candidate.
- `history.json`: search settings, named coordinates, protocol and search history.
- `objective_evaluations.jsonl`: one record per actual evaluation, including all
  repeats, controls, endpoint diagnostics and SEM.
- `progress.png`, `parameters.png`: score and coordinate histories.
- `gp_slice.png`: Bayesian surrogate mean/uncertainty at the best remaining
  coordinates. This is a model prediction, not a measured phase diagram.

Successful raw simulations are deleted unless `--keep` is supplied. Failed
simulations retain inputs and logs. An unavailable endpoint measurement stops
optimisation with a diagnostic; it is never converted to a zero score. In a
parameter diagram it is recorded as a missing measurement so other points can
continue. The best candidate is not automatically revalidated on fresh repeats.

`--seed` seeds Python proposals only. C++ still seeds initial conditions and
thermal noise independently; exact trajectory reproducibility would require a
future C++ seed parameter. There is no claim of common random numbers between
forward/backward/control experiments.

Bayesian warm start accepts a **new-pipeline** `objective_evaluations.jsonl`:

```bash
python analysis/optimisation.py --method bayesian --search search.json --set total_time 100 --warm-start results/metro/objective_evaluations.jsonl --out results/warm_bayes
```

Protocol hashes, fixed inputs and bounds must match. This intentionally rejects
old histories and histories from edited code; never mix incompatible objectives.
Warm start adds observations, it does not resume a Metropolis chain. New initial
points are still evaluated. Warm observations remain in their original audit;
`history.json` records the warm-start source path.

## 6. Characterisation

### Mechanism: every bead, three directions/controls

```bash
python analysis/diode_characterisation.py mechanism --config results/metro/best.json --runs 5 --time 100 --out results/mechanism
```

`relative_phase.png` shows the first repeat's unwrapped phase relative to the
signal for every bead, forwards/backwards/control. `mechanism.png` shows mean
per-bead R, relative frequency mismatch and winding-candidate rate. All repeat
measurements and their actual time intervals are in `mechanism.json` and the
audit. Missing interior measurements are gaps; they are not zeros. Keep raw
trajectories with `--keep` if you want to inspect other repeats.

### Unforced frequency spectra

```bash
python analysis/diode_characterisation.py frequency --config results/metro/best.json --time 100 --runs 5 --frequency-min 0.1 --frequency-max 20 --out results/frequency
```

Outputs include per-bead Welch spectra, dominant frequencies, the peak of the
mean spectrum and ensemble uncertainty. These are frequencies of the unforced
**coupled chain**, not necessarily isolated-oscillator natural frequencies.

### Two-dimensional phase diagrams

```bash
python analysis/diode_characterisation.py diagram --config results/metro/best.json --x alpha 0.5 2 --y T_signal 0.1 0.5 --points 15 --runs 3 --time 100 --out results/alpha_period
```

A 15x15 grid with three repeats costs 2,025 simulations. Start with a small grid.
Outputs are saved after every point. `diagram.json` contains complete point
configurations and raw diagnostics, allowing reclassification without rerunning.
`diagram.png` contains score, forward/backward R, locking regime and control
locking fractions. Blank cells indicate invalid or unavailable measurements.

A repeat is classified as locked when R >= `--r-min` (default 0.9), absolute
relative frequency mismatch <= `--mismatch-max` (default 0.01), and winding
candidate rate <= `--slip-rate-max` (default 0). A direction is labelled locked
when at least `--lock-fraction` of repeats pass (default 1). These are explicit
finite-window choices, not universal physical thresholds. Controls are shown
separately rather than being assumed to establish causality automatically.

Scalar inputs can be diagram axes. A per-bead axis uses e.g. `kx_per_bead[2]`.
Geometry axes are:

- `asymmetry`: interpolate gaps from equal spacing (0) to the supplied geometry
  (1), at fixed length. Does not remove non-geometrical trap asymmetry.
- `d_avg`: scale all supplied gaps to a new mean spacing.
- `d_grad`: construct linear gaps at the current mean spacing.
- `g_factor`: construct geometrically varying gaps at the current mean spacing.

Mean spacing is applied before the second geometry control. Conflicting axes
are rejected. Invalid gaps are never clipped, because clipping would change the
claimed geometry. The legacy implementation did clip them; this is an explicit
correctness change.

### One-dimensional and legacy sweeps

Omit `--y` to get a line sweep. For an unforced gradient sweep:

```bash
python analysis/diode_characterisation.py diagram --config templates/6beads.input --metric frequency --x d_grad -0.5 0.5 --points 11 --time 100 --out results/frequency_gradient
```

Use `--metric legacy` to retain the previous sweep's real-sum endpoint coherence
and distance-reversal protocol. It reverses **distances only**, not the full
per-bead physical configuration. It is distinct from swapping the driven endpoint
and is labelled accordingly. All values and repeats are stored in JSON.

### Existing trajectory plots

```bash
python analysis/diode_characterisation.py trajectory path/to/run.dat --signal-period 0.184715 --analysis-start 7 --outdir results/trajectory
```

This preserves the previous six plots: trajectories, coherence, crossings,
bead-to-bead phase differences, signal-relative phase and signal-locking summary,
plus the JSON summary. Omit signal arguments for a purely trajectory/coherence
analysis. `trajectory --help` lists the retained options.

For compatibility, trajectory mode retains its previous reference-level choice
(settled mean when signal analysis is enabled), whereas the original optimisation
and matched-experiment evaluator use full-record mean crossing levels. Thus
standalone trajectory measurements need not exactly equal evaluator measurements.
The evaluator's audit is authoritative for its score. Use `--reference-x` to
choose explicit levels in trajectory mode. Cached phases assume an analyser's
trajectory and hysteresis settings are not modified after construction.

## 7. Input file reference

The first line is a space-separated numerical record. Fields below are numbered
from **1**; Python indices in `simulation.py` start at **0**. The first 22 fields
are scalars, followed by geometry and per-bead blocks. Do not insert text labels
or inline comments into the numerical line.

The templates use the following configurations:

| Template | Beads | Reference gaps | Total reference length |
| --- | --- | --- | --- |
| `6beads.input` | 6 | 6.046, 7.068, 7.961, 9.037, 9.929 µm | 40.041 µm |
| `8beads.input` | 8 | Seven gaps of 7.5 µm | 52.5 µm |

The table below lists the six-bead template's scalar values. Quantities must use
the simulator's consistent length, time, force and thermal-energy conventions;
do not mix SI viscosity with micrometre-based numerical inputs without conversion.

| Field | Python name | Value | Meaning |
| --- | --- | --- | --- |
| 1 | `num_beads` | 6 | Number of beads; choose a corresponding template. |
| 2 | `amplitude` | 3 | Excursion parameter A; switching thresholds are at ±A/2 relative to each reference position. |
| 3 | `base_distance` | 6 | Reference spacing used in ring geometry and chain trap-strength scaling. |
| 4 | `bead_radius` | 1.5 | Bead radius entering mobility. |
| 5 | `dt` | 0.00005 | Integration timestep. |
| 6 | `total_time` | 30 | Simulation duration. |
| 7 | `viscosity` | 0.006 | Fluid viscosity in the model's unit convention. |
| 8 | `alpha` | 0.5 | Exponent of the longitudinal **potential**; the force magnitude scales as displacement to the power alpha−1. |
| 9 | `kx` | 10 | Default longitudinal potential coefficient, not a universal spring constant for all alpha. |
| 10 | `beta` | 1.5 | Exponent of the transverse potential. |
| 11 | `ky` | 10 | Default transverse potential coefficient. |
| 12 | `temperature` | 70 | Physical/effective temperature entering thermal displacement. Zero disables stepwise thermal noise. |
| 13 | `sampling_feedback` | 50 | Integration steps between trap-switching checks. |
| 14 | `sampling_write` | 200 | Integration steps between output records. |
| 15 | `epsilon` | 0.3 | Places active trap centres epsilon beyond the switching thresholds. |
| 16 | `velox` | 0 | Uniform background flow velocity in x. |
| 17 | `ks` | 0 | Parsed legacy field; unused by the supplied force implementation. |
| 18 | `fl0` | 0.25 | Parsed legacy field; unused by the supplied force implementation. |
| 19 | `veloy` | 0 | Uniform background flow velocity in y. |
| 20 | `F0` | 2.5 | External square-wave force amplitude. |
| 21 | `T_signal` | 0.184715 | External driving period. |
| 22 | `default_lambda` | 0.06 | Fallback parameter for the per-bead trap-strength profile. |

After these scalar fields, **chain inputs** contain:

| Block | Number of values | Meaning |
| --- | --- | --- |
| `distances` | N−1 | Gaps between neighbouring reference positions. |
| `lambdas` | N | Per-bead trap-profile parameters. |
| `kx_per_bead` | N, optional | Per-bead longitudinal coefficients. |
| `ky_per_bead` | N, optional | Per-bead transverse coefficients. |
| Signal list | 1 + count, optional | Number of driven beads followed by their zero-based indices. |

The supplied templates include distances and lambdas but omit the stiffness and
signal blocks. The defaults are global kx/ky and signal bead 0. To specify signal
beads, include both full stiffness blocks first. The Python input writer always
emits the complete blocks to avoid ambiguity.

For example, this complete six-bead input has increasing x coefficients, uniform
y coefficients, and external forcing at beads 0 and 5:

```text
6 3 6.0 1.5 0.00005 100 0.006 0.5 10 1.5 10 70 50 200 0.3 0.0 0 0.25 0.0 2.5 0.18 0.06 6.0 7.1 8.0 9.0 9.9 0.06 0.06 0.06 0.06 0.06 0.06 8 9 10 11 12 13 10 10 10 10 10 10 2 0 5
```

The matched signal experiments intentionally override the signal list: first
bead for forward, last bead for backward, and zero force for control. They keep
the geometry and per-bead coefficients in place. They do **not** reverse those
arrays. Legacy sweep mode instead reverses distances only, retaining the older
protocol.

For arbitrary geometry, the C++ parser expects coordinate pairs rather than the
chain-distance block; this table and the Python input reader are for chains.

### Trap force and active oscillation

Writing delta x for displacement from the active trap centre, the x force is
`−kx_i * H_i * alpha * delta_x * abs(delta_x)^(alpha−2)`, plus the signal on
selected beads. Thus alpha=2 gives a harmonic potential; kx is a potential
coefficient and the corresponding force includes the factor alpha.

The factor H_i depends on local reference spacing and lambda_i. Lambda changes
trap strength, rather than directly multiplying the hydrodynamic tensor. Moving
trap centres between opposite sides sustains oscillation even at zero thermal
temperature. Random initial positions and initial trap sides remain at zero
temperature.

The external signal is `F0 * sign(cos(2*pi*time/T_signal))` after time 5. Its phase
clock is not reset when forcing begins. The code uses a definite branch at cosine
zero, so this expression is shorthand for its implemented square wave.

## 8. Interpretation and numerical checks

Alpha changes the existing force coefficient as well as the shape. This release
does not normalise force scale when changing exponent. Geometry also changes
both mobility and the simulator's spacing-dependent trap-strength factor. These
are retained model features, not refactoring side effects. Check timestep and
observation-time convergence across any expanded search range, especially near
steep/singular force profiles and locking boundaries.

The information-transfer mode (phase jump, response delay, message transmission)
is deliberately not implemented: the current C++ signal cannot express those
protocols. It requires an explicit future simulator extension. Per-bead alpha
also requires C++ support; alpha here is global.

This package does not assert a new licence over your existing source. Before
public release, choose a licence with the project's owners and preserve existing
attributions. `TESTING.md` records what was checked for this refactor.

## 9. Migration from the old scripts

Keep a backup or commit before replacing the old analysis folder. The supported
Python folder now contains exactly four modules; the existing C++ headers and
sources remain separate.

| Previous script | New entry point or location |
| --- | --- |
| `optimisation_metropolis.py` | `optimisation.py --method metropolis` |
| `optimisation_bayesian.py` | `optimisation.py --method bayesian` |
| `optimisation_common.py` | Shared evaluator inside `optimisation.py`; input/runner functions in `simulation.py`. |
| Root `diode_analysis.py` | `diode_characterisation.py mechanism` |
| `bead_analysis_output.py` | `diode_characterisation.py trajectory` |
| `check_natural_frequency.py` | `diode_characterisation.py frequency` or `diagram --metric frequency` |
| `sweep.py` | `diode_characterisation.py diagram`; add `--metric legacy` for the old coherence protocol. |

After migration, remove the superseded scripts from the working tree; retain
`bead_analysis.py` using its new version. Existing trajectories remain readable
when they have valid physical timestamps. Old optimisation histories do not
contain complete configurations and are not accepted as new-format `best.json`
or warm-start observations.

Output folders are explicitly named with `--out`, rather than automatically
timestamped. For trajectory mode the corresponding option is `--outdir`.
Optimisation duration is set with `--set total_time VALUE`; characterisation also
has `--time VALUE`. Derived geometry controls such as `d_avg` are diagram axes,
not accepted fixed-input names for `--set`; use explicit `distances` for a fixed
geometry.

## 10. Help and reproducibility

```bash
python analysis/optimisation.py --help
python analysis/diode_characterisation.py --help
python analysis/diode_characterisation.py trajectory --help
```

Record the complete configuration, search file, code version, executable and
analysis settings alongside reported results. The pipeline stores configuration
and code/executable hashes with its records. Consult `TESTING.md` for the scope
of the delivered validation; short smoke tests are not substitutes for timestep,
repeat-count and observation-time convergence checks.
