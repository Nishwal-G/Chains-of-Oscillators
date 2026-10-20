# Computing experiments with the rower simulator

This extension supplies experiment infrastructure, not a demonstrated computer.
The example junction is not an optimised switch or AND gate. The ring example
compares two writing protocols; a nonzero score alone does not establish memory.
No equations for the existing hydrodynamic mobility or thermal drift were changed.

## Start here

Copy the changed/new files into the repository root, preserving their paths.
Rebuild and install the existing Python requirements:

```bash
make -C src clean
make -C src
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

All commands below run from the repository root. Use a new output directory for
each invocation. Existing diode commands work without an `--experiment` option.

```bash
# Four matched cases: neither drive, input only, control only, both.
python analysis/experiments.py --config templates/computing/junction.json --experiment templates/computing/control.json --runs 3 --cores 3 --keep --out results/control

# Control-strength response curve. Quotes protect brackets in shell arguments.
python analysis/experiments.py --config templates/computing/junction.json --experiment templates/computing/control.json --sweep 'drives[1].amplitude' --values '[0,1,2.5,4]' --runs 3 --cores 3 --out results/control_sweep

# The same Metropolis/Bayesian code, using the selected experiment's score.
python analysis/optimisation.py --config templates/computing/junction.json --experiment templates/computing/control.json --search templates/computing/search.json --method metropolis --iterations 10 --runs 3 --cores 3 --out results/control_search

python analysis/optimisation.py --config templates/computing/junction.json --experiment templates/computing/and.json --search templates/computing/search.json --method bayesian --initial 4 --iterations 5 --candidates 1000 --runs 3 --cores 3 --out results/logic_search

# Characterise a saved candidate through precisely the same experiment protocol.
python analysis/experiments.py --config results/control_search/best.json --experiment templates/computing/control.json --runs 5 --cores 3 --keep --out results/control_validation

# Timed control: compare before, during and after the control pulse.
python analysis/experiments.py --config templates/computing/control_pulse.json --experiment templates/computing/control_sequence.json --runs 3 --cores 3 --keep --out results/sequence

# Ring: compare early/late readout after two temporary writing drives.
python analysis/experiments.py --config templates/computing/ring.json --experiment templates/computing/memory.json --runs 3 --cores 3 --keep --out results/memory
```

Examples use deterministic dynamics (temperature=0), a fixed simulator seed and
Rotne-Prager mobility for fast pilot runs. Set mobility to `blake` and a valid
wall_height to study the wall. Thermal Blake retains its expensive divergence
calculation. To validate on fresh initial states, change the configuration seed
(or use --set seed NEW_INTEGER in optimisation). Reusing a seed reuses the same
initial-condition ensemble. Increasing --runs alone retains the earlier seeds
and appends new ones; it is not wholly fresh validation.

## Configuration: one description of the physical system

New templates are JSON objects with a `configuration` member, just like saved
best.json files. They contain all existing scalar/per-bead settings, plus:

| Field | Meaning |
| --- | --- |
| reference_x, reference_y | Two length-N arrays of planar trap reference coordinates. Both empty means the old chain layout. |
| groups | A dictionary mapping names to nonempty lists of unique zero-based bead indices. |
| direct_traps | true gives H_i=1 in the trap potential. false preserves the legacy spacing/index multiplier. |
| drives | null preserves the legacy F0/T_signal force; [] explicitly disables external drives; a list defines independent drives. |
| seed | -1 selects random seeds; integer 0..4294967295 seeds initial state and thermal noise. |

Groups are labels for drives and analysis. They do not introduce connections or
remove interactions: the engine evaluates the hydrodynamic coupling between all
beads. A ring is simply explicit coordinates plus a cyclically ordered group.
Arbitrary layout coordinates override the first-line chain reference geometry.
The retained `distances` array is a legacy input container for explicit layouts;
it does not control their actual positions. Search reference_x/y instead.

Keep one bead fixed and usually fix a second coordinate when searching geometry
to avoid spending effort on translations/rotations of the same arrangement.
All reference pair separations are checked against the existing minimum gap and
bead diameter. Blake also requires at least four radii; instantaneous Blake
separations remain checked by the engine. Reference checks do not prevent later
close encounters or replace timestep/physical-validity studies.

Set direct_traps=true for the new circuit templates. Each bead then has
U_i = kx_i |d_parallel|^alpha_i + ky_i |d_transverse|^beta_i.
The familiar kx is the potential coefficient: at alpha=2 the harmonic spring
constant is 2*kx under this convention. Changing bead indices cannot change H_i
in direct mode. Legacy lambda/base_distance trap multipliers remain available
with direct_traps=false; their searches are rejected in direct mode because
these multipliers then have no effect.

## Independent drives

Each drive has these fields (JSON needs no comments):

```json
{
  "targets": "input",
  "amplitude": 2.5,
  "period": 0.184715,
  "phase": 0.0,
  "start": 1.0,
  "stop": 12.0,
  "waveform": "square"
}
```

`targets` is a group name or a list such as [0,2]. Amplitude is the force on EACH
target, not a total divided among targets. Several drives on a bead add. Forces
act along that bead's local stroke axis, then rotate into laboratory coordinates
before the mobility calculation. They are external forces, not background flow.

phase is in radians. For start <= t < stop, the waveforms are:

- square: amplitude * sign(cos(2*pi*t/period + phase)), with zero mapped to -1.
- sine: amplitude * sin(2*pi*t/period + phase).
- constant: amplitude (period and phase do not affect the force).

Outside this interval the contribution is zero. The clock uses ABSOLUTE time,
not time since start. To start a sine at phase zero at start, set
phase=-2*pi*start/period. Negative constant forcing can be represented by a
reversed stroke direction; amplitudes are nonnegative. Missing phase/start/stop/
waveform default to 0/0/total_time/square when loading configuration JSON.

An explicit list replaces the legacy drive; it never adds to it. F0/T_signal are
then ignored by the engine and rejected as search coordinates. An empty list
means unforced, even if F0 is nonzero. Legacy mode retains the exact t>5 square
wave behaviour. To make a phase jump, use consecutive, non-overlapping drives
with different phases. To provide a pulse, give a finite start/stop interval.

## Experiment JSON and what is measured

Experiment files describe a set of matched simulations. They are separate from
physical configuration and search bounds. The common fields are:

- kind: control, logic or memory.
- output: a group name or list. Report the arithmetic mean of per-output-bead R.
- reference: a group/list; its FIRST bead is the phase reference. Output cannot
  contain this bead, because self-comparison would give R=1 trivially.
- period: a positive diagnostic timescale; windows must span at least twice it.
- cases: list of uniquely named cases. Each has a scoring `window: [start,stop]`.
- cases may have scales: one nonnegative amplitude multiplier per drive.
- cases may have overrides: `drives` or indexed drive fields only. They cannot
  silently change the trap physics or geometry between truth-table cases.
- optional case windows: additional [start,stop] intervals from the SAME run,
  for sequence/retention diagnostics; these do not alter the score.
- threshold: optional logic readout threshold (default 0.8).
- ring: required for memory; names/lists the beads in cyclic neighbour order.

The new protocols use upward crossings of each bead's fixed trap reference,
projected along its stroke direction. The existing diode metric keeps its old
full-record mean-crossing convention. Phase is interpolated between confirmed
crossings with no extrapolation. R_out,in is the modulus of the time-weighted
mean of exp(i*(phi_out-phi_in)) over the requested window. Trapezoidal weights
sum to one. Pair results also include mean phase, frequency mismatch in cycles
per unit time, and confirmed slip count using the existing hysteresis detector.
Every sample in a scoring window must have defined pair phase; otherwise the
candidate fails with its input/logs retained. Low R is not used as a substitute
for an unavailable measurement. Check window length against the actual oscillator
frequencies, not only the diagnostic period.

Signal metrics additionally compare output to the FIRST drive's absolute clock,
including its configured phase. In no-drive/control-off cases this is a
hypothetical reference clock; it does not mean that an external force is active.
Constant first drives have no signal-clock metric. The score uses bead-to-bead R.

These are offline diagnostics: crossing interpolation uses a later crossing.
The window comparisons do not give a causal, high-resolution switching latency.
A high R can coexist with a nonzero lag; it does not distinguish phase-coded
bits 0 and pi. The initial logic protocol instead thresholds phase consistency.

## Scores (larger is better)

Control requires cases named neither, input, control, both. If their mean
output-to-reference consistencies are R00, R10, R01, R11, the score is

    R11 - R10 - max(0, R01 - R00).

This searches for a control that OPENS transmission, penalising control-only
entrainment. It is a deliberately explicit first objective, not an assertion
that every positive score is a transistor. Inspect all cases, frequency mismatch,
phase slips and parameter robustness. A closing switch needs a different score.

Logic requires four cases with expected=0 or 1. The score is

    1 - mean_over_cases(abs(R_case - expected_case)).

Correctness is also reported separately using R>=threshold as logical 1. The
AND template expects only both=1; change expected values to investigate OR or
another two-input truth table. A sharp threshold in analysis does not create a
physical threshold or regenerate a signal for a downstream gate. These four
static cases do not exhaust input histories; use timed drives and extra windows
to examine history dependence before claiming reusable logic.

Memory requires two writing cases and a ring group. All nonzero external drives
must end before every readout window. For each neighbour pair, measure
z_i = time_mean(exp(i*(phi_(i+1)-phi_i))). The score is

    mean_over_neighbours(abs(z_i_write0 - z_i_write1)) / 2.

It rewards distinct, temporally consistent phase patterns. Per-case neighbour
angles and |z_i|, plus the winding values visited during each window, are saved.
Comparing early/late windows tests observed persistence. This is not a measured
retention lifetime, proof of metastability, or a demonstrated write/erase/read
cycle. Readout here observes simulated phases without applying a physical reader;
its backaction and coupling to future circuitry remain research tasks.

Each score is calculated separately for every repeat; the outer evaluator then
reports its arithmetic mean and SEM (SEM is null for a single repeat). Existing
Metropolis acceptance and Bayesian acquisition code are shared without copying.
Case count determines cost: four simulations per control/logic repeat, two per
memory repeat; extra windows do not require extra simulations.

## Reproducibility and outputs

Fixed simulator seed s produces replicate seeds s, s+1, ... modulo 2^32. Cases
within a replicate share the seed and therefore initial conditions and thermal
random draws, while their evolving configurations can differ. With seed=-1,
new computing protocols draw and record a random seed per replicate before
matching their cases. Legacy diode seed=-1 behaviour remains independent random
runs. Reproducibility is within the same executable/platform; C++ normal draws
need not match bit-for-bit across standard-library implementations.

The optimiser's --seed still controls proposals only. Its --set seed INTEGER
controls the C++ simulator. The new templates use 12345 for repeatable pilots;
change this for genuinely fresh validation ensembles.

Standalone runs save result.json and response.png; sweeps save sweep.json and
sweep.png. The JSON contains all repeats, cases, windows and exact realised seeds.
Optimisation retains best.json, history.json, progress plots and the existing
audit JSONL, now with the experiment specification and module hash. Raw .dat,
.axes, .hydro, .forces and .traps outputs retain their formats. --keep preserves
per-case inputs, logs and metrics; otherwise successful raw work is removed.
The response plot displays pair R, including for memory; inspect the stored
neighbour-pattern diagnostics for the memory objective itself.

## File organisation and next steps

- src/Types.h: independent Drive records and optional trap/seed settings.
- src/Simulation.cpp: parse labelled extensions, explicit coordinates and seeds;
  emit pipeline version metadata to detect stale executables.
- src/PhysicsEngine.cpp: sum gated drives and optionally use direct trap strengths.
- analysis/simulation.py: one configuration/validation/search/serialization path.
- analysis/optimisation.py: retain both algorithms and execution/audit machinery;
  select the new evaluator only when --experiment is supplied.
- analysis/experiments.py: one shared module for cases, windows, scores and reports.
- tests/test_computing.py: compatibility, schema and physical force checks.

No extra per-device simulation wrappers are needed. First optimise/validate a
controlled junction. Then establish a truth table and its response to changing
inputs. To test cascading, put both candidate components in one explicit layout
and drive only the intended external ports: all beads already interact through
the same mobility matrix. No software bit is injected into a downstream gate.
A working multigate circuit remains a physical result to demonstrate.
