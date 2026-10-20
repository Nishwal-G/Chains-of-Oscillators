"""Shared input handling and execution for the chain simulator.

The catalogue below is the only mapping between Python names and C++ input
columns. Search bounds belong in a run's JSON file, not in the physics code.
Explicit planar coordinates and independent drives share the legacy input path.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np
from bead_analysis import BeadDataAnalyzer

MIN_GAP = 3.1
# One catalogue defines scalar defaults, broadcast overrides and search conflicts.
PER_BEAD_DEFAULTS = {
    "lambdas": "default_lambda", "kx_per_bead": "kx", "ky_per_bead": "ky",
    "alpha_per_bead": "alpha", "beta_per_bead": "beta",
    "amplitude_per_bead": "amplitude", "epsilon_per_bead": "epsilon",
}
# These structural fields are fixed during a search; individual numerical drive
# fields and reference coordinates can be selected explicitly below.
DRIVE_FIELDS = {"amplitude", "period", "phase", "start", "stop"}
WAVEFORMS = {"square": 0, "sine": 1, "constant": 2}

def drive_coordinate(name):
    match = re.fullmatch(r"drives\[(\d+)\]\.(amplitude|period|phase|start|stop)", name)
    return (int(match[1]), match[2]) if match else None

EXTENSION_SCALARS = ("wall_height", "shear_x", "shear_y")

def indexed_name(name):
    match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\[(\d+)\]", name)
    return (match.group(1), int(match.group(2))) if match else (name, None)

def reject_conflicts(names):
    """Reject global/array/index selections that overwrite the same property."""
    names = list(names)
    for i, first in enumerate(names):
        a, ai = indexed_name(first)
        for second in names[i+1:]:
            b, bi = indexed_name(second)
            if (a == b and (ai is None or bi is None or ai == bi)) or \
               PER_BEAD_DEFAULTS.get(a) == b or PER_BEAD_DEFAULTS.get(b) == a:
                raise ValueError(f"Conflicting parameters: {first}, {second}")
SCALARS = (
    "num_beads amplitude base_distance bead_radius dt total_time viscosity "
    "alpha kx beta ky temperature sampling_feedback sampling_write epsilon "
    "velox ks fl0 veloy F0 T_signal default_lambda"
).split()
PARAMETERS = {
    name: {"index": i, "label": name.replace("_", " ")}
    for i, name in enumerate(SCALARS)
}
ARRAYS = ("distances", "lambdas", "kx_per_bead", "ky_per_bead", "signal_beads", "angles_deg", "alpha_per_bead", "beta_per_bead", "amplitude_per_bead", "epsilon_per_bead", "reference_x", "reference_y")
# These are supported continuous search coordinates. Numerical settings and
# unused C++ fields are intentionally excluded from the optimisation catalogue.
SEARCHABLE = {
    "reference_x", "reference_y",
    "amplitude",
    "base_distance",
    "viscosity",
    "alpha",
    "beta",
    "temperature",
    "epsilon",
    "velox",
    "veloy",
    "F0",
    "T_signal",
    "default_lambda",
    "kx",
    "ky",
    "lambdas",
    "kx_per_bead",
    "ky_per_bead",
    "positions",
    "angles_deg", "alpha_per_bead", "beta_per_bead", "amplitude_per_bead",
    "epsilon_per_bead", "bead_radius", "wall_height", "shear_x", "shear_y",
}


def json_value(value):
    """Convert NumPy values and missing numerical measurements to strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    )


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Configuration:
    """Complete physical input, independent of the file it originally came from."""

    values: dict

    def __post_init__(self):
        # Old templates and saved best.json files retain horizontal strokes.
        n = int(self.values["num_beads"])
        self.values.setdefault("angles_deg", [0.0] * n)
        for array, scalar in PER_BEAD_DEFAULTS.items():
            self.values.setdefault(array, [self.values[scalar]] * n)
        self.values.setdefault("mobility", "rotne_prager")
        self.values.setdefault("wall_height", 6.0)
        self.values.setdefault("shear_x", 0.0)
        self.values.setdefault("shear_y", 0.0)
        self.values.setdefault("reference_x", [])
        self.values.setdefault("reference_y", [])
        self.values.setdefault("groups", {})
        self.values.setdefault("drives", None) # None = legacy; [] = deliberately unforced.
        self.values.setdefault("direct_traps", False)
        self.values.setdefault("seed", -1)
        for drive in self.values["drives"] or []:
            for key, default in {"phase": 0.0, "start": 0.0,
                                 "stop": self.values["total_time"], "waveform": "square"}.items():
                drive.setdefault(key, default)

    @property
    def reference_positions(self):
        """Coordinates used by both geometry validation and layout plots."""
        if self.values["reference_x"]:
            return np.column_stack([self.values["reference_x"], self.values["reference_y"]])
        return np.column_stack([np.r_[0, np.cumsum(self.values["distances"])], np.zeros(self.n)])

    def targets(self, targets):
        """Resolve a named group or an explicit list, without adding physical links."""
        beads = self.values["groups"].get(targets) if isinstance(targets, str) else targets
        if not isinstance(beads, list) or not beads or any(
            not isinstance(b, (int, float)) or b != int(b) or not 0 <= b < self.n for b in beads
        ) or len(set(beads)) != len(beads):
            raise ValueError(f"Invalid bead group/targets: {targets}")
        return [int(b) for b in beads]

    @classmethod
    def read(cls, path):
        p = [float(x) for x in Path(path).read_text().splitlines()[0].split()]
        if len(p) < 22:
            raise ValueError("Need at least 22 scalar input values.")
        v = dict(zip(SCALARS, p[:22]))
        n = int(v["num_beads"])
        if n != v["num_beads"] or n < 2:
            raise ValueError("A chain needs an integer bead count >= 2.")
        cursor = 22
        v["distances"] = p[cursor : cursor + n - 1]
        cursor += n - 1
        if len(v["distances"]) != n - 1:
            raise ValueError("Missing chain distances.")
        # Match the existing parser's optional-block defaults, then always
        # write complete blocks so signal indices cannot become stiffnesses.
        v["lambdas"] = p[cursor : cursor + n]
        cursor += len(v["lambdas"])
        v["lambdas"] += [v["default_lambda"]] * (n - len(v["lambdas"]))
        for key, default in [("kx_per_bead", "kx"), ("ky_per_bead", "ky")]:
            if len(p) - cursor >= n:
                v[key] = p[cursor : cursor + n]
                cursor += n
            else:
                v[key] = [v[default]] * n
        tail = p[cursor:]
        if tail:
            count = int(tail[0])
            if count != tail[0] or count < 0 or len(tail) != count + 1:
                raise ValueError("Malformed signal bead block.")
            v["signal_beads"] = tail[1:] or [0]
        else:
            v["signal_beads"] = [0]
        # Labelled values override legacy blocks; no positional layout changes.
        seen = set()
        parsed_drives = []
        allowed = set(PER_BEAD_DEFAULTS) | {"angles_deg", "mobility", "reference_x", "reference_y", "direct_traps", "seed", "explicit_drives", *EXTENSION_SCALARS}
        for line in Path(path).read_text().splitlines()[1:]:
            if line.startswith("# groups "):
                v["groups"] = json.loads(line[len("# groups "):])
            fields = line.split("#", 1)[0].split()
            if not fields: continue
            key = fields[0]
            if key == "drive":
                a, period, phase, start, stop = map(float, fields[1:6])
                wave, count = map(int, fields[6:8])
                beads = list(map(int, fields[8:]))
                if wave not in WAVEFORMS.values() or len(beads) != count:
                    raise ValueError("Malformed drive line")
                parsed_drives.append(dict(amplitude=a, period=period, phase=phase,
                    start=start, stop=stop, waveform=list(WAVEFORMS)[wave], targets=beads))
                continue
            if key not in allowed or key in seen:
                raise ValueError(f"Unknown/duplicate extension: {key}")
            seen.add(key)
            if key == "mobility":
                if len(fields) != 2: raise ValueError("mobility needs one name")
                v[key] = fields[1]
            elif key in (*EXTENSION_SCALARS, "direct_traps", "seed", "explicit_drives"):
                if len(fields) != 2: raise ValueError(f"{key} needs one value")
                v[key] = float(fields[1])
            else:
                v[key] = [float(x) for x in fields[1:]]
        explicit = v.pop("explicit_drives", 0)
        if explicit not in (0, 1) or (parsed_drives and not explicit):
            raise ValueError("Drive lines require explicit_drives 1")
        v["drives"] = parsed_drives if explicit else None
        result = cls(v)
        result.validate()
        return result

    @property
    def n(self):
        return int(self.values["num_beads"])

    @property
    def length(self):
        return float(sum(self.values["distances"]))

    def get(self, name):
        coordinate = drive_coordinate(name)
        if coordinate:
            index, field = coordinate
            if self.values["drives"] is None or index >= len(self.values["drives"]):
                raise ValueError(f"Invalid drive coordinate: {name}")
            return self.values["drives"][index][field]
        root, index = indexed_name(name)
        if index is not None:
            if root not in ARRAYS or root == "signal_beads" or not 0 <= index < len(self.values[root]):
                raise ValueError(f"Invalid indexed parameter: {name}")
            return deepcopy(self.values[root][index])
        if name == "positions":
            if self.values["reference_x"]:
                raise ValueError("Use reference_x[i]/reference_y[i] for explicit geometry")
            return np.cumsum(self.values["distances"])[:-1].tolist()
        return deepcopy(self.values[name])

    def updated(self, **overrides):
        """Return a copy. Global kx/ky/lambda overrides update their full blocks.

        Explicit per-bead overrides take priority over global ones. Position
        overrides hold both endpoints and total length fixed.
        """
        result = Configuration(deepcopy(self.values))
        # Broadcast global settings first, then explicit arrays/indices take priority.
        for array, scalar in PER_BEAD_DEFAULTS.items():
            if scalar in overrides:
                result.values[array] = [float(overrides[scalar])] * self.n
        reject_conflicts([key for key in overrides if indexed_name(key)[0] in ARRAYS])
        for name, value in overrides.items():
            root, index = indexed_name(name)
            coordinate = drive_coordinate(name)
            if coordinate:
                result.get(name)
                di, field = coordinate
                result.values["drives"][di][field] = float(value)
            elif index is not None:
                result.get(name)  # validate index before assignment
                result.values[root][index] = float(value)
            elif name == "positions":
                result.get(name) # Chain coordinates must not silently alter an explicit layout.
                result.values["distances"] = np.diff([0, *value, self.length]).tolist()
            elif name in result.values:
                result.values[name] = deepcopy(value)
            else:
                raise ValueError(f"Unknown parameter: {name}")
        result.validate()
        return result

    def validate(self):
        v = self.values
        if not all(np.all(np.isfinite(np.asarray(x, dtype=float))) for key, x in v.items() if key not in ("mobility", "groups", "drives")):
            raise ValueError("All input values must be finite.")
        for name in ("num_beads", "sampling_feedback", "sampling_write"):
            if v[name] < 1 or v[name] != int(v[name]):
                raise ValueError(f"{name} must be a positive integer.")
        if v["direct_traps"] not in (0, 1) or v["seed"] != int(v["seed"]) or not -1 <= v["seed"] <= 4294967295:
            raise ValueError("direct_traps must be boolean; seed must be -1 or a uint32")
        if not isinstance(v["groups"], dict):
            raise ValueError("groups must map names to bead lists")
        for group in v["groups"]:
            self.targets(group)
        if v["drives"] is not None and not isinstance(v["drives"], list):
            raise ValueError("drives must be a list or null")
        for drive in v["drives"] or []:
            if set(drive) != DRIVE_FIELDS | {"waveform", "targets"}:
                raise ValueError("Each drive needs amplitude, period, phase, start, stop, waveform, targets")
            self.targets(drive["targets"])
            if not all(np.isfinite(drive[k]) for k in DRIVE_FIELDS) or drive["amplitude"] < 0 or drive["period"] <= 0 or drive["start"] < 0 or drive["stop"] <= drive["start"] or drive["waveform"] not in WAVEFORMS:
                raise ValueError("Invalid drive values")
            if drive["waveform"] != "constant" and drive["period"] < 20*v["dt"]:
                raise ValueError("Need >=20 integration steps per drive period")
        if bool(v["reference_x"]) != bool(v["reference_y"]) or (v["reference_x"] and
            (len(v["reference_x"]) != self.n or len(v["reference_y"]) != self.n)):
            raise ValueError("Explicit reference_x/reference_y each need N coordinates")
        if self.n < 2:
            raise ValueError("Need at least two beads.")
        for name in (
            "amplitude",
            "base_distance",
            "bead_radius",
            "dt",
            "total_time",
            "viscosity",
            "alpha",
            "beta",
            "epsilon",
            "T_signal",
            "kx",
            "ky",
        ):
            if v[name] <= 0:
                raise ValueError(f"{name} must be positive.")
        if v["temperature"] < 0 or v["F0"] < 0:
            raise ValueError("Temperature and signal amplitude cannot be negative.")
        for key in (*PER_BEAD_DEFAULTS, "angles_deg"):
            if len(v[key]) != self.n:
                raise ValueError(f"{key} needs {self.n} values.")
        if min(v["kx_per_bead"] + v["ky_per_bead"]) <= 0:
            raise ValueError("Trap coefficients must be positive.")
        for key in PER_BEAD_DEFAULTS:
            if key != "lambdas" and np.any(np.asarray(v[key]) <= 0):
                raise ValueError(f"{key} must be positive.")
        if v["mobility"] not in ("oseen", "rotne_prager", "blake"):
            raise ValueError("mobility must be oseen, rotne_prager or blake")
        if v["wall_height"] <= 0: raise ValueError("wall_height must be positive")
        if v["mobility"] == "blake":
            if v["wall_height"] < 3*v["bead_radius"]:
                raise ValueError("Blake requires wall_height >= 3 radii")
            if v["velox"] != 0 or v["veloy"] != 0:
                raise ValueError("Use shear_x/shear_y for wall-compatible flow")
            if np.any(self.pair_distances() < 4*v["bead_radius"]):
                raise ValueError("Blake reference gaps must be >= 4 radii; instantaneous gaps are also checked")
        elif v["shear_x"] != 0 or v["shear_y"] != 0:
            raise ValueError("shear_x/shear_y require Blake mobility")
        gaps = np.asarray(v["distances"])
        if len(gaps) != self.n - 1 or np.any(gaps < MIN_GAP - 1e-12):
            raise ValueError(f"Reference gaps must be >= {MIN_GAP}.")
        if np.any(self.pair_distances() < MIN_GAP - 1e-12):
            raise ValueError(f"All reference pair distances must be >= {MIN_GAP}")
        if np.any(self.pair_distances() <= 2 * v["bead_radius"]):
            raise ValueError("Reference gaps must exceed bead diameter.")
        # Negative effective trap strength would not be a confining trap.
        i = np.arange(self.n) - (self.n - 1) / 2
        if not v["direct_traps"] and np.any(1 + np.asarray(v["lambdas"]) * (i * i - 0.25) <= 0):
            raise ValueError("Lambda profile gives nonpositive trap strength.")
        if any(b != int(b) or not 0 <= b < self.n for b in v["signal_beads"]):
            raise ValueError("Signal bead indices are invalid.")

    def pair_distances(self):
        points = self.reference_positions
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        return distances[np.triu_indices(self.n, 1)]

    def write(self, path):
        self.validate()
        p = [self.values[k] for k in SCALARS]
        # Angles are deliberately not inserted into the positional blocks.
        for key in ("distances", "lambdas", "kx_per_bead", "ky_per_bead"):
            p.extend(self.values[key])
        beads = self.values["signal_beads"]
        p += [len(beads), *beads]
        lines = [" ".join(format(float(x), ".17g") for x in p)]
        lines.append("mobility " + self.values["mobility"])
        for key in (*EXTENSION_SCALARS, "angles_deg", *PER_BEAD_DEFAULTS):
            values = np.atleast_1d(self.values[key])
            lines.append(key + " " + " ".join(format(float(x), ".17g") for x in values))
        # Keep all structural extensions labelled; the legacy first line is intact.
        if self.values["reference_x"]:
            for key in ("reference_x", "reference_y"):
                lines.append(key + " " + " ".join(format(x, ".17g") for x in self.values[key]))
        lines += [f"direct_traps {int(self.values['direct_traps'])}",
                  f"seed {int(self.values['seed'])}",
                  f"explicit_drives {int(self.values['drives'] is not None)}"]
        for d in self.values["drives"] or []:
            beads = self.targets(d["targets"])
            fields = [d[k] for k in ("amplitude", "period", "phase", "start", "stop")]
            fields += [WAVEFORMS[d["waveform"]], len(beads), *beads]
            lines.append("drive " + " ".join(format(float(x), ".17g") for x in fields))
        lines.append("# groups " + json.dumps(self.values["groups"]))
        Path(path).write_text("\n".join(lines) + "\n")



def load_configuration(path):
    """Accept a template, a saved best.json, or a new optimisation history."""
    path = Path(path)
    if path.suffix == ".input":
        return Configuration.read(path)
    data = json.loads(path.read_text())
    data = data.get("best", data)
    if "configuration" not in data:
        raise ValueError(
            "JSON must contain a complete configuration; old best-only histories need their original template."
        )
    config = Configuration(data["configuration"])
    config.validate()
    return config


class ParameterSpace:
    """Flatten named scalar/array variables for either optimisation algorithm."""

    def __init__(self, base, specification):
        self.base, self.specification = base, deepcopy(specification)
        self.slices, self.labels = {}, []
        lower, upper, steps, initial = [], [], [], []
        if not specification:
            raise ValueError("Select at least one free parameter.")
        reject_conflicts(specification)
        for name, settings in specification.items():
            if base.values["direct_traps"] and indexed_name(name)[0] in ("lambdas", "default_lambda", "base_distance"):
                raise ValueError(f"{name} has no force effect with direct_traps=true")
            if base.values["drives"] is not None and name in ("F0", "T_signal"):
                raise ValueError("Search drives[i].amplitude/period when using explicit drives")
            coordinate = drive_coordinate(name)
            if coordinate:
                base.get(name) # Validate drive index before inspecting its waveform.
                di, field = coordinate
                if base.values["drives"][di]["waveform"] == "constant" and field in ("phase", "period"):
                    raise ValueError(f"{name} has no effect for a constant waveform")
            root, index = indexed_name(name)
            if not drive_coordinate(name) and (root not in SEARCHABLE or (index is not None and root not in PER_BEAD_DEFAULTS and root not in ("angles_deg", "reference_x", "reference_y"))):
                raise ValueError(f"Unsupported search parameter: {name}")
            if root in EXTENSION_SCALARS and base.values["mobility"] != "blake":
                raise ValueError(f"{root} only affects Blake mode")
            if root in ("velox", "veloy") and base.values["mobility"] == "blake":
                raise ValueError("Search shear_x/shear_y in Blake mode")
            values = np.atleast_1d(settings.get("initial", base.get(name))).astype(
                float
            )
            size = np.size(base.get(name))
            if len(values) != size or size == 0:
                raise ValueError(f"Wrong initial dimension for {name}.")
            bounds = settings.get(
                "bounds", (0, base.length) if name == "positions" else None
            )
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"Supply [lower, upper] bounds for {name}.")
            lo, hi = [np.broadcast_to(x, (size,)).astype(float) for x in bounds]
            step = np.broadcast_to(settings.get("step", (hi - lo) * 0.05), (size,))
            if (
                not np.all(np.isfinite([lo, hi, step, values]))
                or np.any(hi <= lo)
                or np.any(step < 0)
            ):
                raise ValueError(f"Invalid bounds/steps for {name}.")
            self.slices[name] = slice(len(lower), len(lower) + size)
            self.labels.extend(
                [name] if size == 1 else [f"{name}[{i}]" for i in range(size)]
            )
            lower.extend(lo)
            upper.extend(hi)
            steps.extend(step)
            initial.extend(values)
        self.lo, self.hi, self.steps, self.initial = map(
            np.asarray, (lower, upper, steps, initial)
        )
        self.configuration(self.initial)

    def configuration(self, vector):
        x = np.asarray(vector)
        if (
            x.shape != self.lo.shape
            or not np.all(np.isfinite(x))
            or np.any(x < self.lo)
            or np.any(x > self.hi)
        ):
            raise ValueError("Candidate outside parameter bounds.")
        overrides = {}
        for name, section in self.slices.items():
            values = x[section]
            overrides[name] = (
                values.tolist()
                if name in ARRAYS or name == "positions"
                else float(values[0])
            )
        return self.base.updated(**overrides)

    def sample(self, rng, count=1):
        """Sample geometry on the fixed-length gap simplex; reject other invalid inputs."""
        result = []
        for _ in range(max(1000, 100 * count)):
            x = rng.uniform(self.lo, self.hi)
            if "positions" in self.slices:
                excess = self.base.length - (self.base.n - 1) * MIN_GAP
                gaps = MIN_GAP + rng.dirichlet(np.ones(self.base.n - 1)) * max(
                    0, excess
                )
                x[self.slices["positions"]] = np.cumsum(gaps)[:-1]
            try:
                self.configuration(x)
            except ValueError:
                continue
            result.append(x)
            if len(result) == count:
                return np.asarray(result)
        raise ValueError(
            "Could not sample valid configurations; check bounds and geometry."
        )

    def normalise(self, x):
        return (np.asarray(x) - self.lo) / (self.hi - self.lo)


def run_simulation(config, executable, folder):
    """Run in a fresh folder. Failure leaves input, log and partial output intact."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    config.write(folder / "run.input")
    with (folder / "simulation.log").open("w") as log:
        completed = subprocess.run(
            [str(executable), "-C", {"oseen": "-O", "rotne_prager": "-R", "blake": "-B"}[config.values["mobility"]], str(folder / "run.input")],
            cwd=folder,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    files = list((folder / "run").glob("*.dat"))
    if completed.returncode or len(files) != 1:
        raise RuntimeError(f"Simulation failed; inspect {folder}/simulation.log")
    reader = BeadDataAnalyzer()
    reader.read_data_file(str(files[0]))
    # Detect a stale executable that would silently ignore the new input line.
    if not np.allclose(reader.angles_deg, config.values["angles_deg"], rtol=0, atol=1e-10):
        raise RuntimeError("Trajectory axes disagree with input; rebuild src/rowers.")
    # A sidecar confirms that the executable understood the new model options.
    metadata = files[0].with_suffix(".hydro")
    if not metadata.exists():
        raise RuntimeError("Missing mobility metadata; rebuild the updated src/rowers")
    hydro = dict(line.split() for line in metadata.read_text().splitlines())
    if hydro.get("pipeline_version") != "2":
        raise RuntimeError("Missing computing-pipeline support; rebuild src/rowers")
    if hydro.get("mobility") != config.values["mobility"] or any(
        not np.isclose(float(hydro[k]), config.values[k], rtol=1e-12, atol=1e-12)
        for k in EXTENSION_SCALARS):
        raise RuntimeError("Simulator mobility settings disagree with configuration")
    spacing = config.values["dt"] * config.values["sampling_write"]
    if (
        reader.num_beads != config.n
        or not np.all(np.isfinite(reader.positions))
        or not np.allclose(np.diff(reader.time_series), spacing, rtol=1e-4, atol=1e-8)
        or abs(reader.time_series[-1] - config.values["total_time"]) > 2 * spacing
    ):
        raise RuntimeError(f"Invalid trajectory dimensions/timestamps: {files[0]}")
    return reader
