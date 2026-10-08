# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Simulate braiding yarns whose bottom ends follow circular arcs from JSON.

Run from the repository root:
    uv run --extra examples agent/example_cable_braid.py

The interactive viewer starts paused. Press Space to simulate. Use --viewer null
--test for a headless smoke test. --input selects a JSON file.

Input semantics (see README_braid.md for the schema):
  * ``yarns[].bottom`` / ``top`` define straight initial cables, sampled evenly into
    ``samples`` centerline vertices (default DEFAULT_SAMPLES, endpoints included, JSON only).
  * Both end vertices are pinned in position with ball joints, so their orientation
    is free. All interior capsule bodies are dynamic.
  * Every turn of every step runs in order, each lasting ``turn_duration`` seconds (default 1.0);
    how turns are grouped into steps does not change the motion. The movements of one turn
    (one actuator) move simultaneously. Each moves one yarn's bottom vertex along the circular
    arc through its ``start``, ``mid`` and ``end`` at uniform angular speed, or straight from
    ``start`` to ``end`` when ``mid`` equals ``start``. Yarns absent from a turn hold position.
    After the last turn all targets hold while physics continues. ``--steps N`` uses only the
    first N steps.
  * ``feed`` is validated but not simulated; a warning is issued if any is nonzero.
  * Movements of type ``takeup`` are ignored with a warning; a turn of takeups only takes no time.
  * On CUDA, one frame's substeps are captured as a CUDA graph and replayed each frame; bottom
    targets for every substep are precomputed into a device table for that reason.

Optional physics keys override DEFAULT_PHYSICS individually. Units: gravity [m/s²],
density [kg/m³], stretch/shear/contact stiffness [N/m], stretch/shear/contact
damping [N·s/m], bend/twist stiffness [N·m], bend/twist damping [N·m·s/rad],
ground_height and yarn_diameter [m]. Friction is dimensionless. ground_height=null
means no floor. Materials are illustrative, not calibrated yarn properties.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Keep prototype-generated caches beside the script, including on direct invocation.
AGENT_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault("WARP_CACHE_PATH", str(AGENT_DIR / "cache" / "warp"))

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_PHYSICS = {
    "gravity": [0.0, 0.0, -9.81],
    "density": 1000.0,
    "stretch_stiffness": 100000.0,
    "shear_stiffness": 100000.0,
    "bend_stiffness": 0.01,
    "twist_stiffness": 0.01,
    "stretch_damping": 0.0,
    "shear_damping": 0.0,
    "bend_damping": 0.0001,
    "twist_damping": 0.0001,
    "contact_stiffness": 2500.0,
    "contact_damping": 100.0,
    "friction": 1.0,
    "ground_height": None,
}
COLORS = (
    (0.9, 0.25, 0.18),
    (0.15, 0.65, 0.95),
    (0.25, 0.8, 0.35),
    (0.95, 0.65, 0.12),
    (0.65, 0.35, 0.9),
    (0.95, 0.4, 0.7),
    (0.1, 0.8, 0.75),
    (0.6, 0.45, 0.25),
    (0.85, 0.85, 0.2),
    (0.9, 0.9, 0.9),
)
# Tolerance for a step's start matching the yarn's current bottom position [m].
CONTINUITY_TOLERANCE = 1.0e-6
# Centerline vertices per yarn when the JSON omits "samples", endpoints included.
DEFAULT_SAMPLES = 30


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return np.array([_number(v, name) for v in value], dtype=np.float64)


@dataclass
class Arc:
    """Circular arc through three points, traversed from start to end at uniform angular speed."""

    center: np.ndarray
    """Circle center [m]."""
    radius: float
    """Circle radius [m]."""
    u: np.ndarray
    """Unit vector from the center to the start point."""
    v: np.ndarray
    """Unit vector in the arc plane, a quarter turn ahead of ``u`` toward ``mid``."""
    sweep: float
    """Signed angle [rad] from start to end passing through mid, in (0, 2π)."""

    @classmethod
    def through(cls, start: np.ndarray, mid: np.ndarray, end: np.ndarray, name: str) -> Arc:
        """Construct the unique circle through three non-collinear points and its start→mid→end sweep."""
        a = mid - start
        b = end - start
        normal = np.cross(a, b)
        normal_sq = float(normal @ normal)
        scale = max(float(a @ a), float(b @ b))
        if scale <= 0.0 or normal_sq <= (1.0e-9 * scale) ** 2:
            raise ValueError(f"{name}: start, mid, and end must be distinct and not collinear")
        center = start + np.cross(float(a @ a) * b - float(b @ b) * a, normal) / (2.0 * normal_sq)
        radius = float(np.linalg.norm(start - center))
        u = (start - center) / radius
        # Orient the plane so start→mid→end runs counterclockwise about the normal.
        v = np.cross(normal / math.sqrt(normal_sq), u)

        def angle(point: np.ndarray) -> float:
            offset = point - center
            return math.atan2(float(offset @ v), float(offset @ u)) % (2.0 * math.pi)

        sweep = angle(end)
        if not 0.0 < angle(mid) < sweep:
            raise ValueError(f"{name}: mid does not lie between start and end on the arc")
        return cls(center, radius, u, v, sweep)

    def position(self, fraction: float) -> np.ndarray:
        """Position [m] at a fraction in [0, 1] of the arc, clamped."""
        theta = self.sweep * min(1.0, max(0.0, fraction))
        return self.center + self.radius * (math.cos(theta) * self.u + math.sin(theta) * self.v)


@dataclass
class Line:
    """Straight move from start to end at uniform speed: an arc movement whose mid equals its start."""

    start: np.ndarray
    """Start point [m]."""
    end: np.ndarray
    """End point [m]."""

    def position(self, fraction: float) -> np.ndarray:
        """Position [m] at a fraction in [0, 1] of the segment, clamped."""
        return self.start + min(1.0, max(0.0, fraction)) * (self.end - self.start)


@dataclass
class SceneConfig:
    """Validated initial geometry, bottom-vertex motion, and settings in SI units."""

    @dataclass
    class Cable:
        """One yarn's straight initial centerline [m], including both endpoints."""

        yarn: int
        bottom: np.ndarray
        top: np.ndarray
        points: np.ndarray
        """Evenly sampled centerline, shape [samples, 3], from bottom to top."""

    samples: int
    diameter: float
    physics: dict
    cables: list[Cable]
    turn_duration: float
    """Duration of every turn [s]."""
    turns: list[dict[int, Arc | Line]]
    """Per turn, in playback order, the moving cables by cable index; absent cables hold position."""
    turn_steps: list[int]
    """Index of the JSON step each turn belongs to."""
    bottom_keyframes: np.ndarray
    """Bottom positions [m] after k completed turns, shape [turn_count + 1, cable_count, 3]."""
    feed: np.ndarray
    """Feed values parsed from the input (unused), shape [turn_count, cable_count]."""

    @property
    def playback_duration(self) -> float:
        """Total bottom motion duration [s]."""
        return len(self.turns) * self.turn_duration

    @property
    def default_num_frames(self) -> int:
        """Cover all motion and one second of holding at 60 simulation frames/s."""
        return max(120, math.ceil(60 * (self.playback_duration + 1)))

    def bottom_positions(self, time: float) -> np.ndarray:
        """Bottom vertex positions [m] at simulation time [s], shape [cable_count, 3]."""
        if time >= self.playback_duration or not self.turns:
            return self.bottom_keyframes[-1].copy()
        phase = max(0.0, time) / self.turn_duration
        index = min(math.floor(phase), len(self.turns) - 1)
        positions = self.bottom_keyframes[index].copy()
        for cable_index, path in self.turns[index].items():
            positions[cable_index] = path.position(phase - index)
        return positions

    def moving_yarns(self, turn_index: int) -> list[int]:
        """Yarn IDs that move during a turn, in cable order."""
        return [self.cables[i].yarn for i in sorted(self.turns[turn_index])]

    @classmethod
    def load(cls, path: str | Path, step_count: int | None = None) -> SceneConfig:
        """Load initial yarns and the turns of the first ``step_count`` steps (all when None).

        Defaults apply only to missing properties.
        """
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        units = data.get("units", {"length": "m"})
        if not isinstance(units, dict) or units.get("length", "m") != "m":
            raise ValueError("units.length must be 'm'")
        samples = data.get("samples", DEFAULT_SAMPLES)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
            raise ValueError("samples must be an integer >= 4 (including endpoints)")
        diameter = _number(data.get("yarn_diameter", 0.002), "yarn_diameter")
        if diameter <= 0:
            raise ValueError("yarn_diameter must be positive")
        turn_duration = _number(data.get("turn_duration", 1.0), "turn_duration")
        if turn_duration <= 0:
            raise ValueError("turn_duration must be positive [s]")
        physics = cls._load_physics(data.get("physics", {}))

        yarns = data.get("yarns")
        if not isinstance(yarns, list) or not yarns:
            raise ValueError("yarns must be a nonempty list")
        cables = []
        cable_index = {}
        for item in yarns:
            if not isinstance(item, dict):
                raise ValueError("Each yarn must be an object")
            yarn = item.get("yarn")
            if isinstance(yarn, bool) or not isinstance(yarn, int) or yarn < 0 or yarn in cable_index:
                raise ValueError("Yarn IDs must be distinct nonnegative integers")
            bottom = _vector(item.get("bottom"), f"yarn {yarn} bottom")
            top = _vector(item.get("top"), f"yarn {yarn} top")
            if np.linalg.norm(top - bottom) / (samples - 1) <= 1.0e-8:
                raise ValueError(f"Yarn {yarn} has zero or too-small initial segment length")
            cable_index[yarn] = len(cables)
            cables.append(cls.Cable(yarn, bottom, top, np.linspace(bottom, top, samples)))

        steps_data = data.get("steps", [])
        if not isinstance(steps_data, list):
            raise ValueError("steps must be a list")
        if step_count is not None:
            if step_count < 0:
                raise ValueError("the step count must be nonnegative")
            steps_data = steps_data[:step_count]
        current = np.array([cable.bottom for cable in cables])
        keyframes = [current.copy()]
        turns = []
        turn_steps = []
        feed = []
        takeup = False
        for index, step in enumerate(steps_data):
            if not isinstance(step, list) or not step:
                raise ValueError(f"steps[{index}] must be a nonempty list of turns")
            for turn_index, turn in enumerate(step):
                if not isinstance(turn, list) or not turn:
                    raise ValueError(f"Each turn in step {index} must be a nonempty list of movements")
                paths = {}
                turn_feed = np.zeros(len(cables))
                for item in turn:
                    if not isinstance(item, dict):
                        raise ValueError(f"Each movement in step {index} must be an object")
                    if item.get("type") == "takeup":
                        takeup = True
                        continue
                    yarn = item.get("yarn")
                    if yarn not in cable_index:
                        raise ValueError(f"Step {index} references unknown yarn {yarn!r}")
                    name = f"step {index} turn {turn_index} yarn {yarn}"
                    i = cable_index[yarn]
                    if i in paths:
                        raise ValueError(f"{name} appears more than once in the turn")
                    if item.get("type") != "arc":
                        raise ValueError(
                            f"{name}: only types 'arc' and 'takeup' are supported, got {item.get('type')!r}"
                        )
                    start = _vector(item.get("start"), f"{name} start")
                    mid = _vector(item.get("mid"), f"{name} mid")
                    end = _vector(item.get("end"), f"{name} end")
                    if np.linalg.norm(start - current[i]) > CONTINUITY_TOLERANCE:
                        raise ValueError(
                            f"{name}: start {start.tolist()} does not match the current bottom {current[i].tolist()}"
                        )
                    if np.linalg.norm(mid - start) <= CONTINUITY_TOLERANCE:
                        paths[i] = Line(start, end)
                    else:
                        paths[i] = Arc.through(start, mid, end, name)
                    turn_feed[i] = _number(item.get("feed", 0.0), f"{name} feed")
                if not paths:
                    continue
                for i, path in paths.items():
                    current[i] = path.position(1.0)
                turns.append(paths)
                turn_steps.append(index)
                feed.append(turn_feed)
                keyframes.append(current.copy())
        if not math.isfinite(60 * (len(turns) * turn_duration + 1)):
            raise ValueError("turn_duration produces a nonfinite playback length")
        if takeup:
            warnings.warn("takeup movements are present but not simulated; they are ignored.", stacklevel=2)
        feed = np.array(feed).reshape(len(turns), len(cables))
        if np.any(feed != 0.0):
            warnings.warn(
                "Nonzero feed values are present but feed is not simulated; yarn length stays fixed.", stacklevel=2
            )
        return cls(samples, diameter, physics, cables, turn_duration, turns, turn_steps, np.array(keyframes), feed)

    @staticmethod
    def _load_physics(supplied) -> dict:
        if not isinstance(supplied, dict):
            raise ValueError("physics must be an object")
        unknown = supplied.keys() - DEFAULT_PHYSICS.keys()
        if unknown:
            raise ValueError(f"Unknown physics keys: {', '.join(sorted(unknown))}")
        physics = DEFAULT_PHYSICS | supplied
        physics["gravity"] = _vector(physics["gravity"], "physics.gravity").tolist()
        for key, value in physics.items():
            if key == "gravity" or (key == "ground_height" and value is None):
                continue
            physics[key] = _number(value, f"physics.{key}")
            if key != "ground_height" and (physics[key] < 0 or (key == "density" and physics[key] == 0)):
                raise ValueError(f"physics.{key} must be {'positive' if key == 'density' else 'nonnegative'}")
        return physics


@wp.kernel
def update_pin_targets(
    joint_ids: wp.array[wp.int32],
    target_table: wp.array2d[wp.vec3],
    substep_count: wp.array[wp.int32],
    substep_offset: int,
    joint_frames: wp.array[wp.transform],
):
    """Translate selected world-parent joint frames to the next substep's targets without changing orientation.

    Row ``k`` of ``target_table`` holds the targets after ``k`` completed substeps; rows past the end hold.
    """
    i = wp.tid()
    row = wp.min(substep_count[0] + substep_offset + 1, target_table.shape[0] - 1)
    joint = joint_ids[i]
    joint_frames[joint] = wp.transform(target_table[row, i], wp.transform_get_rotation(joint_frames[joint]))


@wp.kernel
def advance_substep_count(substep_count: wp.array[wp.int32], substeps: int):
    """Advance the device-side substep counter by one frame (single thread)."""
    substep_count[0] = substep_count[0] + substeps


def add_endpoint_pins(builder: newton.ModelBuilder, bodies: list[int], points: np.ndarray) -> tuple[int, int]:
    """Pin both centerline endpoints to the world with ball joints; returns (bottom, top) joint IDs."""
    length = float(np.linalg.norm(points[1] - points[0]))
    joints = []
    # Rod bodies use body_frame_origin="com": the segment runs along local z from -L/2 to +L/2.
    for body, point, sign in ((bodies[0], points[0], -1.0), (bodies[-1], points[-1], 1.0)):
        joints.append(
            builder.add_joint_ball(
                parent=-1,
                child=body,
                parent_xform=wp.transform(wp.vec3(*point), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, sign * length * 0.5), wp.quat_identity()),
                label=f"pin_{body}",
            )
        )
    return joints[0], joints[1]


def build_scene(config: SceneConfig, device=None) -> tuple[newton.Model, list[list[int]], list[int], list[int]]:
    """Build dynamic rods, ball-joint endpoint pins, and optional ground geometry."""
    physics = config.physics
    builder = newton.ModelBuilder(gravity=tuple(physics["gravity"]))
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=physics["density"],
        ke=physics["contact_stiffness"],
        kd=physics["contact_damping"],
        mu=physics["friction"],
        gap=0.0,
    )
    rod_material = {
        f"{mode}_{kind}": physics[f"{mode}_{kind}"]
        for mode in ("stretch", "shear", "bend", "twist")
        for kind in ("stiffness", "damping")
    }
    cable_bodies = []
    bottom_pins = []
    top_pins = []
    for index, cable in enumerate(config.cables):
        bodies, _ = builder.add_rod(
            positions=[wp.vec3(*point) for point in cable.points],
            radius=config.diameter * 0.5,
            body_frame_origin="com",
            label=f"yarn_{cable.yarn}",
            color=COLORS[index % len(COLORS)],
            **rod_material,
        )
        cable_bodies.append(bodies)
        bottom, top = add_endpoint_pins(builder, bodies, cable.points)
        bottom_pins.append(bottom)
        top_pins.append(top)
    if physics["ground_height"] is not None:
        builder.add_ground_plane(height=physics["ground_height"])
    builder.color()
    return builder.finalize(device=device), cable_bodies, bottom_pins, top_pins


class Example:
    """Advance cable physics while commanding bottom vertex positions along JSON arcs, one turn at a time."""

    def __init__(self, viewer, args, config: SceneConfig | None = None):
        self.viewer = viewer
        self.config = config if config is not None else SceneConfig.load(args.input, args.steps)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Even, so the captured state_0/state_1 swaps return to the starting buffers each frame.
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.substep_count = 0
        self.model, self.cable_bodies, self.bottom_pins, self.top_pins = build_scene(self.config, args.device)
        self.pins = self.bottom_pins + self.top_pins
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.initial_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        if self.initial_contact_count:
            raise ValueError(
                f"Initial geometry has {self.initial_contact_count} contacts; check duplicate yarns, "
                "diameter, samples, and ground_height. Input coordinates were not changed."
            )
        # Correct the full pin residual each substep instead of retaining motion lag.
        self.solver = newton.solvers.SolverVBD(self.model, iterations=20, rigid_avbd_joint_alpha=0.0)
        self.bottom_pin_ids = wp.array(self.bottom_pins, dtype=wp.int32, device=self.model.device)
        self.target_table = wp.array(self._target_table(), dtype=wp.vec3, device=self.model.device)
        self.device_substep_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.viewer.set_model(self.model)
        self._frame_camera()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        count = len(self.config.cables)
        print(f"Yarns: {[c.yarn for c in self.config.cables]}")
        print(
            f"Samples/cable: {self.config.samples}; vertices: {count * self.config.samples}; "
            f"capsule bodies: {self.model.body_count}; initial contacts: {self.initial_contact_count}"
        )
        print(f"Yarn diameter [m]: {self.config.diameter}; physics: {json.dumps(self.config.physics, sort_keys=True)}")
        print(
            f"Turns: {len(self.config.turns)}; turn duration: {self.config.turn_duration:g} s; "
            f"playback: {self.config.playback_duration:g} s, then hold"
        )
        for index in range(len(self.config.turns)):
            print(
                f"  turn {index} (step {self.config.turn_steps[index]}): moving yarns {self.config.moving_yarns(index)}"
            )

    def _frame_camera(self):
        points = [self.config.bottom_keyframes.reshape(-1, 3), np.array([c.top for c in self.config.cables])]
        for turn in self.config.turns:
            points.extend(path.position(0.5)[None, :] for path in turn.values())
        points = np.concatenate(points)
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), self.config.diameter * 10)
        # Look across the row from +X so the yarns spread along Y do not overlap in projection.
        eye = center + np.array([1.6, 0.0, 0.3]) * extent
        self.viewer.set_camera(pos=wp.vec3(*eye), pitch=-10.0, yaw=180.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at(wp.vec3(*center))

    def _target_table(self) -> np.ndarray:
        """Bottom targets [m] after k completed substeps, shape [row_count, cable_count, 3].

        Precomputed so each substep needs no host work and a frame can be captured as a CUDA graph.
        The last row is the final hold, reached once playback ends.
        """
        substeps_per_second = self.fps * self.sim_substeps
        row_count = math.ceil(self.config.playback_duration * substeps_per_second) + 2
        return np.array([self.config.bottom_positions(k / substeps_per_second) for k in range(row_count)], np.float32)

    def simulate(self):
        """Advance one frame of substeps on the device, with no host work, so it can be graph-captured."""
        for substep in range(self.sim_substeps):
            # Only ball-joint frames change, so no solver.notify_model_changed() is needed: its JOINT_PROPERTIES
            # refresh covers cable-joint rest invariants only.
            wp.launch(
                update_pin_targets,
                dim=len(self.bottom_pins),
                inputs=[self.bottom_pin_ids, self.target_table, self.device_substep_count, substep],
                outputs=[self.model.joint_X_p],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            advance_substep_count,
            dim=1,
            inputs=[self.device_substep_count, self.sim_substeps],
            device=self.model.device,
        )

    def step(self):
        """Advance one frame with substep arc interpolation and collision detection."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.substep_count += self.sim_substeps
        self.sim_time = self.substep_count / (self.fps * self.sim_substeps)

    def render(self):
        """Render the current cable state without adding collision geometry."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def pin_error(self) -> float:
        """Return the maximum distance [m] between pinned endpoints and their targets."""
        poses = self.state_0.body_q.numpy()
        children = self.model.joint_child.numpy()
        local = self.model.joint_X_c.numpy()
        target = self.model.joint_X_p.numpy()
        error = 0.0
        for joint in self.pins:
            row = poses[children[joint]]
            body_pose = wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:]))
            actual = wp.transform_point(body_pose, wp.vec3(*local[joint, :3]))
            error = max(error, float(wp.length(actual - wp.vec3(*target[joint, :3]))))
        return error

    def test_final(self):
        """Verify finite states and bounded pinned endpoint errors."""
        for state in (self.state_0, self.state_1):
            if not np.isfinite(state.body_q.numpy()).all() or not np.isfinite(state.body_qd.numpy()).all():
                raise AssertionError("Cable state contains nonfinite positions or velocities")
        error = self.pin_error()
        if error >= 1.0e-3:
            raise AssertionError(f"Pinned endpoint error exceeds tolerance: {error:.6g} m")
        print(f"Pinned endpoint error: {error:.6g} m after {self.sim_time:.3f} s")

    @staticmethod
    def create_parser():
        """Expose input selection and standard viewer options; samples belong in JSON."""
        parser = newton.examples.create_parser()
        parser.add_argument("--input", type=Path, default=AGENT_DIR / "braid4_feed.json")
        parser.add_argument("--steps", type=int, default=None, help="use only the input's first N steps")
        parser.set_defaults(
            paused=True, num_frames=None, render_fps=60.0, output_path=str(AGENT_DIR / "braid_initial.usd")
        )
        return parser


def load_run_config(parser, argv=None) -> SceneConfig:
    """Validate input and set a duration-based frame default before creating the viewer."""
    args = parser.parse_args(argv)
    config = SceneConfig.load(args.input, args.steps)
    parser.set_defaults(num_frames=config.default_num_frames)
    return config


if __name__ == "__main__":
    parser = Example.create_parser()
    try:
        config = load_run_config(parser)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    viewer, args = newton.examples.init(parser)
    if getattr(viewer, "gui", None) is not None and viewer.gui.ui.is_available:
        viewer.gui.ui.io.set_ini_filename(str(AGENT_DIR / "braid_viewer.ini"))
        viewer.gui.ui.io.set_log_filename(str(AGENT_DIR / "braid_viewer_log.txt"))
    try:
        example = Example(viewer, args, config)
    except (ValueError, OSError) as exc:
        viewer.close()
        parser.error(str(exc))
    newton.examples.run(example, args)
