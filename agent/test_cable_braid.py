# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the braid prototype: arc geometry, JSON loading, and pinned-endpoint tracking.

Run from the repository root:
    uv run --extra examples -m unittest -v agent.test_cable_braid
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import warp as wp

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
os.environ.setdefault("WARP_CACHE_PATH", str(AGENT_DIR / "cache" / "warp"))

from example_cable_braid import DEFAULT_SAMPLES, Arc, Example, Line, SceneConfig  # noqa: E402

INPUT_PATH = AGENT_DIR / "braid4_feed.json"
WOVEN_PATH = AGENT_DIR / "woven3x3_tension.json"


def _write_json(data: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    with handle:
        json.dump(data, handle)
    return Path(handle.name)


def _load_quiet(path: Path) -> SceneConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SceneConfig.load(path)


class TestArc(unittest.TestCase):
    def test_semicircle_through_three_points(self):
        """Reconstruct the 21 mm semicircle used by the braid file and hit start, mid, and end."""
        start, mid, end = np.array([0.0, 0.0, 0.0]), np.array([-0.021, 0.021, 0.0]), np.array([0.0, 0.042, 0.0])
        arc = Arc.through(start, mid, end, "test")
        np.testing.assert_allclose(arc.center, [0.0, 0.021, 0.0], atol=1e-12)
        self.assertAlmostEqual(arc.radius, 0.021)
        self.assertAlmostEqual(arc.sweep, math.pi)
        np.testing.assert_allclose(arc.position(0.0), start, atol=1e-12)
        np.testing.assert_allclose(arc.position(0.5), mid, atol=1e-12)
        np.testing.assert_allclose(arc.position(1.0), end, atol=1e-12)

    def test_uniform_angular_speed(self):
        """Advance the arc parameter evenly and verify equal angular increments."""
        arc = Arc.through(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), "test")
        points = [arc.position(f) for f in np.linspace(0.0, 1.0, 5)]
        angles = [math.atan2(p[1], p[0]) for p in points]
        np.testing.assert_allclose(np.diff(angles), math.pi / 4, atol=1e-12)

    def test_mid_selects_long_way_round(self):
        """Choose the 270-degree arc when mid lies on the far side of the chord."""
        arc = Arc.through(np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), "test")
        self.assertAlmostEqual(arc.sweep, 1.5 * math.pi)
        np.testing.assert_allclose(arc.position(1.0 / 3.0), [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(arc.position(2.0 / 3.0), [-1.0, 0.0, 0.0], atol=1e-12)

    def test_arc_in_tilted_plane(self):
        """Handle an arc whose plane is not axis-aligned."""
        start, mid, end = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, -1.0])
        arc = Arc.through(start, mid, end, "test")
        np.testing.assert_allclose(arc.position(0.5), mid, atol=1e-12)
        for f in np.linspace(0.0, 1.0, 7):
            self.assertAlmostEqual(float(np.linalg.norm(arc.position(f) - arc.center)), arc.radius)

    def test_rejects_collinear_points(self):
        """Reject three points on a line, which define no circle."""
        with self.assertRaises(ValueError):
            Arc.through(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), "test")
        with self.assertRaises(ValueError):
            Arc.through(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), "test")


class TestSceneConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with INPUT_PATH.open(encoding="utf-8") as stream:
            cls.data = json.load(stream)
        cls.config = _load_quiet(INPUT_PATH)

    def test_initial_cables_from_yarns(self):
        """Build straight, evenly sampled cables from each yarn's bottom and top."""
        config = self.config
        self.assertEqual([c.yarn for c in config.cables], [0, 1, 2, 3])
        self.assertEqual(DEFAULT_SAMPLES, 30)
        self.assertEqual(config.samples, DEFAULT_SAMPLES)
        for cable, item in zip(config.cables, self.data["yarns"], strict=True):
            self.assertEqual(cable.points.shape, (DEFAULT_SAMPLES, 3))
            np.testing.assert_allclose(cable.points[0], item["bottom"])
            np.testing.assert_allclose(cable.points[-1], item["top"])
            lengths = np.linalg.norm(np.diff(cable.points, axis=0), axis=1)
            np.testing.assert_allclose(lengths, lengths[0])

    def test_turn_schedule(self):
        """Run the twelve turns of eight steps one after another, with a thirteen-second default run."""
        config = self.config
        self.assertEqual(len(config.turns), 12)
        self.assertEqual(
            [config.moving_yarns(i) for i in range(12)],
            [sorted(item["yarn"] for item in turn) for step in self.data["steps"] for turn in step],
        )
        self.assertEqual(config.turn_steps, [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7])
        self.assertEqual(config.turn_duration, 1.0)
        self.assertEqual(config.playback_duration, 12.0)
        self.assertEqual(config.default_num_frames, 780)

    def test_bottom_positions_follow_arcs_and_hold(self):
        """Evaluate bottom targets mid-arc, at turn boundaries, for holding yarns, and after playback."""
        config = self.config
        bottoms = np.array([y["bottom"] for y in self.data["yarns"]])
        np.testing.assert_allclose(config.bottom_positions(0.0), bottoms)
        np.testing.assert_allclose(config.bottom_positions(-1.0), bottoms)
        # Halfway through turn 0 (step 0), yarns 0 and 1 are at their mid points; yarns 2 and 3 wait.
        at_0_5 = config.bottom_positions(0.5)
        np.testing.assert_allclose(at_0_5[:2], [[-0.021, 0.021, 0.0], [0.021, 0.021, 0.0]], atol=1e-12)
        np.testing.assert_allclose(at_0_5[2:], bottoms[2:], atol=1e-12)
        # A quarter through turn 0, yarn 0 is 45 degrees along its semicircle.
        quarter = np.array([-0.021 * math.sin(math.pi / 4), 0.021 - 0.021 * math.cos(math.pi / 4), 0.0])
        np.testing.assert_allclose(config.bottom_positions(0.25)[0], quarter, atol=1e-12)
        # Halfway through turn 1 (still step 0), yarns 2 and 3 move while 0 and 1 hold their ends.
        at_1_5 = config.bottom_positions(1.5)
        np.testing.assert_allclose(at_1_5[:2], [[0.0, 0.042, 0.0], [0.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(at_1_5[2:], [[-0.021, 0.105, 0.0], [0.021, 0.105, 0.0]], atol=1e-12)
        # After both turns of step 0, every yarn is at its end.
        expected_end = np.array([[0.0, 0.042, 0.0], [0.0, 0.0, 0.0], [0.0, 0.126, 0.0], [0.0, 0.084, 0.0]])
        np.testing.assert_allclose(config.bottom_positions(2.0), expected_end, atol=1e-12)
        # During turn 2 (step 1) only yarns 0 and 3 move.
        at_2_5 = config.bottom_positions(2.5)
        np.testing.assert_allclose(at_2_5[[1, 2]], expected_end[[1, 2]], atol=1e-12)
        np.testing.assert_allclose(at_2_5[[0, 3]], [[0.021, 0.063, 0.0], [-0.021, 0.063, 0.0]], atol=1e-12)
        # The sequence returns to the initial layout and then holds.
        np.testing.assert_allclose(config.bottom_positions(12.0), bottoms, atol=1e-12)
        np.testing.assert_allclose(config.bottom_positions(50.0), bottoms, atol=1e-12)

    def test_step_grouping_does_not_change_motion(self):
        """Regrouping all turns into one step gives the same turns and targets."""
        data = copy.deepcopy(self.data)
        data["steps"] = [[turn for step in data["steps"] for turn in step]]
        config = _load_quiet(_write_json(data))
        self.assertEqual(len(config.turns), 12)
        for time in np.linspace(0.0, 13.0, 53):
            np.testing.assert_allclose(config.bottom_positions(time), self.config.bottom_positions(time), atol=1e-12)

    def test_turn_duration_override(self):
        """Scale the schedule by a JSON turn_duration."""
        data = copy.deepcopy(self.data)
        data["turn_duration"] = 0.5
        config = _load_quiet(_write_json(data))
        self.assertEqual(config.playback_duration, 6.0)
        self.assertEqual(config.default_num_frames, 420)
        np.testing.assert_allclose(config.bottom_positions(0.25), self.config.bottom_positions(0.5), atol=1e-12)

    def test_step_count_limit(self):
        """Load only the turns of the first steps."""
        config = SceneConfig.load(INPUT_PATH, 1)
        self.assertEqual(len(config.turns), 2)
        self.assertEqual(config.playback_duration, 2.0)
        np.testing.assert_allclose(config.bottom_positions(5.0), self.config.bottom_positions(2.0), atol=1e-12)

    def test_takeup_is_ignored_and_warned(self):
        """Skip takeup movements with a warning; a turn of takeups only takes no time."""
        data = copy.deepcopy(self.data)
        data["steps"][0].insert(1, [{"type": "takeup", "distance": 0.002, "weft_axis": "y"}])
        data["steps"][1][0].append({"type": "takeup", "distance": 0.002, "weft_axis": "x"})
        with self.assertWarnsRegex(UserWarning, "takeup"):
            config = SceneConfig.load(_write_json(data))
        self.assertEqual(len(config.turns), 12)
        np.testing.assert_allclose(config.bottom_keyframes, self.config.bottom_keyframes, atol=1e-12)

    def test_mid_equal_to_start_moves_straight(self):
        """An arc movement whose mid equals its start moves straight to its end."""
        data = copy.deepcopy(self.data)
        data["steps"].append(
            [
                [
                    {
                        "yarn": 0,
                        "type": "arc",
                        "start": [0.0, 0.0, 0.0],
                        "mid": [0.0, 0.0, 0.0],
                        "end": [0.021, -0.021, 0.0],
                    }
                ]
            ]
        )
        config = _load_quiet(_write_json(data))
        self.assertIsInstance(config.turns[-1][0], Line)
        np.testing.assert_allclose(config.bottom_positions(12.5)[0], [0.0105, -0.0105, 0.0], atol=1e-12)
        np.testing.assert_allclose(config.bottom_positions(13.0)[0], [0.021, -0.021, 0.0], atol=1e-12)

    def test_woven_input(self):
        """Load the 10-yarn woven input: 100 turns of arcs and straight moves; takeups ignored."""
        with self.assertWarnsRegex(UserWarning, "takeup"):
            config = SceneConfig.load(WOVEN_PATH)
        self.assertEqual(len(config.cables), 10)
        self.assertEqual(len(config.turns), 100)
        self.assertEqual(sum(isinstance(p, Line) for turn in config.turns for p in turn.values()), 64)
        self.assertEqual(len(SceneConfig.load(WOVEN_PATH, 1).turns), 14)

    def test_feed_is_parsed_and_warned(self):
        """Record feed values and warn that they are not simulated."""
        with self.assertWarns(UserWarning):
            config = SceneConfig.load(INPUT_PATH)
        self.assertEqual(config.feed.shape, (12, 4))
        self.assertAlmostEqual(config.feed[0, 0], -0.00514)
        np.testing.assert_allclose(config.feed.sum(axis=0), 0.0, atol=1e-12)
        data = copy.deepcopy(self.data)
        for step in data["steps"]:
            for turn in step:
                for item in turn:
                    item["feed"] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SceneConfig.load(_write_json(data))

    def test_no_steps_is_stationary(self):
        """Accept an input without steps and hold the initial bottoms."""
        data = copy.deepcopy(self.data)
        data["steps"] = []
        config = _load_quiet(_write_json(data))
        self.assertEqual(config.playback_duration, 0.0)
        self.assertEqual(config.default_num_frames, 120)
        np.testing.assert_allclose(config.bottom_positions(3.0), [y["bottom"] for y in self.data["yarns"]])

    def _assert_rejects(self, mutate, message_fragment: str):
        data = copy.deepcopy(self.data)
        mutate(data)
        with self.assertRaisesRegex(ValueError, message_fragment):
            _load_quiet(_write_json(data))

    def test_rejects_discontinuous_start(self):
        """Reject a step whose start does not match the yarn's current bottom."""

        def mutate(data):
            data["steps"][1][0][0]["start"] = [0.0, 0.05, 0.0]

        self._assert_rejects(mutate, "does not match the current bottom")

    def test_rejects_unknown_yarn(self):
        """Reject a movement for a yarn not declared in yarns."""

        def mutate(data):
            data["steps"][0][0][0]["yarn"] = 7

        self._assert_rejects(mutate, "unknown yarn")

    def test_rejects_duplicate_yarn_within_turn(self):
        """Reject a yarn that moves twice within a single turn."""

        def mutate(data):
            data["steps"][1][0].append(copy.deepcopy(data["steps"][1][0][0]))

        self._assert_rejects(mutate, "more than once")

    def test_rejects_non_arc_type(self):
        """Reject movement types other than arc and takeup."""

        def mutate(data):
            data["steps"][0][0][0]["type"] = "line"

        self._assert_rejects(mutate, "only types 'arc' and 'takeup'")

    def test_rejects_bad_geometry(self):
        """Reject collinear arc points and duplicate yarn IDs."""

        def collinear(data):
            data["steps"][0][0][0]["mid"] = [0.0, 0.021, 0.0]

        def duplicate(data):
            data["yarns"][1]["yarn"] = 0

        self._assert_rejects(collinear, "collinear")
        self._assert_rejects(duplicate, "distinct")

    def test_rejects_unknown_physics_key(self):
        """Reject unknown physics overrides while accepting known ones."""

        def mutate(data):
            data["physics"] = {"gravity": [0.0, 0.0, 0.0], "viscosity": 1.0}

        self._assert_rejects(mutate, "Unknown physics keys")
        data = copy.deepcopy(self.data)
        data["physics"] = {"gravity": [0.0, 0.0, 0.0], "friction": 0.5}
        config = _load_quiet(_write_json(data))
        self.assertEqual(config.physics["gravity"], [0.0, 0.0, 0.0])
        self.assertEqual(config.physics["friction"], 0.5)
        self.assertEqual(config.physics["density"], 1000.0)


class TestTracking(unittest.TestCase):
    def test_bottom_pins_track_arcs_cpu(self):
        """Simulate the first two steps (three turns) on CPU and keep pinned endpoints on their targets every frame.

        Uses a half-second turn so the run covers semicircle exchanges while other yarns hold, and final
        holding, within 150 frames.
        """
        import newton  # noqa: PLC0415

        with INPUT_PATH.open(encoding="utf-8") as stream:
            data = json.load(stream)
        data["steps"] = data["steps"][:2]
        data["turn_duration"] = 0.5
        path = _write_json(data)
        config = _load_quiet(path)
        parser = Example.create_parser()
        args = parser.parse_args(["--viewer", "null", "--device", "cpu", "--input", str(path)])
        example = Example(newton.viewer.ViewerNull(), args, config)
        self.assertEqual(example.initial_contact_count, 0)
        self.assertEqual(example.model.body_count, 4 * (DEFAULT_SAMPLES - 1))
        self.assertEqual(len(example.bottom_pins), 4)
        self.assertEqual(len(example.top_pins), 4)

        interior = [bodies[len(bodies) // 2] for bodies in example.cable_bodies]
        initial = example.state_0.body_q.numpy()[interior, :3].copy()
        children = example.model.joint_child.numpy()
        max_error = 0.0
        for _ in range(config.default_num_frames):
            example.step()
            self.assertTrue(np.isfinite(example.state_0.body_q.numpy()).all())
            max_error = max(max_error, example.pin_error())
            # The commanded targets must equal the independently evaluated arc positions.
            targets = example.model.joint_X_p.numpy()[example.bottom_pins, :3]
            np.testing.assert_allclose(targets, config.bottom_positions(example.sim_time), atol=1e-6)
        self.assertLess(max_error, 1.0e-4)
        # Top pins never moved.
        tops = example.model.joint_X_p.numpy()[example.top_pins, :3]
        np.testing.assert_allclose(tops, [c.top for c in config.cables], atol=1e-6)
        # Interior segments are dynamic and moved with the braid.
        moved = np.linalg.norm(example.state_0.body_q.numpy()[interior, :3] - initial, axis=1)
        self.assertGreater(float(moved.max()), 1.0e-3)
        # Ball joints leave orientation free: the bottom segment of a swapped yarn no longer points along its
        # initial direction.
        first_bodies = [children[j] for j in example.bottom_pins]
        body_q = example.state_0.body_q.numpy()
        tilts = []
        for body, cable in zip(first_bodies, config.cables, strict=True):
            q = body_q[body, 3:]
            # Rotate local +z by the quaternion (xyzw).
            x, y, z, w = q
            axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
            rest = cable.points[1] - cable.points[0]
            tilts.append(math.degrees(math.acos(np.clip(axis @ rest / np.linalg.norm(rest), -1, 1))))
        self.assertGreater(max(tilts), 1.0)
        example.test_final()


@unittest.skipUnless(wp.is_cuda_available(), "requires a CUDA device")
class TestGraphCapture(unittest.TestCase):
    def test_graph_replay_matches_eager_cuda(self):
        """Replay the captured frame graph on CUDA and match eager execution of the same substeps.

        Covers the first 1.5 s of the supplied input: one full turn and half of the next, two yarns
        moving and two holding in each. The graph run must also advance its device-side targets every frame.
        """
        import newton  # noqa: PLC0415

        config = _load_quiet(INPUT_PATH)
        args = Example.create_parser().parse_args(["--viewer", "null", "--device", "cuda:0"])
        graph = Example(newton.viewer.ViewerNull(), args, config)
        eager = Example(newton.viewer.ViewerNull(), args, config)
        self.assertIsNotNone(graph.graph)
        eager.graph = None
        for _ in range(90):
            graph.step()
            eager.step()
            targets = graph.model.joint_X_p.numpy()[graph.bottom_pins, :3]
            np.testing.assert_allclose(targets, config.bottom_positions(graph.sim_time), atol=1e-6)
            self.assertLess(graph.pin_error(), 1.0e-4)
        np.testing.assert_allclose(graph.model.joint_X_p.numpy(), eager.model.joint_X_p.numpy(), atol=1e-6)
        # Atomic accumulation order may differ between runs, so allow small drift in the dynamic bodies.
        np.testing.assert_allclose(graph.state_0.body_q.numpy()[:, :3], eager.state_0.body_q.numpy()[:, :3], atol=1e-3)
        graph.test_final()


if __name__ == "__main__":
    unittest.main()
