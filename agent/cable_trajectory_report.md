**Cable simulation with user-prescribed point trajectories**

**Newton supports the underlying physics, but this checkout does not provide the requested JSON-to-simulation CLI workflow.** You can prescribe moving anchors in Python and let the attached cables respond to physics. There is no existing cable trajectory JSON schema or CLI option that consumes `bx2k/braid9_sample.json`. A small application layer should be sufficient for ordinary position tracking; a new cable solver is not needed.

Assessment date: September 9, 2026. Checkout: `67313bab1ad5869bf5498c9a2318a0ddbfa9ea44`, project version `1.5.0.dev0`. Existing source and user input files were left unchanged. This report, verification logs, and tool caches are all under `agent/`.

The distinction matters because “fixed moving points” can mean either a precisely prescribed segment pose or a cable point attached to a precisely moving anchor:

| Requested behavior | Current support |
| --- | --- |
| Move selected rigid cable segments along a programmed motion; simulate the rest | Yes, through Python state updates and kinematic bodies. Prescribing the full segment pose also controls its orientation. |
| Move a cable endpoint while leaving its rotation free | Yes, through a moving kinematic anchor and a ball joint. The attachment has numerical tracking error. |
| Select several endpoints or interior attachment locations | The body/joint APIs provide the building blocks; a runner must map user point identifiers to segment-local attachment positions. There is no ready-made point-selection input interface. |
| Load arbitrary position keyframes from JSON through an existing cable CLI | No implementation found. |
| Run `bx2k/braid9_sample.json` directly | No implementation found; the file also lacks timing and scene information needed for a determined simulation. |

**Evidence from the implementation.** `newton.ModelBuilder.add_rod()` constructs capsule bodies connected by cable joints; the input `positions` describes the initial centerline, not a time series. For N segments there are N+1 centerline positions but N rigid bodies. Consequently, a centerline point index is not generally a body index. With `body_frame_origin="com"`, a segment's endpoints are at local `(0, 0, -length/2)` and `(0, 0, length/2)`. See [rod construction](/home/bx2k/github/newton/newton/_src/sim/builder.py:7645), [body-frame conventions](/home/bx2k/github/newton/newton/_src/sim/builder.py:7729), and [segment construction](/home/bx2k/github/newton/newton/_src/sim/builder.py:8020).

The strongest matching example is actually an existing regression test: [moving ball-joint cable attachment](/home/bx2k/github/newton/newton/tests/test_cable.py:1227). It creates a kinematic anchor, attaches a dynamic rod endpoint with `builder.add_joint_ball()`, updates the anchor position at each substep, and advances `newton.solvers.SolverVBD`. The endpoint is checked against a final attachment error below **0.001 m**. That is the test's acceptance threshold, not a universal accuracy guarantee or a measurement of its actual error. Its [motion kernel](/home/bx2k/github/newton/newton/tests/test_cable.py:139) uses a sine function; replacing that function with interpolated user keyframes is the principal missing motion feature.

There is also a [translating and rotating attachment test](/home/bx2k/github/newton/newton/tests/test_cable.py:4196), and the [plectoneme example](/home/bx2k/github/newton/newton/examples/cable/example_cable_plectoneme.py:137) makes the first and last segments kinematic, then counter-rotates them while the intervening cable simulates. Its endpoint positions stay constant; its motion is not loaded from a file.

The solver explicitly [skips integrating zero-effective-inverse-mass bodies](/home/bx2k/github/newton/newton/_src/solvers/vbd/rigid_vbd_kernels.py:2722), while [ball joints constrain anchor coincidence](/home/bx2k/github/newton/newton/_src/solvers/vbd/solver_vbd.py:1223). These support the proposed approach. Setting a whole cable segment kinematic would also prescribe its orientation, so the ball-joint anchor approach better matches position-only control.

**There is no working JSON command to give you today.** I inspected the [example dispatcher](/home/bx2k/github/newton/newton/examples/__init__.py:966), [shared CLI parser](/home/bx2k/github/newton/newton/examples/__init__.py:632), cable example entry points, and JSON-loading call sites throughout the checkout. The JSON loader in [the file viewer](/home/bx2k/github/newton/newton/_src/viewer/viewer_file.py:1286) restores recorded model/states; it does not turn selected point trajectories into physics boundary conditions. The [USD cable importer](/home/bx2k/github/newton/newton/_src/utils/import_usd_deformable_cable.py:55) reads initial curve points and builds rods; it supplies no equivalent JSON trajectory runner.

You can run an existing related demonstration from the repository root:

```bash
uv run --extra examples -m newton.examples cable_plectoneme
```

This runs the built-in twisting scenario. It does **not** read your braid file. The supported options were verified through `--help`; see [captured help](/home/bx2k/github/newton/agent/cable_plectoneme_help.txt). I did not launch its graphical viewer.

**Your braid file describes spatial paths and scheduling order, but not a complete timed simulation.** [The supplied JSON](/home/bx2k/github/newton/bx2k/braid9_sample.json:1) contains nine yarn IDs, two steps, four turns per step, and two concurrent yarn motions per turn: sixteen individual arc descriptions. It specifies meters, a 0.002 m yarn diameter, and 0.004 m take-up per row. No Python consumer of its characteristic fields was found.

To implement this input faithfully, the runner would need the following information or explicit defaults:

1. **Timing:** seconds per turn or timestamps, plus the speed profile along each arc. Order alone cannot determine velocities or inertial response. The two yarn motions within a turn must share that turn's time interval; subsequent turns run sequentially.
2. **Initial yarn geometry and attachments:** the initial centerline, length, and discretization of each yarn; which material point moves; and what happens at the other end. The `start`, `mid`, and `end` values describe actuator paths, not the full initial shape of each yarn. Yarn 8 first appears in the second step, so its earlier position needs a definition.
3. **Motion between turns and rows:** whether inactive anchors hold position and how take-up happens over time. For example, yarn 0 ends the first row at `[0, 0.021, 0]` and starts the next at `[0, 0.021, 0.004]`; the intervening 4 mm movement needs a trajectory or an explicit jump policy. The runner must also define whether take-up moves these anchors, an opposite support, or changes a reference frame.
4. **Material and environment:** density, stretch/bend/twist properties, damping, friction/contact settings, and gravity direction in your axes. Diameter alone does not determine these. The Newton rod radius corresponding to your diameter is 0.001 m.
5. **Meaning of tension:** `1 = nominal` is not a physical force specification. With fully prescribed motion, tension is largely a reaction to the motion and material. A tension field needs a defined mapping, such as pretension or a separately controlled feed/tensioning mechanism; it cannot independently specify both force and displacement in the same direction.

The arc adapter should reconstruct the circle through `start`, `mid`, and `end` and choose the sweep through `mid`. Linear interpolation through those three positions would produce two straight chords, not the circular motion declared by the file.

**Estimated implementation size.** These are engineering estimates of new or substantially changed Python lines, including normal comments/docstrings. They assume fixed cable topology, ordinary numerical attachment tolerances, and reuse of the existing public APIs, Warp, NumPy, and stdlib JSON support.

| Scope | Estimated Python lines | Likely files |
| --- | ---: | --- |
| Minimal demonstration: one cable, a few timed moving anchors, linear interpolation, simple CLI | 150–250 | 1 new runner |
| Reusable runner: several cables/anchors, validated input, initial geometry, substep interpolation, viewer integration, tracking checks | 350–550 | 1–2 new implementation files |
| Regression tests for that reusable runner | 150–250 | 1 new `unittest` file |
| Additional adapter for your ordered circular-arc braid format | 150–300 | 1 additional implementation file |
| Additional tests for arc geometry, concurrency, and row transitions | 80–150 | 1 additional test file, or extend the runner tests |

For budgeting, expect **500–800 Python lines for a tested general trajectory runner**, or **730–1,250 lines including your braid-format adapter and its tests**. The adapter estimate assumes timing, initial scene, take-up, and inactive-yarn policies have been agreed. It excludes physical calibration, a yarn-feeding/spooling model, and an independently regulated tension mechanism.

**Expected edits to existing solver code: zero.** This can be delivered as new application/helper files. If incorporated as a standard repository example, also add its sample input, README command/screenshot, and changelog entry. Those documentation/assets are outside the Python estimates. A new example is discovered automatically by the [existing example scanner](/home/bx2k/github/newton/newton/examples/__init__.py:611).

The proposed runner would build the initial rods once, create kinematic anchors, attach selected cable locations through ball joints, then evaluate each position trajectory at every physics substep before collision detection and the solver step. Unselected cable bodies would remain dynamic. For interior points and multiple anchors, attachment mapping and articulation construction require care: [articulations reject multiple parents](/home/bx2k/github/newton/newton/_src/sim/builder.py:3217), so extra constraints must follow the supported arrangement for joints outside the articulation tree. For GPU graph replay, changing time/targets must live in device arrays or be updated outside capture, as illustrated by the existing examples. The solver already [derives body velocities from pose changes and copies output poses](/home/bx2k/github/newton/newton/_src/solvers/vbd/rigid_vbd_kernels.py:4934).

If you require a cable material point to match the input **exactly with zero attachment error while its orientation remains free**, the ball-joint approach does not establish that guarantee. The estimates above cover practical numerical tracking. Exact position-only enforcement would require a separate constraint-design investigation; simply making a whole segment kinematic changes the rotational behavior.

**Verification performed:** both existing CPU tests passed, using the installed environment without dependency synchronization:

```bash
PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=./agent/cache/uv \
WARP_CACHE_PATH=./agent/cache/warp \
TMPDIR=./agent/tmp \
uv run --no-sync --offline -m unittest -v \
  newton.tests.test_cable.TestCable.test_cable_ball_joint_attaches_rod_endpoint_cpu \
  newton.tests.test_cable.TestCable.test_cable_fixed_joint_tracks_moving_kinematic_cpu
```

See [test output](/home/bx2k/github/newton/agent/cable_motion_tests.txt). The two tests completed in 14.589 seconds. CUDA was unavailable, so GPU execution was not verified. These tests verify existing moving-anchor behavior; the proposed JSON loader, braid adapter, and multi-anchor scene were not implemented or tested during this read-only assessment.
