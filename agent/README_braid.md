**Cable braid prototype — circular-arc bottom motion**

From the repository root:

```bash
uv run --extra examples agent/example_cable_braid.py
```

The default input is `agent/braid4_feed.json`. Use `--input PATH` to select another file. The interactive viewer starts paused; press Space to simulate. Use `--no-paused` to start simulation immediately. Rendering is capped at 60 FPS by default.

**Input format**

- `yarns`: one object per yarn with an integer `yarn` ID, `bottom`, and `top` in meters. Each yarn starts as a straight cable from `bottom` to `top`, sampled evenly into `samples` centerline vertices (default 30, including both endpoints, JSON only, integer of at least four).
- Both end vertices are pinned in position with ball joints to the world. Their orientation is free, so the cable may lean and twist at the pins. All interior capsule bodies are dynamic.
- `steps`: a list of steps that run in order. Each step is a list of turns and each turn a list of movements. All movements within a step, across all its turns, run simultaneously over the same time interval. Each movement has `yarn`, `type` (only `"arc"` is accepted), `start`, `mid`, `end`, and optional `feed`. The yarn's bottom vertex moves along the unique circle through the three points, from `start` through `mid` to `end`, at uniform angular speed. `mid` selects the side of the chord, so arcs larger than 180° are also supported. Arcs may lie in any plane.
- A movement's `start` must equal the yarn's current bottom position, chained from `bottom` through all earlier steps, within 1 µm. Yarns absent from a step hold position. After the last step all targets hold while physics continues.
- `feed` is validated as a finite number and recorded but not simulated: cable length stays fixed. A warning is issued when any feed is nonzero.
- `units.length` must be `"m"` if present. `yarn_diameter` defaults to 0.002 m. `axes`, `arc_convention`, `order`, `feed_scale`, `convergence_point`, and `collector_height` are ignored.

**Timing**

Set optional top-level `"step_duration": 1.0` to control the duration of every step in seconds. It defaults to one second when omitted, must be finite and positive, and is read only from JSON. Step 0 runs from time zero to `step_duration`, step 1 to `2 * step_duration`, and so on. The default run length covers all steps plus one second of holding, with a minimum of 120 frames: `max(120, ceil(60 * (step_count * step_duration + 1)))`. An explicit `--num-frames` overrides this. Motion uses simulation time, not wall-clock time; durations are not rounded to whole frames. The interactive viewer stays open until closed.

For the supplied file: four yarns of 30 vertices each (116 capsule bodies), eight steps, 8 s of motion, 540 default frames. Every arc is a 21 mm-radius semicircle in the XY plane. Steps alternate between all four yarns moving and only two moving; the bottoms return to their initial layout after step 7.

**Physics**

Optional top-level `physics` values override defaults individually:

| JSON key within `physics` | Default | Units / meaning |
| --- | --- | --- |
| `gravity` | `[0, 0, -9.81]` | m/s² |
| `density` | `1000` | kg/m³ |
| `stretch_stiffness`, `shear_stiffness` | `100000` | N/m, per joint |
| `bend_stiffness`, `twist_stiffness` | `0.01` | N·m, per joint |
| `stretch_damping`, `shear_damping` | `0` | N·s/m, per joint |
| `bend_damping`, `twist_damping` | `0.0001` | N·m·s/rad, per joint |
| `contact_stiffness` | `2500` | N/m |
| `contact_damping` | `100` | N·s/m |
| `friction` | `1` | Dimensionless |
| `ground_height` | `null` | No floor; a number adds a floor at that Z coordinate [m] |

For example, adding `"physics": {"gravity": [0, 0, 0], "friction": 0.5}` changes only gravity and friction. Density must be positive, material coefficients nonnegative, and all supplied numbers finite. Unknown `physics` keys are rejected. Contact gap is zero. Simulation uses 60 frames/s, ten substeps/frame, and twenty solver iterations. Joint stabilization uses `rigid_avbd_joint_alpha=0.0` to correct the full pin residual each substep.

The pin targets are written into the world-frame joint anchors before each physics substep. The targets of every substep are precomputed at start into a device table, so a frame needs no host work: on CUDA its ten substeps are captured once as a CUDA graph and replayed each frame, which removes the Python kernel-launch overhead that otherwise dominates. On CPU the same substeps run directly. Collision detection is enabled throughout playback, with directly connected neighbors filtered as in Newton's rod builder. An initial contact causes an error; the script does not move coordinates to hide intersections.

**Tests**

```bash
uv run --extra examples agent/example_cable_braid.py \
  --viewer null --device cpu --test

PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=./agent/cache/uv \
  WARP_CACHE_PATH=./agent/cache/warp TMPDIR=./agent/tmp \
  uv run --extra examples -m unittest -v agent.test_cable_braid
```

The graph-replay test (`TestGraphCapture`) runs only when a CUDA device is available and is skipped otherwise. To measure simulation speed without the default 60 FPS render cap:

```bash
time uv run --extra examples agent/example_cable_braid.py \
  --viewer null --no-paused --num-frames 300 --render-fps 10000
```

For the existing installed environment, replace `--extra examples` with `--no-sync --offline` to run without dependency synchronization. The script directs its default Warp cache and viewer output files into `agent/`.

The unit tests cover arc reconstruction (semicircle, uniform angular speed, greater-than-180° arcs, tilted planes, collinear rejection), the loader (initial cables, step schedule, arc evaluation at sample times, holding yarns, final hold, `step_duration` override, feed warning, empty steps, and rejection of discontinuous starts, unknown yarns, duplicates, non-arc types, bad geometry, and unknown physics keys), and a CPU tracking run of the first two steps at 0.5 s per step. The tracking run checks every frame that the commanded targets equal the independently evaluated arc positions, that pinned endpoints stay within 0.1 mm of their targets, that the top pins never move, that interior segments move, and that a bottom segment tilts away from its initial direction, which confirms the ball joints leave orientation free.

**Current limitations**

- `feed` is not simulated. Because both ends are pinned and cable length is fixed, the cable stretches or slackens as its bottom moves; the stiff stretch mode resists this.
- Newton's inertia validation automatically adjusts the rotational inertia of these small capsules. The defaults are suitable for a construction/stability check, not calibrated yarn dynamics.
- The default 30 samples gives 12 mm segments on a 34 cm cable against 21 mm-radius arcs. Raise `samples` in the JSON for finer wrapping, at higher simulation cost.
