# Software Testing Project — Pure Python 2D Game Engine

## Team Members
- **Nishanth** — Blackbox testing, whitebox testing, mutation testing
- **Hsuan** — Combinatorial testing, integration testing

## Project Overview

This project applies systematic software testing techniques to a **Pure Python 2D Game Engine** — a comprehensive game engine built entirely from Python's standard library. We tested the engine's core modules using five testing methodologies: blackbox, whitebox, mutation, combinatorial, and greybox integration testing. Our test suite contains **1,354 tests** and achieves **96% branch coverage** on the SUT modules, uncovering **4 real bugs** in the engine.

## System Under Test (SUT)

The SUT consists of the following engine modules:

| Module | Path |
|--------|------|
| Vector2 | `engine/math/vector2.py` |
| Vector3 | `engine/math/vector3.py` |
| Quaternion | `engine/math/quaternion.py` |
| Transform | `engine/math/transform.py` |
| GameObject | `engine/scene/game_object.py` |
| Scene | `engine/scene/scene.py` |
| Logger | `engine/core/logger.py` |
| InputManager | `engine/input/input_manager.py` |
| SoundGenerator | `engine/audio/sound_generator.py` |
| Sprite | `engine/graphics/sprite.py` |
| Renderer | `engine/graphics/renderer.py` |

Non-SUT modules (excluded from coverage metrics): `engine.py`, `window.py`, `hot_reload.py`, and the `ecs/` subsystem.

---

## Prerequisites

- **Python 3.10+**
- pip (Python package manager)

## Setup Instructions

1. Clone or extract the project folder.

2. Install the required testing dependencies:

```bash
pip install pytest coverage mutmut
```

Specific versions used during development:
- pytest 9.0.3
- coverage 7.13.5
- mutmut 2.5.1

3. Navigate to the project root directory (the folder containing the `engine/` and `tests/` directories):

```bash
cd pure-python-game-engine
```

---

## Running the Tests

### 1. Run All Tests

```bash
python -m pytest tests/ -v
```

**Expected output:** 1,350 passed, 4 failed (the 4 failures are real bugs in the SUT — see [Bugs Found](#bugs-found) below).

### 2. Run Tests by Category

**Blackbox tests (545 tests):**
```bash
python -m pytest tests/ -k "not whitebox and not mutation and not combinatorial and not integration" -v
```

**Whitebox tests (78 tests):**
```bash
python -m pytest tests/ -k "whitebox" -v
```

**Mutation-killing tests (129 tests):**
```bash
python -m pytest tests/ -k "mutation" -v
```

**Combinatorial tests (569 tests):**
```bash
python -m pytest tests/ -k "combinatorial" -v
```

**Integration tests (33 tests):**
```bash
python -m pytest tests/ -k "integration" -v
```

---

## Reproducing Coverage Results

### Branch Coverage (SUT Modules Only — 96%)

To see the **96% branch coverage** for the SUT modules, run coverage with the `--omit` flag to exclude non-SUT files:

```bash
coverage run --branch \
  --omit="tests/*,engine/core/engine.py,engine/core/window.py,engine/core/hot_reload.py,engine/graphics/renderer.py,engine/ecs/*,breakout_game.py,asteroids_game.py,centipede_game.py,space_shooter.py,ui_game.py,example_*.py" \
  -m pytest tests/ -q
```

Then view the report:

```bash
coverage report
```

To generate an interactive HTML report:

```bash
coverage html
```

Open `htmlcov/index.html` in your browser to explore per-file coverage.

### Full Codebase Coverage

To see coverage for the entire codebase (including non-SUT modules):

```bash
coverage run --branch -m pytest tests/ -q
coverage report
```

This shows ~86% branch coverage / ~92% total — the lower numbers reflect non-SUT modules like `engine.py` and `window.py` that are not the focus of our testing.

---

## Reproducing Mutation Testing Results

Mutation testing was run using **mutmut 2.5.1** on the SUT math modules. mutmut requires all tests to pass on the unmodified source code first (clean baseline). Since 4 tests fail due to real SUT bugs, we exclude those specific test files from the mutmut runner.

### Running mutmut

```bash
mutmut run \
  --paths-to-mutate=engine/math/ \
  --tests-dir=tests/test_math/ \
  --runner="python -m pytest tests/test_math/ -x -q --tb=no \
    --ignore=tests/test_math/test_vector2_combinatorial.py \
    --ignore=tests/test_math/test_quaternion_combinatorial.py \
    --ignore=tests/test_math/test_transform_combinatorial.py"
```

**Expected result:** 697 mutants generated, 579 killed, 118 survived.

### Viewing Results

```bash
mutmut results
```

To inspect a specific surviving mutant:

```bash
mutmut show <mutant_id>
```

### Why some tests are excluded from mutmut

The 4 failing tests (caused by real SUT bugs) would cause mutmut's baseline check to fail. The `--ignore` flags exclude Hsuan's combinatorial test files that contain these failing tests, so the baseline passes cleanly. This does not affect the mutation testing results — the excluded tests target different code paths.

---

## Test File Reference

### Blackbox Tests (Nishanth)

| Test File | SUT Module |
|-----------|------------|
| `tests/test_math/test_vector2.py` | Vector2 |
| `tests/test_math/test_vector3.py` | Vector3 |
| `tests/test_math/test_quaternion.py` | Quaternion |
| `tests/test_math/test_transform.py` | Transform |
| `tests/test_scene/test_game_object.py` | GameObject |
| `tests/test_scene/test_scene.py` | Scene |
| `tests/test_core/test_logger.py` | Logger |
| `tests/test_input/test_input_manager.py` | InputManager |
| `tests/test_audio/test_sound_generator.py` | SoundGenerator |
| `tests/test_graphics/test_sprite.py` | Sprite |
| `tests/test_graphics/test_renderer.py` | Renderer |

### Whitebox Tests (Nishanth)

| Test File | SUT Module |
|-----------|------------|
| `tests/test_math/test_quaternion_whitebox.py` | Quaternion |
| `tests/test_scene/test_game_object_whitebox.py` | GameObject |
| `tests/test_scene/test_scene_whitebox.py` | Scene |
| `tests/test_core/test_logger_whitebox.py` | Logger |
| `tests/test_input/test_input_manager_whitebox.py` | InputManager |
| `tests/test_audio/test_sound_whitebox.py` | SoundGenerator |
| `tests/test_graphics/test_sprite_whitebox.py` | Sprite |

### Mutation-Killing Tests (Nishanth)

| Test File | SUT Module |
|-----------|------------|
| `tests/test_math/test_vector2_mutation.py` | Vector2 |
| `tests/test_math/test_vector3_mutation.py` | Vector3 |
| `tests/test_math/test_quaternion_mutation.py` | Quaternion |
| `tests/test_scene/test_game_object_mutation.py` | GameObject |
| `tests/test_core/test_logger_mutation.py` | Logger |
| `tests/test_input/test_input_manager_mutation.py` | InputManager |
| `tests/test_audio/test_sound_mutation.py` | SoundGenerator |
| `tests/test_graphics/test_sprite_mutation.py` | Sprite |

### Combinatorial Tests (Shawn)

| Test File | SUT Module |
|-----------|------------|
| `tests/test_math/test_vector2_combinatorial.py` | Vector2 |
| `tests/test_math/test_quaternion_combinatorial.py` | Quaternion |
| `tests/test_math/test_transform_combinatorial.py` | Transform |
| `tests/test_audio/test_sound_combinatorial.py` | SoundGenerator |
| `tests/test_graphics/test_sprite_combinatorial.py` | Sprite |
| `tests/test_graphics/test_renderer_combinatorial.py` | Renderer |

### Integration Tests (Shawn)

| Test File | Description |
|-----------|-------------|
| `tests/test_integration/test_breakout_integration.py` | Greybox integration testing of engine subsystems |

---

## Bugs Found

Our testing uncovered **4 real bugs** in the SUT:

### Bug 1: Mouse Release State Not Cleared

- **Location:** `engine/input/input_manager.py`
- **Description:** The `_mouse_just_released` state is never cleared between frames. Once a mouse button is released, `is_mouse_just_released()` returns `True` indefinitely instead of only for one frame.
- **Discovered by:** Whitebox testing (`test_input_manager_whitebox.py`)
- **Failing test:** The whitebox test passes — it asserts the buggy behavior, confirming the bug exists.

### Bug 2: Gamepad Just-Pressed Logic Inverted

- **Location:** `engine/input/input_manager.py`
- **Description:** The `is_gamepad_just_pressed()` method checks `self._gamepad_buttons` (currently held) instead of `self._gamepad_just_pressed` (newly pressed this frame), making it behave identically to `is_gamepad_pressed()`.
- **Discovered by:** Whitebox testing (`test_input_manager_whitebox.py`)
- **Failing test:** The whitebox test passes — it asserts the buggy behavior, confirming the bug exists.

### Bug 3: Sprite Animation Frame Timing

- **Location:** `engine/graphics/sprite.py` (SpriteAnimation.update)
- **Description:** The animation frame advancement logic miscalculates elapsed time against `frame_duration`, causing looping animations to land on the wrong frame after wrapping.
- **Discovered by:** Blackbox testing, confirmed by combinatorial testing
- **Failing tests:**
  - `tests/test_graphics/test_sprite.py::test_animation_looping_behavior`
  - `tests/test_graphics/test_sprite_combinatorial.py::test_sprite_animation_update_pwc[True-True-True-True]`

### Bug 4: Triangle Containment Point Check

- **Location:** `engine/graphics/sprite.py` (Sprite.contains_point)
- **Description:** The `contains_point()` method returns incorrect results for triangle-shaped sprites. Points that should be inside/outside the triangle are misclassified.
- **Discovered by:** Combinatorial testing
- **Failing tests:**
  - `tests/test_graphics/test_sprite_combinatorial.py::test_sprite_contains_point_pwc[triangle-0.5]`
  - `tests/test_graphics/test_sprite_combinatorial.py::test_sprite_contains_point_pwc[triangle-1.5]`

---

## Summary of Results

| Metric | Value |
|--------|-------|
| Total tests | 1,354 |
| Tests passing | 1,350 |
| Tests failing (real bugs) | 4 |
| Branch coverage (SUT modules) | 96% |
| Statement coverage (SUT modules) | 99% |
| Mutants generated | 697 |
| Mutants killed | 579 |
| Mutants survived | 118 |
| Mutation score | 83% |
| Real bugs found | 4 |
| Testing methodologies used | 5 (blackbox, whitebox, mutation, combinatorial, integration) |
