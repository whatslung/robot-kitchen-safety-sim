# Collision, Braking, and Avoidance Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PR #24's presentation-only corridor drama with deterministic simulator-path safety control, smooth braking, robust swept contact, and hold/retract/safe-lift fallback behavior.

**Architecture:** Put Babylon-independent geometry, planned-path sampling, speed limiting, and maneuver arbitration in `safety_motion.js`, exposed as both `window.SafetyMotion` and `module.exports`. Keep scene adaptation and robot actuation in `sim.html`; test the pure module with Node's built-in test runner and verify integration in the browser.

**Tech Stack:** JavaScript (UMD/CommonJS), Node `node:test`, Babylon.js, existing Python/pytest regression suite.

---

### Task 1: Simulator planned-path sampling

**Files:**
- Create: `safety_motion.js`
- Create: `tests/js/safety_motion.test.cjs`

- [ ] **Step 1: Write failing planned-path tests**

```js
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../safety_motion.js');

test('samplePlannedPath follows remaining waypoints at walking speed', () => {
  const out = S.samplePlannedPath({x:0,y:0,z:0}, [{x:1,y:0,z:0},{x:1,y:0,z:1}], 0, 1, [0, 0.5, 1, 1.5, 2]);
  assert.deepEqual(out.map(p => [p.x,p.z]), [[0,0],[0.5,0],[1,0],[1,0.5],[1,1]]);
});

test('samplePlannedPath treats invalid input as stationary', () => {
  const out = S.samplePlannedPath({x:2,y:0,z:3}, null, 0, NaN, [0, 1]);
  assert.deepEqual(out.map(p => [p.x,p.z]), [[2,3],[2,3]]);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test tests/js/safety_motion.test.cjs`

Expected: FAIL because `safety_motion.js` or `samplePlannedPath` does not exist.

- [ ] **Step 3: Implement minimal path sampler**

Create a UMD module exporting `samplePlannedPath(current, path, pi, speed, times)`. It must validate finite coordinates and positive speed, consume distance along `current -> path[pi] -> ...`, and hold the last point after the route ends.

- [ ] **Step 4: Run GREEN and commit**

Run: `node --test tests/js/safety_motion.test.cjs`

Expected: 2 passing, 0 failing.

Commit: `feat: sample simulator planned paths for safety`

### Task 2: Three-dimensional swept contact

**Files:**
- Modify: `safety_motion.js`
- Modify: `tests/js/safety_motion.test.cjs`

- [ ] **Step 1: Write failing geometry tests**

Add tests that call:

```js
S.sweptSegmentCapsuleContact({
  previous:{a:{x:-1,y:1,z:0},b:{x:0,y:1,z:0}},
  current:{a:{x:1,y:1,z:0},b:{x:2,y:1,z:0}},
  linkRadius:0.1,
  person:{center:{x:0.5,y:0.9,z:0},radius:0.25,halfHeight:0.85},
  maxStep:0.04,
});
```

Verify contact during the swept interval, no contact when the person capsule is above the arm, and identical results for symmetric approach angles around the same swept path.

- [ ] **Step 2: Run RED**

Run: `node --test tests/js/safety_motion.test.cjs`

Expected: FAIL because `sweptSegmentCapsuleContact` is missing.

- [ ] **Step 3: Implement segment distance and swept sampling**

Implement finite-input validation, closest distance between a 3D segment and the vertical center segment of the person capsule, and temporal interpolation. Choose sample count as `ceil(max endpoint movement / maxStep)` with both endpoints included. Return `{hit, clearance, t}` and return a conservative hit for malformed contact input.

- [ ] **Step 4: Run GREEN and commit**

Run: `node --test tests/js/safety_motion.test.cjs`

Expected: all tests pass.

Commit: `feat: detect swept robot person contact`

### Task 3: Smooth speed governor and maneuver arbitration

**Files:**
- Modify: `safety_motion.js`
- Modify: `tests/js/safety_motion.test.cjs`

- [ ] **Step 1: Write failing speed-governor tests**

Test `approachFactor(current, target, dtSec, decelRate, accelRate, immediateStop)` for a 0.25 decrease in 0.1s at rate 2.5, a 0.08 increase at rate 0.8, and immediate zero on a stop command.

- [ ] **Step 2: Run RED, implement, and run GREEN**

Run: `node --test tests/js/safety_motion.test.cjs`

Implement clamped `[0,1]` factor changes with invalid inputs returning zero. Re-run and expect all tests pass.

- [ ] **Step 3: Write failing arbitration tests**

Use `chooseManeuver({danger, beforeCross, holdMs, minHoldMs, proceed, retract, safeLift})` where candidates are `{safe, clearance}`. Verify `PROCEED`, `HOLD`, `RETRACT`, `SAFE_LIFT`, and `STOP`, including a 300ms hold that prevents a downgrade.

- [ ] **Step 4: Run RED, implement, and run GREEN**

Run: `node --test tests/js/safety_motion.test.cjs`

Implement the exact priority in the design and return `{mode, reason, clearance}`. Re-run and expect all tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: choose smooth safety maneuvers`

### Task 4: Integrate simulator paths and robot maneuvers

**Files:**
- Modify: `sim.html` near script imports, `SAFE`, `AVOID`, `avoidDecide`, and `stepUpdate`
- Create: `tests/test_sim_safety_wiring.py`

- [ ] **Step 1: Write failing static wiring tests**

```python
from pathlib import Path

SIM = Path(__file__).parents[1] / "sim.html"

def test_sim_loads_safety_motion_before_inline_controller():
    text = SIM.read_text(encoding="utf-8")
    assert '<script src="./safety_motion.js"></script>' in text
    assert text.index('safety_motion.js') < text.index('const SAFE =')

def test_default_avoidance_uses_simulator_planned_paths():
    text = SIM.read_text(encoding="utf-8")
    assert 'SafetyMotion.samplePlannedPath' in text
    assert 'plannedPeopleOccupancy' in text
```

- [ ] **Step 2: Run RED**

Run: `uv run --with pytest python -m pytest tests/test_sim_safety_wiring.py -q --basetemp .venv/pytest-tmp`

Expected: FAIL because the script and adapters are not wired.

- [ ] **Step 3: Add the browser adapter**

Load `safety_motion.js` before the inline script. Add `plannedPeopleOccupancy(times)` that adapts the main person and every extra using current position, remaining `path`, `pi`, speed, and movement state. Use velocity extrapolation only for manual movement without a route, and hold invalid/stationary people at their current position.

- [ ] **Step 4: Add candidate scoring and state machine**

Extend `AVOID` with `mode`, `since`, `targetFactor`, `appliedFactor`, and safe-lift state. Score the original forward segment, reversed segment, and a 12-sample interpolation to the existing reachable pot-hover pose against planned person occupancy. Apply `chooseManeuver` and use `approachFactor` before advancing `state.seqT`. `SAFE.blind` disables avoidance but not contact monitoring.

- [ ] **Step 5: Run GREEN and existing Python tests**

Run: `uv run --with pytest python -m pytest tests/test_sim_safety_wiring.py -q --basetemp .venv/pytest-tmp`

Run: `uv run --with pytest python -m pytest -q --basetemp .venv/pytest-tmp`

Expected: all tests pass.

- [ ] **Step 6: Commit**

Commit: `feat: drive avoidance from simulator paths`

### Task 5: Unify main and extra-person contact handling

**Files:**
- Modify: `sim.html` near `armContactUpdate`, `physWatch`, and run-tab routing
- Modify: `tests/test_sim_safety_wiring.py`

- [ ] **Step 1: Write failing integration-shape tests**

Assert the source contains `safetyPeople`, previous link snapshots, `SafetyMotion.sweptSegmentCapsuleContact`, and routes `로봇 접촉 사고` into the run tab. Assert contact iteration is not limited to `EXTRAS`.

- [ ] **Step 2: Run RED**

Run: `uv run --with pytest python -m pytest tests/test_sim_safety_wiring.py -q --basetemp .venv/pytest-tmp`

Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement unified contact adapter**

Add `safetyPeople()` returning main and extra person capsule adapters with per-target contact handlers. Snapshot all arm link endpoints after each frame, run swept contact for every person, trigger one E-STOP, and delegate to existing main/extra fall behavior. Preserve collision monitoring during `SAFE.blind`.

- [ ] **Step 4: Repair run-tab routing and status text**

Include `로봇 접촉 사고` in the run tab section list. Render applied speed and the selected mode (`SLOW`, `HOLD`, `RETRACT`, `SAFE_LIFT`, `STOP`) so an idle robot is not labeled as slowing.

- [ ] **Step 5: Run GREEN and commit**

Run the focused and full pytest commands from Task 4, then `node --test tests/js/safety_motion.test.cjs`.

Commit: `feat: unify collision handling and safety status`

### Task 6: Browser scenarios and final verification

**Files:**
- Create: `tests/browser/simulator-path-safety.spec.mjs`
- Modify: `README.md` only if the user-facing controls or run command need documentation

- [ ] **Step 1: Add browser regression scenarios**

Cover deterministic simulator path input without YOLO, smooth multi-frame slowdown, HOLD before crossing, RETRACT while crossing, SAFE_LIFT when reverse is blocked, STOP when every path is blocked, and E-STOP contact for both the main and an extra person.

- [ ] **Step 2: Run browser verification**

Serve the worktree with `uv run python -m http.server 5199 --bind 127.0.0.1` and execute the scenarios through the local browser. Record the actual mode sequence and contact status from `window.__sim`.

- [ ] **Step 3: Run full verification**

Run:

```powershell
node --test tests/js/safety_motion.test.cjs
uv run --with pytest python -m pytest -q --basetemp .venv/pytest-tmp
git diff --check origin/main...HEAD
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 4: Commit final tests/docs**

Commit: `test: cover simulator path safety scenarios`

- [ ] **Step 5: Request code review**

Review `origin/main...HEAD` against the design document. Fix every Critical and Important finding, rerun the full verification commands, and report remaining Minor findings explicitly.
