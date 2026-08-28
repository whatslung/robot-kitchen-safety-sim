# LSTM Demo Route Cart Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove only environment carts that block the fixed LSTM demo route, then restore their complete physics state on every exit path.

**Architecture:** Generalize the existing temporary rack-removal lifecycle into route-blocker removal. Static racks retain enabled-state handling; dynamic environment carts are identified by their `ENV_PROPS` index, parked outside the room with physics synchronization, and restored from a saved snapshot. The robot delivery cart is not part of `ENV_PROPS`/`PUSH.bodies`, so it remains untouched.

**Tech Stack:** Babylon.js/Havok browser simulation, Python wiring tests, Playwright end-to-end tests.

---

### Task 1: Specify cart selection and restoration wiring

**Files:**
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `tests/browser/lstm-active-yield.spec.mjs`

- [ ] **Step 1: Write the failing wiring test**

Assert that the generalized blocker lifecycle names exist, both `env_pancart.glb` and
`env_basketcart.glb` are allowed, push bodies retain their `ENV_PROPS` index, and cart snapshots
include transform, velocities, and `disablePreStep`.

- [ ] **Step 2: Run the focused wiring test and verify RED**

Run: `uv run --with pytest python -m pytest tests/test_sim_safety_wiring.py -q`

Expected: FAIL because the implementation still exposes only `disableBlockingDemoRacks` and
`restoreDemoRacks`.

- [ ] **Step 3: Extend the Playwright test before implementation**

Place an environment push cart across a known approach segment, rebuild colliders/navigation,
call the blocker-removal lifecycle, and assert that the cart is parked during the demo and returns
to its saved position, rotation, velocities, and pre-step state after restore.

### Task 2: Generalize route blocker removal

**Files:**
- Modify: `sim.html`

- [ ] **Step 1: Preserve environment identity on push bodies**

Add `idx:it.idx` to each `PUSH.bodies` entry so the dynamic body maps back to its source
`ENV_PROPS` record without relying on labels or scene names.

- [ ] **Step 2: Replace rack-only state with blocker snapshots**

Rename the demo collection to `blockers` and implement `restoreDemoRouteBlockers()`.
Static snapshots restore holder/collider enabled state. Dynamic snapshots restore box position,
rotation quaternion or Euler rotation, linear/angular velocity, and `disablePreStep`, then request
two-frame transform synchronization through the existing `restore` mechanism.

- [ ] **Step 3: Remove only intersecting supported props**

Implement `disableBlockingDemoProps(route)`. For intersecting `env_rack.glb`, preserve the existing
disable behavior. For intersecting `env_pancart.glb` and `env_basketcart.glb`, snapshot the matching
push body, zero both velocities, move its proxy below and outside `LAYOUT.room`, force its world
matrix, and open pre-step synchronization. Do not inspect or mutate the robot delivery cart.

- [ ] **Step 4: Rebuild derived collision state and retain fail-closed behavior**

After remove or restore, call `buildPersonColliders()` and `buildNavGrid()`. Return false when the
same approach route remains blocked so startup still aborts safely.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
uv run --with pytest python -m pytest tests/test_sim_safety_wiring.py -q
node --test tests/js/*.test.cjs
```

Expected: 13 Python wiring tests and 22 Node tests pass.

### Task 3: Verify and deliver

**Files:**
- Modify: `tests/browser/lstm-active-yield.spec.mjs` if the test needs only deterministic timing adjustments

- [ ] **Step 1: Run the LSTM browser scenario**

Run: `$env:BASE_URL='http://127.0.0.1:8001'; npx playwright test tests/browser/lstm-active-yield.spec.mjs --workers=1 --reporter=line`

Expected: PASS, including temporary environment-cart removal and exact restoration.

- [ ] **Step 2: Run full verification**

Run: `uv run --with pytest python -m pytest tests/ -q --basetemp=.pytest-tmp-cart-final`

Expected: all tests pass with only the repository's documented skips. Then run `git diff --check`.

- [ ] **Step 3: Review, commit, push, and restart the server**

Commit the implementation and tests, push `codex/pr24-collision-avoidance-rebuild` to PR #32,
restart port 8001 from the worktree, and verify `/health` plus the simulator URL return HTTP 200.

