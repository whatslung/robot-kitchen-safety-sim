# LSTM Active-Yield Demo Implementation Plan

> **Execution:** Apply the `executing-plans` and `test-driven-development` skills task by task. Every behavior change starts with a failing focused test, then the smallest implementation, then focused and regression verification.

**Goal:** Add a deterministic demo in which the robot starts immediately, an actual LSTM prediction drives an active `RETRACT`/`SAFE_LIFT` maneuver for a fixed-route person, the held basket participates in safety geometry, and work resumes after 300 ms of clear space.

**Architecture:** Keep the normal automatic job on simulator-known paths. Add an opt-in demo state that gates the person at a safe approach point until a fresh `/predict` result exists, converts LSTM modes into conservative timed occupancy, and feeds only that occupancy to the existing candidate scorer. Extend the shared arm geometry with per-segment radii so the same robot-plus-basket envelope is used for candidate clearance and swept contact.

**Tech stack:** Browser JavaScript/Babylon.js in `sim.html`, CommonJS safety helpers in `safety_motion.js`, Node's built-in test runner, pytest wiring tests, Playwright browser tests, Python prediction server.

---

## Task 1: Active-yield policy and learned-prediction freshness

**Files:**
- Modify: `tests/js/safety_motion.test.cjs`
- Modify: `safety_motion.js`

1. Add failing tests proving that `chooseManeuver` selects `SAFE_LIFT` before the crossing when lift is safe, still prioritizes `RETRACT` after progress, and never overrides an explicit emergency stop. Add tests for a pure freshness helper at 999 ms and 1000 ms.
2. Run `node --test tests/js/safety_motion.test.cjs` and confirm the new assertions fail for the old early `HOLD` policy or missing helper.
3. Implement the smallest policy change: accept `emergencyStop`, return `STOP` first, prefer safe `RETRACT` after progress, otherwise prefer safe `SAFE_LIFT`, and use `HOLD` only when no verified moving candidate exists. Export `predictionFresh(record, nowMs, maxAgeMs = 1000)` with finite timestamp validation.
4. Re-run the focused Node suite and commit with `feat: enable active yield from fresh predictions`.

## Task 2: Robot-plus-basket safety envelope

**Files:**
- Modify: `tests/js/safety_motion.test.cjs`
- Modify: `safety_motion.js`
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `sim.html`

1. Add a failing unit test where a person clears the arm links but intersects a larger payload segment, and assert `armTrajectoryClearance` uses `segment.radius` instead of the default link radius. Add wiring assertions for `basketPayloadSegments`, `state.basketHeld`, and shared use by candidate and actual-contact paths.
2. Run the focused Node and pytest files and confirm failure.
3. Update clearance and swept-contact calls to resolve radius as `segment.radius ?? linkRadius`. Add `basketPayloadSegments()` that derives a conservative cross-shaped capsule envelope from the basket world bounds, returning invalid geometry fail-closed. Append those segments in `armLinksSnapshot()` only while held.
4. Re-run focused tests and commit with `feat: include held basket in safety envelope`.

## Task 3: LSTM-only occupancy source for the dedicated demo

**Files:**
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `sim.html`

1. Add failing wiring tests requiring a distinct `learnedPeopleOccupancy` function, a demo-only source selection in `avoidDecide`, fresh `MPRED.pred` records, and no call from that function to `plannedPeopleOccupancy` or `person.path`.
2. Run `uv run pytest tests/test_sim_safety_wiring.py -q` and confirm failure.
3. Implement `learnedPeopleOccupancy(times, now)` using the fresh `gt:0` LSTM modes and their uncertainty. Sample every candidate time with existing `modePosAt`/`modeSigAt`, inflate the person radius conservatively, and return an explicit unavailable result when no fresh record exists. Make `avoidDecide` select it only when the demo sets `predictionSource: "lstm"`; keep the existing planned source for ordinary auto work.
4. Surface prediction source and stale/unavailable status through demo state, expose the state in `window.__sim`, re-run focused tests, and commit with `feat: drive demo avoidance from LSTM occupancy`.

## Task 4: Deterministic robot-first fixed-route scenario and UI

**Files:**
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `sim.html`

1. Add failing wiring tests for the `고정 동선 LSTM 회피` button, an immediate `startAutoWork()` call, a 400 ms person delay, eight observations before crossing release, dedicated cleanup, and rack restoration.
2. Run the focused pytest file and confirm failure.
3. Extract the existing auto-button body into idempotent `startAutoWork()` and call it synchronously from the demo start. Add `LSTM_YIELD_DEMO` state and a deterministic route whose approach portion accumulates history outside the hazard zone, then crosses the robot transfer path once after a fresh prediction.
4. Add preflight route validation. If rack obstacles alone make the route invalid, record and disable only those rack meshes/colliders for this demo, rebuild navigation, and restore them on reset, cancellation, failure, and completion. Do not remove non-rack equipment.
5. Add the button and concise phases: robot lead, observation count, LSTM crossing prediction, active retreat/lift, route clear/resume, completion/failure. Preserve WASD as the secondary live-input route through the same multi-person prediction stream.
6. Re-run focused tests and the full Node/pytest suites, then commit with `feat: add robot-first LSTM yield scenario`.

## Task 5: End-to-end deterministic browser contract

**Files:**
- Add: `tests/browser/lstm-active-yield.spec.mjs`
- Modify: `README.md`

1. Add a Playwright test that launches the page through `backend/detect_server.py`, intercepts `/predict` with deterministic LSTM-shaped modes while retaining the real application request path, starts the dedicated demo, and samples exposed telemetry.
2. Assert: joint motion within 200 ms, person start at 400 ms, eight observations before LSTM control, at least one `RETRACT` or `SAFE_LIFT`, no non-emergency stationary interval above 800 ms, resume after 300 ms clear, no robot/basket contact, basket delivered, home reached, and no console errors.
3. Run the new browser test and confirm it fails before any missing telemetry/behavior is added. Implement only the telemetry and timing corrections required by the test, then re-run it to green.
4. Document the required LSTM server command and distinguish it from a static server. Commit with `test: cover deterministic LSTM active-yield demo`.

## Task 6: Verification, review, and branch handoff

1. Run `node --test tests/js/*.test.cjs`.
2. Run `uv run pytest -q`.
3. Start `uv run python backend/detect_server.py --port 8001`, open `http://127.0.0.1:8001/sim.html?person=1`, and run the focused Playwright test or equivalent browser verification against the live page.
4. Inspect `git diff origin/main...HEAD`, verify no silent LSTM fallback and no unrelated changes, and perform the requested code review with severity-ranked findings. Fix any confirmed findings with failing regression tests first.
5. Apply `verification-before-completion`, then `finishing-a-development-branch`: push the branch, update PR #32, and report exact verification evidence and any remaining environmental limitation.
