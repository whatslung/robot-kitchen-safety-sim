# Radius-Driven Active Yield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the LSTM yield demo visibly move the robot through a verified escape maneuver while a person enters and leaves the stop radius, without weakening emergency-stop behavior.

**Architecture:** Keep distance-based SSM as the safety authority, but separate an unconditional emergency stop from a verified escape-motion exception. The avoidance planner must mark only collision-checked `RETRACT` and `SAFE_LIFT` candidates as verified; the motion loop may then apply their limited speed factors inside the stop radius. Demo telemetry and browser tests prove that the person crosses the radius and robot joints move during that interval.

**Tech Stack:** Vanilla JavaScript simulation, Python wiring tests, Node unit tests, Playwright browser tests.

---

### Task 1: Lock the motion-stop policy with unit tests

**Files:**
- Modify: `tests/js/safety_motion.test.cjs`
- Modify: `safety_motion.js`

1. Add failing cases for verified `RETRACT`/`SAFE_LIFT`, unverified escape, `PROCEED`, zero target speed, and emergency stop.
2. Run the focused Node test and confirm the new helper is missing.
3. Implement `motionStopRequired` as a pure helper and export it.
4. Re-run the focused test and confirm it passes.

### Task 2: Wire verified escape motion into the simulator

**Files:**
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `sim.html`

1. Replace the old assertion that `SAFE.stopped` always forces zero speed with assertions for the new policy and verification state.
2. Run the focused Python test and confirm it fails.
3. Track `AVOID.escapeVerified` from the selected collision-checked candidate.
4. Use `motionStopRequired` in the frame update so only verified escape modes bypass the ordinary SSM stop; physical emergency stop, unsafe candidates, and zero-speed decisions still stop immediately.
5. Make the status bar show the active escape mode and actual speed instead of coloring every SSM stop as a severe stop.
6. Re-run Node and Python tests.

### Task 3: Drive the person across the safety radius and expose proof telemetry

**Files:**
- Modify: `tests/test_sim_safety_wiring.py`
- Modify: `sim.html`
- Modify: `tests/browser/lstm-active-yield.spec.mjs`

1. Add failing wiring/browser expectations for an outside-slow-radius start, a waypoint clearly inside the stop radius, and joint motion while inside it.
2. Update the fixed LSTM demo route while preserving permanent-obstacle clearance.
3. Record minimum distance, stop-radius entry, escape joint delta, and successful motion-inside-radius in demo state.
4. Assert those signals in Playwright together with the existing completion and stationary-time requirements.

### Task 4: Verify, review, and deliver

**Files:**
- Review all changed implementation and test files.

1. Run focused Node, Python, and Playwright tests.
2. Run the full test suite and both browser scenarios.
3. Review the diff for emergency-stop regressions, stale status masking, and route/fixture conflicts.
4. Commit the implementation, push the feature branch, update PR 32, and confirm the local demo URL remains healthy.
