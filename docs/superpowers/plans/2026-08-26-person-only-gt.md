# Person-only Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate only per-person YOLO ground-truth labels without multi-class palette leakage.

**Architecture:** Reuse the existing RGB and GT render pipeline, but reduce its semantic contract to one class and one object family. The instance pass retains unique person colors and uses exact-color decoding so antialiased mixtures cannot create remote bbox pixels.

**Tech Stack:** Babylon.js in `sim.html`, Playwright browser regression tests, pytest project regression suite.

---

### Task 1: Add failing person-only GT regressions

**Files:**
- Create: `tests/browser/person-only-gt.spec.mjs`

- [ ] Add a test that loads the simulator, captures `cvN` ground truth, and requires `GT_CLASSES`, `labels`, and `instances` to contain only `person` with class id `0`.
- [ ] Add a test that feeds two palette colors and one blended color to the exported exact decoder and requires the blended pixel to return `-1`.
- [ ] Run the new browser spec and confirm it fails because six classes are exposed and the exact decoder does not exist.

### Task 2: Implement the minimal person-only mask contract

**Files:**
- Modify: `sim.html:7151-7165`
- Modify: `sim.html:7225-7235`
- Modify: `sim.html:7307-7355`
- Modify: `sim.html:7437-7654`
- Modify: `sim.html:11517-11521`

- [ ] Reduce `GT_CLASSES` to `person` id `0`.
- [ ] Make `instanceOfMesh` return an instance key only when `classOfMesh(mesh) === "person"`.
- [ ] Add `classifyByExactColor` and use it for instance bbox extraction.
- [ ] Disable particles during semantic and instance mask passes and remove fire/smoke mask injection from the person-only path.
- [ ] Export the exact decoder for the browser regression test.
- [ ] Run the new browser spec and confirm it passes.

### Task 3: Verify repository and rendered output

**Files:**
- Modify only if verification exposes a defect: `sim.html`, `tests/browser/person-only-gt.spec.mjs`

- [ ] Run the full pytest suite with a worktree-local temporary directory.
- [ ] Capture actual GT from `cvN` and `cvNE`; assert every label is `person`, class id is `0`, and no non-person instance exists.
- [ ] Draw the returned boxes over the captured RGB images and inspect alignment.
- [ ] Review `git diff --check` and the final worktree status.
