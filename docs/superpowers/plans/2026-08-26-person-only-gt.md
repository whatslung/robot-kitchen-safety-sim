# Person-only Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate only per-person YOLO ground-truth labels without multi-class palette leakage.

**Architecture:** Reuse the existing RGB and GT render pipeline, but reduce its semantic contract to one class and one object family. The instance pass retains unique person colors and uses exact-color decoding so antialiased mixtures cannot create remote bbox pixels. The returned class mask is rebuilt as a binary person/background image from those exact instance pixels.

**Tech Stack:** Babylon.js in `sim.html`, Playwright browser regression tests, pytest project regression suite.

---

### Task 1: Add failing person-only GT regressions

**Files:**
- Create: `tests/browser/person-only-gt.spec.mjs`
- Create: `tests/test_person_only_class_contract.py`

- [x] Add a test that loads the simulator, captures `cvN` ground truth, and requires `GT_CLASSES`, `labels`, and `instances` to contain only `person` with class id `0`.
- [x] Add a test that feeds two palette colors and one blended color to the exported exact decoder and requires the blended pixel to return `-1`.
- [x] Add a test that requires both YOLO YAML generators to expose only `person`.
- [x] Run the new browser spec and confirm it fails because six classes are exposed and the exact decoder does not exist.

### Task 2: Implement the minimal person-only mask contract

**Files:**
- Modify: `sim.html:7151-7165`
- Modify: `sim.html:7225-7235`
- Modify: `sim.html:7307-7355`
- Modify: `sim.html:7437-7654`
- Modify: `sim.html:11517-11521`
- Modify: `train/prepare_yolo_split.py:23-25`
- Modify: `train/prep_3way.py:1-22`

- [x] Reduce `GT_CLASSES` to `person` id `0`.
- [x] Make `instanceOfMesh` return an instance key only when `classOfMesh(mesh) === "person"`.
- [x] Add `classifyByExactColor` and use it for instance bbox extraction.
- [x] Disable particles during semantic and instance mask passes and remove fire/smoke mask injection from the person-only path.
- [x] Export the exact decoder for the browser regression test.
- [x] Change both training YAML class declarations to `["person"]`.
- [x] Run the new browser spec and confirm it passes.

### Task 3: Verify repository and rendered output

**Files:**
- Modify only if verification exposes a defect: `sim.html`, `tests/browser/person-only-gt.spec.mjs`

- [x] Run the full pytest suite with a worktree-local temporary directory.
- [x] Capture actual GT from `cvN` and `cvS`; assert every label is `person`, class id is `0`, and no non-person instance exists.
- [x] Draw the returned boxes over the captured RGB images and inspect alignment.
- [x] Review `git diff --check` and the final worktree status.
