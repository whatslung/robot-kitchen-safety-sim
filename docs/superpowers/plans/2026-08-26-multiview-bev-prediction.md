# Multiview BEV Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse person detections from four quadrant cameras plus one central camera into stable global BEV tracks and return multimodal future trajectories without changing robot avoidance control.

**Architecture:** `backend/multiview.py` owns camera calibration, image-footpoint projection, online constant-velocity Kalman state, and cross-camera global IDs. `backend/detect_server.py` keeps per-camera ByteTrack, adds calibration/reset/health contracts, enriches `/detect` with global tracks, and batches the existing LSTM `/predict` endpoint. `sim.html` adds the five approved cameras, schedules at most one HTTP inference at a time, consumes only server-provided BEV coordinates in multiview mode, and renders predicted paths; the existing single-camera modes remain intact.

**Tech Stack:** Python 3.11+, NumPy, SciPy, FastAPI, ByteTrack, Babylon.js, existing PyTorch LSTM, pytest, in-app browser.

---

### Task 1: Homography calibration and projection

**Files:**
- Create: `backend/multiview.py`
- Create: `tests/test_multiview.py`

- [ ] **Step 1: Write failing calibration tests**

```python
import numpy as np
import pytest

from backend.multiview import CalibrationError, CameraCalibration


def test_homography_restores_unseen_floor_point():
    cal = CameraCalibration.from_points(
        image=[[0.1, 0.2], [0.9, 0.2], [0.8, 0.9], [0.2, 0.9]],
        world=[[-4, -3], [4, -3], [3, 5], [-3, 5]],
        valid_world_polygon=[[-4, -3], [4, -3], [3, 5], [-3, 5]],
    )
    assert np.allclose(cal.project((0.5, 0.55)), (0.0, 1.0), atol=1e-6)


def test_calibration_rejects_collinear_points():
    with pytest.raises(CalibrationError):
        CameraCalibration.from_points(
            image=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
            world=[[0, 0], [1, 1], [2, 2], [3, 3]],
            valid_world_polygon=[[-1, -1], [4, -1], [4, 4], [-1, 4]],
        )
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `uv run python -m pytest tests/test_multiview.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.multiview'`.

- [ ] **Step 3: Implement normalized DLT and polygon validation**

Create `CalibrationError`, `_normalise_points`, `_solve_homography`, `_inside_polygon`, and immutable `CameraCalibration`. `from_points()` must require at least four non-collinear pairs, reject a singular matrix, store image-to-world `H`, and calculate RMS reprojection error. `project()` must divide homogeneous coordinates, reject a near-zero denominator, and return `None` outside `valid_world_polygon`.

```python
@dataclass(frozen=True)
class CameraCalibration:
    matrix: np.ndarray
    valid_world_polygon: tuple
    reprojection_rms: float

    def project(self, image_xy):
        p = self.matrix @ np.array([image_xy[0], image_xy[1], 1.0])
        if abs(p[2]) < 1e-10:
            raise CalibrationError("homography denominator is zero")
        world = (float(p[0] / p[2]), float(p[1] / p[2]))
        return world if _inside_polygon(world, self.valid_world_polygon) else None
```

- [ ] **Step 4: Run projection tests**

Run: `uv run python -m pytest tests/test_multiview.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add backend/multiview.py tests/test_multiview.py
git commit -m "feat: add multiview homography calibration"
```

### Task 2: Global BEV tracker

**Files:**
- Modify: `backend/multiview.py`
- Modify: `tests/test_multiview.py`

- [ ] **Step 1: Write failing global-ID tests**

Add tests that calibrate two identity cameras and prove:

```python
def test_two_cameras_merge_same_person_into_one_global_id():
    fusion = calibrated_fusion("mvNW", "mvCenter")
    a = fusion.update("mvNW", [{"label":"person", "id":2, "conf":.9,
                                "cx":.4, "cy":.3, "w":.1, "h":.2}], 1000)
    b = fusion.update("mvCenter", [{"label":"person", "id":7, "conf":.8,
                                    "cx":.41, "cy":.3, "w":.1, "h":.2}], 1080)
    assert a[0]["global_id"] == b[0]["global_id"]
    assert len(fusion.snapshot(1080)) == 1


Use separate assertions for: identical local numeric IDs in different cameras, measurements farther than `gate`, an older timestamp arriving after a newer one, and reset preserving `fusion.calibrations` while emptying `fusion.tracks`.
```

- [ ] **Step 2: Run the new tests and verify missing APIs**

Run: `uv run python -m pytest tests/test_multiview.py -q`

Expected: FAIL because `MultiViewFusion` and `update()` do not exist.

- [ ] **Step 3: Implement measurements and online CV Kalman state**

Add `Measurement`, `GlobalTrack`, and `MultiViewFusion`. A detection uses the bottom-centre footpoint `(cx, cy + h/2)`. Local keys are `(camera_id, local_id)`, never a bare integer. Predict each state to the measurement timestamp, gate by BEV distance, prefer an existing local-key binding when it remains inside the gate, and otherwise use Hungarian assignment with a deterministic greedy fallback. Store history for at least 4 seconds.

```python
DEFAULT_FUSION_CONFIG = {
    "gate": 0.8,
    "fusion_window_ms": 250,
    "coast_ms": 750,
    "remove_ms": 1500,
}
```

Each enriched box must include `global_id` and `world: {x,z}`. Each snapshot item must include `id`, `x`, `z`, `vx`, `vz`, `age_ms`, `stale`, `sources`, and chronological `history`.

- [ ] **Step 4: Run all multiview tests**

Run: `uv run python -m pytest tests/test_multiview.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add backend/multiview.py tests/test_multiview.py
git commit -m "feat: fuse local tracks into global BEV identities"
```

### Task 3: FastAPI calibration, detection, health, and reset contracts

**Files:**
- Modify: `backend/detect_server.py`
- Create: `tests/test_detect_server_multiview.py`

- [ ] **Step 1: Write failing API tests**

Set `DETECT_DISABLE_MODEL=1` before importing the server, replace `run_detect` and `track_and_measure` with deterministic fakes, and use `fastapi.testclient.TestClient`.

The four tests must assert exact HTTP statuses and response keys: calibrated `/detect` has `world`, `global_id`, and one `global_tracks` item; uncalibrated `/detect` omits global fields; reset returns `{"ok": true}` and keeps the camera in `/health`; health reports `global_track_count` as an integer.

- [ ] **Step 2: Run the endpoint tests and verify 404 responses**

Run: `uv run python -m pytest tests/test_detect_server_multiview.py -q`

Expected: FAIL because `/calibrate` and `/tracks/reset` are absent.

- [ ] **Step 3: Wire the shared fusion service**

Add `FUSION = MultiViewFusion(gate=0.8, fusion_window_ms=250, coast_ms=750, remove_ms=1500)`, honor `DETECT_DISABLE_MODEL=1` before model resolution, and add the two explicit routes:

```python
@app.post("/tracks/reset")
async def tracks_reset():
    FUSION.reset_tracks()
    CAMS.clear()
    return {"ok": True, "calibrated_cameras": sorted(FUSION.calibrations)}
```

The `/calibrate` route parses `camera`, `points[].image`, `points[].world`, and `valid_world_polygon`, constructs `CameraCalibration.from_points`, stores it through `FUSION.calibrate`, and returns the camera ID plus reprojection RMS.

`/detect` must preserve old fields, echo `seq`, call `FUSION.update()` only after local tracking, and return `global_tracks`. `/health` must add `calibrated_cameras`, per-camera update ages, and `global_track_count`. Malformed calibration must return HTTP 422 with a stable `error` string.

- [ ] **Step 4: Run API and legacy trajectory tests**

Run: `uv run python -m pytest tests/test_detect_server_multiview.py tests/test_kalman.py tests/test_learned_predictor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add backend/detect_server.py tests/test_detect_server_multiview.py
git commit -m "feat: expose multiview fusion APIs"
```

### Task 4: Batch future prediction and risk-entry output

**Files:**
- Modify: `backend/detect_server.py`
- Create: `backend/prediction_contract.py`
- Create: `tests/test_prediction_contract.py`
- Modify: `tests/test_detect_server_multiview.py`

- [ ] **Step 1: Write failing prediction-contract tests**

```python
def test_risk_entry_returns_first_time_per_radius():
    mode = {"prob": 1.0, "path": [[0.4, 3.5, 0.0], [0.8, 2.8, 0.0], [1.2, 1.9, 0.0]]}
    risk = risk_entry([mode], center=(0, 0), stop_radius=2.0, slow_radius=3.1)
    assert risk == {"stop_entry_s": 1.2, "slow_entry_s": 0.8}


The second test supplies two IDs and asserts output order `[2, 9]`, three descending-probability modes per ID, and `[time,x,z]` path rows. The third supplies two observations, asserts `source == "kalman"`, and asserts every numeric response value is finite.
```

- [ ] **Step 2: Run and verify missing contract module**

Run: `uv run python -m pytest tests/test_prediction_contract.py -q`

Expected: FAIL with missing module/API.

- [ ] **Step 3: Implement pure prediction formatting helpers**

`prediction_contract.py` must resample timestamped history to 8 points at 0.4 seconds, format LSTM modes as `[t,x,z]`, calculate first entry into nominal radii for every accepted mode, and provide a constant-velocity/Kalman fallback when history is short or model inference fails. The response must carry `source: "lstm"|"kalman"` and never emit NaN/Infinity.

- [ ] **Step 4: Extend `/predict` without breaking the single-history request**

Keep the legacy request `{"hist": [[0.0,0.0],[0.2,0.0],[0.4,0.0],[0.6,0.0],[0.8,0.0],[1.0,0.0],[1.2,0.0],[1.4,0.0]]}` unchanged. Add `{"tracks":[{"id":2,"history":[{"t":1000,"x":0.0,"z":0.0}],"age_ms":20,"stale":false}], "robot":{"x":-1.1,"z":0.795,"stop_radius":3.1,"slow_radius":3.9}}`. Call `LearnedPredictor.predict_batch()` once for all eligible histories. Return one result per input global ID, including stale tracks with no active prediction.

- [ ] **Step 5: Run prediction and endpoint tests**

Run: `uv run python -m pytest tests/test_prediction_contract.py tests/test_detect_server_multiview.py tests/test_learned_predictor.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```text
git add backend/prediction_contract.py backend/detect_server.py tests/test_prediction_contract.py tests/test_detect_server_multiview.py
git commit -m "feat: return global multimodal future predictions"
```

### Task 5: SIM 4+1 cameras and single-flight scheduler

**Files:**
- Modify: `sim.html`
- Create: `tests/test_sim_multiview_contract.py`

- [ ] **Step 1: Write failing static contract tests**

Read `sim.html` and assert that the five IDs, approved pose values, `MV_SCHED`, calibration bootstrap, and global response handler are present. Assert the multiview handler body does not contain `person.node.position`, `EXTRAS`, or `milAssocTracks`.

- [ ] **Step 2: Run and verify missing camera IDs**

Run: `uv run python -m pytest tests/test_sim_multiview_contract.py -q`

Expected: FAIL because `mvNW`…`mvCenter` are absent.

- [ ] **Step 3: Add camera definitions without changing legacy cameras**

Append the five definitions before `CAM_FACTORY` is captured. Use the approved positions, targets, and vertical FOVs from the design. Assign one shared housing layer bit because all five are dataset/sensor views.

- [ ] **Step 4: Add calibration bootstrap**

Choose at least six fixed walkable floor anchors. Project anchors with each Babylon camera to normalized image coordinates and POST image/world pairs plus the walkable valid polygon to `/calibrate`. Calibration uses fixed floor anchors only; it must never use person nodes.

- [ ] **Step 5: Add a weighted single-flight multiview scheduler**

Keep the legacy `milCapture()` untouched. Add `MV_SCHED` with target weights `4,4,4,4,6`, one in-flight request maximum, monotonic `seq`, stale response rejection, and frame dropping instead of request queueing. The response handler replaces `MIL.tracks` from `global_tracks` and stores timestamped histories by global ID.

- [ ] **Step 6: Connect batch prediction display only**

At a bounded rate, POST fresh global histories to `/predict`, store returned modes by global ID, and draw BEV future paths and uncertainty. Do not call `avoidDecide`, mutate `SAFE.factor`, change robot joints, or alter the avoidance PR contract.

- [ ] **Step 7: Run static and Python regression tests**

Run: `uv run python -m pytest tests/test_sim_multiview_contract.py tests -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```text
git add sim.html tests/test_sim_multiview_contract.py
git commit -m "feat: schedule 4 plus 1 BEV prediction views"
```

### Task 6: Browser integration verification

**Files:**
- Modify if required by observed failures: `sim.html`, `backend/detect_server.py`, `backend/multiview.py`

- [ ] **Step 1: Start the local FastAPI server in model-disabled mode**

Run: `DETECT_DISABLE_MODEL=1 uv run python backend/detect_server.py --port 8001` (PowerShell equivalent: `$env:DETECT_DISABLE_MODEL='1'; uv run python backend/detect_server.py --port 8001`).

Expected: server starts in dummy mode without downloading weights.

- [ ] **Step 2: Open the SIM with the in-app browser**

Open `http://127.0.0.1:8001/sim.html?person=1&scenario=prepWash`, wait for readiness, and enable multiview HTTP validation.

- [ ] **Step 3: Verify live contracts**

Confirm all five cameras calibrate, only one request is in flight, `global_tracks` uses stable IDs as the same worker crosses a view seam, and each fresh track shows K=3 LSTM paths or a labeled Kalman fallback. Change the visible main camera and confirm the background scheduler continues.

- [ ] **Step 4: Verify degradation behavior**

Disable one camera and confirm the global track coasts then becomes stale; reset the scene and confirm `/tracks/reset` clears IDs but calibration remains. Stop the prediction model and confirm the UI labels Kalman fallback without changing robot motion.

- [ ] **Step 5: Run final verification**

Run: `uv run python -m pytest tests -q` and `git diff --check`.

Expected: all tests pass and no whitespace errors.

- [ ] **Step 6: Commit any integration fixes**

```text
git add backend sim.html tests
git commit -m "fix: harden multiview prediction integration"
```
