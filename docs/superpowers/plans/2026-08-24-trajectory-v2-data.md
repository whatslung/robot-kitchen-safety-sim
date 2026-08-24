# Trajectory v2 Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 120개 v1을 건드리지 않고 서로 다른 세 레이아웃에서 150개 v2 scene을 수집해, 고정 split과 SHA-256 manifest로 검증한다.

**Architecture:** `/traj` 저장 책임을 작은 Python 모듈로 분리해 허용된 dataset 이름만 받는다. 브라우저 수집기는 `island_h58`, `corridor`, `legacy`를 서로 다른 tag로 전송하고, 별도 감사기가 50 seed × 3 layout 전수 조합과 파일 내용을 검사한다. v2 JSON은 무시되는 `dataset/trajectories_v2/`에 두고 추적 가능한 manifest만 `docs/chanwoo/results/`에 둔다.

**Tech Stack:** Python 3.11+, FastAPI, pytest, 브라우저 JavaScript, SHA-256, PowerShell, uv

**Spec:** `docs/chanwoo/specs/2026-08-24-transformer-autoresearch-design.md`

## Global Constraints

- 작업 경로는 `C:\Users\chanwoo\workspace\robot-kitchen-safety-sim-autoresearch-transformer`뿐이다.
- 기준 브랜치는 `chanwoo/autoresearch-transformer-2026-08-24`; v1 `dataset/trajectories/`는 수정하지 않는다.
- v2 레이아웃은 `island_h58`, `corridor`, `legacy`; seed는 1~50이다.
- split은 train 1~30, val 31~40, locked test 41~50이며 같은 seed의 세 레이아웃은 같은 split이다.
- 모든 scene은 `dt=0.4`, `hz=2.5`, `steps=150`, `room`, `robot`, `mPerAU`를 가져야 한다.
- JSON 원본은 Git에 넣지 않고 `docs/chanwoo/results/traj-v2-manifest.json`만 추적한다.
- test 궤적에서 정답 기반 safety 통계는 이 계획에서 계산하지 않는다.
- 파일 편집은 `apply_patch`, Python 실행은 `uv run --group serve ...`를 사용한다.

---

### Task 1: 안전한 v1/v2 저장 경계

**Files:**
- Create: `trajectory/collection.py`
- Modify: `backend/detect_server.py:380-401`
- Create: `tests/test_traj_collection.py`

**Interfaces:**
- Consumes: `/traj` JSON body의 `scene_id`와 선택적 `dataset`.
- Produces: `resolve_dataset_dir(project_root: Path, dataset: str | None) -> Path`, `save_trajectory_scene(body: dict, project_root: Path) -> dict`.

- [ ] **Step 1: 허용 이름과 기본 v1 저장을 검증하는 실패 테스트를 작성한다**

```python
# tests/test_traj_collection.py
import json
import pytest
from trajectory.collection import resolve_dataset_dir, save_trajectory_scene


def _scene(dataset=None):
    out = {"scene_id": "island_h58_seed1_0000", "schema": 1, "seed": 1, "nodes": []}
    if dataset is not None:
        out["dataset"] = dataset
    return out


def test_default_dataset_stays_v1(tmp_path):
    result = save_trajectory_scene(_scene(), tmp_path)
    assert result["dataset"] == "trajectories"
    assert (tmp_path / "dataset" / "trajectories" / "island_h58_seed1_0000.json").exists()


def test_v2_dataset_uses_separate_directory(tmp_path):
    result = save_trajectory_scene(_scene("trajectories_v2"), tmp_path)
    path = tmp_path / "dataset" / "trajectories_v2" / "island_h58_seed1_0000.json"
    assert result["dataset"] == "trajectories_v2"
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 1


@pytest.mark.parametrize("name", ["../trajectories", "x/y", "trajectories_v3", ""])
def test_unknown_or_path_dataset_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="허용되지 않은 trajectory dataset"):
        resolve_dataset_dir(tmp_path, name)
```

- [ ] **Step 2: 테스트가 import 오류로 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_traj_collection.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory.collection'`.

- [ ] **Step 3: 허용 목록과 저장 함수를 최소 구현한다**

```python
# trajectory/collection.py
from __future__ import annotations
import json
from pathlib import Path

ALLOWED_DATASETS = frozenset({"trajectories", "trajectories_v2"})


def resolve_dataset_dir(project_root: Path, dataset: str | None) -> Path:
    name = "trajectories" if dataset is None else str(dataset)
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"허용되지 않은 trajectory dataset: {name!r}")
    return Path(project_root) / "dataset" / name


def _scene_id(value: object) -> str:
    sid = "".join(c for c in str(value) if c.isalnum() or c in "-_")[:80]
    if not sid:
        raise ValueError("scene_id가 비어 있음")
    return sid


def save_trajectory_scene(body: dict, project_root: Path) -> dict:
    dataset = body.get("dataset")
    root = resolve_dataset_dir(project_root, dataset)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_scene_id(body.get('scene_id', ''))}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"ok": True, "count": len(list(root.glob("*.json"))),
            "dataset": dataset or "trajectories", "dir": str(root), "file": path.name}
```

`backend/detect_server.py`의 `/traj`는 위 함수를 호출하고 `ValueError`만 HTTP 400으로 바꾼다.

```python
from trajectory.collection import save_trajectory_scene

@app.post("/traj")
async def traj(req: Request):
    try:
        return save_trajectory_scene(await req.json(), _ROOT)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(exc)})
```

- [ ] **Step 4: 저장 단위 테스트와 서버 기존 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_traj_collection.py tests/test_detect_server_none.py -q`

Expected: PASS.

- [ ] **Step 5: 저장 경계만 커밋한다**

```powershell
git add -- trajectory/collection.py backend/detect_server.py tests/test_traj_collection.py
git commit -m "feat: isolate trajectory v2 collection storage"
```

### Task 2: 세 레이아웃의 고유 tag와 v2 전달

**Files:**
- Modify: `sim.html:12445-12585`
- Create: `tests/test_sim_traj_collection_contract.py`

**Interfaces:**
- Consumes: `__sim.trajRun({scenes, seed, seconds, jobs, dataset})`와 URL `layout`.
- Produces: POST body의 `dataset`; `trajGeom()`의 `layout/tag` 조합 `island/island_h58`, `corridor/corridor`, `legacy/legacy`.

- [ ] **Step 1: HTML 수집 계약의 실패 테스트를 작성한다**

```python
# tests/test_sim_traj_collection_contract.py
from pathlib import Path

SIM = (Path(__file__).parents[1] / "sim.html").read_text(encoding="utf-8")


def test_corridor_has_distinct_trajectory_layout_and_tag():
    assert 'CORRIDOR ? "corridor"' in SIM
    assert 'lay === "island" ? "island_h"' in SIM


def test_traj_run_accepts_only_v1_or_v2_dataset():
    assert 'new Set(["trajectories","trajectories_v2"])' in SIM
    assert 'dataset: datasetName' in SIM


def test_collection_log_names_target_dataset():
    assert '"· dataset", datasetName' in SIM
```

- [ ] **Step 2: 현재 corridor 표기와 dataset 전달이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_sim_traj_collection_contract.py -q`

Expected: FAIL on all three contract assertions.

- [ ] **Step 3: `trajGeom`, `trajRunScene`, `trajRun`을 수정한다**

```javascript
function trajGeom() {
  const island = !(typeof ISLAND !== "undefined" && !ISLAND);
  const lay = CORRIDOR ? "corridor" : (island ? "island" : "legacy");
  const half = lay === "island" && typeof ISLAND_HALF !== "undefined" && isFinite(ISLAND_HALF)
    ? +ISLAND_HALF.toFixed(2) : null;
  const tag = lay === "island" ? "island_h" + Math.round(half * 10) : lay;
  const R = LAYOUT.room, b = LAYOUT.robot.base;
  return { layout:lay, half, tag,
           room:{x0:+R.x0.toFixed(2), x1:+R.x1.toFixed(2), z0:+R.z0.toFixed(2), z1:+R.z1.toFixed(2)},
           robot:{x:+b.x.toFixed(3), z:+b.z.toFixed(3)}, mPerAU:+SCALE.mPerAU.toFixed(4) };
}
```

```diff
// trajRunScene의 기존 수집/복원 로직은 유지하고 signature와 POST 두 줄만 바꾼다.
- async function trajRunScene(sceneSeed, seconds, jobs, idx) {
+ async function trajRunScene(sceneSeed, seconds, jobs, idx, datasetName) {
-   body:JSON.stringify(scene)
+   body:JSON.stringify({ ...scene, dataset: datasetName })

// trajRun의 기존 running guard, warmup, try/finally는 유지하고 다음 줄만 추가·교체한다.
+ const datasetName = opts.dataset || "trajectories";
+ const allowedDatasets = new Set(["trajectories","trajectories_v2"]);
+ if (!allowedDatasets.has(datasetName)) throw new Error("허용되지 않은 trajectory dataset: " + datasetName);
- console.log("[TRAJ] 시작 — scene", scenes, "· 기준시드", base, "· 각", seconds + "초");
+ console.log("[TRAJ] 시작 — scene", scenes, "· 기준시드", base, "· 각", seconds + "초", "· dataset", datasetName);
- const scene = await trajRunScene(base + k, seconds, jobs, k);
+ const scene = await trajRunScene(base + k, seconds, jobs, k, datasetName);
```

- [ ] **Step 4: 계약 테스트와 전체 Python 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_sim_traj_collection_contract.py tests/ -q`

Expected: PASS, 기존 테스트 회귀 없음.

- [ ] **Step 5: 브라우저 수집 계약을 커밋한다**

```powershell
git add -- sim.html tests/test_sim_traj_collection_contract.py
git commit -m "feat: distinguish trajectory v2 layouts"
```

### Task 3: v2 manifest 생성과 무결성 감사

**Files:**
- Create: `trajectory/traj_v2.py`
- Create: `train/audit_traj_v2.py`
- Create: `tests/test_traj_v2.py`

**Interfaces:**
- Consumes: `dataset/trajectories_v2/*.json`, `code_commit: str`, 기존 manifest 선택값.
- Produces: `build_manifest(dataset_dir: Path, code_commit: str, generated_at: str | None = None) -> dict`, `validate_manifest(dataset_dir: Path, manifest: dict) -> None`, `DatasetAuditError`.

- [ ] **Step 1: 전수 조합, split, hash 실패 테스트를 작성한다**

```python
# tests/test_traj_v2.py
import hashlib
import json
import pytest
from trajectory.traj_v2 import build_manifest, validate_manifest, DatasetAuditError

LAYOUT_META = {
    "island_h58": {"layout": "island", "half": 5.75},
    "corridor": {"layout": "corridor", "half": None},
    "legacy": {"layout": "legacy", "half": None},
}
JOB_SETS = [("cook", "carry"), ("cook", "lead", "wash"), ("prep", "wash"),
            ("cook", "carry", "wash", "lead"), ("prep", "cook", "carry")]


def _write_v2(root, seed, tag):
    jobs = JOB_SETS[(seed - 1) % 5]
    meta = LAYOUT_META[tag]
    frames = [{"t": round(i * .4, 3), "x": 0.01 * i, "z": 0.0,
               "goal": None, "gx": None, "gz": None, "moving": True} for i in range(150)]
    scene = {"scene_id": f"{tag}_seed{seed}_{seed - 1:04d}", "schema": 1, "seed": seed,
             "layout": meta["layout"], "half": meta["half"],
             "room": {"x0": -5, "x1": 5, "z0": -5, "z1": 5},
             "robot": {"x": -1.1, "z": .815}, "mPerAU": 1.0,
             "wf": True, "hz": 2.5, "dt": .4, "steps": 150, "discarded": False,
             "nodes": [{"id": f"extra_{i}", "job": job, "role": "danger",
                        "discarded": False, "frames": frames} for i, job in enumerate(jobs)]}
    (root / f"{scene['scene_id']}.json").write_text(json.dumps(scene), encoding="utf-8")


def _full_dataset(root):
    root.mkdir()
    for seed in range(1, 51):
        for tag in LAYOUT_META:
            _write_v2(root, seed, tag)


def test_manifest_has_fixed_90_30_30_split_and_150_hashes(tmp_path):
    data = tmp_path / "v2"; _full_dataset(data)
    manifest = build_manifest(data, code_commit="abc123", generated_at="2026-08-24T00:00:00Z")
    assert manifest["meta"]["counts"] == {"train": 90, "val": 30, "test": 30}
    assert len(manifest["files"]) == 150
    assert set(manifest["meta"]["layouts"]) == {"island_h58", "corridor", "legacy"}
    assert set(manifest["summary"]["train"]) >= {"scenes", "people", "job_sets", "safety_eligible", "safety_positive"}
    assert manifest["summary"]["test"] == {"scenes": 30, "labels_locked": True}
    validate_manifest(data, manifest)


def test_manifest_detects_changed_file(tmp_path):
    data = tmp_path / "v2"; _full_dataset(data)
    manifest = build_manifest(data, code_commit="abc123")
    path = data / manifest["train"][0]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(DatasetAuditError, match="SHA-256"):
        validate_manifest(data, manifest)


def test_manifest_rejects_missing_layout(tmp_path):
    data = tmp_path / "v2"; _full_dataset(data)
    next(data.glob("corridor_seed1_*.json")).unlink()
    with pytest.raises(DatasetAuditError, match="누락"):
        build_manifest(data, code_commit="abc123")
```

- [ ] **Step 2: 감사 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_traj_v2.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory.traj_v2'`.

- [ ] **Step 3: 고정 상수와 파일 검사기를 구현한다**

```python
# trajectory/traj_v2.py 핵심 계약
SPLIT_SEEDS = {"train": range(1, 31), "val": range(31, 41), "test": range(41, 51)}
LAYOUTS = ("island_h58", "corridor", "legacy")
JOB_SETS = (("cook", "carry"), ("cook", "lead", "wash"), ("prep", "wash"),
            ("cook", "carry", "wash", "lead"), ("prep", "cook", "carry"))


class DatasetAuditError(ValueError):
    pass


def split_for_seed(seed: int) -> str:
    for split, seeds in SPLIT_SEEDS.items():
        if seed in seeds:
            return split
    raise DatasetAuditError(f"범위 밖 seed: {seed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_scene(path: Path) -> dict:
    match = re.fullmatch(r"(island_h58|corridor|legacy)_seed(\d+)_(\d{4})\.json", path.name)
    if match is None:
        raise DatasetAuditError(f"예상하지 않은 파일명: {path.name}")
    tag, raw_seed, _index = match.groups(); seed = int(raw_seed)
    scene = json.loads(path.read_text(encoding="utf-8"))
    expected_layout = "island" if tag == "island_h58" else tag
    expected_half = 5.75 if tag == "island_h58" else None
    if scene.get("scene_id") != path.stem or scene.get("seed") != seed:
        raise DatasetAuditError(f"scene 식별 불일치: {path.name}")
    if scene.get("layout") != expected_layout or scene.get("half") != expected_half:
        raise DatasetAuditError(f"layout metadata 불일치: {path.name}")
    if scene.get("dt") != .4 or scene.get("hz") != 2.5 or scene.get("steps") != 150:
        raise DatasetAuditError(f"시간 계약 불일치: {path.name}")
    if not scene.get("room") or not scene.get("robot") or scene.get("mPerAU") is None:
        raise DatasetAuditError(f"geometry metadata 누락: {path.name}")
    if scene.get("discarded") or any(node.get("discarded") for node in scene.get("nodes", [])):
        raise DatasetAuditError(f"폐기 scene/node 포함: {path.name}")
    jobs = tuple(node.get("job") for node in scene.get("nodes", []))
    if jobs != JOB_SETS[(seed - 1) % len(JOB_SETS)]:
        raise DatasetAuditError(f"job 조합 불일치: {path.name}")
    for node in scene["nodes"]:
        frames = node.get("frames", [])
        if len(frames) != 150 or any(abs(frame["t"] - index * .4) > 1e-6
                                     for index, frame in enumerate(frames)):
            raise DatasetAuditError(f"frame 계약 불일치: {path.name}/{node.get('id')}")
    return {"name": path.name, "seed": seed, "layout": tag, "split": split_for_seed(seed),
            "jobs": list(jobs), "people": len(scene["nodes"]), "frames": 150, "dt": .4,
            "has_room": True, "has_robot": True, "mPerAU": scene["mPerAU"],
            "sha256": sha256_file(path)}


def build_manifest(dataset_dir: Path, code_commit: str, generated_at: str | None = None) -> dict:
    expected = {(seed, layout) for seed in range(1, 51) for layout in LAYOUTS}
    rows = [inspect_scene(path) for path in sorted(dataset_dir.glob("*.json"))]
    actual = {(row["seed"], row["layout"]) for row in rows}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra or len(rows) != 150:
        raise DatasetAuditError(f"전수 조합 누락={missing} 추가={extra} files={len(rows)}")
    names = {split: sorted(row["name"] for row in rows if row["split"] == split)
             for split in SPLIT_SEEDS}
    manifest = {"schema": 2,
            "meta": {"dataset": "traj-v2", "code_commit": code_commit,
                     "generated_at": generated_at, "layouts": list(LAYOUTS),
                     "generation": {
                         "urls": {"island_h58": "sim.html?layout=island",
                                  "corridor": "sim.html?layout=corridor",
                                  "legacy": "sim.html?layout=legacy"},
                         "command": "__sim.trajRun({scenes:50,seed:1,seconds:60,dataset:'trajectories_v2'})"},
                     "seeds": {split: list(seeds) for split, seeds in SPLIT_SEEDS.items()},
                     "counts": {split: len(names[split]) for split in names}},
            **names, "files": {row["name"]: row for row in rows}}
    manifest["summary"] = summarize_dataset(dataset_dir, manifest)
    return manifest


def summarize_dataset(dataset_dir: Path, manifest: dict) -> dict:
    summary = {}
    for split in ("train", "val"):
        people = eligible = positive = 0; job_sets = Counter()
        for name in manifest[split]:
            scene = json.loads((dataset_dir / name).read_text(encoding="utf-8"))
            people += len(scene["nodes"]); job_sets[tuple(node["job"] for node in scene["nodes"])] += 1
            rx, rz = scene["robot"]["x"], scene["robot"]["z"]
            for node in scene["nodes"]:
                frames = node["frames"]
                for start in range(len(frames) - (8 + 12) + 1):
                    current = frames[start + 7]
                    if math.hypot(current["x"] - rx, current["z"] - rz) < 3.10:
                        continue
                    eligible += 1
                    future = frames[start + 8:start + 12]
                    positive += any(math.hypot(frame["x"] - rx, frame["z"] - rz) < 3.10
                                    for frame in future)
        summary[split] = {"scenes": len(manifest[split]), "people": people,
                          "job_sets": {"|".join(key): count for key, count in sorted(job_sets.items())},
                          "safety_eligible": eligible, "safety_positive": positive}
    summary["test"] = {"scenes": len(manifest["test"]), "labels_locked": True}
    return summary


def validate_manifest(dataset_dir: Path, manifest: dict) -> None:
    current = {path.name for path in dataset_dir.glob("*.json")}
    recorded = set(manifest["files"])
    if current != recorded:
        raise DatasetAuditError(f"파일 집합 불일치 누락={sorted(recorded-current)} 추가={sorted(current-recorded)}")
    changed = [name for name, row in manifest["files"].items()
               if sha256_file(dataset_dir / name) != row["sha256"]]
    if changed:
        raise DatasetAuditError("SHA-256 불일치: " + ", ".join(changed))
    for split, seeds in SPLIT_SEEDS.items():
        expected_names = sorted(name for name, row in manifest["files"].items()
                                if row["seed"] in seeds and row["split"] == split)
        if manifest.get(split) != expected_names or len(expected_names) != len(seeds) * len(LAYOUTS):
            raise DatasetAuditError(f"split membership 불일치: {split}")
    if manifest.get("meta", {}).get("counts") != {"train": 90, "val": 30, "test": 30}:
        raise DatasetAuditError("split count 불일치")
```

- [ ] **Step 4: CLI를 구현한다**

```python
# train/audit_traj_v2.py 핵심 흐름
from datetime import datetime, timezone

parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "trajectories_v2")
parser.add_argument("--manifest", type=Path,
                    default=ROOT / "docs" / "chanwoo" / "results" / "traj-v2-manifest.json")
parser.add_argument("--code-commit", default=subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
parser.add_argument("--write", action="store_true")

if args.manifest.exists():
    previous = json.loads(args.manifest.read_text(encoding="utf-8"))
    generated_at = previous.get("meta", {}).get("generated_at")
else:
    generated_at = datetime.now(timezone.utc).isoformat()
manifest = build_manifest(args.dataset_dir, code_commit=args.code_commit, generated_at=generated_at)
validate_manifest(args.dataset_dir, manifest)
if args.write:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest["meta"]["counts"], ensure_ascii=False))
```

- [ ] **Step 5: 감사 테스트를 통과시킨다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_traj_v2.py -q`

Expected: PASS.

- [ ] **Step 6: 감사기만 커밋한다**

```powershell
git add -- trajectory/traj_v2.py train/audit_traj_v2.py tests/test_traj_v2.py
git commit -m "feat: audit trajectory v2 dataset manifest"
```

### Task 4: 외부 v2 manifest를 읽는 로더

**Files:**
- Modify: `trajectory/sim_traj.py:43-102`
- Modify: `tests/test_sim_traj.py`

**Interfaces:**
- Consumes: `load_windows(split, traj_dir=..., manifest_path=...)`.
- Produces: v1의 같은-directory manifest 기본 동작과 v2의 `docs/chanwoo/results/traj-v2-manifest.json` 외부 경로 동작.

- [ ] **Step 1: 외부 manifest 경로 테스트를 추가한다**

```python
def test_loader_accepts_external_manifest(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    manifest_dir = tmp_path / "docs"; manifest_dir.mkdir()
    frs = [(0.1 * i, 0.0, "kettle", 2.0, 0.0) for i in range(22)]
    _write_scene(data, 31, {"extra_0": frs})
    manifest = manifest_dir / "traj-v2-manifest.json"
    manifest.write_text(json.dumps({"train": [], "val": ["t_seed31.json"], "test": []}), encoding="utf-8")
    assert len(load_windows("val", traj_dir=data, manifest_path=manifest)) == 3
```

- [ ] **Step 2: 새 인자가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_sim_traj.py::test_loader_accepts_external_manifest -q`

Expected: FAIL with `TypeError: load_windows() got an unexpected keyword argument 'manifest_path'`.

- [ ] **Step 3: manifest 경로 선택을 최소 확장한다**

```python
def load_manifest(traj_dir=None, manifest_path=None):
    d = Path(traj_dir) if traj_dir else TRAJ_DIR
    p = Path(manifest_path) if manifest_path else d / MANIFEST_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_windows(split="val", traj_dir=None, manifest_path=None):
    d = Path(traj_dir) if traj_dir else TRAJ_DIR
    members = None
    if split in ("train", "val", "test"):
        manifest = load_manifest(d, manifest_path)
        if manifest is None:
            source = Path(manifest_path) if manifest_path else d / MANIFEST_NAME
            raise FileNotFoundError(f"manifest 없음: {source}")
        members = set(manifest[split])
    elif split != "all":
        raise ValueError(f"알 수 없는 split: {split!r}")
    # 아래 JSON 순회와 Window 생성은 현재 함수 내용을 그대로 사용한다.
```

- [ ] **Step 4: v1과 v2 로더 회귀 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_sim_traj.py tests/test_traj_split.py tests/test_traj_v2.py -q`

Expected: PASS.

- [ ] **Step 5: 로더 확장을 커밋한다**

```powershell
git add -- trajectory/sim_traj.py tests/test_sim_traj.py
git commit -m "feat: load versioned trajectory manifests"
```

### Task 5: v2 150개 수집과 감사

**Files:**
- Create locally, ignored: `dataset/trajectories_v2/*.json`
- Create: `docs/chanwoo/results/traj-v2-manifest.json`

**Interfaces:**
- Consumes: 실행 중인 `backend/detect_server.py`, 세 sim URL, `__sim.trajRun`.
- Produces: 감사 통과한 150개 v2 JSON과 추적 manifest.

- [ ] **Step 1: v1을 PR worktree 원본과 비교한다**

```powershell
$sourceDir = 'C:\Users\chanwoo\workspace\robot-kitchen-safety-sim-traj-cvae\dataset\trajectories'
$targetDir = (Resolve-Path 'dataset/trajectories').Path
$source = @{}; Get-ChildItem -LiteralPath $sourceDir -File | ForEach-Object { $source[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$target = @{}; Get-ChildItem -LiteralPath $targetDir -File | ForEach-Object { $target[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$different = @($source.Keys | Where-Object { -not $target.ContainsKey($_) -or $source[$_] -ne $target[$_] })
if ($different.Count -or $source.Count -ne $target.Count) { throw "v1 불일치: $($different -join ', ')" }
```

Expected: 두 경로 모두 123개, 불일치 0개.

- [ ] **Step 2: 검출 없이 저장 서버를 숨김 창으로 실행한다**

```powershell
$env:DETECT_MODEL = 'none'
$server = Start-Process -FilePath 'uv' -ArgumentList @('run','--group','serve','python','backend/detect_server.py') -WorkingDirectory (Get-Location) -PassThru -WindowStyle Hidden
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
```

Expected: health 응답의 `mode`가 `off`.

- [ ] **Step 3: 세 URL에서 정확히 50개씩 v2로 수집한다**

각 페이지가 `window.__simReady === true`가 된 뒤 브라우저 콘솔에서 아래 명령을 한 번씩 실행한다.

```javascript
await __sim.trajRun({scenes:50, seed:1, seconds:60, dataset:"trajectories_v2"})
```

URLs:

```text
http://127.0.0.1:8000/sim.html?layout=island
http://127.0.0.1:8000/sim.html?layout=corridor
http://127.0.0.1:8000/sim.html?layout=legacy
```

각 실행 뒤 서버 응답 경로가 `dataset/trajectories_v2`이고 브라우저 로그가 유효 scene 50개인지 확인한다.

- [ ] **Step 4: manifest를 생성하고 전수 감사를 실행한다**

Run: `uv run --group serve python train/audit_traj_v2.py --write`

Expected: `{"train": 90, "val": 30, "test": 30}`, 파일 150개, 오류 0개.

- [ ] **Step 5: 같은 비교를 다시 실행해 v1 불변성을 확인한다**

```powershell
$sourceDir = 'C:\Users\chanwoo\workspace\robot-kitchen-safety-sim-traj-cvae\dataset\trajectories'
$targetDir = (Resolve-Path 'dataset/trajectories').Path
$source = @{}; Get-ChildItem -LiteralPath $sourceDir -File | ForEach-Object { $source[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$target = @{}; Get-ChildItem -LiteralPath $targetDir -File | ForEach-Object { $target[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$different = @($source.Keys | Where-Object { -not $target.ContainsKey($_) -or $source[$_] -ne $target[$_] })
if ($different.Count -or $source.Count -ne $target.Count) { throw "v1 변경: $($different -join ', ')" }
```

Expected: 비교 예외 없음.

- [ ] **Step 6: 서버를 종료하고 manifest만 커밋한다**

```powershell
Stop-Process -Id $server.Id
git status --short
git add -- docs/chanwoo/results/traj-v2-manifest.json
git commit -m "data: lock trajectory v2 manifest"
```

`git status`에서 v2 JSON과 `training/` 산출물이 보이지 않아야 한다.

### Task 6: 데이터 단계 최종 회귀 검증

**Files:**
- Verify only.

**Interfaces:**
- Consumes: Task 1~5의 코드와 manifest.
- Produces: 다음 평가 계획이 사용할 감사 통과 상태.

- [ ] **Step 1: manifest를 쓰지 않는 검증 모드로 다시 검사한다**

Run: `uv run --group serve python train/audit_traj_v2.py`

Expected: 150개 전수와 90/30/30 split 통과.

- [ ] **Step 2: 전체 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 3: diff와 브랜치 상태를 확인한다**

```powershell
git diff --check
git status --short --branch
git log --oneline -6
```

Expected: 추적 파일 변경 없음, v1 데이터 변경 없음, Task별 커밋 존재.
