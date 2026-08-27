# Autoresearch Evaluation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer 후보를 validation safety F2로 공정하게 비교하고, 정확도·지연·크기 보호 조건과 잠긴 test 1회 규칙을 코드로 강제한다.

**Architecture:** 고정 평가 계약은 `train/autoresearch_contract.py`에 두고 후보 모델은 `trajectory/autoresearch_candidate.py` 한 파일로 격리한다. 개발 로더는 train/val만 열 수 있고, test는 별도 CLI가 명시적 확인 문자열·승자 커밋·출력 파일 부재를 모두 검사한 뒤 한 번만 연다. 안전 평가는 현재 배포 경로인 `trajectory.risk.track_risk`를 재사용한다.

**Tech Stack:** Python 3.11+, PyTorch 2.7+, NumPy, pytest, dataclasses, scene bootstrap

**Spec:** `docs/chanwoo/specs/2026-08-24-transformer-autoresearch-design.md`

## Global Constraints

- 데이터 계획 `docs/superpowers/plans/2026-08-24-trajectory-v2-data.md`가 먼저 완료되어야 한다.
- v2 경로는 `dataset/trajectories_v2`, manifest는 `docs/chanwoo/results/traj-v2-manifest.json`이다.
- 관측 8점, 예측 12점, K=3, 상대 좌표 정규화, MTP 손실 계열은 고정한다.
- 1차 판정은 `stopR=3.10`, `horizon=1.6`, `safeKsig=1.0`, `safeTau=0.1`인 `track_risk`다.
- 보호 조건은 recall 하락 1%p 이내, FDE@1.6s 1.02배 이내, CPU p95 1.20배 이내, 파라미터 1.20배 이내다.
- 기준 Transformer 대비 F2 `+0.01`을 최소 성공으로 쓴다.
- 자동 탐색 코드에서는 test split을 요청하거나 import하지 않는다.
- minADE@K는 참고 상한일 뿐 선택 점수나 안전 성능으로 사용하지 않는다.

---

### Task 1: F2와 보호 조건 순수 계약

**Files:**
- Create: `train/autoresearch_contract.py`
- Create: `tests/test_autoresearch_contract.py`

**Interfaces:**
- Consumes: candidate/baseline metric dict.
- Produces: `f2_score(precision: float, recall: float) -> float`, `evaluate_guards(candidate: Metrics, baseline: Metrics) -> GuardReport`, `rank_key(metrics: Metrics) -> tuple`.

- [ ] **Step 1: 경계값 실패 테스트를 작성한다**

```python
# tests/test_autoresearch_contract.py
from train.autoresearch_contract import Metrics, f2_score, evaluate_guards, rank_key


def _m(**overrides):
    base = dict(precision=.60, recall=.70, fde16=.30, cpu_p95_ms=2.0,
                parameters=100_000, ade16=.20, tp=70, fp=47, fn=30)
    base.update(overrides)
    return Metrics(**base)


def test_f2_weights_recall_four_times():
    assert abs(f2_score(.60, .70) - (5 * .60 * .70 / (4 * .60 + .70))) < 1e-12
    assert f2_score(0.0, 0.0) == 0.0


def test_guards_accept_exact_boundaries():
    baseline = _m()
    candidate = _m(recall=.69, fde16=.306, cpu_p95_ms=2.4, parameters=120_000)
    report = evaluate_guards(candidate, baseline)
    assert report.passed
    assert report.failures == ()


def test_each_guard_reports_its_failure():
    baseline = _m()
    candidate = _m(recall=.689, fde16=.307, cpu_p95_ms=2.401, parameters=120_001)
    report = evaluate_guards(candidate, baseline)
    assert set(report.failures) == {"recall", "fde16", "cpu_p95_ms", "parameters"}


def test_rank_prefers_f2_then_recall_then_lower_fde_and_latency():
    a = _m(precision=.70, recall=.70, fde16=.25, cpu_p95_ms=2.0)
    b = _m(precision=.70, recall=.70, fde16=.30, cpu_p95_ms=1.0)
    assert rank_key(a) > rank_key(b)
```

- [ ] **Step 2: 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'train.autoresearch_contract'`.

- [ ] **Step 3: dataclass와 보호 조건을 구현한다**

```python
# train/autoresearch_contract.py
from dataclasses import asdict, dataclass

RECALL_DROP_MAX = .01
FDE16_RATIO_MAX = 1.02
LATENCY_RATIO_MAX = 1.20
PARAMETER_RATIO_MAX = 1.20
MIN_F2_GAIN = .01


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    fde16: float
    cpu_p95_ms: float
    parameters: int
    ade16: float
    tp: int
    fp: int
    fn: int

    @property
    def f2(self) -> float:
        return f2_score(self.precision, self.recall)


@dataclass(frozen=True)
class GuardReport:
    passed: bool
    failures: tuple[str, ...]


def f2_score(precision: float, recall: float) -> float:
    denominator = 4 * precision + recall
    return 0.0 if denominator == 0 else 5 * precision * recall / denominator


def evaluate_guards(candidate: Metrics, baseline: Metrics) -> GuardReport:
    failures = []
    if candidate.recall < baseline.recall - RECALL_DROP_MAX: failures.append("recall")
    if candidate.fde16 > baseline.fde16 * FDE16_RATIO_MAX: failures.append("fde16")
    if candidate.cpu_p95_ms > baseline.cpu_p95_ms * LATENCY_RATIO_MAX: failures.append("cpu_p95_ms")
    if candidate.parameters > baseline.parameters * PARAMETER_RATIO_MAX: failures.append("parameters")
    return GuardReport(not failures, tuple(failures))


def rank_key(metrics: Metrics) -> tuple[float, float, float, float]:
    return (metrics.f2, metrics.recall, -metrics.fde16, -metrics.cpu_p95_ms)
```

- [ ] **Step 4: 계약 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py -q`

Expected: PASS.

- [ ] **Step 5: 순수 계약을 커밋한다**

```powershell
git add -- train/autoresearch_contract.py tests/test_autoresearch_contract.py
git commit -m "feat: define autoresearch safety score guards"
```

### Task 2: 개발 split 전용 로더

**Files:**
- Modify: `train/autoresearch_contract.py`
- Modify: `tests/test_autoresearch_contract.py`

**Interfaces:**
- Consumes: `development_windows(split: str, dataset_dir: Path = V2_DIR, manifest_path: Path = V2_MANIFEST)`.
- Produces: train/val window 목록; test와 알 수 없는 split에는 `TestSplitLockedError`.

- [ ] **Step 1: test 잠금 실패 테스트를 추가한다**

```python
import pytest
from train.autoresearch_contract import development_windows, TestSplitLockedError


@pytest.mark.parametrize("split", ["test", "locked_test", "all"])
def test_development_loader_rejects_non_development_splits(split):
    with pytest.raises(TestSplitLockedError, match="train/val만"):
        development_windows(split)
```

- [ ] **Step 2: 새 함수가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py::test_development_loader_rejects_non_development_splits -q`

Expected: FAIL with import error for `development_windows`.

- [ ] **Step 3: v2 위치를 고정한 개발 로더를 구현한다**

```python
from pathlib import Path
from trajectory.sim_traj import load_windows
from trajectory.traj_v2 import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "dataset" / "trajectories_v2"
V2_MANIFEST = ROOT / "docs" / "chanwoo" / "results" / "traj-v2-manifest.json"


class TestSplitLockedError(ValueError):
    pass


def development_windows(split: str, dataset_dir: Path = V2_DIR,
                        manifest_path: Path = V2_MANIFEST):
    if split not in {"train", "val"}:
        raise TestSplitLockedError(f"자동 실험은 train/val만 허용: {split!r}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(Path(dataset_dir), manifest)
    return load_windows(split, traj_dir=dataset_dir, manifest_path=manifest_path)
```

- [ ] **Step 4: 잠금 테스트와 v2 manifest 검증을 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py tests/test_traj_v2.py -q`

Expected: PASS.

- [ ] **Step 5: 개발 로더 잠금을 커밋한다**

```powershell
git add -- train/autoresearch_contract.py tests/test_autoresearch_contract.py
git commit -m "feat: lock autoresearch to train and validation"
```

### Task 3: 실제 런타임 규칙을 쓰는 safety/FDE 평가

**Files:**
- Modify: `train/autoresearch_contract.py`
- Modify: `tests/test_autoresearch_contract.py`

**Interfaces:**
- Consumes: `evaluate_windows(predictor, windows, cpu_p95_ms=0.0, parameters=0, bootstrap_samples=2000) -> Evaluation`.
- Produces: `Metrics`, scene 단위 F2/recall/precision/ADE16/FDE16 95% CI, 원시 scene confusion.

- [ ] **Step 1: 확률질량·sigma·최빈 FDE 계약 테스트를 추가한다**

```python
from types import SimpleNamespace
from trajectory.types import Track, TrackScene
from trajectory.sim_traj import Window
from train.autoresearch_contract import evaluate_windows


class FixedPredictor:
    def predict_batch(self, hists):
        # 첫 모드는 정답 위치오차용 최빈, 두 번째 모드는 stopR 진입; 합 mass=.15 >= tau=.1
        return [[
            {"path": [(4.0, 0.0)] * 12, "w": .85, "sigma": [0.0] * 12},
            {"path": [(3.0, 0.0)] * 12, "w": .15, "sigma": [0.0] * 12},
            {"path": [(5.0, 0.0)] * 12, "w": 0.0, "sigma": [0.0] * 12},
        ] for _ in hists]


def _window():
    hist = [(i * .4, 5.0, 0.0) for i in range(8)]
    gt = [((i + 8) * .4, 3.0, 0.0) for i in range(12)]
    return Window("scene-1", 31, "extra_0", TrackScene(2.8, 4.8, [Track(0, hist)]),
                  gt, None, True, (0.0, 0.0))


def test_evaluation_uses_probability_mass_for_runtime_alert():
    out = evaluate_windows(FixedPredictor(), [_window()], bootstrap_samples=20)
    assert (out.metrics.tp, out.metrics.fp, out.metrics.fn) == (1, 0, 0)
    assert out.metrics.recall == 1.0 and out.metrics.precision == 1.0
    assert abs(out.metrics.fde16 - 1.0) < 1e-9  # 최빈 path x=4, GT x=3
```

- [ ] **Step 2: 평가 함수가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py::test_evaluation_uses_probability_mass_for_runtime_alert -q`

Expected: FAIL with import error for `evaluate_windows`.

- [ ] **Step 3: 고정 안전 상수와 per-scene 집계를 구현한다**

```python
STOP_R, SLOW_R, HORIZON, KSIG, TAU = 3.10, 3.90, 1.6, 1.0, .1
HORIZON_STEPS = 4


@dataclass(frozen=True)
class Evaluation:
    metrics: Metrics
    ci: dict[str, tuple[float, float, float]]
    by_scene: dict[str, dict[str, object]]


def _predicted_entry(modes, robot) -> bool:
    return track_risk(modes, robot, STOP_R, SLOW_R, HORIZON, KSIG, TAU)["tEntryStop"] is not None


def _actual_entry(window) -> bool:
    gt = [(x, z) for _, x, z in window.gt[:HORIZON_STEPS]]
    return enters_radius(gt, window.robot, STOP_R)


def _evaluation_from_scene_rows(by_scene, cpu_p95_ms, parameters, bootstrap_samples):
    rows = list(by_scene.values())
    tp = sum(row["confusion"][0] for row in rows)
    fp = sum(row["confusion"][1] for row in rows)
    fn = sum(row["confusion"][2] for row in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    ades = [value for row in rows for value in row["ade16"]]
    fdes = [value for row in rows for value in row["fde16"]]
    metrics = Metrics(precision, recall, sum(fdes) / len(fdes), cpu_p95_ms, parameters,
                      sum(ades) / len(ades), tp, fp, fn)

    def confusion_stat(sample, kind):
        stp = sum(row["confusion"][0] for row in sample)
        sfp = sum(row["confusion"][1] for row in sample)
        sfn = sum(row["confusion"][2] for row in sample)
        sr = stp / (stp + sfn) if stp + sfn else 0.0
        sp = stp / (stp + sfp) if stp + sfp else 0.0
        return {"recall": sr, "precision": sp, "f2": f2_score(sp, sr)}[kind]

    def mean_field(sample, field):
        values = [value for row in sample for value in row[field]]
        return sum(values) / len(values)

    ci = {kind: scene_bootstrap_ci(rows, lambda sample, k=kind: confusion_stat(sample, k),
                                   B=bootstrap_samples, seed=0)
          for kind in ("recall", "precision", "f2")}
    ci["ade16"] = scene_bootstrap_ci(rows, lambda sample: mean_field(sample, "ade16"),
                                     B=bootstrap_samples, seed=0)
    ci["fde16"] = scene_bootstrap_ci(rows, lambda sample: mean_field(sample, "fde16"),
                                     B=bootstrap_samples, seed=0)
    return Evaluation(metrics, ci, dict(by_scene))


def evaluate_windows(predictor, windows, cpu_p95_ms=0.0, parameters=0,
                     bootstrap_samples=2000) -> Evaluation:
    hists = [[(x, z) for _, x, z in w.scene.agents[0].history] for w in windows]
    predicted = predictor.predict_batch(hists)
    by_scene = defaultdict(lambda: {"confusion": [0, 0, 0], "ade16": [], "fde16": []})
    for window, modes in zip(windows, predicted):
        x, z = window.scene.agents[0].history[-1][1:3]
        if math.hypot(x - window.robot[0], z - window.robot[1]) < STOP_R:
            continue
        actual = _actual_entry(window); alert = _predicted_entry(modes, window.robot)
        cell = entry_confusion(STOP_R + 1.0, actual, alert, STOP_R)
        index = {"TP": 0, "FP": 1, "FN": 2}.get(cell)
        if index is not None: by_scene[window.scene_id]["confusion"][index] += 1
        pred = modes[0]["path"][:HORIZON_STEPS]
        gt = [(gx, gz) for _, gx, gz in window.gt[:HORIZON_STEPS]]
        errors = [math.hypot(px - gx, pz - gz) for (px, pz), (gx, gz) in zip(pred, gt)]
        by_scene[window.scene_id]["ade16"].append(sum(errors) / len(errors))
        by_scene[window.scene_id]["fde16"].append(errors[-1])
    return _evaluation_from_scene_rows(by_scene, cpu_p95_ms, parameters, bootstrap_samples)
```

현재 반경 안인 창은 제외한다. 각 scene에 `[tp, fp, fn]`, ADE16 목록, FDE16 목록을 모으고
`scene_bootstrap_ci`로 각각 95% CI를 계산한다. `Metrics` point 값은 전체 scene 합산 confusion과
전체 window 평균 ADE/FDE로 만든다. FDE16은 `modes[0]["path"][3]`과 GT 네 번째 점의 거리다.

- [ ] **Step 4: 평가 테스트와 기존 위험 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py tests/test_risk.py tests/test_bootstrap.py -q`

Expected: PASS.

- [ ] **Step 5: 안전 평가를 커밋한다**

```powershell
git add -- train/autoresearch_contract.py tests/test_autoresearch_contract.py
git commit -m "feat: evaluate candidates with runtime safety rules"
```

### Task 4: CPU p95와 파라미터 수 측정

**Files:**
- Modify: `train/autoresearch_contract.py`
- Modify: `tests/test_autoresearch_contract.py`

**Interfaces:**
- Consumes: `measure_cpu_p95(predictor, hist, warmups=100, repeats=1000, clock_ns=time.perf_counter_ns)`, `count_trainable_parameters(net)`.
- Produces: 밀리초 p95와 정수 파라미터 수.

- [ ] **Step 1: 주입 clock으로 측정 절차를 고정하는 테스트를 추가한다**

```python
def test_cpu_p95_excludes_warmups_and_uses_95th_percentile():
    ticks = iter(range(0, 10_000_000, 1_000_000))
    predictor = SimpleNamespace(predict_batch=lambda _: None)
    p95 = measure_cpu_p95(predictor, [(0.0, 0.0)] * 8,
                            warmups=1, repeats=3, clock_ns=lambda: next(ticks))
    assert p95 == 1.0


def test_parameter_count_only_includes_trainable_values():
    import torch.nn as nn
    net = nn.Linear(2, 3)
    net.bias.requires_grad_(False)
    assert count_trainable_parameters(net) == 6
```

- [ ] **Step 2: 측정 함수가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py -q`

Expected: FAIL with import errors for measurement functions.

- [ ] **Step 3: 단일 thread 측정을 구현한다**

```python
def measure_cpu_p95(predictor, hist, warmups=100, repeats=1000,
                    clock_ns=time.perf_counter_ns) -> float:
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(warmups):
            predictor.predict_batch([hist])
        elapsed = []
        for _ in range(repeats):
            start = clock_ns(); predictor.predict_batch([hist]); elapsed.append(clock_ns() - start)
        return float(np.percentile(np.asarray(elapsed, dtype=np.float64), 95) / 1_000_000)
    finally:
        torch.set_num_threads(previous_threads)


def count_trainable_parameters(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
```

- [ ] **Step 4: 측정 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py -q`

Expected: PASS.

- [ ] **Step 5: 측정 계약을 커밋한다**

```powershell
git add -- train/autoresearch_contract.py tests/test_autoresearch_contract.py
git commit -m "feat: measure autoresearch model cost guards"
```

### Task 5: 단일 변경 지점인 Transformer 후보

**Files:**
- Create: `trajectory/autoresearch_candidate.py`
- Create: `tests/test_autoresearch_candidate.py`

**Interfaces:**
- Consumes: top-level `CONFIG: CandidateConfig`.
- Produces: `build_candidate(config: CandidateConfig = CONFIG) -> nn.Module` with `(paths, logits, logsig)`.

- [ ] **Step 1: 출력 형상과 잘못된 head 조합 테스트를 작성한다**

```python
# tests/test_autoresearch_candidate.py
import pytest
import torch
from trajectory.autoresearch_candidate import CandidateConfig, build_candidate
from trajectory.learned_predictor import OBS, PRED, K


def test_candidate_output_contract():
    net = build_candidate(CandidateConfig(hidden=32, layers=1, heads=4, ff_ratio=2,
                                          dropout=0.0, norm_first=True, pooling="last",
                                          learning_rate=1e-3, weight_decay=0.0, batch_size=64))
    paths, logits, logsig = net(torch.zeros(5, OBS, 2))
    assert paths.shape == (5, K, PRED, 2)
    assert logits.shape == (5, K)
    assert logsig.shape == (5, K, PRED)


def test_hidden_must_be_divisible_by_heads():
    with pytest.raises(ValueError, match="hidden.*heads"):
        build_candidate(CandidateConfig(hidden=30, heads=4))


def test_pooling_is_explicit():
    with pytest.raises(ValueError, match="pooling"):
        build_candidate(CandidateConfig(pooling="max"))
```

- [ ] **Step 2: 후보 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_candidate.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 현재 Transformer를 기본 후보로 옮긴다**

```python
@dataclass(frozen=True)
class CandidateConfig:
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    ff_ratio: int = 4
    dropout: float = 0.0
    norm_first: bool = False
    pooling: str = "last"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512


CONFIG = CandidateConfig()


def build_candidate(config: CandidateConfig = CONFIG):
    if config.hidden % config.heads:
        raise ValueError("hidden은 heads로 나누어져야 함")
    if config.pooling not in {"last", "mean"}:
        raise ValueError(f"지원하지 않는 pooling: {config.pooling}")
    import torch
    import torch.nn as nn

    class CandidateTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            h = config.hidden
            self.inp = nn.Linear(2, h)
            self.pos = nn.Parameter(torch.zeros(1, OBS, h))
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=config.heads, dim_feedforward=h * config.ff_ratio,
                dropout=config.dropout, batch_first=True, norm_first=config.norm_first)
            self.enc = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                      nn.Linear(h, K * PRED * 2 + K + K * PRED))

        def forward(self, x):
            z = self.enc(self.inp(x) + self.pos[:, :x.shape[1], :])
            pooled = z[:, -1, :] if config.pooling == "last" else z.mean(dim=1)
            out = self.head(pooled); batch = out.shape[0]
            paths = out[:, :K * PRED * 2].reshape(batch, K, PRED, 2)
            logits = out[:, K * PRED * 2:K * PRED * 2 + K]
            logsig = out[:, K * PRED * 2 + K:].reshape(batch, K, PRED)
            return paths, logits, logsig

    return CandidateTransformer()
```

- [ ] **Step 4: 후보와 기존 Transformer 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_candidate.py tests/test_transformer_predictor.py -q`

Expected: PASS.

- [ ] **Step 5: 후보 seam을 커밋한다**

```powershell
git add -- trajectory/autoresearch_candidate.py tests/test_autoresearch_candidate.py
git commit -m "feat: add isolated transformer candidate seam"
```

### Task 6: locked test 1회 실행 장치

**Files:**
- Create: `train/eval_autoresearch_locked_test.py`
- Create: `tests/test_autoresearch_test_gate.py`

**Interfaces:**
- Consumes: `--winner training/autoresearch/winner.json`, `--confirm RUN_LOCKED_TEST_ONCE`.
- Produces: 새 파일 `docs/chanwoo/results/autoresearch-final.json`; 이미 있으면 기본 실패.

- [ ] **Step 1: 확인 문자열·커밋·덮어쓰기 거부 테스트를 작성한다**

```python
# tests/test_autoresearch_test_gate.py
import pytest
from train.eval_autoresearch_locked_test import validate_gate, LockedTestError


def test_gate_requires_exact_confirmation(tmp_path):
    with pytest.raises(LockedTestError, match="확인 문자열"):
        validate_gate({"minimum_success": True, "selected_commit": "HEAD",
                       "weights": str(tmp_path / "model.pt"), "weights_sha256": "x"},
                      "no", tmp_path / "final.json", head_commit="HEAD")


def test_gate_requires_frozen_candidate_commit(tmp_path):
    with pytest.raises(LockedTestError, match="커밋"):
        validate_gate({"minimum_success": True, "selected_commit": "old",
                       "weights": str(tmp_path / "model.pt"), "weights_sha256": "x"},
                      "RUN_LOCKED_TEST_ONCE", tmp_path / "final.json", head_commit="new")


def test_gate_refuses_existing_result(tmp_path):
    out = tmp_path / "final.json"; out.write_text("{}", encoding="utf-8")
    with pytest.raises(LockedTestError, match="이미 존재"):
        validate_gate({"minimum_success": True, "selected_commit": "HEAD",
                       "weights": str(tmp_path / "model.pt"), "weights_sha256": "x"},
                      "RUN_LOCKED_TEST_ONCE", out, head_commit="HEAD")


def test_gate_rejects_weight_hash_mismatch(tmp_path):
    weights = tmp_path / "model.pt"; weights.write_bytes(b"actual")
    with pytest.raises(LockedTestError, match="weights SHA-256"):
        validate_gate({"minimum_success": True, "selected_commit": "HEAD",
                       "weights": str(weights), "weights_sha256": "wrong"},
                      "RUN_LOCKED_TEST_ONCE", tmp_path / "final.json", head_commit="HEAD")
```

- [ ] **Step 2: CLI 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_test_gate.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: test 전용 로더와 배타적 결과 생성을 구현한다**

```python
CONFIRM = "RUN_LOCKED_TEST_ONCE"


class LockedTestError(RuntimeError):
    pass


def validate_gate(winner, confirmation, output, head_commit):
    if confirmation != CONFIRM:
        raise LockedTestError(f"확인 문자열은 {CONFIRM!r} 이어야 함")
    if not winner.get("minimum_success"):
        raise LockedTestError("validation 최소 성공 기준을 통과하지 못함")
    if winner["selected_commit"] != head_commit:
        raise LockedTestError("승자 후보 커밋과 현재 HEAD 커밋이 다름")
    if output.exists():
        raise LockedTestError(f"locked-test 결과가 이미 존재: {output}")
    weights = Path(winner["weights"])
    if not weights.is_file() or sha256_file(weights) != winner["weights_sha256"]:
        raise LockedTestError("winner weights SHA-256 불일치")


def _locked_test_windows():
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(V2_DIR, manifest)
    return load_windows("test", traj_dir=V2_DIR, manifest_path=V2_MANIFEST)
```

CLI는 winner JSON이 가리키는 seed-0 재검증 weights를 `build_candidate()`에 로드하고
`evaluate_windows()`를 호출한다. 결과는
`os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL)`로 새 파일일 때만 쓰며 candidate commit,
weights SHA-256, manifest SHA-256, metrics와 CI를 포함한다. test 정답 기반 요약은 이 파일에 처음 기록한다.

- [ ] **Step 4: gate 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_test_gate.py tests/test_autoresearch_contract.py -q`

Expected: PASS.

- [ ] **Step 5: locked-test 장치를 커밋한다**

```powershell
git add -- train/eval_autoresearch_locked_test.py tests/test_autoresearch_test_gate.py
git commit -m "feat: guard one-time autoresearch test evaluation"
```

### Task 7: 평가 계약 전체 검증

**Files:**
- Verify only.

**Interfaces:**
- Consumes: Task 1~6.
- Produces: 자동 실험 runner가 사용할 고정 계약.

- [ ] **Step 1: 관련 테스트를 묶어서 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_contract.py tests/test_autoresearch_candidate.py tests/test_autoresearch_test_gate.py tests/test_risk.py tests/test_bootstrap.py -q`

Expected: PASS.

- [ ] **Step 2: 전체 회귀 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 3: test 결과가 아직 없는지 확인한다**

```powershell
if (Test-Path 'docs/chanwoo/results/autoresearch-final.json') { throw '자동 탐색 전에 locked-test 결과가 생성됨' }
git diff --check
git status --short --branch
```

Expected: final JSON 없음, 추적 파일 변경 없음.
