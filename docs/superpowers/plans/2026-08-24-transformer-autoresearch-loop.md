# Transformer Autoresearch Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 고정된 v2 train/validation 계약 안에서 Transformer 후보 하나만 반복 변경하며, 300초 동일 예산의 모든 성공·실패를 추가 전용 로그에 남기고 검증된 승자를 고정한다.

**Architecture:** child worker가 모델 한 개를 학습·validation 평가하고 parent runner가 600초 watchdog, 고정 파일 hash, JSONL 기록을 맡는다. 모델 후보와 학습 hyperparameter는 `trajectory/autoresearch_candidate.py` 한 파일에만 있고 나머지는 lock manifest로 고정한다. 빠른 탐색 뒤 개선 커밋 중 상위 3개를 seed 3개로 재실행해 중앙값 승자를 정한다.

**Tech Stack:** Python 3.11+, PyTorch 2.7+, subprocess, JSONL, SHA-256, pytest, Git, PowerShell, uv

**Spec:** `docs/chanwoo/specs/2026-08-24-transformer-autoresearch-design.md`

## Global Constraints

- 데이터 계획과 평가 계약 계획이 모두 완료되어야 한다.
- 공식 `karpathy/autoresearch` 코드는 복사하거나 import하지 않고 운영 원칙만 적용한다.
- warmup 10 step 뒤 학습 시간은 300초, child 전체 timeout은 600초다.
- 빠른 탐색은 후보 20개 또는 전체 2시간 중 먼저 도달한 시점에 끝낸다.
- 빠른 탐색 seed는 0, 상위 3개 재검증 seed는 0·1·2다.
- 자동 실험 중 수정 허용 파일은 `trajectory/autoresearch_candidate.py` 하나다.
- `training/autoresearch/results.jsonl`은 추가 전용이며 Git에서 무시한다.
- NaN, 무한값, OOM, timeout, child 비정상 종료는 실패 record로 남긴다.
- `git reset --hard`, `git clean`, 다른 worktree 변경, test 평가, push, PR, merge를 금지한다.

---

### Task 1: 시간 예산 학습 루프

**Files:**
- Create: `train/autoresearch_training.py`
- Create: `tests/test_autoresearch_training.py`

**Interfaces:**
- Consumes: `train_for_budget(net, optimizer, loss_fn, x, y, batch_size, seed, budget_seconds, warmup_steps=10, clock=time.perf_counter)`.
- Produces: `TrainingResult(steps, warmup_steps, train_seconds, final_loss)`; NaN/Inf에는 `NonFiniteTrainingError`.

- [ ] **Step 1: warmup 제외와 NaN 실패 테스트를 작성한다**

```python
# tests/test_autoresearch_training.py
import pytest
import torch
from train.autoresearch_training import train_for_budget, NonFiniteTrainingError


def test_training_budget_starts_after_warmup():
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(net.parameters(), lr=.01)
    x = torch.zeros(8, 2); y = torch.zeros(8, 2)
    ticks = iter([0.0, 0.0, .4, .8, 1.2, 1.2])
    result = train_for_budget(net, opt, lambda model, bx, by, progress: ((model(bx) - by) ** 2).mean(),
                              x, y, batch_size=8, seed=0, budget_seconds=1.0,
                              warmup_steps=2, clock=lambda: next(ticks))
    assert result.warmup_steps == 2
    assert result.steps == 3


def test_non_finite_loss_aborts_trial():
    net = torch.nn.Linear(2, 2); opt = torch.optim.SGD(net.parameters(), lr=.01)
    x = torch.zeros(2, 2); y = torch.zeros(2, 2)
    with pytest.raises(NonFiniteTrainingError, match="finite"):
        train_for_budget(net, opt, lambda model, bx, by, progress: model(bx).sum() * float("nan"),
                         x, y, 2, seed=0, budget_seconds=.1, warmup_steps=0)
```

- [ ] **Step 2: 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_training.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 주입 clock과 결정적 batch 순서를 구현한다**

```python
@dataclass(frozen=True)
class TrainingResult:
    steps: int
    warmup_steps: int
    train_seconds: float
    final_loss: float


class NonFiniteTrainingError(RuntimeError):
    pass


def train_for_budget(net, optimizer, loss_fn, x, y, batch_size, seed,
                     budget_seconds=300.0, warmup_steps=10, clock=time.perf_counter):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(len(x), generator=generator, device=x.device)
    cursor = 0

    def step(progress):
        nonlocal order, cursor
        if cursor + batch_size > len(order):
            order = torch.randperm(len(x), generator=generator, device=x.device); cursor = 0
        idx = order[cursor:cursor + batch_size]; cursor += batch_size
        optimizer.zero_grad(); loss = loss_fn(net, x[idx], y[idx], progress)
        if not torch.isfinite(loss): raise NonFiniteTrainingError("loss is not finite")
        loss.backward(); optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(warmup_steps):
        final_loss = step(0.0)
    start = clock(); steps = 0
    while True:
        now = clock()
        if now - start >= budget_seconds: break
        final_loss = step(min(1.0, (now - start) / budget_seconds)); steps += 1
    return TrainingResult(steps, warmup_steps, clock() - start, final_loss)
```

실제 후보의 `loss_fn` adapter는 `(paths, logits, logsig)`를 `mtp_loss(..., y)`로 넘기고, CVAE만
`net.elbo(x, y, beta)["loss"]`를 사용한다.

- [ ] **Step 4: 학습 루프 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_training.py -q`

Expected: PASS.

- [ ] **Step 5: 시간 예산 루프를 커밋한다**

```powershell
git add -- train/autoresearch_training.py tests/test_autoresearch_training.py
git commit -m "feat: add fixed-budget trajectory training loop"
```

### Task 2: 한 모델을 실행하는 child worker

**Files:**
- Create: `train/autoresearch_worker.py`
- Create: `tests/test_autoresearch_worker.py`

**Interfaces:**
- Consumes: `WorkerConfig(model, seed, budget_seconds, output_json, weights_path)`.
- Produces: 원자적으로 교체되는 child result JSON과 weights; model은 `lstm|transformer|cvae|candidate`.

- [ ] **Step 1: builder 선택과 짧은 합성 학습 테스트를 작성한다**

```python
# tests/test_autoresearch_worker.py
import torch
from train.autoresearch_worker import build_model, model_hyperparameters


def test_worker_builds_all_fixed_baselines_and_candidate():
    for name in ("lstm", "transformer", "cvae", "candidate"):
        net = build_model(name)
        assert isinstance(net, torch.nn.Module)


def test_candidate_hyperparameters_come_from_candidate_file():
    hp = model_hyperparameters("candidate")
    assert hp.batch_size > 0 and hp.learning_rate > 0
```

- [ ] **Step 2: worker가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_worker.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 고정 builder와 candidate builder를 명시적으로 연결한다**

```python
@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    weight_decay: float
    batch_size: int


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    seed: int
    budget_seconds: float
    output_json: Path
    weights_path: Path


def build_model(name: str):
    if name == "lstm": return build_net(h=64)
    if name == "transformer": return build_transformer_net(h=64, layers=2, heads=4)
    if name == "cvae": return build_cvae_net(h=64, layers=2, heads=4)
    if name == "candidate": return build_candidate(CONFIG)
    raise ValueError(f"지원하지 않는 model: {name}")


def model_hyperparameters(name: str) -> Hyperparameters:
    if name == "candidate":
        return Hyperparameters(CONFIG.learning_rate, CONFIG.weight_decay, CONFIG.batch_size)
    return Hyperparameters(1e-3, 0.0, 512)


def build_training_arrays(windows, seed):
    rng = np.random.default_rng(seed)
    x_values, y_values = [], []
    for window in windows:
        hist = np.asarray([(x, z) for _, x, z in window.scene.agents[0].history], dtype=float)
        future = [(x, z) for _, x, z in window.gt]
        for observed in (hist, hist + rng.normal(0, .06, hist.shape),
                         hist + rng.normal(0, .06, hist.shape)):
            origin, angle = frame_of(observed)
            x_values.append(to_frame(observed, origin, angle))
            y_values.append(to_frame(future, origin, angle))
    return np.asarray(x_values, np.float32), np.asarray(y_values, np.float32)


def mtp_loss_adapter(net, batch_x, batch_y, progress):
    return mtp_loss(*net(batch_x), batch_y)


def cvae_loss_adapter(net, batch_x, batch_y, progress):
    beta = min(1.0, progress / .5)  # 300초 예산의 앞 절반에서 0→1
    return net.elbo(batch_x, batch_y, beta=beta)["loss"]
```

worker main은 `development_windows("train")`를 `build_training_arrays()`에 넘기고 candidate/LSTM/
Transformer는 `mtp_loss_adapter`, CVAE는 `cvae_loss_adapter`로
`train_for_budget()`을 호출한다. 학습 뒤 `development_windows("val")`만 `evaluate_windows()`에
넘긴다. CPU p95는 CPU에 새로 만든 같은 구조에 state dict를 로드해 재며, 파라미터 수와 함께 result
JSON에 기록한다. `Metrics.f2`는 계산 property이므로 `asdict(metrics)` 결과에 `f2` 키를 명시적으로
추가한다. JSON과 `.pt`는 같은 디렉터리의 임시 파일을 완성한 뒤 `Path.replace()`로 교체한다.

```python
metric_record = asdict(evaluation.metrics)
metric_record["f2"] = evaluation.metrics.f2
record = {"status": "ok", "model": config.model, "seed": config.seed,
          "training": asdict(training_result), "metrics": metric_record,
          "ci": evaluation.ci, "candidate_sha256": candidate_sha256(),
          "weights": str(config.weights_path), "weights_sha256": sha256_file(config.weights_path),
          "environment": {"python": platform.python_version(), "torch": torch.__version__,
                          "cuda": torch.version.cuda, "device": device_name,
                          "cpu_threads": torch.get_num_threads()}}
```

- [ ] **Step 4: 0.05초 합성/fixture worker smoke test를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_worker.py -q`

Expected: PASS; 실제 300초 학습은 실행하지 않는다.

- [ ] **Step 5: worker를 커밋한다**

```powershell
git add -- train/autoresearch_worker.py tests/test_autoresearch_worker.py
git commit -m "feat: train and evaluate one autoresearch model"
```

### Task 3: 추가 전용 JSONL과 600초 watchdog parent

**Files:**
- Create: `train/run_autoresearch_experiment.py`
- Create: `tests/test_autoresearch_runner.py`

**Interfaces:**
- Consumes: `--trial-id`, `--model`, `--seed`, `--budget-seconds`, `--timeout-seconds`, baseline JSON.
- Produces: `training/autoresearch/results.jsonl` 한 줄과 성공 시 trial weights/result.

- [ ] **Step 1: append 보존과 timeout 실패 record 테스트를 작성한다**

```python
# tests/test_autoresearch_runner.py
import json
from train.run_autoresearch_experiment import append_jsonl, classify_child


def test_append_jsonl_preserves_existing_records(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"trial_id": "a", "status": "ok"})
    append_jsonl(path, {"trial_id": "b", "status": "failed"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["trial_id"] for row in rows] == ["a", "b"]


def test_timeout_is_a_failed_record():
    row = classify_child("trial-1", "candidate", seed=0, returncode=None,
                         timed_out=True, child_result=None, stderr="")
    assert row["status"] == "failed" and row["failure"] == "timeout"
```

- [ ] **Step 2: parent 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_runner.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: append와 child 분류를 구현한다**

```python
def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line); stream.flush(); os.fsync(stream.fileno())


def classify_child(trial_id, model, seed, returncode, timed_out, child_result, stderr):
    if timed_out:
        return {"trial_id": trial_id, "model": model, "seed": seed,
                "status": "failed", "failure": "timeout"}
    if returncode != 0:
        reason = "oom" if "out of memory" in stderr.lower() else "child_exit"
        return {"trial_id": trial_id, "model": model, "seed": seed,
                "status": "failed", "failure": reason, "returncode": returncode,
                "stderr_tail": stderr[-4000:]}
    return child_result
```

parent는 `subprocess.run([...autoresearch_worker.py...], timeout=600, capture_output=True, text=True)`를
사용한다. candidate 성공이면 Transformer baseline을 읽어 `evaluate_guards`와 F2 gain을 계산하고
기존 JSONL의 `status=ok, verdict=keep` F2 중 최고값(없으면 baseline F2)과 비교한다. 보호 조건을 모두
통과하고 새 F2가 기존 최고보다 클 때만 `verdict=keep`, 그 밖에는 `revert`다. `f2_gain`은 기준
Transformer 대비 차이로 별도 기록한다. `--rerun`은 `verdict`를 만들지 않고 `rerun=true`만 기록해
빠른 탐색 순위에 섞이지 않게 한다. rerun ID는 `base-rerun-s0` 형식으로 검증하고
`base_trial_id = trial_id.rsplit("-rerun-s", 1)[0]`도 record에 넣는다. 어떤 경로에서도 최종적으로
`append_jsonl()`을 한 번 호출한다. candidate의 `environment`가 Transformer baseline과 다르면
순위를 계산하지 않고 `failure=environment_mismatch`로 기록한다.

- [ ] **Step 4: runner 단위 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_runner.py -q`

Expected: PASS.

- [ ] **Step 5: parent를 커밋한다**

```powershell
git add -- train/run_autoresearch_experiment.py tests/test_autoresearch_runner.py
git commit -m "feat: supervise and record autoresearch trials"
```

### Task 4: 고정 파일 hash lock

**Files:**
- Create: `train/lock_autoresearch_contract.py`
- Modify: `train/run_autoresearch_experiment.py`
- Modify: `tests/test_autoresearch_runner.py`
- Create during setup: `docs/chanwoo/results/autoresearch-contract-lock.json`

**Interfaces:**
- Consumes: 고정 파일 경로 목록.
- Produces: `build_lock(root: Path) -> dict`, `verify_lock(root: Path, lock: dict) -> None`.

- [ ] **Step 1: 고정 파일 변경을 거부하는 테스트를 추가한다**

```python
def test_contract_lock_detects_changed_fixed_file(tmp_path):
    fixed = tmp_path / "fixed.py"; fixed.write_text("A", encoding="utf-8")
    lock = build_lock(tmp_path, files=[Path("fixed.py")])
    fixed.write_text("B", encoding="utf-8")
    with pytest.raises(ContractLockError, match="fixed.py"):
        verify_lock(tmp_path, lock)
```

- [ ] **Step 2: lock 함수가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_runner.py::test_contract_lock_detects_changed_fixed_file -q`

Expected: FAIL with import error.

- [ ] **Step 3: lock 파일 목록과 SHA-256 검사를 구현한다**

```python
FIXED_FILES = (
    "docs/chanwoo/results/traj-v2-manifest.json",
    "trajectory/traj_v2.py", "trajectory/sim_traj.py", "trajectory/risk.py",
    "trajectory/evaluator.py", "trajectory/bootstrap.py",
    "trajectory/learned_predictor.py", "train/autoresearch_contract.py",
    "train/autoresearch_training.py", "train/autoresearch_worker.py",
    "train/run_autoresearch_experiment.py",
    "tests/test_autoresearch_candidate.py", "tests/test_autoresearch_contract.py",
)


class ContractLockError(RuntimeError):
    pass


def build_lock(root, files=FIXED_FILES):
    return {"schema": 1, "files": {str(path): sha256_file(root / path) for path in files}}


def verify_lock(root, lock):
    mismatches = [path for path, expected in lock["files"].items()
                  if not (root / path).is_file() or sha256_file(root / path) != expected]
    if mismatches: raise ContractLockError("고정 계약 변경: " + ", ".join(mismatches))
```

candidate 파일, weights, JSONL은 lock에서 제외한다. parent는 child 실행 전에 lock을 검증하고 불일치도
`failure=contract_lock` record로 남긴다.

- [ ] **Step 4: lock을 생성하고 runner 테스트를 실행한다**

```powershell
uv run --group serve python train/lock_autoresearch_contract.py --write
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_runner.py -q
```

Expected: lock JSON 생성, PASS.

- [ ] **Step 5: lock 구현과 최초 lock을 커밋한다**

```powershell
git add -- train/lock_autoresearch_contract.py train/run_autoresearch_experiment.py tests/test_autoresearch_runner.py docs/chanwoo/results/autoresearch-contract-lock.json
git commit -m "feat: lock autoresearch evaluation contract"
```

### Task 5: 세 기준 모델 측정

**Files:**
- Create: `train/run_autoresearch_baselines.py`
- Create: `tests/test_autoresearch_baselines.py`
- Create locally, ignored: `training/autoresearch/baselines.json`
- Create locally, ignored: `training/autoresearch/baseline-*.pt`

**Interfaces:**
- Consumes: child worker의 `lstm`, `transformer`, `cvae` model과 seed 0.
- Produces: 세 기준의 validation metrics; 보호 조건 기준은 `transformer` 행.

- [ ] **Step 1: 세 모델 완전성 테스트를 작성한다**

```python
# tests/test_autoresearch_baselines.py
import pytest
from train.run_autoresearch_baselines import validate_baselines, BaselineError


def test_baselines_require_all_three_models():
    with pytest.raises(BaselineError, match="cvae"):
        validate_baselines({"lstm": {}, "transformer": {}})


def test_transformer_is_guard_reference():
    rows = {name: {"metrics": {"f2": i}} for i, name in enumerate(("lstm", "transformer", "cvae"))}
    assert validate_baselines(rows)["guard_reference"] == "transformer"
```

- [ ] **Step 2: baseline 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_baselines.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: baseline orchestration과 원자적 JSON 쓰기를 구현한다**

```python
MODELS = ("lstm", "transformer", "cvae")


def validate_baselines(rows):
    missing = [name for name in MODELS if name not in rows]
    if missing: raise BaselineError("기준 모델 누락: " + ", ".join(missing))
    return {"guard_reference": "transformer", "models": rows}
```

main은 model마다 parent runner를 `baseline-{model}-seed0`으로 실행하고 성공 record/result JSON을 읽는다.
세 모델이 모두 성공했을 때만 임시 `baselines.json.tmp`를 최종 `baselines.json`으로 교체한다.

- [ ] **Step 4: orchestration 단위 테스트를 실행한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_baselines.py -q`

Expected: PASS.

- [ ] **Step 5: baseline 코드를 커밋한다**

```powershell
git add -- train/run_autoresearch_baselines.py tests/test_autoresearch_baselines.py
git commit -m "feat: benchmark trajectory autoresearch baselines"
```

- [ ] **Step 6: 세 기준을 각각 300초로 실제 측정한다**

Run: `uv run --group serve python train/run_autoresearch_baselines.py --budget-seconds 300 --timeout-seconds 600`

Expected: LSTM, Transformer, CVAE 모두 `status=ok`, `baselines.json` 생성. 실패하면 탐색을 시작하지 않고 원인을 고친 뒤 세 개 모두 다시 잰다.

### Task 6: 에이전트 작업 규칙과 짧은 end-to-end 연습

**Files:**
- Create: `docs/chanwoo/autoresearch-program.md`
- Verify locally: `training/autoresearch/results.jsonl`

**Interfaces:**
- Consumes: candidate seam, runner, contract lock, baseline JSON.
- Produces: 한 시도당 허용 행동, keep/revert 판단, commit 규칙.

- [ ] **Step 1: 프로그램 문서를 실제 명령과 함께 작성한다**

문서에는 아래 규칙을 그대로 포함한다.

```text
수정 허용: trajectory/autoresearch_candidate.py 하나
수정 금지: 데이터, manifest, loss, evaluator, guard, worker, runner, lock, test gate
한 시도: apply_patch 후보 변경 -> 관련 shape test -> run_autoresearch_experiment.py
keep: record의 status=ok, guards.passed=true, f2_gain>0, 현재 최고 F2 초과
revert: 그 밖의 모든 결과; candidate 파일만 마지막 keep 커밋으로 복원
금지: test 실행, results.jsonl 수정/삭제, reset --hard, clean, 다른 worktree 조작
```

실행 예시는 다음과 같이 고정한다.

```powershell
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_candidate.py -q
uv run --group serve python train/run_autoresearch_experiment.py --trial-id trial-001 --model candidate --seed 0 --budget-seconds 300 --timeout-seconds 600
```

- [ ] **Step 2: 2초 smoke trial을 실행한다**

Run: `uv run --group serve python train/run_autoresearch_experiment.py --trial-id smoke-001 --model candidate --seed 0 --budget-seconds 2 --timeout-seconds 60 --smoke`

Expected: child 성공, JSONL 한 줄 추가, candidate weights/result 생성, test 파일 접근 없음.

- [ ] **Step 3: 실패 smoke도 record되는지 실행한다**

Run: `uv run --group serve python train/run_autoresearch_experiment.py --trial-id smoke-timeout --model candidate --seed 0 --budget-seconds 30 --timeout-seconds 1 --smoke`

Expected: `failure=timeout` 한 줄 추가, 이전 smoke 줄 보존.

- [ ] **Step 4: 프로그램 문서만 커밋한다**

```powershell
git add -- docs/chanwoo/autoresearch-program.md
git commit -m "docs: define trajectory autoresearch agent rules"
```

### Task 7: 최대 20개 Transformer 빠른 탐색

**Files:**
- Modify repeatedly: `trajectory/autoresearch_candidate.py`
- Append locally: `training/autoresearch/results.jsonl`
- Commit repeatedly: keep 후보만.

**Interfaces:**
- Consumes: 기준 Transformer metrics와 아래 유한 탐색 공간.
- Produces: 최대 20개 validation record와 단조 개선 keep 커밋.

- [ ] **Step 1: 탐색 시작 시각과 기준 commit을 기록한다**

```powershell
$searchStart = Get-Date
$baselineCommit = git rev-parse HEAD
$baselineCandidateHash = (Get-FileHash -LiteralPath 'trajectory/autoresearch_candidate.py' -Algorithm SHA256).Hash
```

- [ ] **Step 2: 아래 값 안에서 한 번에 한두 변수만 바꾼다**

```text
hidden: 48, 64, 80, 96
layers: 1, 2, 3, 4
heads: 2, 4, 8 (hidden으로 나누어지는 경우만)
ff_ratio: 2, 3, 4
dropout: 0.0, 0.05, 0.1
norm_first: false, true
pooling: last, mean
learning_rate: 0.0003, 0.0006, 0.001, 0.002
weight_decay: 0.0, 0.0001, 0.001, 0.01
batch_size: 256, 512, 1024
```

먼저 pre-norm/pooling, 다음 width/depth/head, 다음 feed-forward/dropout, 마지막 optimizer/batch 순서로
탐색한다. 이전 최고 설정에서 출발하고 같은 조합은 후보 file SHA-256으로 건너뛴다.

유효 trial 6개 연속으로 새 keep가 없을 때만 "정체"로 판정한다. 정체 뒤 남은 20개 예산 안에서는
candidate 파일에 아래 작은 gate를 추가한 `gated_residual: false|true`만 비교한다. 이는 기존 token과
encoder 출력의 혼합 비율을 학습하는 잔차 연결이며 강화학습용 GTrXL 전체 구조가 아니다.

```python
class GatedResidual(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.new_gate = nn.Linear(hidden, hidden)
        self.old_gate = nn.Linear(hidden, hidden, bias=False)
        nn.init.constant_(self.new_gate.bias, -2.0)  # 시작은 기존 표현을 더 보존

    def forward(self, old, new):
        gate = torch.sigmoid(self.new_gate(new) + self.old_gate(old))
        return gate * new + (1.0 - gate) * old
```

`gated_residual=true`일 때 `old = inp(x)+pos`, `new = enc(old)`, `z = gate(old,new)`를 pooling한다.

- [ ] **Step 3: 각 후보의 shape test와 300초 trial을 실행한다**

```powershell
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_candidate.py -q
$trialId = 'trial-001-pre-norm'
uv run --group serve python train/run_autoresearch_experiment.py --trial-id $trialId --model candidate --seed 0 --budget-seconds 300 --timeout-seconds 600
```

다음 실행에서는 `$trialId = 'trial-002-mean-pool'`처럼 순번과 변경점을 함께 올린다.

- [ ] **Step 4: keep 또는 후보 파일 하나만 revert한다**

keep record이면:

```powershell
git add -- trajectory/autoresearch_candidate.py
git commit -m "experiment: keep $trialId transformer candidate"
```

revert record이면:

```powershell
git restore --source=HEAD -- trajectory/autoresearch_candidate.py
```

`results.jsonl`은 두 경우 모두 수정하거나 stage하지 않는다.

- [ ] **Step 5: 20개 또는 2시간에서 즉시 종료한다**

각 trial 전 `(Get-Date) - $searchStart`를 확인한다. 완료 trial이 20개이거나 경과 시간이 2시간
이상이면 새 후보를 시작하지 않는다. OOM/timeout도 20개 안에 포함한다.
유효 Transformer 후보 6개 연속이 모두 CPU p95 보호 조건만 실패하면 탐색을 중단하고, 이 계획의
범위를 넓히지 않은 채 LSTM 구조 탐색을 별도 설계해야 한다고 보고한다.

### Task 8: 상위 3개를 seed 0·1·2로 재검증하고 승자 고정

**Files:**
- Create: `train/select_autoresearch_winner.py`
- Create: `tests/test_autoresearch_selection.py`
- Temporarily restore: `trajectory/autoresearch_candidate.py`
- Append locally: `training/autoresearch/results.jsonl`
- Create locally, ignored: `training/autoresearch/winner.json`
- Commit: 최종 승자 candidate.

**Interfaces:**
- Consumes: 보호 조건을 통과한 keep 커밋과 빠른 탐색 F2.
- Produces: `top_candidates(rows, commits, limit=3)`, `select_winner(rows, commits)`, 후보별 3-seed 중앙값, 승자.

- [ ] **Step 1: 상위 후보와 3-seed 중앙값 선택 테스트를 작성한다**

```python
# tests/test_autoresearch_selection.py
import pytest
from train.select_autoresearch_winner import top_candidates, select_winner, SelectionError


def _fast(trial, f2):
    return {"trial_id": trial, "status": "ok", "verdict": "keep", "rerun": False,
            "metrics": {"f2": f2}}


def _rerun(trial, seed, f2, recall=.8, fde16=.2, latency=2.0, passed=True):
    return {"trial_id": f"{trial}-rerun-s{seed}", "base_trial_id": trial,
            "status": "ok", "rerun": True, "seed": seed, "guards": {"passed": passed},
            "metrics": {"f2": f2, "recall": recall, "fde16": fde16, "cpu_p95_ms": latency},
            "weights": f"training/{trial}-s{seed}.pt", "weights_sha256": f"sha-{trial}-{seed}"}


def test_top_candidates_join_keep_commits_and_sort_f2():
    rows = [_fast("trial-a", .71), _fast("trial-b", .75)]
    top = top_candidates(rows, {"trial-a": "aaa", "trial-b": "bbb"}, limit=2)
    assert [row["trial_id"] for row in top] == ["trial-b", "trial-a"]
    assert top[0]["commit"] == "bbb"


def test_winner_uses_three_seed_median_and_all_seed_guards():
    rows = [_rerun("baseline-transformer", 0, .70), _rerun("baseline-transformer", 1, .70),
            _rerun("baseline-transformer", 2, .70),
            _rerun("trial-a", 0, .70), _rerun("trial-a", 1, .80), _rerun("trial-a", 2, .90),
            _rerun("trial-b", 0, .79), _rerun("trial-b", 1, .79), _rerun("trial-b", 2, .79)]
    winner = select_winner(rows, {"baseline-transformer": "base", "trial-a": "aaa", "trial-b": "bbb"})
    assert winner["trial_id"] == "trial-a" and winner["median"]["f2"] == .80
    assert winner["f2_gain"] == pytest.approx(.10) and winner["minimum_success"]
    assert winner["source_commit"] == "aaa"
    assert winner["weights"].endswith("trial-a-s0.pt")


def test_failed_or_guard_failing_seed_falls_back_to_baseline():
    rows = [_rerun("baseline-transformer", 0, .7), _rerun("baseline-transformer", 1, .7),
            _rerun("baseline-transformer", 2, .7),
            _rerun("trial-a", 0, .9), _rerun("trial-a", 1, .9, passed=False), _rerun("trial-a", 2, .9)]
    winner = select_winner(rows, {"baseline-transformer": "base", "trial-a": "aaa"})
    assert winner["trial_id"] == "baseline-transformer"
    assert not winner["minimum_success"] and winner["source_commit"] == "base"
```

- [ ] **Step 2: selector가 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_selection.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 결정적 top/median 선택기를 구현한다**

```python
# train/select_autoresearch_winner.py 핵심 함수
from statistics import median


class SelectionError(RuntimeError):
    pass


def top_candidates(rows, commits, limit=3):
    candidates = []
    for row in rows:
        if row.get("status") != "ok" or row.get("verdict") != "keep" or row.get("rerun"):
            continue
        trial_id = row["trial_id"]
        if trial_id not in commits: raise SelectionError(f"keep commit 누락: {trial_id}")
        candidates.append({"trial_id": trial_id, "commit": commits[trial_id],
                           "f2": float(row["metrics"]["f2"])})
    return sorted(candidates, key=lambda row: (-row["f2"], row["trial_id"]))[:limit]


def select_winner(rows, commits):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("rerun"):
            grouped[row["base_trial_id"]].append(row)
    eligible = []
    for trial_id, runs in grouped.items():
        seeds = {run.get("seed") for run in runs if run.get("status") == "ok"}
        if seeds != {0, 1, 2} or any(not run.get("guards", {}).get("passed") for run in runs):
            continue
        med = {name: median(float(run["metrics"][name]) for run in runs)
               for name in ("f2", "recall", "fde16", "cpu_p95_ms")}
        eligible.append({"trial_id": trial_id, "commit": commits[trial_id], "median": med})
    baseline = next((row for row in eligible if row["trial_id"] == "baseline-transformer"), None)
    candidates = [row for row in eligible if row["trial_id"] != "baseline-transformer"]
    if baseline is None: raise SelectionError("3-seed Transformer baseline 누락")
    winner = (max(candidates, key=lambda row: (row["median"]["f2"], row["median"]["recall"],
                                                -row["median"]["fde16"], -row["median"]["cpu_p95_ms"],
                                                row["trial_id"]))
              if candidates else dict(baseline))
    winner["baseline_median"] = baseline["median"]
    winner["f2_gain"] = winner["median"]["f2"] - baseline["median"]["f2"]
    winner["minimum_success"] = winner["f2_gain"] >= .01
    winner["source_commit"] = winner["commit"] if winner["minimum_success"] else baseline["commit"]
    selected_trial = winner["trial_id"] if winner["minimum_success"] else "baseline-transformer"
    seed_zero = next(run for run in grouped[selected_trial] if run["seed"] == 0)
    winner["weights"] = seed_zero["weights"]
    winner["weights_sha256"] = seed_zero["weights_sha256"]
    return winner
```

CLI는 Git의 `experiment: keep $trialId transformer candidate` 형식인 commit 메시지와 최초 candidate
commit을 읽어 trial→commit map을 만든다. `--print-top`은 keep JSON 배열을, `--print-baseline`은
`{"trial_id":"baseline-transformer","commit":"..."}`를 stdout에 쓴다. `--write-winner`는
배타적으로 `training/autoresearch/winner.json`을 쓴다. 승자 후보를 새 HEAD로 커밋한 뒤
`--finalize-commit $selectionCommit`이 winner JSON의 `selected_commit`을 한 번 추가한다.

- [ ] **Step 4: selector 테스트를 실행하고 커밋한다**

```powershell
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_selection.py -q
git add -- train/select_autoresearch_winner.py tests/test_autoresearch_selection.py
git commit -m "feat: select autoresearch candidates across seeds"
```

- [ ] **Step 5: keep record를 F2 내림차순으로 읽고 상위 세 commit을 정한다**

`trial_id`, candidate SHA-256, keep commit SHA를 표로 만든다. keep가 3개 미만이면 존재하는 keep만
사용한다. rejected 후보는 복원 가능한 commit이 없으므로 포함하지 않는다. 기준 Transformer는
후보 세 개와 별도로 항상 3-seed 재검증한다.

```powershell
$topCandidates = @(uv run --group serve python train/select_autoresearch_winner.py --print-top |
  ConvertFrom-Json)
$baselineCandidate = uv run --group serve python train/select_autoresearch_winner.py --print-baseline |
  ConvertFrom-Json
$rerunCandidates = @($topCandidates) + @($baselineCandidate)
```

- [ ] **Step 6: 각 commit의 후보 파일로 seed 0·1·2를 실행한다**

```powershell
foreach ($candidate in $rerunCandidates) {
  git restore --source=$candidate.commit -- trajectory/autoresearch_candidate.py
  if ($LASTEXITCODE -ne 0) { throw "candidate 복원 실패: $($candidate.commit)" }
  foreach ($seed in 0,1,2) {
    $rerunId = "$($candidate.trial_id)-rerun-s$seed"
    uv run --group serve python train/run_autoresearch_experiment.py --trial-id $rerunId --model candidate --seed $seed --budget-seconds 300 --timeout-seconds 600 --rerun
    if ($LASTEXITCODE -ne 0) { throw "rerun 명령 실패: $rerunId" }
  }
}
```

세 줄 중 하나라도 실패하면 해당 후보는 승자 자격이 없다.

- [ ] **Step 7: 중앙값과 동률 규칙으로 승자를 계산한다**

Run: `uv run --group serve python train/select_autoresearch_winner.py --write-winner`

Expected: 후보별 세 record의 F2, recall, FDE16, CPU p95 중앙값을 계산해
`training/autoresearch/winner.json`을 만든다. 네 보호 조건은 각 seed에서 모두 통과해야 한다.
`minimum_success`는 승자 F2 중앙값이 Transformer baseline 중앙값보다 0.01 이상 높을 때만 true다.

- [ ] **Step 8: 승자 파일을 복원하고 명시적 승자 커밋을 만든다**

```powershell
$winner = Get-Content -LiteralPath 'training/autoresearch/winner.json' -Raw | ConvertFrom-Json
$winnerCommit = $winner.source_commit
git restore --source=$winnerCommit -- trajectory/autoresearch_candidate.py
git add -- trajectory/autoresearch_candidate.py
git commit --allow-empty -m "experiment: select transformer autoresearch winner"
$selectionCommit = git rev-parse HEAD
uv run --group serve python train/select_autoresearch_winner.py --finalize-commit $selectionCommit
```

Expected: `minimum_success=true`면 HEAD의 candidate SHA-256이 승자 record와 같고,
false면 기준 Transformer candidate가 유지된다.

### Task 9: validation 보고서와 locked-test 전 정지점

**Files:**
- Create: `train/summarize_autoresearch.py`
- Create: `tests/test_autoresearch_summary.py`
- Create: `docs/chanwoo/results/autoresearch-validation-summary.json`
- Create: `docs/chanwoo/autoresearch-validation-report.md`

**Interfaces:**
- Consumes: JSONL, baseline JSON, winner commit/SHA.
- Produces: 실패 포함 전체 시도 수, keep 이력, 상위 3개 3-seed 중앙값, 승자, 보호 조건; test 값 없음.

- [ ] **Step 1: 요약이 실패 시도와 test 부재를 보존하는 테스트를 작성한다**

```python
# tests/test_autoresearch_summary.py
from train.summarize_autoresearch import summarize


def test_summary_counts_failures_and_omits_test_metrics():
    rows = [{"trial_id": "a", "status": "ok", "verdict": "keep"},
            {"trial_id": "b", "status": "failed", "failure": "timeout"}]
    out = summarize(rows, winner={"trial_id": "a", "commit": "abc"})
    assert out["counts"] == {"total": 2, "ok": 1, "failed": 1, "keep": 1}
    assert "test" not in out
```

- [ ] **Step 2: 요약 모듈이 없어 실패하는지 확인한다**

Run: `uv run --group serve --with pytest python -m pytest tests/test_autoresearch_summary.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 결정적 JSON/Markdown 요약기를 구현한다**

```python
def summarize(rows, winner, baselines=None):
    failed_by = Counter(row.get("failure") for row in rows if row.get("status") == "failed")
    counts = {"total": len(rows),
              "ok": sum(row.get("status") == "ok" for row in rows),
              "failed": sum(row.get("status") == "failed" for row in rows),
              "keep": sum(row.get("verdict") == "keep" for row in rows)}
    return {"schema": 1, "counts": counts,
            "failure_counts": {key: failed_by[key] for key in sorted(failed_by) if key is not None},
            "baselines": baselines or {},
            "keep_trials": [row for row in rows if row.get("verdict") == "keep"],
            "reruns": [row for row in rows if row.get("rerun")],
            "winner": winner,
            "boundary": "validation-only simulator trajectory result; locked test not run"}
```

CLI는 JSONL의 record 순서를 유지하고 위 결과를 `sort_keys=True, indent=2`로 쓴다. Markdown에는
baseline 표, keep 이력, 실패 사유, rerun 중앙값, winner 보호 조건과 다음 경계를 적는다.

```text
이 결과는 v2 validation에서 고른 시뮬레이터 궤적 예측 성능이다.
locked test는 아직 실행하지 않았고 실제 급식실 안전 성능을 뜻하지 않는다.
```

- [ ] **Step 4: 요약 테스트와 전체 테스트를 실행한다**

```powershell
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_summary.py -q
uv run --group serve --with pytest python -m pytest tests/ -q
```

Expected: PASS.

- [ ] **Step 5: 실제 validation 요약을 생성하고 커밋한다**

```powershell
uv run --group serve python train/summarize_autoresearch.py
git add -- train/summarize_autoresearch.py tests/test_autoresearch_summary.py docs/chanwoo/results/autoresearch-validation-summary.json docs/chanwoo/autoresearch-validation-report.md
git commit -m "report: summarize transformer autoresearch validation"
```

- [ ] **Step 6: locked test 직전 상태를 검증하고 사용자 승인을 요청한다**

```powershell
if (Test-Path 'docs/chanwoo/results/autoresearch-final.json') { throw '승인 전 locked test가 실행됨' }
git diff --check
git status --short --branch
git log -1 --oneline
```

Expected: clean worktree, final JSON 없음. 이 시점에서 validation 승자와 보호 조건 결과를 사용자에게
보고한다. `training/autoresearch/winner.json`의 `minimum_success=true`일 때만
`eval_autoresearch_locked_test.py` 한 번 실행 승인을 별도로 받는다. false면 최소 개선 목표를 못
달성했다고 보고하고 locked test를 열지 않는다.
