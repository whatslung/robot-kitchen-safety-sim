# train/val/test 완전 분리 + scene-level CI — 설계 (감사 P0-1)

> 작성 2026-08-24 · 담당 chanwoo · 상태: **설계 — 승인 대기**
> 근거: [handoff/2026-08-22-motion-quality-audit.md](../handoff/2026-08-22-motion-quality-audit.md) §P0-1

## 0. 목표와 현재 한계

북극성 한 문장:

> **모델·운영점을 val에서만 고르고 test는 마지막 한 번만 봐서, 보고 수치가 과적합이 아님을 보장한다.**

**현재 한계**
- `trajectory/sim_traj.py`의 `is_val(seed) = seed%5==0` — **val 하나뿐**(train/val), test 없음.
  같은 val에서 모델 비교와 운영점(τ) 선택이 함께 일어나 **운영점 과적합**을 피할 수 없다(감사 §P0-1).
- **교차 레이아웃 누수(신규 발견)**: 한 seed가 레이아웃마다 한 번씩 있다(`island_seed1`·`island_h58_seed1`·
  `legacy_seed1`). 실측 — 같은 seed의 `island`/`island_h58`는 **목표가 동일**(예: seed1 goal
  (-1.525,0.225)), 시작도 근접. 즉 같은 seed는 레이아웃만 다른 **근사 중복**이다. 파일(scene) 단위로
  나누면 train과 test에 근사 중복이 갈려 누수가 된다.

## 1. 범위 결정 (확정 — 사용자 승인: 재학습 O, 70/15/15 계층화)

- **분할 단위 = seed**(scene 단위의 더 엄격한 형태). 한 seed의 **모든 레이아웃 변형을 같은 split**에
  둬 교차 레이아웃 누수를 원천 차단한다. seed마다 3개 레이아웃을 다 가지므로 **레이아웃 계층화가 자동**.
- **비율 70/15/15**: 40 seed → **28 train / 6 val / 6 test** → ×3 레이아웃 = **84 / 18 / 18 scene**.
  `real_test_sample.json`은 sim 분할에서 제외하고 별도 보관(실사 예시).
- **LSTM 재학습**: 새 train split으로 다시 학습해 test가 진짜 held-out이 되게 한다(누수 없는 test 수치).
- **선택 프로토콜**: 모델·τ·k는 **val에서만** 선택하고 로그로 남긴다. **test는 마지막 1회**만 평가.
- **scene-level bootstrap 95% CI**: stride-1 중첩 윈도우가 아니라 **scene(=seed·레이아웃) 단위**로
  리샘플해 ADE/FDE·safety recall/precision의 불확실성을 낸다.

## 2. split 정의 · manifest

- **생성기** `train/make_traj_split.py`: sim scene 파일 열거 → seed 목록 추출 → **고정 시드(=0)로
  결정적 셔플** → 28/6/6 seed 배정 → 각 split에 그 seed의 전 레이아웃 파일 수집.
- **manifest** `dataset/trajectories/split_manifest.json`(커밋):
  ```json
  {"meta": {"unit":"seed","ratios":[0.7,0.15,0.15],"shuffle_seed":0,
            "counts":{"train":84,"val":18,"test":18},"seeds":{"train":[…28],"val":[…6],"test":[…6]}},
   "train": ["island_seed2_0001.json", …], "val": […], "test": […]}
  ```
- **결정성**: 같은 명령 → 같은 manifest(정렬된 seed 목록 + 고정 셔플 시드). 재생성 시 diff 없음.

## 3. 컴포넌트

### 3-1. `trajectory/sim_traj.py` (로더 수정)
- `load_windows(split)`가 `seed%5` 대신 **manifest 멤버십**으로 필터. `split ∈ {train,val,test,all}`.
- manifest 없으면 명확히 오류(“먼저 make_traj_split 실행”) — silent 폴백 금지.
- `is_val`는 하위호환 위해 남기되 내부적으로 manifest 기반(또는 deprecated 주석). 윈도우 생성 로직 불변.

### 3-2. `trajectory/bootstrap.py` (신규, 순수 — pytest)
- `scene_bootstrap_ci(per_scene, statistic, B=2000, alpha=0.05, seed=0) -> (point, lo, hi)`:
  scene 키별로 묶은 값(또는 카운트)을 **scene 단위 복원추출** B회 → 통계 재계산 → 백분위 CI.
- ADE/FDE: scene별 윈도우 평균을 값으로. recall/precision: scene별 [TP,FP,FN]를 합산 재계산.
- 결정적(고정 seed). numpy만.

### 3-3. `train/make_traj_split.py` (신규)
manifest 생성 + 요약 출력(각 split seed/scene 수, 레이아웃 분포).

### 3-4. `train/train_traj_predictor.py` (재학습)
`load_windows("train")`가 새 split을 자동 반영 — 코드 변경 최소. 새 train으로 재학습해 `model.pt` 갱신.
학습 결정성(가능하면 고정 시드)·재현 명령 기록.

### 3-5. `train/eval_traj_safety.py`·`eval_traj_baselines.py` (평가)
- **val**: 운영점 τ 스윕·모델/k 비교 → 선택 로그(JSON).
- **test**: 선택된 설정으로 **1회** 평가 + scene-level CI.
- 선택 로그와 최종 test 결과를 **분리 출력**(감사 완료조건).

## 4. scene-level bootstrap CI 방법

- 윈도우는 stride-1이라 강한 상관 → 윈도우 단위 CI는 과소추정. **scene(seed·레이아웃) 단위**로 리샘플.
- ADE/FDE: 각 scene의 윈도우 평균 → scene 리스트를 B회 복원추출 → 각 회 평균 → 2.5/97.5 백분위.
- safety: 각 scene의 [TP,FP,FN] → 리샘플 합산 → recall/precision 재계산 → 백분위.
- B=2000, 고정 seed로 재현.

## 5. 선택 프로토콜 (누수 방지 핵심)

1. `make_traj_split` → manifest 커밋.
2. LSTM 재학습(train) → `model.pt`.
3. **val**에서 모델(CV·칼만·LSTM) 비교 + τ·k 선택 → `docs/chanwoo/results/oppoint-selection.json`.
4. 선택 고정 후 **test 1회** → ADE/FDE·recall/precision + 95% CI → docs.
5. test는 개발 중 절대 열람/튜닝 금지(문서에 “test 1회” 명시).

## 6. 테스트 (TDD seam)

- **`tests/test_traj_split.py`**: split 결정성(같은 입력→같은 배정)·무중복(train∩val∩test=∅)·
  전수 커버·비율(28/6/6 seed)·레이아웃 계층(각 split에 3레이아웃 존재)·seed 단위 무누수
  (한 seed의 파일이 한 split에만).
- **`tests/test_bootstrap.py`**: 결정성(고정 seed)·point가 전체 통계와 일치·CI가 point를 포함·
  단일 scene이면 폭 0에 수렴·알려진 입력의 대칭성.
- 전체: `uv run --group serve --with pytest python -m pytest tests/ -q`.

## 7. 재현 (단일 흐름)

```
uv run python train/make_traj_split.py                    # manifest
uv run python train/train_traj_predictor.py               # 새 train 재학습 → model.pt
uv run python train/eval_traj_safety.py --split val ...    # 운영점 선택(로그)
uv run python train/eval_traj_safety.py --split test ...   # 최종 1회 + CI
```
정확한 인자·산출 경로는 구현 시 확정, 완료 조건은 “동일 명령으로 split·결과 재생성”.

## 8. 파일 · 완료 정의

**건드릴 파일**: `train/make_traj_split.py`(신규) · `dataset/trajectories/split_manifest.json`(신규 커밋) ·
`trajectory/bootstrap.py`(신규) · `trajectory/sim_traj.py`(로더) · `train/train_traj_predictor.py`(재학습) ·
`train/eval_traj_safety.py`·`eval_traj_baselines.py`(test+CI+선택 분리) ·
`tests/test_traj_split.py`·`tests/test_bootstrap.py`(신규) · `docs/chanwoo/prediction-eval.md`·
`prediction-safety-eval.md`(재생성) · `docs/chanwoo/results/oppoint-selection.json`(선택 로그).

**완료 정의**(감사 §P0-1)
- 동일 명령으로 split·결과 재생성 가능(manifest 커밋, 결정적).
- ADE/FDE·safety recall/precision에 **scene-level 95% CI** 제공.
- **운영점 선택 로그와 최종 test 결과 분리**. test는 1회.
- 순수 함수 pytest 통과, 전체 스위트 green.

## 9. 후속 (이 스펙 밖)

- P0-2 LSTM 수치를 새 test split 기준으로 갱신(교차 참조).
- P0-4 CI에서 split 무결성 테스트를 회귀로 편입.
- real split(실사) 별도 test는 P0-2 확장에서.
