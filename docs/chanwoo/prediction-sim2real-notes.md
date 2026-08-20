# 궤적 예측 — 다양성·sim-to-real 조사 (이슈 #2 후속)

> 2026-08-20. "Trajectron++로 확장" 요청에서 출발해, **짓기 전에 재는** 방식으로 스파이크
> 3개를 돌린 결론. 데이터·수치는 재현 가능(스파이크 스크립트 `train/spike_*.py`).

## 한 줄 결론

**Trajectron++의 간판 기능(사회적 풀링)은 이 도메인에 값어치 없다. 진짜 지렛대는
(1) 레이아웃 다양성과 (2) 검출 노이즈 강건성이었고, 둘 다 반영했다.** 모델은 경량
멀티모달 LSTM(MTP) 유지 — Trajectron++의 실용 핵심(멀티모달+불확실성)은 이미 갖췄다.

## 스파이크 결과

### 1. 사회적 풀링 (`spike_social.py`) — 무효
이웃(가장 가까운 사람 상대위치) 특징 추가 → val ADE 개선 **~0**(조밀·다양 데이터에선 오히려 -5%).
동선이 직무 사이클(WORKFLOW) 지배라 상호작용 신호가 적다. → 사회적 attention 안 짓는다.

### 2. 교차 레이아웃 일반화 (`spike_crosslayout.py`) — 다양성 필요
island만 학습 → 못 본 legacy 평가: ML ADE/FDE **0.826/1.547**.
legacy 포함 학습 → legacy: **0.630/1.126**. **격차 +31%.**
→ 배포할 레이아웃은 학습에 넣어야 한다. 에이전트 중심 정규화로도 새 배치는 못 메꾼다.
→ 다양성 데이터 수집(2단계 하네스에 자기기술 메타 + 인원 확대)으로 대응. island+legacy 확보.

### 3. 검출 노이즈 갭 (`spike_noise.py`) — 큰 갭, 증강으로 절반 회복
깨끗 GT로 학습한 모델을 노이즈 낀 관측으로 평가(실제 나디르→YOLO→추적 흉내):

| | ADE/FDE | 깨끗 대비 |
|---|---|---|
| 깨끗 val | 0.751/1.428 | 기준 |
| 노이즈 0.06m | 1.022/1.751 | **+36%** |
| 노이즈 0.12m | 1.257/2.038 | **+67%** |

→ 그대로 실배포하면 크게 나빠진다. **노이즈 증강 학습**으로 대응(관측에 σ=0.06 가우시안 사본).

## 반영한 것 (프로덕션)

- **레이아웃 다양성**: `sim.html` trajGeom() — scene에 실제 layout·half·room·robot·mPerAU 기록,
  scene_id에 설정 태그. jobSets 2~4인. 설정별(`?layout=`·`?half=`) 페이지 열어 수집 → 통합.
  현재 데이터셋: island 40 + island_h58 40 + legacy 40 = 120 scene.
- **노이즈 증강**: `train/train_traj_predictor.py` — train에 σ=0.06 노이즈 사본 2개(정규화 전),
  val은 깨끗 유지. 재학습 결과:
  - 깨끗 val 유지: 최빈 **0.748/1.420** (증강 전 0.751/1.428)
  - 노이즈 강건성 개선: 0.06에서 **+14%**(증강 전 +36%), 0.12에서 **+32%**(전 +67%) — 악화 반토막.

## 안 한 것 / 다음

- **사회적 Trajectron++·CVAE** — 지표 이득 없어 보류(원하면 CVAE는 방법론적 완결용으로만 가능).
- **완전한 sim-to-real (다음 세션)**:
  - 실사 오버헤드 트랙으로 평가/파인튜닝(옆 레포 `cooking-robot-safety/trajectory/overhead_lstm.py`가 실사 YOLO 클립을 씀).
  - 렌더 프레임을 `detect_server`로 실제 검출·추적해 나온 트랙(드롭·ID스위치 포함)으로 학습 —
    가우시안 근사보다 실제 노이즈 특성에 맞음.
  - 더 많은 레이아웃(현재 island·legacy 2종뿐 — 새 배치는 지오메트리 제작 필요).

## 스파이크 스크립트 (재현)

- `train/spike_social.py` · `train/spike_crosslayout.py` · `train/spike_noise.py`
- 실행: `uv run python train/spike_*.py`. 데이터는 `dataset/trajectories/`(gitignore).
