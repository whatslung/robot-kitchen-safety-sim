# 베이스라인 ADE/FDE — 설계 (확정)

> 2026-08-19 브레인스토밍 확정. 이슈 #2의 **3단계**. 2단계에서 모은 궤적
> (`dataset/trajectories/*.json`, 40 scene·13,500 사람-스텝) 위에서 학습 없는 예측
> 베이스라인 3종의 ADE/FDE를 재어, 4단계 학습형 예측기의 **비교 기준선**을 만든다.

## 왜 (베이스라인이 먼저인 이유)

- ADE/FDE 수치는 홀로는 의미가 없다 — 등속·칼만·스테이션 휴리스틱과 나란히 놓여야
  4단계 학습형이 "이겼는지"를 말할 수 있다. 이슈 #2 완료 조건이 **ADE/FDE 3자 비교표**다.
- 데이터 위생 점검: 등속만으로 오차가 거의 0이면 데이터가 너무 쉬운 것(다들 정지) →
  학습 투자 전에 걸러낸다.

## 결정 (브레인스토밍)

| 결정 | 값 |
|---|---|
| 구현 위치 | **이 레포로 핵심 코드 이식**(자체완결). `cooking-robot-safety/trajectory`의 검증된 코드 재사용, 크로스레포 결합 제거. 5단계 배포가 어차피 이 레포(브라우저) |
| 관측/예측 | **관측 8스텝(3.2s) / 예측 12스텝(4.8s)**, stride 1 슬라이딩 |
| 분할 | scene 단위 결정적 — **val = seed%5==0**(8 scene, 20%), 나머지 train(32). 베이스라인은 val에서 평가(4단계와 같은 val 재사용 → 공정 비교) |
| 지표 | ADE(12스텝 평균 L2)·FDE(12스텝째 L2), 미터. **전체 윈도우 + '움직인' 윈도우** 나눠 리포트 |
| 스테이션 휴리스틱 목표 | **기록된 현재 목표**(관측 마지막 스텝의 goal gx,gz). 실제 배포엔 목표 추정기 필요(메모) |

## 이식 (원본 그대로, 출처 표기)

`cooking-robot-safety/trajectory`에서 **핵심만** 가져온다(ETH/ATC·overhead_lstm 등 데이터셋
전용 코드는 제외):

- `trajectory/__init__.py`, `types.py`(Track/TrackScene/Mode/Prediction),
  `predictors.py`(등속·칼만), `evaluator.py`(ade/fde) — **바이트 그대로**(출처 주석 추가).
- `tests/test_kalman.py`, `tests/test_evaluator.py` — 그대로. 이식이 깨지지 않았음을 보증.

## 새로 작성

- `trajectory/sim_traj.py` — 로더. `dataset/trajectories/*.json` → 노드별 윈도우.
  - 폐기 노드(`node.discarded`) 제외. 각 노드 150스텝에서 obs8/pred12 윈도우를 stride 1로.
  - 윈도우 → `TrackScene(now, horizon=4.8, agents=[Track(id, 8×(t,x,z))])` + GT 12×(t,x,z)
    + 관측 마지막 목표 `(goal, gx, gz)` + `moved`(윈도우 동안 실제 이동량 > 임계).
  - scene 단위 train/val 분할(seed%5).
- `trajectory/sim_predictors.py` — `StationHeuristicPredictor(n_steps)`.
  - 관측 마지막 지점에서 관측 속력으로 목표(gx,gz)로 등속 직진, 도달하면 정지.
  - 목표 null(통과 구간)이면 등속으로 폴백.
- `train/eval_traj_baselines.py` — val 윈도우에 3종 예측기 적용, ADE/FDE 집계, 표 출력 +
  `docs/chanwoo/prediction-eval.md` 작성.

## 출력 (표 모양)

```
윈도우 집합: 전체 / 움직인 것만
| 예측기            | ADE(m) | FDE(m) | 윈도우 수 |
| 등속(const-vel)   |  ...   |  ...   |    N     |
| 칼만(Kalman)      |  ...   |  ...   |    N     |
| 스테이션(goal)    |  ...   |  ...   |    N     |
```

## 검증 (성공 기준)

1. **이식 테스트 통과** — `pytest tests/test_kalman.py tests/test_evaluator.py` 새 위치에서 그대로 통과.
2. **로더 sanity** — 윈도우 수·형태(obs8/pred12) 실측, 폐기 노드 제외 확인, train/val 분할 겹침 0.
3. **지표 sanity** — ADE ≤ FDE(대개), 크기 비자명(등속 FDE가 0이 아님), 스테이션 휴리스틱이
   '움직인' 윈도우에서 등속보다 낮음(목표를 알므로). 아니면 데이터·구현 재점검.
4. **재현성** — 같은 데이터·같은 분할 → 같은 표(결정적).

## 범위 밖

- 4단계 학습형(LSTM/Trajectron++ CVAE)·5단계 배포. 여기선 **학습 없는 베이스라인만**.
- 목표 추정기(스테이션 휴리스틱이 기록된 목표를 쓰는 것의 실배포 대체물)는 4단계+.
