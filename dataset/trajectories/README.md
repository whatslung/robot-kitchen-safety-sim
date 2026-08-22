# 궤적 데이터 예시 (LSTM 학습·평가용)

전체 궤적(120 scene·4.3MB)은 용량 문제로 git에서 제외한다. 여기엔 **형태 파악용 예시만** 둔다.
LSTM 데이터 분할(train/val/test)을 하나씩 볼 수 있게 골랐다.

| 파일 | 역할 | 무엇 |
|---|---|---|
| `island_h58_seed11_0010.json` | **train** | 시뮬 궤적 (seed 11, `seed%5≠0`). 사람 3명·150스텝(0.4s)·직무 전이(wf) |
| `island_h58_seed10_0009.json` | **val** | 시뮬 궤적 (seed 10, `seed%5==0`). train과 같은 형식, 깨끗(노이즈 증강 없음) |
| `real_test_sample.json` | **test** | 실사 zero-shot 검증용. `overhead-person-v3`(Roboflow) YOLO 라벨 → IoU 추적. **좌표만**(인물 이미지 없음), 이동량 상위 6트랙 |

## 형식

**시뮬 궤적**(train/val) — 한 파일 = 한 장면(scene):
```
장면: layout·room·robot 위치 · hz 2.5 · dt 0.4s · steps 150 · wf:true
└ nodes: 사람 3명 → 각 id·job·role·frames[150]
   프레임: { t, x, z, goal, gx, gz, moving }   ← 미터 좌표 + 향하는 목표
```

**실사 궤적**(test) — 시뮬과 다르다. 목표(goal)·직무 정보 없음(실사엔 라벨 없음):
```
tracks: [ { id, frames:[{ f, cx, cy }] } ]   ← 정규화 좌표(0~1), 프레임 번호
```

## LSTM 입력으로 자를 때

발자국을 **관측 8스텝(3.2s) → 예측 12스텝(4.8s)** 윈도우로 슬라이딩. 전체에서 train 32,488 / val 8,646 창.
train/val 분할은 파일명이 아니라 **scene 시드의 `seed%5`**로 나눈다(`trajectory/sim_traj.py`).

> LSTM엔 정식 test 분할이 없다 — train/val(둘 다 시뮬)만 있고, 실사는 별도로 zero-shot 검증했다.
> 재생성: 시뮬을 브라우저에서 띄우고 `/traj` 수집. 실사 궤적: `train/spike_real_baseline.py`.
