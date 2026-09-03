# 궤적 데이터 예시 (LSTM 학습·평가용)

전체 궤적(120 scene·4.3MB)은 용량 문제로 git에서 제외한다. 여기엔 **형태 파악용 예시**와
정본 분할 파일 `split_manifest.json`만 둔다.

| 파일 | 역할 | 무엇 |
|---|---|---|
| `island_h58_seed11_0010.json` | **train 예시** | 시뮬 궤적. 사람 3명·150스텝(0.4s)·직무 전이(wf) |
| `island_h58_seed10_0009.json` | **train 예시** | 위와 같은 형식의 다른 seed 시뮬 궤적 |
| `real_test_sample.json` | **별도 실사 평가 예시** | `overhead-person-v3`(Roboflow) YOLO 라벨 → IoU 추적. **좌표만**(인물 이미지 없음), 이동량 상위 6트랙 |
| `split_manifest.json` | **분할 정본** | scene seed 단위 train/val/test 멤버십과 생성 조건 |

## 형식

**시뮬 궤적**(train/val/test) — 한 파일 = 한 장면(scene):
```
장면: layout·room·robot 위치 · hz 2.5 · dt 0.4s · steps 150 · wf:true
└ nodes: 사람 3명 → 각 id·job·role·frames[150]
   프레임: { t, x, z, goal, gx, gz, moving }   ← 미터 좌표 + 향하는 목표
```

**별도 실사 평가 궤적** — 시뮬과 다르다. 목표(goal)·직무 정보 없음(실사엔 라벨 없음):
```
tracks: [ { id, frames:[{ f, cx, cy }] } ]   ← 정규화 좌표(0~1), 프레임 번호
```

## 학습 입력과 분할

발자국을 **관측 8스텝(3.2s) → 예측 12스텝(4.8s)** 윈도우로 슬라이딩한다.
train/val/test는 `split_manifest.json`의 파일 멤버십으로만 나눈다. 현재 정본은 동일 seed의
모든 레이아웃 변형을 한 split에 묶은 70/15/15 분할이며, 고정 셔플 seed 0으로 생성했다.
manifest가 없으면 `trajectory/sim_traj.py`는 임의 분할로 진행하지 않고 오류를 낸다.

- 분할 재생성: `uv run python train/make_traj_split.py`
- 궤적 재생성: 백엔드와 시뮬을 띄우고 `/traj` 수집
- 실사 궤적 예시 생성: `train/spike_real_baseline.py`
