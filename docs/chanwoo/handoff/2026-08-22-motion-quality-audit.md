# 움직임 예측 파이프라인 품질 감사 · 개선 핸드오프 (2026-08-22)

> 목적: 현재 구현을 발표용 데모에서 재현 가능한 연구·제품 후보로 끌어올리기 위한 개선 백로그.
> 감사 기준 커밋: `570a84a`(감사 직전 `main` 최신). 이 문서는 성과 홍보문이 아니라 다음 작업자를 위한 결함·검증 목록이다.

## 0. 결론

이 프로젝트는 **검출 → 추적 → 멀티모달 궤적 예측 → 안전판정 → 로봇 행동**을 한 화면에 연결한
연구 데모로는 강하다. 특히 합성 궤적에서 LSTM 최빈 모드가 칼만보다 ADE/FDE를 낮추고,
여러 미래 모드를 안전판정에 연결한 점은 발표 가치가 높다.

다만 현재 근거로 방어 가능한 범위는 다음까지다.

- 합성 궤적 val에서의 예측 성능
- 일반 오버헤드 영상 라벨 궤적에서의 예비 zero-shot 전이 신호
- 시뮬레이터 안에서의 검출·추적·예측·안전 동작 통합
- localhost에서 동작하는 로컬 엣지 추론

아직 **실제 급식실·물리 로봇·완전 브라우저 온디바이스·안전 성능 0.93**을 한 문장으로 묶을 수는 없다.
다음 개선의 핵심은 모델을 더 크게 만드는 것이 아니라 **평가 분리, 실제 관측 체인 검증, 라이브 지표 측정,
재현성 고정**이다.

## 1. 현재 강점과 품질 판정

| 항목 | 판정 | 근거 |
|---|---:|---|
| 발표용 데모 | 7.5/10 | 디지털 트윈, 안전링, K=3 경로와 밀도 구름, 로봇 동작이 한 화면에 연결됨 |
| 저장소 구성 | 6.5/10 | `uv.lock`, 실행 서버, 학습·평가 코드, 문서와 테스트가 있으나 `sim.html` 단일 파일과 외부 산출물 의존이 큼 |
| 정량 성능 | 5.5/10 | 합성 val 개선은 분명하지만 독립 test·반복실험·실제 폐루프 평가는 부족 |
| 연구 엄밀성 | 5/10 | 여러 베이스라인과 안전 recall/precision을 측정했으나 운영점 선택과 최종 평가가 분리되지 않음 |
| 제품 준비도 | 4/10 | 로컬 서버와 모델 자동 다운로드는 있으나 버전 고정·E2E 테스트·패키징·현장 성능이 없음 |

### 방어 가능한 대표 수치

- 합성 val 최빈 모드: `ADE/FDE = 0.748/1.420m`
- 칼만: `1.031/2.069m` → LSTM이 약 `27%/31%` 낮음
- `minADE@3 = 0.432m`는 정답을 본 뒤 가장 가까운 모드를 고르는 **oracle 상한**이며 배포 성능이 아님
- 안전 전 모드 합집합: `recall 0.756 / precision 0.442`
- 불확실성 운영점 `k=1, τ=0.1`: `recall 0.933 / precision 0.377`
  - 이 값은 합성 평가에서 고른 운영점이다. 실제 라이브/실사 안전 성능으로 표현하지 않는다.

## 2. 발표·문서에서 지켜야 할 주장 경계

### 2-1. “4.8초 예측”

**가능한 표현**

> 2.5Hz 합성 궤적에서 과거 8점으로 미래 12점, 최대 t+4.8초까지 여러 경로를 출력한다.

**피해야 할 표현**

> 실제 주방에서 사고를 5초 먼저 알고 로봇을 멈춘다.

이유:

- 학습·오프라인 평가는 12스텝 × 0.4초 = 4.8초다.
- 현재 라이브 안전판정은 `PRED.horizon = 1.6s`만 사용한다(`sim.html`).
- 실사 샘플은 원본 영상 FPS를 확인하지 않으면 12프레임을 4.8초라고 단정할 수 없다.

### 2-2. “온디바이스”

**가능한 표현**

> 영상과 추론을 외부 클라우드로 보내지 않고 로컬 장치에서 처리하는 엣지 AI 데모다.

**피해야 할 표현**

> 전체 파이프라인이 브라우저 하나에서 서버 없이 동작한다.

검출기는 ONNX를 선택하면 브라우저 WebGPU/WASM 경로가 있지만, 현재 학습형 궤적 예측기의 통합 경로는
localhost Python/PyTorch `/predict` 서버다. 첫 실행 때 HF 가중치 다운로드도 필요할 수 있다.

### 2-3. “Sim-to-Real 검증”

**가능한 표현**

> 실사 파인튜닝 없이 일반 오버헤드 영상의 라벨 궤적에서 CV 대비 전체 ADE/FDE 9~12%, 이동 구간 3~7% 개선된 예비 전이 신호를 확인했다.

**피해야 할 표현**

> 실제 급식실과 물리 로봇에서 zero-shot sim-to-real을 검증했다.

현재 실사 검증은 실제 급식실이 아니며, 검출 결과가 아니라 YOLO 정답 라벨을 연결해 만든 오프라인 트랙이다.

## 3. 즉시 수정할 문서·실행 불일치

### P0-A. 실사 검출 지표 오기 수정

`docs/chanwoo/model-scorecard.md`의 “실사 person recall 0.048”은 지표가 뒤바뀐 표현이다.
`docs/chanwoo/detection-eval.md`의 sim-only → 실사 test 값은 다음이다.

- recall `0.270`
- precision `0.072`
- mAP50 `0.048`

**완료 조건**

- 모델 카드, scorecard, week6 문서에서 `0.048`을 recall로 부르는 문장을 전수 검색해 수정
- 표와 본문의 metric 이름이 자동 검증되는 문서 테스트 추가

### P0-B. `DETECT_MODEL=none` 동작 수정

README와 `docs/chanwoo/HANDOFF.md`는 `DETECT_MODEL=none`으로 예측 전용 실행이 가능하다고 쓰지만,
현재 `_resolve_model_path("none")`은 로컬 파일이 아니므로 HF 기본 모델을 다운로드한다.

**완료 조건**

- `none`을 명시적인 검출 비활성 모드로 처리하거나 문서에서 해당 지침 제거
- 오프라인 상태에서 예측 전용 서버가 모델 다운로드 없이 뜨는 테스트 추가

### P0-C. 4.8초 오프라인 평가와 1.6초 라이브 제어 분리

현재 문서의 4.8초 safety recall과 라이브 제어의 1.6초 horizon이 쉽게 같은 성능으로 읽힌다.

**완료 조건**

- `horizon = 1.6s` 조건으로 라이브와 동일한 평가표를 별도 생성
- 1.6초와 4.8초 결과를 같은 표에서 비교
- 발표 대표 수치는 실제 제어 조건의 값을 사용
- `0.933`을 사용한다면 평가 horizon, 데이터 split, precision `0.377`을 항상 함께 표기

## 4. 개선 백로그

## P0 — 결과의 신뢰도를 먼저 올리는 작업

### P0-1. train / val / test 완전 분리

현재 `seed % 5 == 0` val에서 모델 비교와 운영점 선택이 함께 이뤄진다. 최종 성능을 같은 val로 보고하면
운영점 과적합을 피할 수 없다.

**작업**

1. scene 단위로 train/val/test를 고정하고 manifest를 커밋한다.
2. 모델·`τ`·`k`는 val에서만 선택한다.
3. test는 마지막 한 번만 평가한다.
4. stride-1 중첩 윈도우가 아닌 scene 단위 bootstrap CI도 함께 낸다.

**완료 조건**

- 동일 명령으로 split과 결과를 재생성 가능
- ADE/FDE와 safety recall/precision에 scene-level 95% CI 제공
- 운영점 선택 로그와 최종 test 결과 분리

### P0-2. 실제 관측 체인의 end-to-end 평가

현재 실사 예측 평가는 GT 라벨 궤적 기반이라 검출 누락·박스 지터·ID switch가 예측에 전달되는 영향을 측정하지 않는다.

**작업**

1. 실제 오버헤드 영상 → detector → ByteTrack → 좌표 → predictor 파이프라인으로 트랙 생성
2. GT-track 입력과 detected-track 입력을 같은 클립에서 비교
3. 검출 실패, track fragmentation, ID switch별 ADE/FDE·위험진입 성능 하락을 기록
4. 실제 FPS와 픽셀↔미터 보정 정보를 저장

**완료 조건**

- 실제 클립 단위 재현 명령과 결과 JSON 커밋
- GT-track / detected-track 성능 차이 표 제공
- 최소 1개의 실제 주방 또는 조리실 유사 환경 클립 포함

### P0-3. 라이브 안전 성능 계측

현재 UI에서 정지·감속이 보이는 것과 안전 성능이 입증된 것은 다르다.

**작업**

- 1.6초 horizon에서 위험진입 TP/FP/FN, 선제시간, 불필요 정지시간을 이벤트 로그로 저장
- 반응형 SSM only / CV / Kalman / LSTM을 같은 시나리오 seed로 비교
- 평균뿐 아니라 최악 시나리오와 실패 영상을 보존

**완료 조건**

- `missed entry`, `false hold`, `lead time`, `stop duration` 네 지표 자동 산출
- 고정 seed 회귀 테스트와 발표용 실패 사례 1개 포함

### P0-4. 최소 E2E 테스트와 CI

Python 단위 테스트 16개는 통과하지만 브라우저, 서버 API, 모델 로딩을 포함한 회귀 검증은 없다.

**작업**

- `/health`, `/detect`, `/predict` 계약 테스트
- `DETECT_MODEL=none` 오프라인 시작 테스트
- Playwright로 시뮬 로드, 예측 모드 전환, 정지 이벤트 발생 여부 smoke test
- GitHub Actions에서 CPU/DUMMY 모드 CI 구성

**완료 조건**

- PR마다 unit + API + browser smoke test 자동 실행
- 모델·GPU가 없어도 핵심 배선 테스트 가능

## P1 — 재현성과 제품성을 높이는 작업

### P1-1. 모델 artifact 고정

HF 모델 다운로드에 revision/checksum 고정이 없어 같은 커밋도 나중에 다른 모델을 받을 수 있다.

**완료 조건**

- repo id, filename, revision SHA, checksum을 manifest에 기록
- 서버 시작 로그와 결과 JSON에 모델 SHA 저장
- 오프라인 캐시 사용법과 모델 라이선스 명시

### P1-2. “완전 온디바이스” 범위 결정

둘 중 하나를 명시적으로 선택한다.

1. **로컬 엣지 제품**: 브라우저 + localhost 서버 구조를 공식 아키텍처로 문서화
2. **서버 없는 브라우저 제품**: LSTM도 ONNX Runtime Web으로 이식하고 `/predict` 의존 제거

**완료 조건**

- 선택한 경로의 cold start, 모델 로딩, 평균/p95 지연시간과 메모리 측정
- 대상 장치에서 end-to-end FPS와 전력 측정

### P1-3. 라이브 데모 프리셋과 성능 안정화

현재 화면은 기술 정보가 많아 발표 중 핵심 경로와 위험 이벤트가 묻힐 수 있고 GPU 환경에 따라 WebGL 경고와
프레임 저하가 발생할 수 있다.

**작업**

- “발표 모드” 프리셋: 카메라·K=3 경로·안전링·위험 시나리오 자동 설정
- 디버그 패널/콘솔 접기, 핵심 수치만 크게 표시
- 프레임 저하 시 품질 단계와 안전 계산 주기가 변하지 않는지 테스트
- 라이브 실패를 대비한 동일 시나리오 녹화본 준비

### P1-4. `sim.html` 모듈화

약 1.1만 줄 단일 파일은 기능 수정과 회귀 검토 비용을 키운다.

**권장 분리**

- `sim/prediction.js`
- `sim/safety.js`
- `sim/detection-client.js`
- `sim/ui-demo-mode.js`
- `sim/scenario-runner.js`

우선 테스트 가능한 순수 계산부부터 분리하고 렌더링 코드는 마지막에 옮긴다.

## P2 — 연구 확장

- 실제 급식실 레이아웃과 작업자 동선 데이터 추가
- 한 명의 learned threat만이 아니라 다중 작업자 예측·위험 결합 평가
- 레이아웃 hold-out과 카메라 위치 hold-out 추가
- 모델 크기 확대보다 calibration, domain randomization, detector-noise augmentation 우선
- 물리 로봇 연결 전 shadow mode로 경고만 기록해 false stop/hour와 lead time 측정

## 5. 추천 실행 순서

1. **문서 오기와 `DETECT_MODEL=none` 수정**
2. **1.6초 라이브 조건 평가 생성**
3. **독립 test split + scene-level CI**
4. **실제 detector-track end-to-end 평가**
5. **API/browser CI 추가**
6. **발표 모드와 성능 계측**
7. **artifact 고정 및 온디바이스 아키텍처 확정**

이 순서를 따르면 먼저 과장 위험을 없애고, 그다음 실제 성능을 올릴 수 있다.

## 6. 다음 작업자가 처음 볼 파일

- `docs/chanwoo/model-scorecard.md` — 현재 발표 수치 요약(실사 recall 오기 우선 확인)
- `docs/chanwoo/prediction-eval.md` — 합성 ADE/FDE
- `docs/chanwoo/prediction-safety-eval.md` — 4.8초 위험진입 평가
- `docs/chanwoo/prediction-sim2real-notes.md` — 실사 예비 전이와 운영점
- `docs/chanwoo/detection-eval.md` — 검출 domain gap의 원본 수치
- `trajectory/sim_traj.py` — 8→12 스텝 데이터 계약
- `sim.html`의 `PRED`, `predictionUpdate`, `tStop/tSlow` — 라이브 1.6초 안전판정
- `backend/detect_server.py` — 모델 해석, HF 다운로드, `/predict`

## 7. 최종 완료 정의

다음 조건을 모두 만족하면 “실제 환경 검증 전 단계의 완성된 연구 프로토타입”으로 올릴 수 있다.

- 독립 test split에서 예측·안전 지표와 95% CI 확보
- 실제 detector-track 입력에서 성능 저하와 실패 유형 측정
- 라이브 1.6초 조건의 lead time과 false stop 계측
- 모델 revision과 실행 환경 고정
- unit/API/browser CI 통과
- 발표 문구가 실제 측정 범위를 넘지 않음

물리 로봇·실제 급식실에서 별도 검증하기 전까지는 “안전 보장” 대신 **“선제 안전 보조층”**으로 표현한다.
