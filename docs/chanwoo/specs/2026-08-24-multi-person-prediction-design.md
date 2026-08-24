# 다인원 동시 예측 + 미래 위험 중재 — 설계 (감사 P0-5)

> 작성 2026-08-24 · 담당 chanwoo · 상태: **1+2단계 구현·검증 완료** (3단계 조건부 후속)
> 근거: [handoff/2026-08-22-motion-quality-audit.md](../handoff/2026-08-22-motion-quality-audit.md) §P0-5
> 선결: track ID별 예측 이력 분리(완료 — `predictionUpdate(srcPos, now, srcId)` + `nearestPerson().id`).

## 구현 상태 (2026-08-24)

- ✅ **백엔드**: `trajectory/risk.py`(mode_entry·track_risk·arbitrate, pytest 14) + `/predict` 배치
  확장(하위호환 유지, pytest 4). 무상태.
- ✅ **시뮬 1단계**: `MPRED`(사람별 이력맵·배치 클라이언트) + 전원 실제 LSTM 경로선·화살표 렌더
  (휴리스틱=회색 폴백). 실제 브라우저 검증: 3명 전원 배치 예측·경로선 생성.
- ✅ **시뮬 2단계**: `MPRED.worst` → 제어 결합(`effStop=min(로컬칼만, worst)`), worst 경로 위험색,
  worst 작업자 라벨. 검증: 최근접이 위협 아닐 때 worst가 제어를 당김(effStop=worst), worst 경로 빨강.
- ⏳ **남은 것(후속)**: (a) 검출 모드 관측을 `MIL.tracks` 기반으로(현재 MPRED 관측은 GT 기반이라
  검출 모드에선 worst를 제어에 반영하지 않음 — GT 전지성 누출 방지). (b) 실제 LSTM 모델을 물린
  `/predict` 배치 E2E(브라우저 검증은 mock /predict 사용, 백엔드 로직은 pytest). (c) N=1/3/5/10 지연
  실측. (d) 3단계 상호작용 모델(측정 근거 시).
- **회귀 테스트**: `tests/test_risk.py`·`tests/test_predict_batch.py`(pytest),
  `tests/browser/multi-person-prediction.spec.mjs`(P0-4로 활성).

## 0. 목표와 현재 한계

북극성 한 문장:

> **모든 작업자의 미래를 동시에 예측해 가장 위험한 경로를 찾는 다인원 안전 AI**

**현재**: 로봇에 가장 가까운 **한 명만** LSTM으로 예측하고, 나머지는 휴리스틱 화살표(현재 속도 기반)만
그린다. 로봇 제어도 그 최근접 한 명 기준이다. 그래서 (1) 지금 멀지만 빠르게 진입할 사람, (2) 최빈은
안전하나 낮은 확률 경로가 위험한 사람을 놓친다. 선택 순서를 **"현재 거리 → 예측"에서 "전원 예측 →
미래 위험 비교"로** 바꾼다.

## 1. 범위 결정 (확정)

- **단계**: 1→2 순차 실행, **2단계까지 필수**. 3단계(상호작용 모델)는 독립 배치 예측이 상호작용을
  반복적으로 놓친다는 **측정 근거가 생길 때만** — 이 스펙은 1+2를 커밋 범위로, 3은 조건부 후속으로 둔다.
- **적용 모드**: GT 모드·검출 모드 **둘 다**, 단 **GT를 먼저** 완성하고 검출을 그 위에 얹는다.
  - GT vs 검출은 *라이브 런타임*에 로봇이 사람 위치를 아는 방식일 뿐이며, *오프라인* train/val/test
    분리(P0-1)와는 다른 층이다. 검출 모드는 이후 P0-2(실제 관측 체인 평가)의 토대가 된다.
- **연산 위치**: LSTM 추론은 지금도 백엔드다(변화 없음). **위험 계산·중재도 백엔드(무상태)**로 둔다.
  - 이유: (1) 위험 primitive가 이미 파이썬에 있고 테스트됨(`trajectory/evaluator.py`),
    오프라인 평가(`eval_traj_safety.py`)가 그걸 쓴다 → 라이브·오프라인이 **같은 위험 코드 공유**로
    P0-C(수치 어긋남) 위험 감소. (2) 감사가 요구한 단위테스트(배치==단일, 최근접≠최고위험)를
    **pytest**로 깔끔히 짤 수 있다(JS 모놀리스엔 테스트 인프라 없음). (3) 무상태로 두면 결합·재시작
    문제 없음 — 로봇 자세·반경은 매 호출 인자로 전달.

## 2. 아키텍처 개요 (데이터 흐름)

```
[시뮬]  매 프레임: 활성 사람 전원 관측 → MPRED.hist: Map<id,[{t,x,z}]>   (id별 분리 = 좌표 혼합 불가)
          │ hz=10, 배치
          ▼
   POST /predict { tracks:[{id,hist}], robot, stopR, slowR, horizon, safeKsig, safeTau }
          ▼
[백엔드·무상태]  predict_batch(forward 1회) → 트랙별 K모드
                risk.track_risk(모드, robot, 반경, 운영점) → 트랙별 위험
                risk.arbitrate(전 위험) → worst
          ▼
   { tracks:[{id, modes, risk}], worst }
          ▼
[시뮬]  MPRED.pred: Map<id,{modes,risk,at}> + MPRED.worst 저장
        • 렌더: 전원 K=3 경로선·구름·화살표 (실제 LSTM / 휴리스틱 폴백 구분)
        • 제어(2단계): worst 신선하면 그 tEntry로, 아니면 로컬 칼만 바닥
        • 안전 바닥(항상): 반응형 + 로컬 칼만 — 네트워크를 절대 기다리지 않음
```

## 3. `/predict` 계약 (하위호환 유지)

좌표는 전부 **씬 AU**(모델 학습 단위). 시각(초)은 스텝 i → `t = 0.4·(i+1)`.

**요청**
```json
{
  "tracks": [{"id": 0, "hist": [[x,z], … 8점]}, …],
  "robot":  {"x": .., "z": ..},
  "stopR":  <AU>, "slowR": <AU>,
  "horizon": 1.6,
  "safeKsig": 1.0, "safeTau": 0.1
}
```
**응답**
```json
{
  "tracks": [{
    "id": 0,
    "modes": [{"path": [[x,z]…12], "w": .., "sigma": […12]}],
    "risk": { "tEntryStop": <초|null>, "tEntrySlow": <초|null>,
              "riskMass": <0..1>, "dMin": <AU> }
  }, …],
  "worst": { "id": 0, "tEntryStop": .., "tEntrySlow": .., "riskMass": .., "dMin": .. }
}
```
**하위호환**: 요청에 `tracks`가 없고 옛 `hist`만 있으면 → 옛 응답 `{"modes":[…]}` 그대로(위험 없이).
기존 단일 `__customPredictor`가 1단계 전환 중에도 안 깨진다.

## 4. 백엔드 컴포넌트 (파이썬, 무상태)

### 4-1. `trajectory/risk.py` (신규, 순수 함수)
- `mode_entry(modes, robot, radius, horizon, ksig, tau) -> (t_entry|None, mass)`
  [sim.html:5901](../../sim.html) `entry()`의 1:1 이식: 각 모드에서 σ팽창 거리
  `d = hypot(p - robot) - ksig·sigma[i]`가 `radius` 미만이 되는 **첫 점**(단, `t ≤ horizon`)을 찾고,
  진입 모드 가중치 합 `mass`를 누적. `mass ≥ tau`면 최이른 진입시각, 아니면 `None`.
- `track_risk(modes, robot, stopR, slowR, horizon, ksig, tau) -> dict`
  `{tEntryStop, tEntrySlow, riskMass, dMin}`. `riskMass` = 정지반경 진입 모드 가중치 합,
  `dMin` = 지평선 내 σ팽창 최소거리(보수적).
- `arbitrate(risks) -> worst|None`
  감사 우선순위: **① 정지진입 여부(tEntryStop ≠ None) → ② 가장 이른 tEntryStop →
  ③ 큰 riskMass → ④ 작은 dMin**, 동률은 id 오름차순으로 **결정적**.
- `evaluator.min_dist_to` 등 기존 primitive 재사용, σ팽창만 추가.

### 4-2. `/predict` 확장 ([detect_server.py:434](../../backend/detect_server.py))
- `tracks` 있으면: 전 hist 모아 `LearnedPredictor.predict_batch`(forward 1회) →
  트랙별 `track_risk` → `arbitrate` → `{tracks, worst}`.
- 옛 `hist`만: 현행 단일 경로(하위호환).
- **완전 무상태**: robot·반경·horizon·운영점 전부 요청에서. 서버는 세션 상태 없음.
- `DETECT_MODEL=none`(검출 비활성)에서도 `/predict`는 `_get_predictor()`로 독립 동작.

## 5. 시뮬 컴포넌트

### 5-1. 상태: `PRED`(축소) vs `MPRED`(신규)
- **PRED** = 전역 튜닝 상수(horizon, σ, 운영점, 색…) + **제어 대상 1명의 로컬 칼만** + 상태.
  전역 설정은 사람 수와 무관하고, 로컬 칼만은 백엔드와 독립한 **동기 안전 바닥**이라 여기 남긴다.
- **MPRED** = `hist: Map<id,[{t,x,z}]>` + `pred: Map<id,{modes,risk,at}>` + `worst`.
  겹치는 층이 아니라 **다른 층**: PRED=즉시 반응 바닥(1명), MPRED=선제 다인원 예측(전원).

### 5-2. 관측 — `mpredObserve(now)`
매 프레임 **모든 활성 사람** 좌표를 각자 이력에 push:
- GT 모드: `person`(`"gt:0"`) + `EXTRAS[i]`(`"gt:"+(i+1)`).
- 검출 모드: 신선한 `MIL.tracks`(`"mil:"+id`), 기존 `milAssocTracks` 활용.
각 이력 ~3.2s 트림, 사라진 id 폐기. **id별 Map이라 좌표 혼합이 구조적으로 불가.**

### 5-3. 배치 클라이언트 — `mpredTick(now, dt)`
`customPredictTick`의 배치판: hz=10 스로틀 + in-flight 가드, 각 이력 `trajResample`로 8×0.4s,
`/predict`에 배치 POST, 응답을 `MPRED.pred`/`MPRED.worst`에 저장. 세대 카운터로 stale 응답 무시,
`now-at<1000ms`만 신뢰. **N 상한**(기본 10, 로봇 근접순) 초과 시 잘린 수 `log`(silent cap 금지).

### 5-4. 로컬 칼만 안전 바닥 (동기, 항상)
- 로컬: 반응형(거리) + 최근접 대상 칼만 tStop/tSlow — 매 프레임 즉시(현행).
- 백엔드: worst 멀티모달 위험 — 신선할 때만 결정을 **더 보수적으로** 당긴다.
- 최종(2단계): `tStop = min(로컬 칼만, 신선한 worst.tEntryStop)`. 백엔드 지연·사망 시 로컬만으로 동작.

## 6. 렌더링

세 채널은 각자 **한 가지 정보만** 담는다(시각적 위생):

| 채널 | 담는 정보 |
|---|---|
| **경로선(신규 전원)** | 실제 미래 경로의 **모양**(곡선). 사람별 K=3, 최빈 진하게·나머지 흐리게. |
| **구름(밀도)** | 미래 위치의 **확률 분포**(진할수록 확률↑, 넓을수록 σ↑). 실제 경로 따라감. |
| **화살표** | 갈래별 **진행 방향**(굵기=가중치). |

- **기존 fan-out 재사용**: [predModeArrows](../../sim.html) → `pmodeArrowsFor(node, idx)`가 이미 전원을
  돈다. 모드 소스만 `MPRED.pred.get(id)`로 바꾸면 전원이 실제 LSTM. 단일 `PRED._learnedIdx` 폐기.
- **실제 vs 휴리스틱 구분**: 실제 LSTM = 실선+모드색, 휴리스틱 폴백(예측기 불가/stale) = 흐린 점선.
- **예측 위험색(미래) = 경로 채널에**: 반응형 링/신호등은 **"지금"** 상태(사람이 반경에 들어와야 빨강)라
  예측 임팩트가 약하다. 그래서 **최고위험 사람의 진입 모드 경로에서 반경에 드는 구간을 빨강/주황**으로 +
  진입 지점에 **카운트다운 라벨 "정지 예상 0.8s"**. 링(now)과 경로색(future)은 다른 순간이라 중복 아님.
- **제거되는 선**: 대표선 `PRED.line`(제어 대상 1명 경로+안전색)은 사람별 경로선과 겹치므로 **폐기**하고,
  안전색 역할을 **worst의 경로선**에 흡수한다.
- **씬 그래프(GRAPH)**: 스테이션 구조 설명용(경로 아님) → 데모 기본 숨김 권장(P1-3), 코드 제거는 안 함.
- (선택 폴리시) 예측 경로가 링을 뚫는 호(arc) 하이라이트/점멸.

## 7. 안전 불변식 / 에러처리

- 백엔드 죽음·지연·stale(>1s) → **로컬 칼만 바닥 폴백, 로봇은 네트워크를 안 기다림**(현행 보존).
- 예측기 503 → 시각화는 사람별 휴리스틱 폴백, 제어는 로컬 칼만.
- 트랙 만료·ID 스위치 → 이력 항목 폐기; id별 Map이라 좌표 혼합 불가.
- N 상한 초과 → 잘린 수 `log`.
- 중재 동률 → id 오름차순 결정적. 제어·중재 경로에 `Math.random`/시계 의존 없음.

## 8. 단계 계획

**1단계 (착지 가능)**
- 백엔드: `/predict` 배치 확장(단일 하위호환 유지) + `trajectory/risk.py` 착지(순수·테스트).
- 시뮬: `mpredObserve` + `mpredTick` + `MPRED.pred/worst` 저장 + **전원 실제 LSTM 렌더**(구분).
- **로봇 제어는 현행(최근접 로컬) 유지.** → "모든 작업자가 예측됨"이 화면에.

**2단계**
- 시뮬: `MPRED.worst`를 제어에 배선(신선하면 worst, 아니면 로컬 바닥) + 예측 위험색·진입 라벨 +
  worst UI("위험 작업자 #id · 모드 · 진입 t"). `PRED.line` 흡수.
- → "최근접 아닌 최고위험"으로 로봇 반응.

**3단계 (조건부·문서만)**: 상호작용 모델(social pooling) — 독립 배치가 상호작용을 반복적으로
놓친다는 측정 근거가 생길 때만. 그전엔 "다인원 동시 예측"으로 부르고 "상호작용 공동 예측"으로
과장하지 않는다(감사 §P0-5 1단계 주석).

## 9. 테스트

- **pytest `tests/test_risk.py`**: `mode_entry`/`track_risk`/`arbitrate` 단위 · **최근접≠최고위험**
  시나리오 · 배치==단일 동치(`predict_batch([h])[0] == predict_modes(h)`) · `/predict` 배치 계약 +
  단일 하위호환.
- **Playwright**(P0-4로 활성): 전원 LSTM 연결(휴리스틱 아님) · ID 스위치 시 이력 무혼합(기존
  `prediction-hist-switch.spec.mjs` 확장) · 단일==배치 동일 사람 출력 · nearest≠worst E2E 로봇 반응.
- **지연 계측**: N=1/3/5/10 배치 추론 + 안전 루프 지연 `log`(감사 완료조건).

## 10. 주장 경계 (감사 §2-4)

- **현재(2단계 완료)**: "모든 활성 작업자의 미래경로를 동시에 계산하고, 미래 위험도·진입시각을
  비교해 가장 위험한 경로를 로봇 제어에 반영한다."
- **금지**: "화면의 모든 화살표가 상호작용 기반 공동 예측 결과다"(3단계 전엔 독립 배치 예측).

## 11. 파일 · 완료 정의

**건드릴 파일**: `backend/detect_server.py`(배치) · `trajectory/risk.py`(신규) · `tests/test_risk.py`(신규)
· `sim.html`(MPRED·관측·렌더·2단계 제어·worst UI) · `tests/browser/*`(확장) · `docs/chanwoo/*`(주장 경계).

**완료 정의**
- 활성 N명 모두 실제 LSTM K=3가 연결되고 휴리스틱과 시각적으로 구분됨.
- 대상 전환·ID 스위치 상황에서 다른 사람 좌표가 관측창에 섞이지 않음(테스트).
- 단일==배치 동일 사람 출력이 허용오차 안.
- 로봇 제어가 "현재 최근접"이 아니라 "예측상 최고위험"을 반영(테스트 시나리오 통과).
- N=1/3/5/10 지연 기록.

## 12. 후속 (이 스펙 밖)

- **로봇 능동 회피기동**: 이 스펙은 위험 **판단**(누가·언제·어느 방향)과 그에 따른 **선제 감속·정지**까지다.
  로봇이 능동적으로 **경로를 틀어 피하는** 회피기동은 별도 후속 스펙 — 단, 본 스펙의 중재 출력
  (worst id·진입시각·진입 방향)이 회피 플래너의 입력이 되도록 설계했다.
- P0-2(실제 검출 트랙 E2E 평가)가 검출 모드 위에서 곧바로 성립.
