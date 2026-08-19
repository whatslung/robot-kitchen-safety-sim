# 학습형 예측기 브라우저 배포 — 설계 (확정)

> 2026-08-20. 이슈 #2의 **5단계(마지막)**. 4단계 학습형(경량 LSTM+혼합 헤드)을 브라우저
> 시뮬에 물려 실시간으로 돌리고, 멀티모달 봉우리를 밀도 시각화로 띄운다. 이걸로 이슈 #2
> 완료 조건(학습형이 브라우저에서 돎 · ADE/FDE 3자표 · 멀티모달 봉우리)이 모두 충족된다.

## 배포 경로 — Python 백엔드 (`/predict`)

브레인스토밍 결정: **인브라우저 ONNX가 아니라 Python 백엔드**. 근거:
- 검출(YOLO)이 이미 `detect_server`에서 돈다 — 예측도 서버에 두면 새 인프라 0, 아키텍처 일관.
- torch 그대로 서빙 → ONNX export 제약 없음, 4단계+(Trajectron++: 이웃·CVAE)도 그대로 확장.
- "온디바이스"는 클라우드가 아니라 로컬. 엣지 기기의 로컬 파이썬 = 온디바이스. 실배포에 더 가깝다.
- `__customPredictor`는 이미 async 계약 → `/predict` fetch에 정합.
- ONNX(model.onnx)는 버리지 않고 "나중에 인브라우저" 옵션으로 보존.
- **실측**: 예측 CPU 0.22ms/회 → 10Hz(100ms) 대비 462배 여유. 노트북 CPU로도 실시간.

## 정확성 세부 (구현 착수 시 발견)

1. **단위**: 기본 `SCALE.mPerAU=1.25/1.25=1.0` → 1 AU = 1 m. 2단계 데이터(`P.root.position`=AU)와
   inference obs(AU)와 모델(AU 학습)이 같은 프레임 → **좌표 변환 불필요**. `path`는 AU 그대로,
   `sigma`만 계약(미터)에 맞춰 `SCALE.m()` 적용(기본 항등). `?fryer=` override 시에도 일관.
2. **관측 창 리샘플**: 모델은 8스텝×0.4s(=2.8s) 관측을 먹는데 `PRED.hist`는 매 프레임(~60Hz)
   쌓고 **2.0s만 보관**한다. → (a) 보관을 **≥3.0s로 확장**(sim.html 한 줄), (b) `__customPredictor`가
   dense hist를 **8점(0.4s 간격, now 기준 과거)으로 리샘플**해 POST. 안 맞추면 모델 입력이 왜곡된다.
3. **멀티모달 viz**: 밀도 구름(`PRED.mix`)은 방향모드(`{name,w,dist,ux,uz}`)로 직선 외삽해 그린다.
   학습형 K경로를 **끝점 방향(ux,uz)+거리(dist)+가중치(w)로 변환**해 `predGoalModes` 자리를 대체 →
   렌더 수정 없이 봉우리가 갈라져 보인다(곡률은 근사로 잃지만 "봉우리 가시성"엔 충분).

## 건드리는 것

- `backend/detect_server.py` — `POST /predict`: `{hist:[[x,z]…]}`(AU, 8점 리샘플됨) 받아
  `LearnedPredictor.predict_modes` → `{modes:[{path:[[x,z]…], w, sigma:[…]}]}` 반환.
  모델 가중치는 기동 시 1회 로드(`training/traj_predictor/model.pt`). 없으면 503 + 안내.
- `sim.html`:
  - `PRED.hist` 보관 2000→3200ms.
  - `window.__customPredictor` 등록: dense hist→8×0.4s 리샘플 → `/predict` fetch →
    최빈 모드 `{path,sigma}`(계약) + `res.modes`(viz용) 반환.
  - `predictionUpdate`에서 useCustom+모드 있으면 `modes`(line ~5572)를 학습형 모드로 대체.
- 예측기 소스 모드 선택 UI는 이미 있음(`학습형 — window.__customPredictor`).

## 검증 (성공 기준)

1. **브라우저에서 학습형 구동** — 예측 소스 학습형 선택 → `/predict` 응답으로 예측선/판정이 돈다.
2. **왕복 지연 실측** — `CPRED.lastMs`(fetch+추론 왕복) 기록. 10Hz 주기 안(<100ms)인지.
3. **멀티모달 봉우리** — 갈림길(여러 목표 가능) 관측에서 밀도 구름이 **여러 봉우리로 갈라짐**(정성 + 모드 수·가중치 수치).
4. **폴백** — 서버 죽거나 미등록이면 조용히 칼만으로(로봇 안 멈춤). 계약대로.
5. **실시간** — 시뮬 60fps 유지(async+스로틀). 추론이 늦어도 렌더 안 막힘.

## 범위 밖

- Trajectron++(이웃 풀링·맵·CVAE) — 후속. 그때 `/predict`에 이웃 트랙 입력 추가(계약 확장).
- 인브라우저 ONNX 경로(옵션 보존).
- `?fryer=` 실측 스케일에서의 재검증(기본 스케일로 개발).
