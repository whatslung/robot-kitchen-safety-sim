# 동선 히스토리 나디르 — Figure Labs 핸드오프

슬라이드 08("과거 히스토리 학습")의 왼쪽 참조 그림을 **실제 시뮬레이터의 나디르(직교 top-down) 렌더** 위에 재현하기 위한 핸드오프 자산.

## 파일
- **figurelab_handoff.md** — Figure Labs에 넣는 단일 파일. 프롬프트 + 궤적 데이터(JSON 코드블록)가 한 곳에. (JSON 첨부가 안 될 때 이걸 쓴다.)
- **kitchen_nadir_wholescene.png** — 배경. 1440×1080 직교 나디르 렌더(원근 0 → 픽셀↔미터 선형).
- **kitchen_trajectories_nadir.json** — 조리원 3명 궤적(각 150점, 60s·2.5Hz) + 스테이션 21 + 로봇 + 안전링. 각 점에 `x,z`(미터)와 `u,v`(PNG 픽셀) 동봉 → `u,v`를 이미지에 바로 찍으면 정렬.
- **figurelab_prompt.md** — 프롬프트만(데이터 별도 첨부 시).
- **overlay_preview.png** — 검증/목표 예시. PNG 위에 JSON `u,v`를 그대로 찍은 것(prep 파랑·cook 주황·wash 초록).
- **predictor_architecture.svg / .png** — 슬라이드 09용 구조도(논문 스타일). 관측 8스텝·ego 정규화 → 인코더 교체형(LSTM⇄Transformer) → 64-d 문맥 → 멀티모달 헤드 → K=3 미래 + best-of-K 학습. `trajectory/learned_predictor.py`에 1:1 대응. SVG=벡터(편집용) · PNG=3040×1600(삽입용).

## 넣는 법
Figure Labs에 `figurelab_handoff.md` + `kitchen_nadir_wholescene.png` 두 개만 첨부.

## 생성 방법(재현)
`sim.html` → 📊 데이터 탭 **"동선 히스토리 그리기"** 버튼, 또는 콘솔 `__sim.renderHistoryOverlay()`.
데이터: scene seed 7 · WORKFLOW 동선 고정 dt 기록.

## 주의
- 원래 슬라이드는 prep·cook·**carry**였으나 carry(운반) 역할이 sim NAV 코너-핀 버그로 안 움직여 **wash(세척)** 로 대체(동선 형태는 동일).
- "관측→예측 K=3"(B패널)은 학습형 예측기(백엔드 `/predict`) 출력이 필요해 이 데이터엔 없음.
