# 핸드오프 — 조리로봇 안전: 검출 학습 루프

> 최종 갱신: 2026-08-19

## 한 줄 요약

시뮬에서 합성 데이터를 만들어 YOLO를 학습 → 시뮬에서 검출을 검증하는 루프.
**2026-08-18: 직교 나디르(top-down) 데이터로 파인튜닝해 person 검출 make-or-break PASS**
— recall **0.688** / precision **0.797** (stock 0.175 / 0.374 대비). 설비 오탐이 잡혀
**나디르 top-down 채택 근거 확보**. 다음은 데이터 증량으로 recall↑ + sim-to-real.

## 🟢 2026-08-19 결과 — 스테이션 전이 모델 (이슈 #2 1단계)

사람 동선 생성에 **세 번째 모드**를 넣었다. `wanderTarget`(균등 랜덤)과 `JOB.route`(결정적
순회) 사이가 비어 있었는데, 균등 랜덤은 엔트로피가 최대라 **학습형 예측기가 잡을 구조가
없다** — 이 상태로는 학습형이 스테이션 휴리스틱(`PRED.modes`)을 이길 수 없다.

- **`WORKFLOW` 사이클 + 확률 전이**: 다음 단계 0.65 / 직무 집합 안 거리 가중 0.30(τ=2m) /
  직무 밖 이탈 0.05. `sim.html`의 `WF`·`WORKFLOW`·`workflowTarget`.
- **진입점**: 버튼 `🧭 직무 전이 모드` · `__sim.personWorkflow(true, "prep")` · `__sim.WF.on`.
  추가 인원은 `WF.on`이면 `extraNextTarget`이 확률 전이를 쓴다.
- **실측**: 평균 이동 2.06~3.12 m(균등 랜덤 5.47 m) · 방문 분포 엔트로피 1.85~2.79비트
  (균등 4.39) · 같은 시드 재현 · A* 로 설비를 돌아간다(겹침 1프레임/900).
- **끄면 기존과 동일**하다 — 기존 데이터셋 회차가 재현된다.
- 스펙·계획: `docs/chanwoo/{specs,plans}/2026-08-20-station-transition*.md`
  (실측 표·설계 변경 근거가 스펙 하단에 있다).
- **다음**: 이슈 #2의 2단계 — 이 모드로 궤적 데이터 수집.
- **곁가지로 드러난 문제**: `SAFE.NOM_STOP`이 3.1 m로 커져 5직무의 로봇 최근접이 전부 그
  안에 들어온다 → `jobRole`이 다섯 다 `danger`를 돌려준다. far/caution/danger가 데이터셋
  라벨인데 다 같으면 하드 네거티브가 사라진다. WF 도입 전부터 그런 상태여서 이번엔
  건드리지 않았다. 스펙 하단 "남은 문제" 참조.

---

## 🟢 2026-08-18 결과 (핵심)

- **직교 나디르 카메라(CAM7 `orthotop`) 추가** — 원근 0, 사람 크기가 프레임 전역 균일(112.5px/m).
  `sim.html` 콘솔에서 `__sim.DATAGEN.cams=['orthotop']`로 생성. 미검출 사각 없음.
- **YOLO11s 파인튜닝**(sim-person 200장, 6클래스) → person recall 0.688 · precision 0.797 ·
  mAP50 0.747, 전 클래스 mAP50 0.922. 산출물 `training/yolo11s_orthotop/weights/best.pt`(+`.onnx`).
- **환경/구조**: uv 프로젝트(`pyproject.toml`·`uv.lock`, torch **cu128**, RTX 5070 sm_120 검증).
  검출서버 `backend/`, 학습 스크립트 `train/`(prepare_yolo_split·train_sim·eval_stock)로 분리.
- **판정**: 나디르 top-down = **검출 축 통과**. 단 합성 val 기준이라 **sim-to-real은 다음 축**.
- **바닥 확률밀도 음영 완료(8/19)**: 미래 위치 분포를 가우시안 혼합(`PRED.mix`)으로 만들어 바닥 `DynamicTexture`에 부드러운 색 구름으로 렌더 + 예측선을 `GreasedLine`(두께)으로. orthotop에서 보면 Trajectron++ 논문 그림. 그림용 지평선 3.5s(안전 판정 1.6s와 분리), σ 하한=사람 반폭(0.24). **데이터셋 캡처 중 시각화 억제(`vizSuppressed`)로 학습 데이터 오염 차단**. 설계 `docs/chanwoo/specs/2026-08-19-density-floor-design.md`, 그림 `docs/chanwoo/assets/density_floor_demo.png`.
- **예측 시각화 완료**: 라이브 (x,z) 트랙 → 칼만 예측 + σ 불확실성. **2D 조감 패널**(로봇·정지/감속 원·트랙 점·예측 경로·σ 음영) + **3D 씬 σ 밴드**. `검출→추적→예측` 사슬 시각적 완성. 소스는 `MIL.safety` 켜면 라이브 검출 트랙, 아니면 GT. 스펙·계획 `docs/chanwoo/{specs,plans}/2026-08-18-*`.

---

## 목표 & 전체 파이프라인

```
CCTV(천장) → 검출(YOLO) → 추적(ByteTrack, ID) → 사람별 x,y 궤적 → 예측(칼만/LSTM) → SSM 안전판정 → 로봇제어
```

- **검출**만 이 레포(시뮬)에서 학습·검증. 예측기(칼만/LSTM)·SSM은 이미 별도 레포에 구현됨.
- 천장뷰 탑다운이라 이미지 (x,y) → 바닥 (x,z)가 거의 선형 → 궤적 좌표 확보가 쉬움(overhead 고정의 핵심 이유).

## 관련 레포

| 레포 | 역할 |
|---|---|
| `robot-kitchen-safety-sim` (여기) | 시뮬레이터 + 데이터 생성 + detect_server. **정식본** |
| `cooking-robot-safety` | 궤적 예측기(칼만/LSTM), SSM, ADE/FDE 벤치마크(ETH/ATC) |
| `overhead-person-yolo11` | 실제 CCTV 검출(별개, 실사 도메인) |
| `simulator-robot-cooking` | 옛 로컬 시뮬 사본 — **이 레포로 대체됨, 사용 금지** |

---

## 현재 상태

**✅ 되는 것**
- 시뮬 데이터 생성 파이프라인(합성 프레임 + YOLO 라벨, 6클래스). 데이터 유효성 검증 통과(손상0·형식0).
- 학습 파이프라인(yolo11n 파인튜닝, CPU). kettle 0.995 / robot 0.67 / equipment 0.67로 **"sim→학습→sim검출" 유효 증명**.
- `detect_server.py` 경로로 브라우저 ONNX/스로틀링 우회 가능.
- `sim.html`에 **overhead 전용 생성 옵션** 추가(아래).

**🔴 남은 것 / 막힌 것**
- ~~**person 검출 약함**~~ → **해결(2026-08-18, 위 결과 참조)**: 직교 나디르 200장 파인튜닝으로 recall 0.688 달성. 옛 37장(3~6시점 혼합) mAP50 0.11·Recall 0은 데이터 부족·시점 혼합 탓이었음(방법론 아님).
- **헤드리스 대량 생성 불가**: 브라우저 백그라운드 탭은 GPU 렌더가 멈춤(90초에 0샘플). **데이터 생성은 반드시 실제 브라우저 포그라운드**에서. 학습은 순수 Python으로 가능.

---

## 핵심 결정

1. **overhead(천장) 전용 피벗** — 여러 시점을 섞으니 person이 시점·스케일로 쪼개져 학습 실패. 시점 고정하면 (a) kettle처럼 일관돼 검출이 잘 배워지고 (b) 탑다운 x,y로 궤적 좌표를 바로 얻음.
2. **detect_server.py로 검출을 파이썬 백엔드에서** — 브라우저 안 ONNX/WebGPU/스로틀링을 우회. 시뮬이 프레임을 POST → 박스 반환.
3. **stock YOLO는 시뮬에서 안 됨** — 실사로 학습해 합성 시뮬을 못 잡음(역도메인갭 실측 확인). 그래서 시뮬 데이터로 학습해야 함.

---

## 주요 산출물 & 경로

| 항목 | 경로 |
|---|---|
| 시뮬 (편집됨) | `robot-kitchen-safety-sim/sim.html` |
| 검출 서버 | `robot-kitchen-safety-sim/backend/detect_server.py` |
| 옛 데이터셋(3~6시점 혼합, 37장, **참고용**) | `E:\다운로드\dataset` |
| 학습 워크스페이스(분할·data.yaml·runs) | `E:\VsCodeProjects\cooking-robot-safety\train_sim` |
| 학습된 가중치 | `...\train_sim\runs\detect\runs\sim6\weights\best.pt` |

> ⚠ `sim.html` 편집(overhead 옵션)은 **아직 커밋 안 됨**. 데이터셋/best.pt는 이 레포 밖에 있음.

### sim.html 편집 내용 (uncommitted)
`generateDataset`에 카메라 필터 추가. `__sim.DATAGEN`으로 노출:
- `DATAGEN.cams` — 생성할 카메라 배열. 예 `['overhead']`. `null`=전체(6개: overhead·corner·eye·top·deg45·front).
- `DATAGEN.imagesOnly` — `true`면 `gt/`(mask·inst·depth·meta) 생략 → 더 빠름/가벼움.

---

## 다음 단계 (순서대로)

### 1. overhead 전용 데이터 생성 (사용자 브라우저에서)
```
# 서버 (이 레포 루트에서)
python -m http.server 5173      # 또는 launch/실행하기_Windows.bat
# Chrome/Edge → http://localhost:5173/sim.html
```
콘솔(F12):
```js
__sim.DATAGEN.cams = ['overhead'];
__sim.DATAGEN.imagesOnly = true;
```
→ 📊 데이터 탭 → 샘플 수 200 → **데이터셋 생성** → **새 빈 폴더** 선택 → 탭 앞에 두고 대기.
결과: `overhead_XXXX.png` + `labels/`. 1장/샘플이라 빠름. person이 다양하게(사람 수·위치) 나오게.

### 2. 재학습 (Python, 브라우저 불필요)
train/val 분할(샘플 인덱스 단위로 — 누수 방지) + `data.yaml`(6클래스) 생성 후:
```bash
KMP_DUPLICATE_LIB_OK=TRUE yolo detect train model=yolo11n.pt data=data.yaml \
  epochs=80 imgsz=640 batch=8 workers=0
```
> 환경: anaconda python, **CPU only**, ultralytics 설치됨. `KMP_DUPLICATE_LIB_OK=TRUE` 필수(OMP 충돌 회피). names: `0 person 1 fire 2 smoke 3 robot 4 kettle 5 equipment`.

### 3. detect_server에 모델 물리기 & 검증
`backend/detect_server.py`는 이미 **v2.0(FastAPI + ByteTrack + track id + 속도 vx/vy)** 로 업그레이드돼 있음(워킹카피, 미커밋). 검출→추적→속도까지 서버가 처리.
```bash
pip install fastapi uvicorn ultralytics trackers supervision pillow numpy
python backend/detect_server.py     # http://127.0.0.1:8000/detect  (/health로 상태 확인)
```
우리 모델 연결:
- **49행** `MODEL = YOLO("yolov8n.pt")` → `YOLO(".../best.pt")`
- **66행** `LABEL_MAP` → 6클래스 통과(또는 `None`으로 전부 통과)

응답 계약: `{boxes:[{label,conf,cx,cy,w,h,id,vx,vy}], mode, camera}` (정규화 0~1, id=트랙, vx/vy=이미지평면 속도).
> ⚠ **현재 sim.html 파서는 id·vx·vy를 무시**함(label/conf/cx/cy/w/h만 읽음). 트랙·속도를 실제로 쓰려면 시뮬 쪽 `__customModel` 응답 처리를 트랙 단위로 확장하는 후속 작업 필요.

시뮬을 http(server) 모드로 연결(`MIL.url = http://127.0.0.1:8000/detect`) → **stock vs 우리 모델** 비교, person 잡히는지 확인.

### 4. 이후
- detect_server가 이미 track id + 속도를 주므로, **사람별 x,y 트랙**은 서버에서 나옴 → `cooking-robot-safety`의 칼만/LSTM 예측기 연결 → SSM.
- 남은 배선: 시뮬 파서 확장(id/vx/vy 수용) + 월드 좌표 역투영(카메라 파라미터로 이미지 x,y→바닥 x,z).

---

## 함정 / 주의

- **데이터 생성은 반드시 실제 브라우저 포그라운드.** 백그라운드 탭·헤드리스는 렌더가 멈춤.
- `sim.html`을 `file://`로 열지 말 것 — http 서버로. (ONNX wasm 로드 실패)
- GT 해상도 `GT_W/GT_H`는 `const`(960×720) — 코드 수정 없이 못 바꿈.
- 옛 `E:\다운로드\dataset`(혼합 시점)과 새 overhead 데이터를 **섞지 말 것**.
- 37장 결과(person 0.11)는 데이터 부족 탓 — 방법론 문제 아님. overhead+더 많은 샘플로 해결.
