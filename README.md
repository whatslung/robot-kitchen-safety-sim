# 급식 조리로봇 안전 시뮬레이터 (Web / Babylon.js)

급식 조리 교반솥(국솥) 로봇의 **안전(끼임·충돌 회피)** 연구용 웹 3D 시뮬레이터.
학습 데이터는 이 시뮬레이터에서 합성하고, 실사 촬영본은 검증용으로만 쓴다.

빌드 과정 없이 정적 파일만으로 동작한다. 실행 중 외부 서버에 접속하지 않는다.

---

## 실행

목적에 따라 두 가지로 띄운다.

### 방법 A — 시뮬만 (정적, 빌드 불필요)

3D 시뮬레이터만 본다. 검출·예측은 안 돈다.

```bash
python -m http.server 5173
```

→ <http://localhost:5173/sim.html>

더블클릭 실행이 필요하면 `launch/` 안의 스크립트를 쓴다 (macOS: `실행하기_Mac.command`,
Windows: `실행하기_Windows.bat`). 빈 포트를 찾아 서버를 띄우고 Chrome/Edge로 열어준다.

### 방법 B — 검출·예측까지 (백엔드 서버)

검출(YOLO)·추적·궤적 예측을 함께 돌린다. **한 서버가 시뮬 정적 파일 +
`/detect`·`/traj`·`/predict`를 동일 출처로 서빙**한다(별도 http 서버 불필요).

```bash
uv sync --group serve
uv run python backend/detect_server.py --port 8001
```

→ <http://127.0.0.1:8001/sim.html?person=1>

가중치는 로컬 `training/`에 있으면 그걸, 없으면 허깅페이스에서 자동으로 받는다 —
팀원은 파일 전달 없이 위 두 줄만 실행하면 된다. 두 모델 모두 공개 저장소다:
검출은 [`chanubc/robot-kitchen-nadir-yolo11s`](https://huggingface.co/chanubc/robot-kitchen-nadir-yolo11s),
궤적 예측(LSTM)은 [`chanubc/human-move-lstm`](https://huggingface.co/chanubc/human-move-lstm).
저장소는 env로 교체(`DETECT_MODEL_REPO` / `PREDICT_MODEL_REPO`), 예측만 볼 땐
`DETECT_MODEL=none`(GT 좌표 사용).
파이프라인·수집·학습·평가 상세는 아래 [검출·예측 파이프라인](#검출예측-파이프라인-백엔드) 참조.

그냥 열면 **확정 배치(섬 배치)**가 뜬다 — 방 11.5×11.5 m · 천장 3.9 m ·
구역담당 CCTV 15대 · 안전링 3단계 · 로봇 팔 1.70 m. 화면 우측 정보줄에서 `방 11.5m`로 확인된다.

<details>
<summary>옛 배치로 열어야 할 때</summary>

확정 이전 배치(방 9 m · 천장 3.3 m · 카메라 10대 · 팔 1.30 m)는 비교용으로만 남겨 뒀다.

```
sim.html?layout=legacy
```

**여기서 데이터셋을 뽑지 말 것** — 팀 기준과 맞지 않는다.
`?layout=island`는 예전 주소 호환용으로 계속 받는다(이제 기본값과 같아 아무 효과가 없다).

</details>

> **`sim.html`을 파일로 직접 열지 말 것.** `file://`로 열면 3D 화면은 뜨지만
> ONNX 런타임이 wasm/mjs를 ES 모듈로 로드하지 못해 모델 검증이 동작하지 않는다.
> 그렇게 열면 화면 위에 경고가 뜬다.

**브라우저**: Chrome 또는 Edge 113+ 권장 (WebGPU). Safari는 그림자·MSAA에서 느리고,
WebGPU가 없으면 wasm으로 자동 폴백된다. 브라우저 탭을 앞에 두어야 한다 —
백그라운드로 내리면 렌더 루프가 멈춰 불·연기가 자라지 않고 캡처도 느려진다.

---

## 검출·예측 파이프라인 (백엔드)

백엔드 서버를 띄우는 법은 위 [실행 · 방법 B](#방법-b--검출예측까지-백엔드-서버). 이 절은 그 위에서
데이터를 만들고 모델을 학습·검증하는 흐름이다.

**파이프라인**: 나디르 CCTV → 검출(YOLO) → 추적(ByteTrack) → 사람별 (x,z) 궤적
→ **학습형 멀티모달 예측(경량 LSTM)** → 선제 안전(감속·회피).

- **궤적 데이터 수집**(예측기 학습용): 📊 데이터 탭 → *궤적 수집 시작*
  (또는 콘솔 `__sim.trajRun({scenes:40})`) → `dataset/trajectories/*.json`
  (좌표 시계열 JSON — 이미지 아님). 다양성은 `?layout=legacy`·`?half=`로 열어 각각 수집.
- **학습·평가**:
  ```bash
  uv run python train/train_traj_predictor.py    # 학습 + val ADE/FDE → docs/chanwoo/prediction-eval.md
  uv run --with pytest python -m pytest tests/   # 예측 코어 테스트
  ```
- **학습형(LSTM·Transformer) 모델 쓰는 법** — 재학습 불필요, 3단계:
  1. 백엔드를 띄운다([방법 B](#방법-b--검출예측까지-백엔드-서버)). 예측기 가중치가 로컬에 없으면
     허깅페이스에서 자동으로 받는다 — 기본은 LSTM [`chanubc/human-move-lstm`](https://huggingface.co/chanubc/human-move-lstm).
     **Transformer로 돌리려면** 서버를 `PREDICT_NET=transformer`로 띄운다 — 그러면
     [`chanubc/human-move-transformer`](https://huggingface.co/chanubc/human-move-transformer)를 받아
     self-attention 인코더로 추론한다. 두 모델은 입출력 계약(K모드 · 경로 12스텝 · 스텝별 σ)이
     같아 시뮬 쪽은 그대로다.
     ```bash
     uv run python backend/detect_server.py --port 8001                       # LSTM (기본)
     PREDICT_NET=transformer uv run python backend/detect_server.py --port 8001   # Transformer
     ```
     (Windows PowerShell: `$env:PREDICT_NET="transformer"; uv run python backend/detect_server.py --port 8001`.
      일부 conda 환경에서 torch가 `libiomp5md.dll` 중복으로 죽으면 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`.)
  2. 시뮬 상단 시각화 토글은 기본 꺼짐이다 — ⚙ 고급 탭 *시각화 표시*에서
     **바닥 밀도 음영**(과 원하면 **사람 예측 화살표**)을 켠다.
  3. **예측 모델** 드롭다운 → *학습형 — window.\_\_customPredictor* 선택. 이제 매 프레임
     백엔드 `/predict`가 호출돼(왕복 ~5 ms, 10 Hz) 밀도 구름에 **멀티모달 봉우리**가
     갈라져 뜨고 로봇이 선제 감속한다. 서버가 없거나 미로드면 조용히 칼만으로 폴백한다.
  - 교체 env: 로컬 파일 `PREDICT_MODEL=경로/model.pt` · 다른 허브 저장소
    `PREDICT_MODEL_REPO=... PREDICT_MODEL_FILE=...` · 아키텍처 `PREDICT_NET=lstm|transformer`.
  - 밀도 구름을 나디르(탑뷰·후드 제거)에서 찍은 예시 — 조리원의 곡선 동선을 따라 방향성 구름이
    퍼진다: [docs/chanwoo/assets/prediction/](docs/chanwoo/assets/prediction/) (LSTM 5컷 · Transformer 3컷).

설계·평가·sim2real 조사 문서 색인: **[docs/chanwoo/](docs/chanwoo/README.md)**.

---

## 폴더 구성

```
sim.html                    시뮬레이터 본체 (단일 파일)
index.html                  sim.html로 리다이렉트
babylon.js                  Babylon.js UMD 9.20.0
babylonjs.loaders.min.js    glTF/OBJ/STL 로더
babylon.inspector.bundle.js Babylon Inspector (선택 기능)
HavokPhysics_umd.js         Havok 물리 플러그인
HavokPhysics.wasm.js        Havok wasm (base64 내장)
character-manifest.json     캐릭터 리그 스펙 (보폭·사이클·오디트 결과)

assets/                     sim.html이 실제로 로드하는 GLB 20개
vendor/ort/                 onnxruntime-web (WebGPU/wasm 백엔드)
tools/                      Blender 에셋 생성 스크립트, 모델 어댑터 서버
launch/                     더블클릭 실행 스크립트 (Mac/Windows)
MODEL_HANDOFF.md            모델 담당자용 인계 문서 — 입출력 규격·회차 지표·한계
```

`assets/`에는 시뮬레이터가 실제로 부르는 20개만 둔다. 중간 리그 버전 등
작업 부산물은 저장소에 넣지 않는다 (`.gitignore` 참조).

---

## 에셋 재생성

주방 설비와 로봇 에셋은 Blender 스크립트로 절차 생성한다.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python tools/blender_assets.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python tools/blender_env.py
```

치수·색은 각 스크립트 상단 팔레트와 `build_*` 함수에서 고친 뒤 재실행하고
브라우저를 새로고침하면 반영된다.

**좌표 계약** (sim.html의 앵커와 1:1로 맞물려 있다. 깨면 배치가 어긋난다):

- 솥: 원점 = 바닥 중심, 림 높이 0.76 m
- `robot_jN`: 원점 = 관절 피벗, 팔 방향 = +X
- 단위 m, up-axis +Y

---

## 데이터셋 클래스

```
0 person   1 fire   2 smoke   3 robot   4 kettle   5 equipment
```

**예전에 뽑은 데이터와 섞지 말 것.** id가 바뀌었다.

---

## 알려진 한계

- **치수 미검증 (오차 ±25%)**. 배치와 비례는 실사 기준으로 맞췄지만,
  "몇 미터"라고 수치를 인용하면 안 된다.
- 참조 오버레이(`sim.html?ref=…`)는 `refs/` 폴더를 쓴다. 이 저장소에는 포함하지
  않으므로 해당 기능은 조용히 꺼진 상태로 동작한다.

모델 입출력 규격, 회차 지표, 궤적 예측 연결, 그 밖의 한계는 [MODEL_HANDOFF.md](MODEL_HANDOFF.md)에 있다.

---

## 라이선스

포함된 서드파티 라이브러리와 3D 에셋의 출처는 [ATTRIBUTION.md](ATTRIBUTION.md)를 참조.
이 저장소 자체의 라이선스는 아직 정해지지 않았다.
