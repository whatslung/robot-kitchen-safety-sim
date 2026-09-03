# 급식 조리로봇 안전 시뮬레이터

급식 조리실의 교반솥(국솥) 로봇을 대상으로 사람 접근, 충돌·끼임, 화재 감지와 대피를
재현하는 Babylon.js 기반 웹 3D 시뮬레이터다. 합성 학습 데이터 생성부터 검출·추적·궤적
예측, 모델 기반 안전 정지 검증까지 한 저장소에서 다룬다.

[웹 데모](https://robot-kitchen-safety-sim.vercel.app) ·
[모델·평가 문서](docs/chanwoo/README.md) ·
[모델 연동 규격](MODEL_HANDOFF.md) ·
[출처 및 라이선스](ATTRIBUTION.md)

> 웹 데모는 정적 프런트엔드만 배포한다. 3D 시뮬레이션, 발표 시나리오, 브라우저 ONNX
> 검증과 데이터 생성은 사용할 수 있지만 Python 백엔드의 `/detect`, `/predict`, `/nadir`,
> `/traj` API는 포함하지 않는다.

이 프로젝트는 연구·시연용 시뮬레이터다. 실제 설비의 안전 제어기나 안전 인증을 대신하지
않는다.

## 주요 기능

| 영역 | 현재 구현 |
|---|---|
| 3D 시뮬레이션 | 교반로봇, 조리원, 국솥, 카트와 주방 설비를 Babylon.js로 렌더링하고 Havok 물리를 적용 |
| 안전 시연 | 사람 접근에 따른 감속·정지, 예측 회피, 접촉 E-STOP, 충돌·끼임·실신 시나리오 |
| 화재 시연 | 정상 조리 → 과열·발화 → 연기 → 감지 → 음성 경보 → 대피 흐름의 before/after 비교 |
| 카메라·센서 | 기본 발표 카메라 10대, PiP, 사용자 카메라 추가, CCTV 노이즈·왜곡·김서림·저해상도 재현 |
| 합성 데이터 | 카메라별 RGB, 클래스·인스턴스 마스크, 깊이, YOLO 라벨, 메타데이터와 조건 CSV 생성 |
| 모델 검증 | 내장 모의 모델, 외부 HTTP 모델, 브라우저 ONNX(WebGPU/wasm), 커스텀 JavaScript 모델 |
| 검출·예측 백엔드 | YOLO + ByteTrack, 다중카메라 월드 융합, LSTM/Transformer 멀티모달 궤적 예측과 위험 중재 |

## 실행

### 1. 웹 데모

[https://robot-kitchen-safety-sim.vercel.app](https://robot-kitchen-safety-sim.vercel.app)

설치 없이 최신 `main`의 정적 프런트엔드를 확인할 수 있다. 처음 열면 `발표` 탭이 보인다.

### 2. 로컬 정적 실행

빌드나 패키지 설치 없이 저장소 루트에서 정적 서버만 실행한다.

```bash
git clone https://github.com/whatslung/robot-kitchen-safety-sim.git
cd robot-kitchen-safety-sim
python -m http.server 5173
```

브라우저에서 <http://localhost:5173/sim.html>을 연다.

더블클릭 실행은 `launch/실행하기_Mac.command` 또는 `launch/실행하기_Windows.bat`을
사용한다. macOS 스크립트는 5173~5199에서 빈 포트를 찾고, Windows 스크립트는 5173을
사용한다. 둘 다 Chrome 또는 Edge를 우선해 연다.

정적 실행에서도 3D 시뮬레이션, 시나리오, 데이터 생성과 브라우저 ONNX 검증은 사용할 수
있다. Python 기반 YOLO·ByteTrack과 학습형 궤적 예측은 동작하지 않으며, 궤적 수집 결과를
`dataset/trajectories/`에 저장하려면 백엔드가 필요하다.

> `sim.html`을 `file://`로 직접 열지 않는다. ONNX Runtime의 wasm/mjs 모듈 로딩과 일부
> 브라우저 API가 정상 동작하지 않는다.

### 3. 검출·예측 백엔드 포함 실행

요구 사항은 Python 3.11~3.13과 [uv](https://docs.astral.sh/uv/)다. 저장소의 기본 Python
버전은 `.python-version`에 지정된 3.12다. 현재 `pyproject.toml`은 RTX 5070/Blackwell
학습 환경에 맞춰 PyTorch와 torchvision을 CUDA 12.8 전용 인덱스로 고정한다. 따라서 아래
공식 설치 명령은 cu128 휠이 있는 Linux/Windows 환경을 기준으로 한다.

```bash
uv sync --group serve
uv run python backend/detect_server.py --port 8001
```

브라우저에서 <http://127.0.0.1:8001/sim.html>을 연다. 이 서버 하나가 정적 파일과 API를
같은 출처로 제공하므로 별도의 `http.server`는 필요하지 않다.

서버를 시작할 때 로컬 검출 가중치가 없으면 Hugging Face에서 내려받는다. 궤적 예측기
가중치는 첫 `/predict` 요청 때 지연 로드한다. 따라서 각 모델을 처음 사용할 때는 인터넷
연결이 필요하고, 이후에는 Hugging Face 캐시를 사용한다.

서버 상태는 다음 주소에서 확인한다.

```bash
curl http://127.0.0.1:8001/health
```

정상 검출 구성이라면 `mode`가 `yolo+bytetrack`이다. `predict_net`은 선택된 아키텍처가
`lstm`인지 `transformer`인지만 보여주며, 가중치 로드 성공을 뜻하지는 않는다. 검출
가중치 로드에 실패하면 현재 구현은 서버를 `dummy` 모드로 유지한다. 이는 정상 검출이
아니므로 터미널의 원래 로드 오류와 `/health` 응답을 확인해 원인을 해결해야 한다. 예측기
로드 실패 시 `/predict`는 오류 원인을 담은 HTTP 503을 반환한다.

## 1분 사용법

1. `발표` 탭에서 `▶ ① before`를 실행해 화재를 놓치는 흐름을 본다.
2. `↩ ① 되돌리기` 후 `▶ ① after`를 실행해 감지·경보·대피 흐름을 비교한다.
3. `▶ 충돌 (회피 OFF)`와 `▶ 회피 (예측)`으로 사람 동선 안전 개입 전후를 비교한다.
4. `실행` 탭에서 `▶ 자동 작업`을 누르고, `Tab`으로 사람을 선택한 뒤 `WASD`로 로봇에
   접근해 감속·정지를 확인한다.
5. `데이터` 탭에서 GT 미리보기, 데이터셋 생성, ONNX 또는 HTTP 모델 검증을 실행한다.

`H`를 누르면 화면 안 도움말이 열린다. 주요 조작은 다음과 같다.

| 입력 | 동작 |
|---|---|
| `WASD` | 현재 조작 대상 이동 |
| `Shift` | 빠르게 이동 |
| `Tab` | 자유 시점 → 사람 → 로봇 관절 순환 |
| `Q` / `E` | 자유 시점 높이 이동 |
| `C` | 현재 주 시점 캡처 |
| `Shift+C` | 주 시점과 PiP 함께 캡처 |
| `P` / `X` | PiP 순환 / 주 시점과 교환 |
| `Space` | 로봇 정지 |

최신 Chrome 또는 Edge를 권장한다. 브라우저 ONNX 추론은 WebGPU를 먼저 사용하고,
WebGPU 세션 생성이 안 되면 wasm으로 다시 시도한다. 데이터셋을 폴더 구조 그대로 저장하는
File System Access API도 Chromium 계열에서 가장 안정적이다. 브라우저 탭을 백그라운드로
내리면 렌더 루프가 제한되어 화재·연기 진행과 캡처 속도가 달라질 수 있다.

## 검출·추적·예측 파이프라인

```text
CCTV 프레임
  → YOLO 검출
  → ByteTrack 추적
  → 카메라 좌표를 월드 (x, z)로 융합
  → LSTM 또는 Transformer 멀티모달 궤적 예측
  → 위험도 중재
  → 로봇 감속·정지·회피
```

백엔드의 주요 엔드포인트는 다음과 같다.

| 엔드포인트 | 역할 |
|---|---|
| `GET /health` | 검출 모드와 궤적 예측 아키텍처 확인 |
| `POST /detect` | 단일 카메라 YOLO 검출 + ByteTrack ID·이미지 평면 속도 |
| `POST /predict` | 단일 또는 다인원 궤적의 K=3 멀티모달 예측과 위험 중재 |
| `POST /nadir` | 다중 나디르 카메라 검출 → 월드 융합 → 추적 → 예측 |
| `POST /traj` | 시뮬 궤적 scene을 `dataset/trajectories/`에 저장 |
| `POST /dataset` | 이미지와 YOLO 라벨 쌍을 `dataset/<set>/`에 저장 |
| `POST /shot` | 브라우저 캡처를 `captures/`에 저장 |

기본 모델과 교체 환경변수는 다음과 같다.

| 모델 | 기본값 | 교체 방법 |
|---|---|---|
| 사람 검출 | [`chanubc/robot-kitchen-nadir-yolo11s`](https://huggingface.co/chanubc/robot-kitchen-nadir-yolo11s) | `DETECT_MODEL`, `DETECT_MODEL_REPO`, `DETECT_MODEL_FILE` |
| LSTM 궤적 예측 | [`chanubc/human-move-lstm`](https://huggingface.co/chanubc/human-move-lstm) | `PREDICT_MODEL`, `PREDICT_MODEL_REPO`, `PREDICT_MODEL_FILE` |
| Transformer 궤적 예측 | [`chanubc/human-move-transformer`](https://huggingface.co/chanubc/human-move-transformer) | `PREDICT_NET=transformer`와 위 예측기 변수 |

```bash
# Transformer 백본
PREDICT_NET=transformer uv run python backend/detect_server.py --port 8001

# 검출을 명시적으로 끄고 GT 좌표 기반 예측만 사용
DETECT_MODEL=none uv run python backend/detect_server.py --port 8001

# 로컬 재학습 가중치
DETECT_MODEL=training/real_sim/weights/best.pt \
PREDICT_MODEL=training/traj_predictor/model.pt \
uv run python backend/detect_server.py --port 8001
```

LSTM과 Transformer의 공통 계약은 관측 8스텝(3.2초) → 예측 12스텝(4.8초), K=3 모드와
스텝별 불확실성 `sigma`다. `데이터` 탭에서 예측 모델을 `학습형 —
window.__customPredictor`로 선택하면 백엔드 `/predict`를 사용한다. 백엔드가 없거나 예측
요청이 실패하면 화면의 학습형 경로는 폐기되고 기본 Kalman 예측으로 돌아간다.

## 합성 데이터 생성

`데이터` 탭의 `데이터셋 생성`은 등록된 모든 카메라에 대해 한 scene을 촬영한다. 기본
카메라는 10대이며, 사용자가 카메라를 추가하거나 삭제하면 생성 대상도 함께 바뀐다.

Chrome/Edge에서는 먼저 저장할 폴더를 고른다. 결과 구조는 다음과 같다.

```text
선택한 폴더/
├── images/                  RGB 960×720 PNG
├── labels/                  YOLO txt
├── gt/                      클래스 마스크, 인스턴스 마스크, 깊이 2종, 메타데이터
├── classes.txt              클래스 순서
├── conditions.csv           scene·화재·연기·인원·센서 조건
└── dataset.json             생성 회차와 카메라·클래스 명세
```

`12칸 균등 생성`은 평상시/화재 3단계 × 렌즈 김서림 3단계를 같은 수로 만들어 평가셋을
구성할 때 사용한다. 학습셋은 실제보다 화재 비중이 과도해지지 않도록 일반 생성 흐름을
권장한다.

클래스 ID는 다음 순서로 고정되어 있다.

```text
0 person
1 fire
2 smoke
3 robot
4 kettle
5 equipment
```

이전 클래스 순서로 생성한 데이터와 섞지 않는다.

궤적 예측기용 데이터는 백엔드를 실행한 상태에서 `데이터` 탭의 `궤적 수집 시작`으로
만든다. 기본 40개 scene을 2.5 Hz로 수집해 `dataset/trajectories/*.json`에 저장한다.
형식과 예시는 [dataset/trajectories/README.md](dataset/trajectories/README.md)에 있다.

## 학습과 검증

대표 명령만 아래에 둔다. 검출 학습·평가의 전체 설명과 지표는
[train/README.md](train/README.md), 궤적 예측과 안전 평가는
[docs/chanwoo/README.md](docs/chanwoo/README.md)를 기준으로 한다.

```bash
# YOLO 데이터 분할과 학습
uv run python train/prepare_yolo_split.py sim-person --val-ratio 0.2
uv run python train/train_sim.py

# 궤적 예측기 학습
uv run python train/train_traj_predictor.py
uv run python train/train_traj_transformer.py

# Python 테스트
uv run --with pytest python -m pytest tests/
```

이 테스트 명령도 프로젝트 의존성을 먼저 동기화하므로 현재 패키지 설정에서는 cu128 휠을
설치할 수 있는 환경이 필요하다.

학습 산출물과 전체 데이터셋은 크기가 커서 Git에 포함하지 않는다. 기본 로컬 경로는
`training/`과 `dataset/`이며 `.gitignore` 규칙을 따른다.

## 실행 옵션

자주 쓰는 URL 파라미터만 정리했다.

| 예시 | 용도 |
|---|---|
| `sim.html?layout=legacy` | 확정 전 9 m 비교 배치 |
| `sim.html?layout=corridor` | 13×7 m 복도형 비교 배치 |
| `sim.html?layout=island&half=5.0` | 섬 배치의 방 반너비 변경 |
| `sim.html?reach=1.3` | 로봇 팔 도달거리 변경 |
| `sim.html?fryer=1.40` | 실측 솥 외경을 기준으로 scene 단위 환산 |
| `sim.html?fire=0.7` | 일반 데이터 생성의 화재 scene 비율 변경(0~1) |
| `sim.html?seed=7` | scene 난수 시드 고정 |
| `sim.html?person=0` | 주 조리원 숨김 |
| `sim.html?console=1` | 화면 아래 개발 콘솔 표시 |

`?layout=island`는 이전 링크 호환용이며 현재 기본 배치와 같다. 팀 기준 데이터는 기본 섬
배치에서 생성하고, `legacy`, `corridor`, `half` 변경본은 비교·다양성 실험으로 구분한다.

## 저장소 구성

```text
sim.html                    시뮬레이터 본체
index.html                  sim.html 진입 페이지
backend/detect_server.py    FastAPI 정적 서버 + 검출·추적·예측 API
trajectory/                 궤적 타입, 예측기, 위험도·평가 로직
train/                      YOLO와 궤적 예측기 학습·평가 스크립트
tests/                      Python 단위·통합 테스트와 브라우저 검증 스크립트
dataset/trajectories/       저장소에 포함한 궤적 형식 예시
assets/                     시뮬레이터용 GLB 21개
vendor/ort/                 onnxruntime-web 로컬 런타임
tools/                      Blender 에셋 생성·모델 변환 도구
launch/                     macOS/Windows 더블클릭 실행 스크립트
docs/chanwoo/               검출·예측·안전 평가와 발표 자료
vercel.json                 정적 프런트엔드 배포 설정
MODEL_HANDOFF.md            모델 입출력 계약과 인계 문서
ATTRIBUTION.md              서드파티 라이브러리·에셋 출처
```

## 에셋 재생성

주방 설비와 로봇 에셋은 Blender 스크립트로 절차 생성한다.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python tools/blender_assets.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python tools/blender_env.py
```

좌표 계약은 `sim.html`의 앵커와 직접 연결된다.

- 솥 원점: 바닥 중심
- `robot_jN` 원점: 관절 피벗
- 로봇 팔 방향: +X
- 단위: m, up-axis: +Y

치수와 색을 바꿀 때는 각 스크립트의 팔레트와 `build_*` 정의를 수정한 뒤 에셋을 다시
생성한다.

## 알려진 한계

- 절대 치수는 실측 확정 전이며 오차 범위를 ±25%로 본다. 배치·비례 검증에는 쓸 수 있지만
  안전거리나 설비 치수를 실측값처럼 인용하지 않는다.
- 합성만으로 학습한 검출기는 실사 도메인 갭이 크다. 실사 평가와 real+sim 파인튜닝 결과를
  함께 확인해야 한다.
- 현재 uv 의존성 설정은 CUDA 12.8 PyTorch 휠 전용이다. macOS arm64에서는
  `uv sync --group serve`가 호환 휠 부재로 실패하며, macOS용 실행 스크립트는 정적
  프런트엔드만 실행한다.
- `sim.html?ref=...`가 사용하는 참조 이미지는 저장소에 포함하지 않았다.
- Vercel 배포는 정적 프런트엔드만 포함한다. Python 모델 API와 로컬 파일 저장은 로컬
  백엔드 실행이 필요하다.
- 이 저장소 자체의 라이선스는 아직 정해지지 않았다. 포함된 서드파티 구성요소의 조건은
  [ATTRIBUTION.md](ATTRIBUTION.md)를 확인한다.
