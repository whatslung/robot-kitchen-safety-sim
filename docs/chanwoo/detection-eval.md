# 검출 모델 평가 — top-down person (YOLO11)

> 최종 갱신: 2026-08-18 · 담당: chanwoo
> 목적: 시뮬 합성 데이터로 학습한 top-down 사람 검출기가 **(A) 시뮬 안에서**, **(B) 실사로**
> 얼마나 되는지 정량화하고, **(C) 실사+합성 혼합이 도움이 되는지** 3-way로 검증한다.

관련: [[HANDOFF]] · 세션 [handoff/2026-08-17-session.md](handoff/2026-08-17-session.md) · 참고 실사 레포 `overhead-person-yolo11`

---

## 0. 환경 · 데이터 · 모델

| 항목 | 값 |
|---|---|
| 모델 | YOLO11s (COCO 사전학습 → 파인튜닝), ultralytics 8.4.121 |
| GPU / torch | RTX 5070 (12GB, sm_120) / torch 2.11.0+cu128 |
| 공통 하이퍼파라미터 | imgsz 640, epochs 100, patience 20, seed 42, batch=-1(auto) |
| 클래스(6) | `0 person · 1 fire · 2 smoke · 3 robot · 4 kettle · 5 equipment` |
| sim 데이터 | `sim-person/` 200장(직교 나디르 `orthotop`), train 160 / val 40 |
| 실사 데이터 | Roboflow `overhead-person-szky0` v3 — 실사 top-down, person 단일. test 137장 |

> 실사 라벨은 class 0(person)만 존재 → 6클래스 공간에 그대로 얹어도 무해. 덕분에 sim 모델을
> 재학습 없이 실사에서 바로 평가할 수 있다. 실사 평가는 항상 person(class 0)만 집계.

---

## 1. Sim in-domain — 시뮬 합성 val (40장)

파인튜닝 전/후 (person):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| stock yolo11s (COCO, 파인튜닝 전) | 0.374 | 0.175 | 0.212 | 0.094 |
| **파인튜닝 후** | **0.797** | **0.688** | **0.747** | 0.373 |

클래스별 (파인튜닝 후):

| 클래스 | P | R | mAP50 |
|---|---|---|---|
| person | 0.797 | 0.688 | 0.747 |
| smoke | 0.884 | 1.000 | 0.995 |
| robot | 0.909 | 0.973 | 0.981 |
| kettle | 0.991 | 1.000 | 0.995 |
| equipment | 0.884 | 0.868 | 0.893 |
| **전체** | 0.893 | 0.906 | 0.922 |

**판정: make-or-break PASS.** stock 대비 recall 3.9배·precision 2.1배. 특히 equipment를 825개
인스턴스에서 recall 0.868로 "설비로" 잡음 → **설비를 person으로 오탐하지 않음**(person precision 0.797의 근거).
= 나디르 top-down 검출이 시뮬 파인튜닝으로 살아남. 지난 실사 overhead 모델의 시뮬 top-down 성능
(recall 0.17 / precision 0.30)을 크게 상회.

### 1-1. imgsz 실험 — 해상도는 병목이 아니다

원본 960×720을 640으로 줄여 학습하므로 "작은 객체 때문 아닌가"를 검증(imgsz=960 재학습):

| imgsz | person P | person R | mAP50 |
|---|---|---|---|
| 640 | 0.797 | **0.688** | 0.747 |
| 960 | 0.926 | **0.544** | 0.769 |

→ 고해상도는 precision만 올리고 **recall은 오히려 하락**. 놓치는 사람은 "너무 작아서"가 아니라
**그런 케이스(가림·구석·특이 자세)를 학습에서 못 봐서** = **데이터 양·다양성 문제**로 확정.

---

## 2. Sim-to-real — 시뮬 모델을 실사 test(137장)에

sim-only(위 640 모델, 합성만 학습)을 실사 test에서 person 평가:

| 학습 | 평가 | Recall | Precision | mAP50 |
|---|---|---|---|---|
| sim-only | 실사 test | **0.270** | **0.072** | 0.048 |

**갭이 크다(예상대로).** recall 0.27(실사 사람 27%만 잡음), precision 0.072(친 박스의 93%가 오탐).
합성에서 배운 특징이 실사 픽셀에 전이 안 됨.

**갭의 원인 3중**:
1. **도메인** — 합성 무광 렌더 질감 ≠ 실사 픽셀 (최대 요인, precision 붕괴의 주범).
2. **맥락** — 우리 학습=주방(조리복·솥·설비), 이 실사셋=일반 오버헤드(평상복·비주방).
3. **투영** — 학습=직교·5m 나디르, 실사=원근·다양한 높이/각도.

> 대조: 참고 모델은 이 실사 데이터로 **직접** 학습해 recall 0.98. 데이터·방법은 정상이고,
> **"합성만으로는 실사 전이가 안 된다"**가 정량화된 것.

---

## 3. 3-way — 실사+합성 혼합이 도움이 되나 (limited-real)

**설계**: 실사가 풍부하면(4,120장) real-only가 이미 천장이라 sim 기여가 안 보인다. 그래서
**실사를 500장으로 제한**(주방처럼 실사 라벨이 부족한 현실 모사)하고, 거기에 sim 200장을 더했을 때
실사 test 성능이 오르는지 본다. 세 조건 모두 **동일 실사 test 137장**에서 person 평가.

데이터 분리 검증(파일명 교집합): train(500)∩val(150)=0, train∩test(137)=0, val∩test=0,
real+sim train∩test=0 → **누수 없음**. test 137장은 Roboflow 별도 test 폴더로 세 모델 모두 held-out.

| 조건 | 실사 학습량 | Recall | Precision | mAP50 |
|---|---|---|---|---|
| sim-only | 0 (sim 200) | 0.270 | 0.072 | 0.048 |
| real-only | 500 | 0.829 | 0.848 | 0.879 |
| **real + sim** | 500 (+sim 200) | **0.844** | **0.860** | **0.898** |
| real-full (참고 레포 `overhead-person-yolo11`) | ~3,406 | 0.980 | 0.969 | 0.991 |

**결론**:
1. **합성만으로는 실사 전이 실패** (recall 0.27) — 도메인 갭 확인.
2. **실사가 지배적** — 500장만으로 recall 0.83, 전량이면 0.98(천장).
3. **real+sim > real-only 전 지표** (recall +1.5pp·precision +1.2pp·mAP50 +1.9pp) →
   **제한된 실사 체제에서 합성이 소폭이지만 일관되게 도움**을 주고, **해치지 않음**.
4. 상승폭이 작은 이유: 이 실사 test가 **일반 오버헤드(비주방)**라 우리 합성의 주방 특화
   내용(조리복·솥)이 완전히 맞물리지 않음. **실사 주방 test라면·실사가 더 적을수록** 합성 기여는
   커질 것으로 예상(합성의 본래 용도 = 실사 주방 라벨 부족 보완).

**시사점**: 합성 데이터 파이프라인은 실배포에 **양의 기여**를 하는 것으로 실증됨. 다음은
(a) 실사 주방 라벨 확보 후 재측정, (b) 도메인 랜덤화로 합성 기여폭 확대, (c) 더 적은 실사에서의 기여 곡선.

---

## 4. 재현 방법

```bash
# 0) 환경
uv sync

# 1) sim 데이터 분할 + data.yaml
uv run python train/prepare_yolo_split.py sim-person --val-ratio 0.2

# 2) sim 파인튜닝 (make-or-break)
uv run python train/train_sim.py

# 3) stock 기준선
uv run python train/eval_stock.py

# 4) 실사 데이터 다운로드 (.env에 ROBOFLOW_API_KEY)
uv run --no-project --with roboflow python -c "import os;from roboflow import Roboflow;\
Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']).workspace('riccardo-kxtut')\
.project('overhead-person-szky0').version(3).download('yolov11', location='dataset/overhead-person-v3')"

# 5) sim-to-real 평가
uv run python train/eval_real.py dataset/overhead-person-v3/data.yaml --split test

# 6) 3-way 데이터 구성 + 학습 + 평가
uv run python train/prep_3way.py
uv run yolo detect train model=yolo11s.pt data=dataset/3way/real_only.yaml epochs=100 imgsz=640 \
    batch=-1 device=0 patience=20 seed=42 project=$PWD/training name=real_only
uv run yolo detect train model=yolo11s.pt data=dataset/3way/real_sim.yaml  epochs=100 imgsz=640 \
    batch=-1 device=0 patience=20 seed=42 project=$PWD/training name=real_sim
uv run python train/eval_real.py dataset/3way/real_only.yaml --split test --weights training/real_only/weights/best.pt
uv run python train/eval_real.py dataset/3way/real_sim.yaml  --split test --weights training/real_sim/weights/best.pt
```

산출물: `training/<name>/weights/best.pt`(+`.onnx`), 지표 `training/summary.json`·`training/real_eval.json`.
데이터셋·가중치·`training/`은 gitignore 대상(용량).

---

## 5. 모델 아키텍처 비교 — YOLO vs YOLO26 vs RF-DETR (2026-08-19)

대안 검출기 조사. 스크립트 `train/train_rfdetr.py`·`train/yolo_to_coco.py`(YOLO→COCO 변환).

### 5-1. 합성 val (in-domain, person)
| 모델 | mAP50 | Recall |
|---|---|---|
| YOLO11s | 0.747 | 0.688 |
| YOLO26s (동일 등급) | 0.750 | 0.634 |
| YOLO26n (참고) | 0.521 | 0.345 |

→ YOLO26 ≈ YOLO11s (무승부). STAL(소객체 라벨할당)이 소규모 합성엔 이득 없음 → **YOLO 계열 안에선 아키텍처가 병목 아님**.

### 5-2. 3-way sim-to-real (실사 test 137장, mAP50 / Recall)
| 학습 | YOLO11s | RF-DETR-Nano |
|---|---|---|
| sim-only | 0.048 / 0.270 | **0.411 / 0.479** |
| real-only (500) | 0.879 / 0.829 | **0.943 / 0.917** |
| real+sim (700) | 0.898 / 0.844 | 0.905 / 0.900 |
| real-full (YOLO ~3,406) | 0.980 / — | — |

**핵심**:
1. **sim→real 전이는 RF-DETR 압승**(0.411 vs 0.048). 원인 = RF-DETR의 **DINOv2 자기지도 백본** — 라벨 없이 실사 17억 장으로 배운 도메인 불변 특징이 합성 과적합을 막음. YOLO는 합성 표면에 과적합(sim val 0.688 ↔ 실사 0.048).
2. **데이터 효율**: RF-DETR 실사 500장(0.943) ≈ YOLO 전량 3,406장(0.980).
3. **합성의 값어치는 모델에 따라 반대**: 약한 YOLO엔 도움(0.879→0.898), 강한 RF-DETR엔 소폭 해로움(0.943→0.905, 이미 잘 일반화라 합성이 희석). → 강한 백본엔 "합성 많이"보다 "실사 조금".

### 5-3. 크기·속도 (온디바이스 트레이드오프)
| | params | 체크포인트 | GPU(RTX 5070) | CPU |
|---|---|---|---|---|
| YOLO11s | 9.4M | 19M(.pt)/37M(onnx) | 6.3 ms | 29 ms |
| RF-DETR-Nano | 30.5M | 116M | 10.1 ms | (ViT — 훨씬 느림) |

→ GPU에선 둘 다 실시간(RF-DETR ~1.6× 느림·~3× 큼). **불리함은 CPU/브라우저(onnxruntime-web)에서 큼** — ViT/어텐션이 무겁고 ort-web 배포가 까다로움. YOLO는 브라우저 배포가 가볍고 빠름.

### 5-4. 라이선스
YOLO11/26·YOLO-Master = **AGPL-3.0**(상용 시 소스공개 의무). RF-DETR·YOLOX = **Apache-2.0**(상용 클린). DINOv3 = 커스텀 라이선스(상용 허용).

### 5-5. 결론 / 권고
- **정확도·sim-to-real·실사효율 = RF-DETR 우위**, 라이선스도 상용 클린.
- **배포 경량성 = YOLO 우위**(특히 브라우저).
- **권고**: (a) 정확도 필요한 **서버(detect_server, GPU)엔 RF-DETR**, 브라우저 데모엔 YOLO **하이브리드**; 또는 (b) **RF-DETR(교사)→YOLO(학생) 지식 증류**로 "파운데이션 품질 + 엣지 속도" 결합.
- **YOLOX**: 2021 코드로 최신 스택(py3.12/torch2.11) 설치 실패 + 정확도 열위 → 스킵.

