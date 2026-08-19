# 검출 모델 평가 — top-down person (YOLO11)

> 최종 갱신: 2026-08-19 · 담당: chanwoo
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

## 5. 모델 아키텍처 비교 — YOLO vs YOLO26 vs RF-DETR (2026-08-19)

대안 검출기 조사. 스크립트 `train/train_rfdetr.py`·`train/yolo_to_coco.py`(YOLO→COCO 변환).

### 5-1. 합성 val (in-domain, person)
| 모델 | mAP50 | Recall |
|---|---|---|
| YOLO11s | 0.747 | 0.688 |
| YOLO26s (동일 등급) | 0.750 | 0.634 |
| YOLO26n (참고) | 0.521 | 0.345 |

→ YOLO26 ≈ YOLO11s (무승부). STAL(소객체 라벨할당)이 소규모 합성엔 이득 없음 → **YOLO 계열 안에선 아키텍처가 병목 아님**.

### 5-2. 3-way sim-to-real (실사 test 137장) — Recall / Precision / mAP50
| 학습 | 모델 | Recall | Precision | mAP50 |
|---|---|---|---|---|
| sim-only | YOLO11s | 0.270 | 0.072 | 0.048 |
| sim-only | **RF-DETR** | **0.479** | **0.455** | **0.411** |
| real-only(500) | YOLO11s | 0.829 | 0.848 | 0.879 |
| real-only(500) | **RF-DETR** | **0.917** | **0.931** | **0.943** |
| real+sim(700) | YOLO11s | 0.844 | 0.860 | 0.898 |
| real+sim(700) | RF-DETR | 0.900 | 0.877 | 0.905 |
| real-full(3,970) | YOLO11s | 0.845 | 0.878 | 0.849 |

> ⚠️ real-full(3,970)이 real-only(500)와 사실상 동일(0.849 vs 0.879) — **실사를 8배 늘려도 held-out test에선 안 오름**. 이유는 §5-7(iid vs cross-distribution). 참고 레포의 "recall 0.98"은 iid 재분할 test 기준.

**핵심**:
1. **sim→real 전이는 RF-DETR 압승**(0.411 vs 0.048). 원인 = RF-DETR의 **DINOv2 자기지도 백본** — 라벨 없이 실사 17억 장으로 배운 도메인 불변 특징이 합성 과적합을 막음. YOLO는 합성 표면에 과적합(sim val 0.688 ↔ 실사 0.048).
2. **데이터 효율**: RF-DETR 실사 500장(0.943) ≈ YOLO 전량 3,406장(0.980).
3. **합성의 값어치는 모델에 따라 반대**: 약한 YOLO엔 도움(0.879→0.898), 강한 RF-DETR엔 소폭 해로움(0.943→0.905, 이미 잘 일반화라 합성이 희석). → 강한 백본엔 "합성 많이"보다 "실사 조금".

### 5-3. 크기·속도 (온디바이스 트레이드오프)
| | params | 체크포인트 | GPU(RTX 5070) | CPU |
|---|---|---|---|---|
| YOLO11s | 9.4M | 19M(.pt)/37M(onnx) | 6.3 ms | 29 ms |
| RF-DETR-Nano | 30.5M | 116M | 10.1 ms | 94 ms |

→ GPU에선 둘 다 실시간(RF-DETR ~1.6× 느림·~3× 큼). **CPU는 RF-DETR ~3.2× 느림**(94 vs 29ms, ~10fps). 불리함은 **CPU/브라우저(onnxruntime-web)에서 큼** — ViT/어텐션이 무겁고 ort-web(wasm) 배포는 더 느리고 까다로움. YOLO는 브라우저 배포가 가볍고 빠름.

### 5-4. 라이선스
YOLO11/26·YOLO-Master = **AGPL-3.0**(상용 시 소스공개 의무). RF-DETR·YOLOX = **Apache-2.0**(상용 클린). DINOv3 = 커스텀 라이선스(상용 허용).

### 5-5. 결론 / 권고
- **정확도·sim-to-real·실사효율 = RF-DETR 우위**, 라이선스도 상용 클린.
- **배포 경량성 = YOLO 우위**(특히 브라우저).
- **권고**: (a) 정확도 필요한 **서버(detect_server, GPU)엔 RF-DETR**, 브라우저 데모엔 YOLO **하이브리드**; 또는 (b) **RF-DETR(교사)→YOLO(학생) 지식 증류**로 "파운데이션 품질 + 엣지 속도" 결합.
- **YOLOX**: 2021 코드로 최신 스택(py3.12/torch2.11) 설치 실패 + 정확도 열위 → 스킵.

### 5-6. 지식 증류 — RF-DETR 교사 → YOLO 학생 (실사 GT 0장)
`train/distill_pseudo.py`: RF-DETR(sim-only, 실사 전이 0.411)로 실사 이미지에 pseudo-label 생성 → 그 라벨로만 YOLO11s 학생 학습 → 실사 test(GT) 평가. 교사 conf(pseudo 채택 하한)로 pseudo 밀도 조절.

| 학습 | Recall | Precision | mAP50 | pseudo 박스 |
|---|---|---|---|---|
| YOLO sim-only | 0.270 | 0.072 | 0.048 | — |
| YOLO 증류 conf 0.5 | 0.327 | 0.473 | 0.233 | ~1,600 |
| **YOLO 증류 conf 0.25 (최적)** | **0.533** | **0.568** | **0.491** | ~3,900 |
| YOLO 증류 conf 0.1 | 0.529 | 0.466 | 0.418 | ~11,111 |
| RF-DETR 교사(sim-only) | 0.479 | 0.455 | 0.411 | — |
| YOLO real-GT(500) 참고 | 0.829 | 0.848 | 0.879 | — |

→ **실사 라벨 0장으로 YOLO가 0.048 → 0.491 (~10×), 교사(0.411)까지 초과** (student-beats-teacher).
- **conf 0.25가 최적.** 0.1로 더 낮추면 pseudo 2.8배(→11,111)여도 **recall 정체(0.533→0.529)·precision↓** — 노이즈만 추가.
- **⚠️ recall 상한 ≈ 0.53 (sim-only 증류의 한계).** 교사가 못 본 사람은 pseudo가 안 생겨 학생도 못 배움 → **안전 목표 recall 0.98은 sim-only 증류로 원리적 불가.**
- **0.98 도달 경로**: (a) 실사 주방 수천 장 라벨(참고 레포 3,406장→0.98 실증), (b) 강한 교사(RF-DETR real-only 0.917) 증류, (c) **시스템 recall** — 낮은 conf + ByteTrack + 칼만으로 단일프레임 ~0.9를 시스템 ~0.98로. 최종 목표 지표 = 단일프레임 recall이 아니라 `blindPct→0`.
- 증류의 값어치 = **실사 라벨이 없을 때 부트스트랩**("빠른 YOLO + 파운데이션 전이", 라벨 0).

### 5-7. iid vs cross-distribution — "recall 0.98"의 진짜 조건 (핵심 결론)
YOLO real-full(3,970) 학습 결과를 두 평가셋에서:

| 평가셋 | Recall | Precision | mAP50 |
|---|---|---|---|
| **val (150, train과 동분포 = iid)** | **0.978** | 0.974 | **0.992** |
| **test (137, 원본 v3 held-out = 다른 분포)** | 0.845 | 0.878 | 0.849 |

**0.98은 iid(같은 분포)에서만 난다.** 참고 레포의 recall 0.98도 그들이 재분할한 iid test 기준. 원본 v3의 다른-분포 held-out(137)에선 YOLO가 **실사 3,970장으로도 0.85 포화**(500장 0.879과 동일 — 데이터 양이 답이 아님).

**같은 137 held-out에서 아키텍처 대비**:
| 학습 | Recall | mAP50 |
|---|---|---|
| YOLO real-500 | 0.829 | 0.879 |
| YOLO real-3970 (8×↑) | 0.845 | 0.849 |
| **RF-DETR real-500** | **0.917** | **0.943** |

→ **진짜 난제는 "데이터 양"이 아니라 "분포를 건너뛰는 것(cross-distribution)".** YOLO는 데이터 8배로도 0.85 포화, RF-DETR은 500장으로 0.94 — 분포가 다를 때 **DINOv2 파운데이션 백본이 일반화로 이긴다.**

**안전 목표 recall 0.98의 조건**:
- 배포 현장 = 학습 분포(그 주방·그 카메라를 라벨) → 0.98 가능(iid).
- 배포 현장 ≠ 학습 분포(새 급식실 = 현실) → cross-distribution. 답은 데이터 양이 아니라 **① 파운데이션 백본(RF-DETR/DINOv3) + ② 현장 소량 실사 + ③ 시스템 recall(낮은 conf·ByteTrack·칼만 → `blindPct→0`)**.

---

## 6. 섬 배치 재생성·재학습 (2026-08-19)

시뮬 담당자가 방을 11.5m 섬 배치로 바꾸고 캐릭터(person_cook_v5)·A* 길찾기를 넣었다.
기존 200장은 옛 조리셀 기준이라 **새 배치로 200장을 다시 뽑아 같은 설정으로 재학습**했다.

**생성 자동화**: 시뮬의 `generateDataset`은 폴더 선택창 또는 파일당 다운로드 클릭이 필요해
자동화가 불가능했다 → 서버 `POST /dataset`이 (이미지+라벨) 쌍을 받아 저장하도록 만들어 **클릭 0회**로 생성.
`groundTruth(cam,{noDepth:true})`로 깊이맵 2장을 생략해 캡처를 **4585ms → 1084ms**로 줄였다(라벨 동일).

**데이터 비교**

| | 새(섬 배치) | 옛(조리셀) |
|---|---|---|
| person 인스턴스 | 316 | 289 |
| fire / smoke | 22 / 198 | 0 / 66 |
| equipment | 3,390 | 4,093 |
| person 박스 크기(정규화 w·h) | 0.00774 | 0.00756 |

박스 크기가 거의 같아(CAM11 프레임 재조정 덕) 옛 결과와 공정 비교가 된다.

**학습 결과** (YOLO11s, epochs 100·imgsz 640·seed 42·patience 20 — 옛 회차와 동일, 54ep 조기종료)

| 클래스 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **person** | **0.872** | **0.871** | **0.874** | 0.512 |
| fire | 1.000 | 0.596 | 0.906 | 0.372 |
| smoke | 0.930 | 0.950 | 0.936 | 0.759 |
| robot | 0.849 | 1.000 | 0.995 | 0.735 |
| kettle | 0.984 | 1.000 | 0.995 | 0.885 |
| equipment | 0.827 | 0.842 | 0.878 | 0.665 |
| **전체** | 0.910 | 0.876 | **0.931** | 0.655 |

**옛 배치 대비 person**: recall **0.688 → 0.871**(+0.183), precision 0.797 → 0.872, mAP50 0.747 → 0.874.

원인 추정(데이터 차이에서): ① person 인스턴스 316개로 증가 ② 캐릭터 교체(person_cook_v5)로 형태가
일관·선명 ③ A* 길찾기로 사람이 통로 중앙에 자연스럽게 분포(설비 가림↓) ④ 방이 커져 사람 간·설비 간
겹침 감소. **fire 클래스가 처음으로 학습됐다**(옛 데이터엔 화재 샘플이 0장).

산출물: `training/island_yolo11s/weights/best.pt`(+`best.onnx`, opset 12 — 브라우저 ort-web용).

---

---

## 7. 라이브 검증 — detect_server에 물려 전체 사슬 돌리기 (2026-08-19)

새 섬 모델(`training/island_yolo11s/weights/best.pt`)을 `backend/detect_server.py`에 물려
`검출 → 추적(ByteTrack) → 월드 (x,z) → 예측/회피 → 로봇 기동`을 라이브로 확인했다.

```bash
DETECT_MODEL=training/island_yolo11s/weights/best.pt uv run python backend/detect_server.py --port 8001
# → http://127.0.0.1:8001/sim.html?person=1 · CAM11 직교 상부 · 모델검증 http `/detect` · ▶ 검증 시작
```

### 7-1. 결과 (사람 접근 시나리오)

| 로봇거리 | 검출 | conf | 트랙 id | 월드 트랙 | 로봇 기동 |
|---|---|---|---|---|---|
| 4.66 m | 2 | 0.70 / 0.67 | 3, −1 | 1 | `proceed` |
| 2.27 m | 2 | 0.70 / 0.61 | 3, 4 | 2 | **`stop`** |
| 0.59 m | 3 | — | 5, 6, 7 | 4 | `stop` |

**추적 안정성** (사람 3명 왕복 보행, 26프레임 연속): 트랙 id가 붙은 프레임 **26/26**,
id 지속 **6→26프레임 · 7→25 · 8→25**, ID 스위치 0건, 월드 트랙 3개, 에러 0.

### 7-2. 라이브에서만 드러난 결함 2건 (수정 완료)

**① 시각화가 모델 입력을 오염시켰다** — 가장 중요한 발견.
예측 띠·모드 화살표·그래프 노드/엣지가 **모델에 보내는 프레임에 그려진 채로** 들어갔다.
데이터셋 캡처(`groundTruth`)에서는 억제했지만 모델 입력 캡처(`milCapture`)에는 적용을 빠뜨렸다.

| 5프레임 총 검출 | conf |
|---|---|
| 시각화 ON → **2개**(빈 프레임 다수) | 0.39~0.61 |
| 시각화 OFF → **12개** | 0.54~0.71 |

→ `vizSuppressBegin/End`를 공용 헬퍼로 만들어 **데이터셋·모델 입력 양쪽**에 적용.
화면에는 계속 보이고 캡처에서만 빠진다. 교훈: **주석은 절대 모델 입력에 들어가면 안 된다.**

**② ByteTrack이 트랙을 하나도 만들지 않았다.**
`trackers`의 기본 `track_activation_threshold = 0.7`이 우리 모델의 라이브 conf(0.25~0.58)보다
높아 활성화가 0건 → id가 영영 −1. 기본값은 강한 검출기로 MOT 벤치마크를 도는 전제의 값이다.
안전 시스템에서 낮은 confidence를 버리는 건 위험한 쪽 오류이므로 **0.35로 낮췄다**(env `TRACK_ACT`).
①을 고치자 conf가 0.61~0.70으로 회복돼 근본 원인은 ①이었다.

### 7-3. 남은 것
- 브라우저 ONNX 자립 데모 점검(`best.onnx`, opset 12 — 노트북 발표용, 서버 불필요)
- 학습형 궤적 예측기(`__customPredictor`)로 스테이션 목표 모드 대체
- 실사 주방 라벨 확보 후 sim-to-real 재측정

---

## 8. 모델 배포 — 팀원이 쓰는 방법 (2026-08-19)

### 8-1. 왜 git에 없나
`training/`(9.4GB)·`dataset/`(1.4GB)은 `.gitignore` 대상이다. 가중치를 git에 넣으면
재학습마다 19MB 바이너리가 이력에 영구히 쌓이고 클론이 무거워진다. Git LFS도 무료 할당량을
빠르게 소모한다. 그래서 **가중치는 허깅페이스 허브에, 코드는 git에** 둔다.

### 8-2. 팀원 실행 절차 — 파일 전달 없음

```bash
uv sync --group serve
uv run python backend/detect_server.py --port 8001
# → http://127.0.0.1:8001/sim.html?person=1
```

`detect_server.py`는 로컬 `training/island_yolo11s/weights/best.pt`가 있으면 그걸 쓰고,
**없으면 허브에서 자동으로 내려받는다**(`~/.cache/huggingface`에 캐시, 최초 1회 19MB).
공개 저장소라 토큰도 필요 없다.

| env | 기본값 | 용도 |
|---|---|---|
| `DETECT_MODEL` | `training/island_yolo11s/weights/best.pt` | 로컬 가중치 경로 |
| `DETECT_MODEL_REPO` | `chanubc/robot-kitchen-nadir-yolo11s` | 허브 저장소 |
| `DETECT_MODEL_FILE` | `best.pt` | 받을 파일 (`best.onnx`도 가능) |

> ⚠️ 기본 모델을 옛 `yolo11s_orthotop`(조리셀 배치, person recall 0.688)에서
> **`island_yolo11s`(섬 배치, 0.871)로 바꿨다.** 현재 시뮬 배치에 맞는 모델이다.

### 8-3. 허브 저장소

**https://huggingface.co/chanubc/robot-kitchen-nadir-yolo11s** (공개)

| 파일 | 크기 | 용도 |
|---|---|---|
| `best.pt` | 19 MB | ultralytics / PyTorch |
| `best.onnx` | 37 MB | ONNX Runtime (opset 12, 브라우저 ort-web 호환) |

모델 카드에 클래스 순서·in-domain 지표·학습 조건과 함께 **"실사에는 쓸 수 없다"**(실사 recall
0.048)를 명시했다. §2·§5-7의 결론이라 외부 이용자가 오용하지 않도록 카드에 직접 박아둔다.

라이선스는 **AGPL-3.0** — 베이스 Ultralytics YOLO11이 AGPL이고 학습 가중치도 그 범위로 본다.
GitHub 저장소가 이미 공개라 소스공개 의무는 충족된 상태다. 상용에서 이 의무를 피해야 하면
Apache-2.0인 RF-DETR 계열을 쓸 것(§5-4).

### 8-4. 재학습본을 올릴 때

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='training/<run>/weights/best.pt',
                    path_in_repo='best.pt',
                    repo_id='chanubc/robot-kitchen-nadir-yolo11s',
                    commit_message='<무엇을 바꿨는지>')"
```

git 기반이라 커밋으로 쌓인다. 특정 버전을 고정하려면 `hf_hub_download(..., revision="<sha>")`.
**모델 카드의 지표도 같이 갱신할 것** — 지표가 실제 가중치와 어긋나면 카드가 거짓말이 된다.
