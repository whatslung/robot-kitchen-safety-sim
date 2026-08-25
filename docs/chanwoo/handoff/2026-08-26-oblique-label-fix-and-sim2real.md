# 핸드오프 — 사선 GT 라벨 수정 + sim-to-real 검증 (2026-08-26)

> 시작 과제: sim 사선 카메라 GT의 person 라벨 부풀림 버그 수정.
> 진행 중 라벨 결함 2종을 더 발견해 고쳤고, **"이 sim이 실사 검출에 쓸모 있나"를 실측**했다.
> 결론: **라벨은 고쳐졌고, 검출 관점에선 sim이 실사로 전이되지 않는다(외형 갭). 실사 데이터가 답.**

## 1. 고친 라벨 버그 (커밋됨)

인스턴스 마스크 → bbox 파이프라인의 결함 3종:

1. **박스 부풀림** (`18cc1ff`) — 색별 min/max에 stray 픽셀이 섞여 박스가 프레임 전체로 커졌다.
   최근접색 판정(절대 상한 없음)이 MSAA 경계·어두운 픽셀을 오분류. → 색별 **연결성분 필터**
   (작고 고립된 stray 제거, 가림에 쪼개진 실제 사람 조각은 보존). 순수 로직 `gtboxes.js` +
   TDD `tests/browser/instance-box-filter.test.mjs`. 실측: 대형 박스 43% → 1%, area 최대 0.67 → 0.075.

2. **팬텀 박스 + 미라벨** (`7b89a3a`) — person 인스턴스가 씬 메시 순서상 equip 수십 개 뒤에
   등록돼 격자색(24,x,y)을 받고, 이웃 equip 격자색과 붙어(색거리 ~32) 최근접색이 카트·설비
   픽셀을 person으로 오분류(팬텀)하거나 person 픽셀을 뺏겨 blob이 줄어 n<80로 드롭(보이는 사람
   미라벨). → person·robot·kettle을 **base 색에 먼저 예약**(색거리 32 → 60~109). 실측: 팬텀 0.

3. **반복 생성 크래시** (`878744d`) — `randomizeScene→randomizeEnvironment`가 설비 원위치를
   첫 호출 때만 잡아, 늦게 로드된 프롭에서 `b0` undefined로 죽었다. → 이름 기준 지연 등록.

부수: 센서 열화 완화 + 천장 배관 벽 밀착(`602c7d2`), imagesOnly 깊이 생략(`15ac4fc`).

## 2. 헤드리스 백그라운드 생성 (`acc6e4b`)

브라우저 창을 포그라운드로 유지해야 하던 데이터셋 생성을 **창 없이 백그라운드**로.
`tools/headless_gen/gen.cjs` — Node+Playwright가 화면 밖 headful 창에서 sim을 렌더해 캡처하고
파일을 직접 저장(폴더 선택 불필요). 헤드리스는 WebGL 백버퍼가 0이라 실패 → headful+화면밖+
스로틀해제로 우회. 사용법 `tools/headless_gen/README.md`.

## 3. sim-to-real 실측 (핵심 발견)

배포 뷰(사선 조리실 CCTV)와 맞는 실사 오픈데이터를 확보: **Roboflow chef1 v5**
(실사 사선 조리실 CCTV, 5,387장, CC BY 4.0). human 클래스만 person으로 매핑 →
`dataset/chef1_person` (train 3767 / valid 1074 / **test 546**).

동일 yolo11n·imgsz640·60ep, **chef1 실사 test**에서 person recall:

| 학습 | → 평가 | recall | prec | mAP50 |
|---|---|---|---|---|
| 실사 chef1 (3767) | chef1 실사 test | **0.970** | 0.967 | 0.977 |
| 우리 sim-A (270) | chef1 실사 test | 0.048 | 0.006 | 0.002 |
| real+sim (4037) | chef1 실사 test | 0.961 | 0.962 | 0.977 |
| 실사 chef1 | 우리 sim (held-out) | 0.057 | — | — |
| real+sim (C) | 우리 sim (held-out) | 0.143 | 0.307 | 0.133 |
| sim-only | 우리 sim | ~0.35 | — | — |

**결론:**
- **실사만 = 0.97** → 배포용 사선-조리실 검출기는 실사(chef1)로 끝.
- **sim↔real 외형 갭 양방향 ~0.05** — 뷰·라벨·기하를 맞춰도 질감 없는 회색 렌더 sim과 거친 실제
  CCTV는 픽셀이 너무 달라 서로 전이 안 됨.
- **real+sim = 0.961 ≤ 0.970** — sim을 실사에 더해도 검출 이득 없음(미세 희석).
- **C(혼합)의 sim 검출 = 0.14** (혼합비 14:1로 실사 지배) — 단일 모델로 sim+real 둘 다 커버 안 됨.

## 4. 그래서 방향 (역할 분담)

- **검출 = 실사 데이터**(chef1 0.97). 필요시 대형 CCTV person 데이터 추가.
- **sim = ①** 실사로 못 찍는 위험 시나리오(화재·로봇충돌·대피)의 **완벽 GT** → 궤적예측·안전
  (Trajectron++/risk) **②** 멀티캠 융합·호모그래피·발목 keypoint(기하 기반) **③** 라벨·시나리오 도구.
- **파이프라인(호모+4캠융합+LSTM/Transformer) sim 검증** = **GT 트랙**으로 (검출기 불필요) 또는
  sim-전용 검출기(0.35)+멀티캠 융합(단일 0.35→4캠 ~0.82)으로.
- **ByteTrack** = 실배포에선 실사 0.97 검출로 정상. sim 단일캠 약검출은 융합/저신뢰 연결/ GT로 보완.
- **단일 모델 다리**를 원하면 → **sim 외형 realism 개선**(질감·조명·도메인 랜덤화)이 선결.

## 5. 자산 / 데이터 (gitignore, 재현)

- `dataset/chef1_person/` — 실사 사선 조리실 CCTV person (chef1 v5 remap). **실사 벤치마크·학습.**
- `dataset/sim-oblique-6cam-*` — 우리 사선-(A) sim (헤드리스 생성). 검출엔 전이 안 됨(위 참조).
- `tools/headless_gen/` — 백그라운드 생성 도구.
- ⚠️ chef1 다운로드에 쓴 Roboflow API 키가 대화에 노출됨 — **재발급 권장.**

## 6. 다음 후보

- (가) chef1로 배포 검출기 제대로 학습(yolo11s·imgsz↑) → 즉시 쓸 0.97+ 실사 검출기.
- (나) sim 외형 realism 개선(검출에도 쓰려면).
- (다) GT 트랙으로 호모+융합+궤적 sim 검증 파이프라인 조립(핸드오프 multicam-fusion §"남은 조립").
