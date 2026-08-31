# Figure Labs 핸드오프 — 주방 동선 히스토리(나디르) 그림

슬라이드 08("과거 히스토리 학습")의 왼쪽 참조 그림(worker trajectory sample)을 **우리 실제 시뮬레이터의 나디르(직교 top-down) 렌더** 위에 재현한다. 목적: "조리원이 어디로 움직이는지 = 스테이션마다 경로가 꺾이는 다중 목표(multi-goal) 동선"을 실제 주방 위에서 보여주기.

## 첨부 파일 (같은 폴더)
- `kitchen_nadir_wholescene.png` — 1440×1080, 주방 직교 나디르 렌더(배경). 원근 0이라 픽셀↔미터가 선형이고 전체 방·설비가 균일 스케일로 담겨 있다.
- `kitchen_trajectories_nadir.json` — 위 이미지 좌표계에 정렬된 조리원 3명의 실제 이동 궤적 + 스테이션·로봇·안전링.

## JSON 스키마 (핵심)
- 모든 점은 `x,z`(시뮬 좌표, ≈미터)와 **`u,v`(위 PNG의 픽셀 좌표)** 를 함께 가진다. **`u,v`를 이미지 위에 그대로 찍으면 정렬된다.** `v`는 위(top)=0.
- `workers[]`: `job`(prep/cook/wash) + `points[]`(각 `t,x,z,u,v,goal,moving`). 150점 = 60초 · 2.5Hz.
  - 첫 점 = 시작(○ 빈 원), 끝 점 = 끝(● 채운 원). `moving:false` = 스테이션에서 작업 중(정지) — 같은 좌표가 이어짐.
- `stations[]`: 21개 목적지(`key` 영문 라벨 + `label` 한글 + `u,v`). 사람이 향하는 목표 지점.
- `robot`: 로봇 베이스(`u,v`). `rings`: 정지 3.1m / 감속 3.9m 안전링의 픽셀 반경(`stop_px`,`slow_px`, 로봇 중심 기준).

## 그려야 할 그림 (참조 슬라이드 왼쪽 A패널 스타일)
1. 배경 = `kitchen_nadir_wholescene.png` 그대로. (바닥에 옅은 안전링이 이미 렌더돼 있으니, 링을 새로 그리면 겹칠 수 있음 — 새로 그리려면 배경 링은 무시하고 JSON `rings`로 깔끔하게 다시.)
2. 워커 3명 궤적을 **역할별 색 폴리라인**으로:
   - prep(준비) `#3f6fd1` 파랑 · cook(조리) `#e8703a` 주황 · wash(세척) `#37a05f` 초록.
   - 선 굵기 3~4px, 시작 ○(빈 원)·끝 ●(채운 원). 부드럽게(anti-alias).
3. 스테이션: 작은 회색 사각 + 영문 `key` 라벨(fridge, kettle, isle1 …). 슬라이드처럼 담백하게.
4. 로봇: 중앙 검은 점 + "robot" 라벨. 정지(3.1m)·감속(3.9m) 링을 점선으로(빨강/주황).
5. 범례: 세 역할 색 + "○ 시작 · ● 끝". 제목: "동선 히스토리 · 나디르 — 조리원 3명 · WORKFLOW 동선".
6. 메시지(캡션 후보): "작업자는 스테이션을 이어 가므로 **경로가 스테이션마다 꺾인다** — 직선 예측이 실패하는 지점. 학습형 예측기는 이 다중 목표 운동을 모델링한다."

## 스타일 가이드
- 참조 슬라이드 왼쪽 그림처럼 **깔끔한 2D 도면 톤**. 3D 배경은 살리되 궤적·라벨이 또렷하게(선/글자에 옅은 외곽 헤일로로 가독성 확보).
- 색은 위 3색 고정. 배경이 밝은 베이지라 채도 있는 선이 잘 보인다.

## 참고
- 데이터 출처: `sim.html`(Babylon.js) · scene seed 7 · WORKFLOW 동선 고정 dt 기록.
- 원래 슬라이드는 prep·cook·**carry**였으나, carry(운반) 역할이 현재 sim NAV 코너-핀 버그로 안 움직여 **wash(세척)** 로 대체함. 라벨을 "운반"으로 바꾸고 싶으면 wash→carry로만 표기 교체하면 됨(동선 형태는 동일하게 다중 목표).
- (선택) "관측→예측 K=3" B패널은 학습형 예측기(백엔드 `/predict`)의 출력이 필요해 이 데이터에는 없음. 필요하면 백엔드를 켜고 별도로 K=3 모드를 뽑아 제공 가능.

---

## 궤적 데이터 (JSON — 위 스키마)

아래 JSON 전체를 데이터로 사용한다. 각 점의 `u,v`가 `kitchen_nadir_wholescene.png`(1440×1080)의 픽셀 좌표다.

```json
{
 "meta": {
  "description": "급식 조리로봇 안전셀 시뮬레이터의 나디르(직교 top-down) 렌더와, 그 위 좌표계에 정렬된 조리원 3명의 실제 이동 궤적. Figure Labs 핸드오프용.",
  "base_image": "kitchen_nadir_wholescene.png",
  "image_is": "직교(orthographic) 나디르 렌더. 원근 0 → 픽셀↔미터 선형. 세로=+Z(위=+z), 가로=+X(오른쪽=+x).",
  "source": {
   "sim": "sim.html (Babylon.js)",
   "scene_seed": 7,
   "roles": [
    "prep(준비)",
    "cook(조리)",
    "wash(세척)"
   ],
   "duration_s": 60,
   "sample_hz": 2.5,
   "dt_s": 0.4,
   "note": "WORKFLOW 동선(스테이션 체이닝)을 고정 dt로 스텝해 기록. carry(운반) 역할은 현재 sim NAV 코너-핀 버그로 제외, 대신 wash 사용."
  },
  "coordinate_system": "각 점에 (x,z)=시뮬 좌표(≈미터, mPerAU 참조)와 (u,v)=base_image 픽셀좌표를 함께 제공. Figure Labs는 u,v를 그대로 이미지 위에 찍으면 정렬됨. v는 위(top)=0.",
  "worker_colors_suggested": {
   "prep": "#3f6fd1",
   "cook": "#e8703a",
   "wash": "#37a05f"
  },
  "markers": "각 워커 궤적: 첫 점=시작(○ 빈 원), 끝 점=끝(● 채운 원). moving=false 구간은 스테이션에서 작업 중(정지).",
  "rings_note": "정지/감속 안전링은 base_image 바닥에 이미 렌더돼 있음. rings.*_px는 로봇 중심(robot.u,v) 기준 픽셀 반경."
 },
 "image": {
  "w": 1440,
  "h": 1080
 },
 "frame_world": {
  "xMin": -6.292,
  "xMax": 7.442,
  "zMin": -5.15,
  "zMax": 5.15
 },
 "mPerAU": 1,
 "robot": {
  "x": -1.4,
  "z": 0.515,
  "u": 512.9,
  "v": 486
 },
 "rings": {
  "stop_m": 3.1,
  "slow_m": 3.9,
  "stop_px": 325,
  "slow_px": 409
 },
 "stations": [
  {
   "key": "shelf",
   "label": "선반 앞",
   "x": 0.42,
   "z": 1.84,
   "u": 703.7,
   "v": 347.1
  },
  {
   "key": "prep",
   "label": "조리대 앞",
   "x": 1.67,
   "z": 0.99,
   "u": 834.8,
   "v": 436.2
  },
  {
   "key": "kettle",
   "label": "솥 앞",
   "x": 0.225,
   "z": 2.675,
   "u": 683.3,
   "v": 259.5
  },
  {
   "key": "panel",
   "label": "제어반 앞",
   "x": 2.325,
   "z": 1.625,
   "u": 903.5,
   "v": 369.6
  },
  {
   "key": "aisle",
   "label": "통로 대기",
   "x": 2.325,
   "z": 2.675,
   "u": 903.5,
   "v": 259.5
  },
  {
   "key": "door",
   "label": "입구",
   "x": 4.775,
   "z": 3.375,
   "u": 1160.4,
   "v": 186.1
  },
  {
   "key": "sink",
   "label": "세척대 앞",
   "x": 4.775,
   "z": -2.225,
   "u": 1160.4,
   "v": 773.3
  },
  {
   "key": "store",
   "label": "건조 선반 앞",
   "x": 2.675,
   "z": 4.425,
   "u": 940.2,
   "v": 76
  },
  {
   "key": "serve",
   "label": "배식구 앞",
   "x": -2.2,
   "z": 4.4,
   "u": 429,
   "v": 78.6
  },
  {
   "key": "etable",
   "label": "배선 작업대",
   "x": 4.775,
   "z": 1.625,
   "u": 1160.4,
   "v": 369.6
  },
  {
   "key": "fridge",
   "label": "냉장고 앞",
   "x": -2.925,
   "z": -2.225,
   "u": 353,
   "v": 773.3
  },
  {
   "key": "wrack",
   "label": "재료 선반 앞",
   "x": -2.575,
   "z": 0.575,
   "u": 389.7,
   "v": 479.7
  },
  {
   "key": "island",
   "label": "전처리대",
   "x": -3.625,
   "z": 1.625,
   "u": 279.6,
   "v": 369.6
  },
  {
   "key": "isle1",
   "label": "중앙 준비대(북)",
   "x": 0.3,
   "z": -4.45,
   "u": 691.2,
   "v": 1006.6
  },
  {
   "key": "isle2",
   "label": "중앙 준비대(사이)",
   "x": 2.1,
   "z": -4.45,
   "u": 879.9,
   "v": 1006.6
  },
  {
   "key": "isle3",
   "label": "중앙 배선 준비대",
   "x": -0.2,
   "z": 4.45,
   "u": 638.7,
   "v": 73.4
  },
  {
   "key": "robotside",
   "label": "로봇 앞 점검",
   "x": -0.3,
   "z": 1.28,
   "u": 628.3,
   "v": 405.8
  },
  {
   "key": "cartPick",
   "label": "카트 수거",
   "x": -1.525,
   "z": 1.975,
   "u": 499.8,
   "v": 332.9
  },
  {
   "key": "nkettle",
   "label": "북측 국솥 앞",
   "x": -3.16,
   "z": -4.3,
   "u": 328.4,
   "v": 990.9
  },
  {
   "key": "ntable",
   "label": "북측 배출대",
   "x": -0.475,
   "z": -3.975,
   "u": 609.9,
   "v": 956.8
  },
  {
   "key": "nrack",
   "label": "북측 선반 앞",
   "x": 3.025,
   "z": -4.325,
   "u": 976.9,
   "v": 993.5
  }
 ],
 "workers": [
  {
   "job": "prep",
   "role": "caution",
   "points": [
    {
     "t": 0,
     "x": 0.716,
     "z": 2.78,
     "u": 734.8,
     "v": 248.5,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 0.4,
     "x": 0.697,
     "z": 2.798,
     "u": 732.8,
     "v": 246.6,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 0.8,
     "x": 0.693,
     "z": 2.802,
     "u": 732.4,
     "v": 246.2,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 1.2,
     "x": 0.692,
     "z": 2.803,
     "u": 732.3,
     "v": 246.1,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 1.6,
     "x": 0.692,
     "z": 2.803,
     "u": 732.3,
     "v": 246.1,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 2,
     "x": 0.692,
     "z": 2.803,
     "u": 732.3,
     "v": 246.1,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 2.4,
     "x": 0.754,
     "z": 2.798,
     "u": 738.8,
     "v": 246.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 2.8,
     "x": 1.064,
     "z": 2.774,
     "u": 771.3,
     "v": 249.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 3.2,
     "x": 1.374,
     "z": 2.749,
     "u": 803.8,
     "v": 251.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 3.6,
     "x": 1.684,
     "z": 2.725,
     "u": 836.3,
     "v": 254.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 4,
     "x": 1.995,
     "z": 2.701,
     "u": 868.9,
     "v": 256.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 4.4,
     "x": 2.221,
     "z": 2.622,
     "u": 892.6,
     "v": 265,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 4.8,
     "x": 2.274,
     "z": 2.315,
     "u": 898.2,
     "v": 297.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 5.2,
     "x": 2.335,
     "z": 2.049,
     "u": 904.5,
     "v": 325.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 5.6,
     "x": 2.572,
     "z": 1.848,
     "u": 929.4,
     "v": 346.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 6,
     "x": 2.81,
     "z": 1.647,
     "u": 954.3,
     "v": 367.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 6.4,
     "x": 3.047,
     "z": 1.446,
     "u": 979.2,
     "v": 388.4,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 6.8,
     "x": 3.284,
     "z": 1.244,
     "u": 1004.1,
     "v": 409.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 7.2,
     "x": 3.522,
     "z": 1.043,
     "u": 1029,
     "v": 430.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 7.6,
     "x": 3.759,
     "z": 0.842,
     "u": 1053.9,
     "v": 451.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 8,
     "x": 3.996,
     "z": 0.67,
     "u": 1078.7,
     "v": 469.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 8.4,
     "x": 4.023,
     "z": 0.67,
     "u": 1081.6,
     "v": 469.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 8.8,
     "x": 4.031,
     "z": 0.67,
     "u": 1082.4,
     "v": 469.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 9.2,
     "x": 4.038,
     "z": 0.67,
     "u": 1083.1,
     "v": 469.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 9.6,
     "x": 4.071,
     "z": 0.39,
     "u": 1086.5,
     "v": 499.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 10,
     "x": 4.072,
     "z": 0.079,
     "u": 1086.6,
     "v": 531.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 10.4,
     "x": 4.072,
     "z": -0.232,
     "u": 1086.7,
     "v": 564.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 10.8,
     "x": 4.073,
     "z": -0.543,
     "u": 1086.8,
     "v": 597,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 11.2,
     "x": 4.074,
     "z": -0.854,
     "u": 1086.9,
     "v": 629.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 11.6,
     "x": 4.013,
     "z": -1.073,
     "u": 1080.4,
     "v": 652.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 12,
     "x": 3.701,
     "z": -1.079,
     "u": 1047.8,
     "v": 653.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 12.4,
     "x": 3.39,
     "z": -1.085,
     "u": 1015.2,
     "v": 653.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 12.8,
     "x": 3.079,
     "z": -1.091,
     "u": 982.6,
     "v": 654.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 13.2,
     "x": 2.768,
     "z": -1.096,
     "u": 950,
     "v": 654.9,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 13.6,
     "x": 2.457,
     "z": -1.102,
     "u": 917.4,
     "v": 655.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 14,
     "x": 2.146,
     "z": -1.108,
     "u": 884.7,
     "v": 656.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 14.4,
     "x": 1.835,
     "z": -1.113,
     "u": 852.1,
     "v": 656.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 14.8,
     "x": 1.524,
     "z": -1.119,
     "u": 819.5,
     "v": 657.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 15.2,
     "x": 1.226,
     "z": -1.133,
     "u": 788.2,
     "v": 658.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 15.6,
     "x": 1.131,
     "z": -1.275,
     "u": 778.3,
     "v": 673.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 16,
     "x": 1.028,
     "z": -1.412,
     "u": 767.5,
     "v": 688.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 16.4,
     "x": 0.918,
     "z": -1.544,
     "u": 756,
     "v": 701.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 16.8,
     "x": 0.801,
     "z": -1.668,
     "u": 743.7,
     "v": 714.9,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 17.2,
     "x": 0.679,
     "z": -1.785,
     "u": 730.9,
     "v": 727.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 17.6,
     "x": 0.552,
     "z": -1.893,
     "u": 717.6,
     "v": 738.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 18,
     "x": 0.421,
     "z": -1.994,
     "u": 703.9,
     "v": 749,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 18.4,
     "x": 0.289,
     "z": -2.085,
     "u": 690,
     "v": 758.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 18.8,
     "x": 0.155,
     "z": -2.167,
     "u": 676,
     "v": 767.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 19.2,
     "x": 0.023,
     "z": -2.239,
     "u": 662.1,
     "v": 774.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 19.6,
     "x": -0.108,
     "z": -2.303,
     "u": 648.4,
     "v": 781.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 20,
     "x": -0.235,
     "z": -2.358,
     "u": 635,
     "v": 787.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 20.4,
     "x": -0.358,
     "z": -2.405,
     "u": 622.2,
     "v": 792.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 20.8,
     "x": -0.475,
     "z": -2.444,
     "u": 609.9,
     "v": 796.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 21.2,
     "x": -0.586,
     "z": -2.476,
     "u": 598.3,
     "v": 799.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 21.6,
     "x": -0.689,
     "z": -2.502,
     "u": 587.5,
     "v": 802.4,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 22,
     "x": -0.785,
     "z": -2.523,
     "u": 577.4,
     "v": 804.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 22.4,
     "x": -0.873,
     "z": -2.54,
     "u": 568.2,
     "v": 806.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 22.8,
     "x": -1.015,
     "z": -2.561,
     "u": 553.3,
     "v": 808.5,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 23.2,
     "x": -1.3,
     "z": -2.583,
     "u": 523.4,
     "v": 810.9,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 23.6,
     "x": -1.591,
     "z": -2.579,
     "u": 492.9,
     "v": 810.4,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 24,
     "x": -1.882,
     "z": -2.547,
     "u": 462.4,
     "v": 807.1,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 24.4,
     "x": -2.171,
     "z": -2.487,
     "u": 432,
     "v": 800.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 24.8,
     "x": -2.455,
     "z": -2.4,
     "u": 402.3,
     "v": 791.6,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 25.2,
     "x": -2.734,
     "z": -2.296,
     "u": 373.1,
     "v": 780.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 25.6,
     "x": -2.817,
     "z": -2.265,
     "u": 364.3,
     "v": 777.5,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 26,
     "x": -2.817,
     "z": -2.265,
     "u": 364.3,
     "v": 777.5,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 26.4,
     "x": -2.817,
     "z": -2.265,
     "u": 364.3,
     "v": 777.5,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 26.8,
     "x": -2.817,
     "z": -2.265,
     "u": 364.3,
     "v": 777.5,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 27.2,
     "x": -2.817,
     "z": -2.265,
     "u": 364.3,
     "v": 777.5,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 27.6,
     "x": -2.661,
     "z": -2.399,
     "u": 380.6,
     "v": 791.5,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 28,
     "x": -2.488,
     "z": -2.547,
     "u": 398.8,
     "v": 807.1,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 28.4,
     "x": -2.315,
     "z": -2.696,
     "u": 416.9,
     "v": 822.7,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 28.8,
     "x": -2.142,
     "z": -2.845,
     "u": 435.1,
     "v": 838.3,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 29.2,
     "x": -1.969,
     "z": -2.993,
     "u": 453.3,
     "v": 853.9,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 29.6,
     "x": -1.796,
     "z": -3.142,
     "u": 471.4,
     "v": 869.4,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 30,
     "x": -1.623,
     "z": -3.291,
     "u": 489.6,
     "v": 885,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 30.4,
     "x": -1.45,
     "z": -3.439,
     "u": 507.7,
     "v": 900.6,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 30.8,
     "x": -1.276,
     "z": -3.588,
     "u": 525.9,
     "v": 916.2,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 31.2,
     "x": -1.103,
     "z": -3.736,
     "u": 544,
     "v": 931.8,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 31.6,
     "x": -0.93,
     "z": -3.885,
     "u": 562.2,
     "v": 947.3,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 32,
     "x": -0.747,
     "z": -3.975,
     "u": 581.4,
     "v": 956.8,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 32.4,
     "x": -0.539,
     "z": -4.069,
     "u": 603.2,
     "v": 966.7,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 32.8,
     "x": -0.331,
     "z": -4.164,
     "u": 625,
     "v": 976.6,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 33.2,
     "x": -0.123,
     "z": -4.258,
     "u": 646.8,
     "v": 986.5,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 33.6,
     "x": 0.085,
     "z": -4.352,
     "u": 668.6,
     "v": 996.4,
     "goal": "isle1",
     "moving": true
    },
    {
     "t": 34,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 34.4,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 34.8,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 35.2,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 35.6,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 36,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 36.4,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 36.8,
     "x": 0.209,
     "z": -4.409,
     "u": 681.7,
     "v": 1002.3,
     "goal": "isle1",
     "moving": false
    },
    {
     "t": 37.2,
     "x": 0.093,
     "z": -4.356,
     "u": 669.5,
     "v": 996.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 37.6,
     "x": -0.139,
     "z": -4.251,
     "u": 645.1,
     "v": 985.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 38,
     "x": -0.372,
     "z": -4.146,
     "u": 620.7,
     "v": 974.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 38.4,
     "x": -0.604,
     "z": -4.041,
     "u": 596.4,
     "v": 963.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 38.8,
     "x": -0.836,
     "z": -3.936,
     "u": 572,
     "v": 952.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 39.2,
     "x": -1.069,
     "z": -3.831,
     "u": 547.7,
     "v": 941.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 39.6,
     "x": -1.301,
     "z": -3.726,
     "u": 523.3,
     "v": 930.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 40,
     "x": -1.491,
     "z": -3.603,
     "u": 503.4,
     "v": 917.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 40.4,
     "x": -1.675,
     "z": -3.426,
     "u": 484.1,
     "v": 899.3,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 40.8,
     "x": -1.858,
     "z": -3.25,
     "u": 464.8,
     "v": 880.8,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 41.2,
     "x": -2.042,
     "z": -3.073,
     "u": 445.6,
     "v": 862.2,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 41.6,
     "x": -2.226,
     "z": -2.896,
     "u": 426.3,
     "v": 843.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 42,
     "x": -2.41,
     "z": -2.72,
     "u": 407,
     "v": 825.2,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 42.4,
     "x": -2.594,
     "z": -2.543,
     "u": 387.7,
     "v": 806.7,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 42.8,
     "x": -2.778,
     "z": -2.366,
     "u": 368.4,
     "v": 788.1,
     "goal": "fridge",
     "moving": true
    },
    {
     "t": 43.2,
     "x": -2.851,
     "z": -2.296,
     "u": 360.7,
     "v": 780.7,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 43.6,
     "x": -2.851,
     "z": -2.296,
     "u": 360.7,
     "v": 780.7,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 44,
     "x": -2.851,
     "z": -2.296,
     "u": 360.7,
     "v": 780.7,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 44.4,
     "x": -2.851,
     "z": -2.296,
     "u": 360.7,
     "v": 780.7,
     "goal": "fridge",
     "moving": false
    },
    {
     "t": 44.8,
     "x": -2.679,
     "z": -2.309,
     "u": 378.8,
     "v": 782.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 45.2,
     "x": -2.437,
     "z": -2.406,
     "u": 404.2,
     "v": 792.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 45.6,
     "x": -2.22,
     "z": -2.474,
     "u": 426.9,
     "v": 799.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 46,
     "x": -2.066,
     "z": -2.513,
     "u": 443.1,
     "v": 803.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 46.4,
     "x": -1.986,
     "z": -2.529,
     "u": 451.4,
     "v": 805.2,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 46.8,
     "x": -1.952,
     "z": -2.535,
     "u": 455,
     "v": 805.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 47.2,
     "x": -1.939,
     "z": -2.538,
     "u": 456.4,
     "v": 806.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 47.6,
     "x": -1.937,
     "z": -2.538,
     "u": 456.6,
     "v": 806.1,
     "goal": "island",
     "moving": true
    },
    {
     "t": 48,
     "x": -1.972,
     "z": -2.532,
     "u": 452.9,
     "v": 805.5,
     "goal": "island",
     "moving": true
    },
    {
     "t": 48.4,
     "x": -2.004,
     "z": -2.526,
     "u": 449.5,
     "v": 804.8,
     "goal": "island",
     "moving": true
    },
    {
     "t": 48.8,
     "x": -1.976,
     "z": -2.531,
     "u": 452.5,
     "v": 805.4,
     "goal": "island",
     "moving": true
    },
    {
     "t": 49.2,
     "x": -2.008,
     "z": -2.525,
     "u": 449.1,
     "v": 804.7,
     "goal": "island",
     "moving": true
    },
    {
     "t": 49.6,
     "x": -2.038,
     "z": -2.519,
     "u": 446,
     "v": 804.1,
     "goal": "island",
     "moving": true
    },
    {
     "t": 50,
     "x": -2.066,
     "z": -2.513,
     "u": 443.1,
     "v": 803.5,
     "goal": "island",
     "moving": true
    },
    {
     "t": 50.4,
     "x": -2.009,
     "z": -2.594,
     "u": 449.1,
     "v": 812,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 50.8,
     "x": -1.826,
     "z": -2.807,
     "u": 468.2,
     "v": 834.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 51.2,
     "x": -1.644,
     "z": -3.02,
     "u": 487.3,
     "v": 856.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 51.6,
     "x": -1.461,
     "z": -3.233,
     "u": 506.5,
     "v": 879,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 52,
     "x": -1.279,
     "z": -3.446,
     "u": 525.6,
     "v": 901.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 52.4,
     "x": -1.096,
     "z": -3.659,
     "u": 544.8,
     "v": 923.6,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 52.8,
     "x": -0.929,
     "z": -3.874,
     "u": 562.3,
     "v": 946.2,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 53.2,
     "x": -0.686,
     "z": -3.918,
     "u": 587.8,
     "v": 950.8,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 53.6,
     "x": -0.407,
     "z": -3.949,
     "u": 617,
     "v": 954,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 54,
     "x": -0.128,
     "z": -3.979,
     "u": 646.3,
     "v": 957.2,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 54.4,
     "x": 0.15,
     "z": -4.01,
     "u": 675.5,
     "v": 960.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 54.8,
     "x": 0.429,
     "z": -4.04,
     "u": 704.7,
     "v": 963.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 55.2,
     "x": 0.708,
     "z": -4.071,
     "u": 733.9,
     "v": 966.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 55.6,
     "x": 0.987,
     "z": -4.101,
     "u": 763.2,
     "v": 970.1,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 56,
     "x": 1.266,
     "z": -4.132,
     "u": 792.4,
     "v": 973.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 56.4,
     "x": 1.544,
     "z": -4.163,
     "u": 821.6,
     "v": 976.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 56.8,
     "x": 1.823,
     "z": -4.193,
     "u": 850.9,
     "v": 979.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 57.2,
     "x": 2.102,
     "z": -4.224,
     "u": 880.1,
     "v": 982.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 57.6,
     "x": 2.381,
     "z": -4.254,
     "u": 909.3,
     "v": 986.1,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 58,
     "x": 2.659,
     "z": -4.285,
     "u": 938.5,
     "v": 989.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 58.4,
     "x": 2.91,
     "z": -4.312,
     "u": 964.9,
     "v": 992.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 58.8,
     "x": 2.91,
     "z": -4.312,
     "u": 964.9,
     "v": 992.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 59.2,
     "x": 2.91,
     "z": -4.312,
     "u": 964.9,
     "v": 992.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 59.6,
     "x": 2.91,
     "z": -4.312,
     "u": 964.9,
     "v": 992.2,
     "goal": "nrack",
     "moving": false
    }
   ]
  },
  {
   "job": "cook",
   "role": "caution",
   "points": [
    {
     "t": 0,
     "x": 2.253,
     "z": 1.183,
     "u": 896,
     "v": 416,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 0.4,
     "x": 2.303,
     "z": 1.491,
     "u": 901.2,
     "v": 383.7,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 0.8,
     "x": 2.308,
     "z": 1.522,
     "u": 901.7,
     "v": 380.5,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 1.2,
     "x": 2.308,
     "z": 1.522,
     "u": 901.7,
     "v": 380.5,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 1.6,
     "x": 2.308,
     "z": 1.522,
     "u": 901.7,
     "v": 380.5,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 2,
     "x": 2.308,
     "z": 1.522,
     "u": 901.7,
     "v": 380.5,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 2.4,
     "x": 2.315,
     "z": 1.273,
     "u": 902.5,
     "v": 406.5,
     "goal": "prep",
     "moving": true
    },
    {
     "t": 2.8,
     "x": 2.291,
     "z": 1.023,
     "u": 900,
     "v": 432.7,
     "goal": "prep",
     "moving": true
    },
    {
     "t": 3.2,
     "x": 1.981,
     "z": 1.007,
     "u": 867.4,
     "v": 434.5,
     "goal": "prep",
     "moving": true
    },
    {
     "t": 3.6,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 4,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 4.4,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 4.8,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 5.2,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 5.6,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 6,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 6.4,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 6.8,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 7.2,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 7.6,
     "x": 1.764,
     "z": 0.995,
     "u": 844.7,
     "v": 435.7,
     "goal": "prep",
     "moving": false
    },
    {
     "t": 8,
     "x": 1.861,
     "z": 0.983,
     "u": 854.8,
     "v": 436.9,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 8.4,
     "x": 2.182,
     "z": 0.943,
     "u": 888.5,
     "v": 441.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 8.8,
     "x": 2.231,
     "z": 1.198,
     "u": 893.6,
     "v": 414.4,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 9.2,
     "x": 2.265,
     "z": 1.51,
     "u": 897.2,
     "v": 381.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 9.6,
     "x": 2.282,
     "z": 1.834,
     "u": 898.9,
     "v": 347.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 10,
     "x": 2.298,
     "z": 2.158,
     "u": 900.7,
     "v": 313.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 10.4,
     "x": 2.315,
     "z": 2.482,
     "u": 902.4,
     "v": 279.8,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 10.8,
     "x": 2.126,
     "z": 2.588,
     "u": 882.6,
     "v": 268.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 11.2,
     "x": 1.802,
     "z": 2.603,
     "u": 848.6,
     "v": 267.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 11.6,
     "x": 1.478,
     "z": 2.618,
     "u": 814.6,
     "v": 265.5,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 12,
     "x": 1.153,
     "z": 2.633,
     "u": 780.7,
     "v": 264,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 12.4,
     "x": 0.84,
     "z": 2.658,
     "u": 747.8,
     "v": 261.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 12.8,
     "x": 0.691,
     "z": 2.803,
     "u": 732.2,
     "v": 246.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 13.2,
     "x": 0.584,
     "z": 2.897,
     "u": 721,
     "v": 236.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 13.6,
     "x": 0.523,
     "z": 2.946,
     "u": 714.5,
     "v": 231,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 14,
     "x": 0.492,
     "z": 2.971,
     "u": 711.3,
     "v": 228.5,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 14.4,
     "x": 0.477,
     "z": 2.982,
     "u": 709.7,
     "v": 227.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 14.8,
     "x": 0.515,
     "z": 2.979,
     "u": 713.7,
     "v": 227.6,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 15.2,
     "x": 0.736,
     "z": 2.942,
     "u": 736.8,
     "v": 231.5,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 15.6,
     "x": 0.956,
     "z": 2.905,
     "u": 759.9,
     "v": 235.4,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 16,
     "x": 1.176,
     "z": 2.868,
     "u": 783,
     "v": 239.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 16.4,
     "x": 1.396,
     "z": 2.831,
     "u": 806.1,
     "v": 243.1,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 16.8,
     "x": 1.617,
     "z": 2.794,
     "u": 829.2,
     "v": 247,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 17.2,
     "x": 1.837,
     "z": 2.757,
     "u": 852.3,
     "v": 250.9,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 17.6,
     "x": 2.057,
     "z": 2.72,
     "u": 875.4,
     "v": 254.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 18,
     "x": 2.229,
     "z": 2.653,
     "u": 893.5,
     "v": 261.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 18.4,
     "x": 2.318,
     "z": 2.448,
     "u": 902.8,
     "v": 283.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 18.8,
     "x": 2.407,
     "z": 2.243,
     "u": 912.1,
     "v": 304.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 19.2,
     "x": 2.421,
     "z": 2.043,
     "u": 913.5,
     "v": 325.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 19.6,
     "x": 2.529,
     "z": 1.851,
     "u": 924.9,
     "v": 345.9,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 20,
     "x": 2.649,
     "z": 1.692,
     "u": 937.5,
     "v": 362.6,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 20.4,
     "x": 2.825,
     "z": 1.554,
     "u": 955.9,
     "v": 377,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 20.8,
     "x": 3.001,
     "z": 1.417,
     "u": 974.4,
     "v": 391.5,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 21.2,
     "x": 3.177,
     "z": 1.279,
     "u": 992.8,
     "v": 405.9,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 21.6,
     "x": 3.353,
     "z": 1.141,
     "u": 1011.2,
     "v": 420.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 22,
     "x": 3.528,
     "z": 1.003,
     "u": 1029.7,
     "v": 434.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 22.4,
     "x": 3.704,
     "z": 0.866,
     "u": 1048.1,
     "v": 449.2,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 22.8,
     "x": 3.88,
     "z": 0.728,
     "u": 1066.5,
     "v": 463.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 23.2,
     "x": 4.012,
     "z": 0.67,
     "u": 1080.4,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 23.6,
     "x": 3.999,
     "z": 0.67,
     "u": 1079,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 24,
     "x": 3.986,
     "z": 0.67,
     "u": 1077.7,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 24.4,
     "x": 4.006,
     "z": 0.67,
     "u": 1079.8,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 24.8,
     "x": 3.993,
     "z": 0.67,
     "u": 1078.4,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 25.2,
     "x": 3.981,
     "z": 0.67,
     "u": 1077.1,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 25.6,
     "x": 3.969,
     "z": 0.67,
     "u": 1075.8,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 26,
     "x": 3.97,
     "z": 0.67,
     "u": 1075.9,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 26.4,
     "x": 3.982,
     "z": 0.67,
     "u": 1077.3,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 26.8,
     "x": 4.008,
     "z": 0.67,
     "u": 1079.9,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 27.2,
     "x": 4.015,
     "z": 0.67,
     "u": 1080.7,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 27.6,
     "x": 4.022,
     "z": 0.67,
     "u": 1081.4,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 28,
     "x": 4.028,
     "z": 0.67,
     "u": 1082.1,
     "v": 469.7,
     "goal": "nkettle",
     "moving": true
    },
    {
     "t": 28.4,
     "x": 4.034,
     "z": 0.67,
     "u": 1082.7,
     "v": 469.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 28.8,
     "x": 3.867,
     "z": 0.832,
     "u": 1065.2,
     "v": 452.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 29.2,
     "x": 3.7,
     "z": 0.995,
     "u": 1047.7,
     "v": 435.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 29.6,
     "x": 3.534,
     "z": 1.157,
     "u": 1030.2,
     "v": 418.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 30,
     "x": 3.367,
     "z": 1.319,
     "u": 1012.8,
     "v": 401.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 30.4,
     "x": 3.2,
     "z": 1.482,
     "u": 995.3,
     "v": 384.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 30.8,
     "x": 3.034,
     "z": 1.644,
     "u": 977.8,
     "v": 367.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 31.2,
     "x": 2.867,
     "z": 1.806,
     "u": 960.3,
     "v": 350.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 31.6,
     "x": 2.7,
     "z": 1.969,
     "u": 942.8,
     "v": 333.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 32,
     "x": 2.539,
     "z": 2.002,
     "u": 926,
     "v": 330.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 32.4,
     "x": 2.394,
     "z": 2.059,
     "u": 910.8,
     "v": 324.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 32.8,
     "x": 2.263,
     "z": 2.252,
     "u": 897,
     "v": 303.9,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 33.2,
     "x": 2.252,
     "z": 2.43,
     "u": 895.8,
     "v": 285.2,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 33.6,
     "x": 2.103,
     "z": 2.579,
     "u": 880.2,
     "v": 269.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 34,
     "x": 1.903,
     "z": 2.613,
     "u": 859.2,
     "v": 266,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 34.4,
     "x": 1.67,
     "z": 2.622,
     "u": 834.9,
     "v": 265.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 34.8,
     "x": 1.438,
     "z": 2.63,
     "u": 810.5,
     "v": 264.2,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 35.2,
     "x": 1.205,
     "z": 2.639,
     "u": 786.1,
     "v": 263.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 35.6,
     "x": 0.973,
     "z": 2.647,
     "u": 761.7,
     "v": 262.4,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 36,
     "x": 0.794,
     "z": 2.705,
     "u": 743,
     "v": 256.4,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 36.4,
     "x": 0.691,
     "z": 2.804,
     "u": 732.2,
     "v": 246,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 36.8,
     "x": 0.61,
     "z": 2.875,
     "u": 723.6,
     "v": 238.5,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 37.2,
     "x": 0.553,
     "z": 2.922,
     "u": 717.7,
     "v": 233.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 37.6,
     "x": 0.518,
     "z": 2.951,
     "u": 714,
     "v": 230.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 38,
     "x": 0.496,
     "z": 2.968,
     "u": 711.7,
     "v": 228.8,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 38.4,
     "x": 0.571,
     "z": 2.962,
     "u": 719.6,
     "v": 229.5,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 38.8,
     "x": 0.854,
     "z": 2.915,
     "u": 749.2,
     "v": 234.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 39.2,
     "x": 1.137,
     "z": 2.869,
     "u": 778.9,
     "v": 239.2,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 39.6,
     "x": 1.42,
     "z": 2.823,
     "u": 808.6,
     "v": 244,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 40,
     "x": 1.703,
     "z": 2.777,
     "u": 838.3,
     "v": 248.9,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 40.4,
     "x": 1.986,
     "z": 2.73,
     "u": 868,
     "v": 253.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 40.8,
     "x": 2.224,
     "z": 2.667,
     "u": 892.9,
     "v": 260.4,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 41.2,
     "x": 2.338,
     "z": 2.404,
     "u": 904.9,
     "v": 288,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 41.6,
     "x": 2.421,
     "z": 2.141,
     "u": 913.5,
     "v": 315.5,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 42,
     "x": 2.513,
     "z": 1.881,
     "u": 923.2,
     "v": 342.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 42.4,
     "x": 2.666,
     "z": 1.676,
     "u": 939.2,
     "v": 364.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 42.8,
     "x": 2.892,
     "z": 1.499,
     "u": 962.9,
     "v": 382.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 43.2,
     "x": 3.118,
     "z": 1.323,
     "u": 986.6,
     "v": 401.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 43.6,
     "x": 3.344,
     "z": 1.146,
     "u": 1010.3,
     "v": 419.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 44,
     "x": 3.57,
     "z": 0.97,
     "u": 1034,
     "v": 438.3,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 44.4,
     "x": 3.796,
     "z": 0.793,
     "u": 1057.7,
     "v": 456.9,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 44.8,
     "x": 4.016,
     "z": 0.67,
     "u": 1080.8,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 45.2,
     "x": 4,
     "z": 0.67,
     "u": 1079.1,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 45.6,
     "x": 3.983,
     "z": 0.67,
     "u": 1077.4,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 46,
     "x": 4.011,
     "z": 0.67,
     "u": 1080.3,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 46.4,
     "x": 3.994,
     "z": 0.67,
     "u": 1078.5,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 46.8,
     "x": 3.978,
     "z": 0.67,
     "u": 1076.8,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 47.2,
     "x": 3.962,
     "z": 0.67,
     "u": 1075.2,
     "v": 469.7,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 47.6,
     "x": 3.876,
     "z": 0.715,
     "u": 1066.1,
     "v": 465,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 48,
     "x": 3.62,
     "z": 0.865,
     "u": 1039.3,
     "v": 449.3,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 48.4,
     "x": 3.364,
     "z": 1.015,
     "u": 1012.4,
     "v": 433.5,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 48.8,
     "x": 3.108,
     "z": 1.165,
     "u": 985.6,
     "v": 417.8,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 49.2,
     "x": 2.852,
     "z": 1.316,
     "u": 958.8,
     "v": 402.1,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 49.6,
     "x": 2.596,
     "z": 1.466,
     "u": 932,
     "v": 386.3,
     "goal": "panel",
     "moving": true
    },
    {
     "t": 50,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 50.4,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 50.8,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 51.2,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 51.6,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 52,
     "x": 2.417,
     "z": 1.571,
     "u": 913.2,
     "v": 375.3,
     "goal": "panel",
     "moving": false
    },
    {
     "t": 52.4,
     "x": 2.412,
     "z": 1.629,
     "u": 912.7,
     "v": 369.2,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 52.8,
     "x": 2.388,
     "z": 1.923,
     "u": 910.1,
     "v": 338.4,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 53.2,
     "x": 2.363,
     "z": 2.216,
     "u": 907.5,
     "v": 307.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 53.6,
     "x": 2.339,
     "z": 2.509,
     "u": 905,
     "v": 277,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 54,
     "x": 2.128,
     "z": 2.578,
     "u": 882.9,
     "v": 269.7,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 54.4,
     "x": 1.835,
     "z": 2.593,
     "u": 852.1,
     "v": 268.1,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 54.8,
     "x": 1.541,
     "z": 2.608,
     "u": 821.3,
     "v": 266.6,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 55.2,
     "x": 1.247,
     "z": 2.623,
     "u": 790.5,
     "v": 265,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 55.6,
     "x": 0.954,
     "z": 2.638,
     "u": 759.7,
     "v": 263.4,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 56,
     "x": 0.76,
     "z": 2.739,
     "u": 739.4,
     "v": 252.9,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 56.4,
     "x": 0.641,
     "z": 2.849,
     "u": 726.9,
     "v": 241.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 56.8,
     "x": 0.56,
     "z": 2.917,
     "u": 718.4,
     "v": 234.2,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 57.2,
     "x": 0.514,
     "z": 2.954,
     "u": 713.6,
     "v": 230.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 57.6,
     "x": 0.49,
     "z": 2.973,
     "u": 711,
     "v": 228.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 58,
     "x": 0.477,
     "z": 2.982,
     "u": 709.7,
     "v": 227.3,
     "goal": "kettle",
     "moving": true
    },
    {
     "t": 58.4,
     "x": 0.688,
     "z": 2.948,
     "u": 731.9,
     "v": 230.8,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 58.8,
     "x": 0.993,
     "z": 2.897,
     "u": 763.9,
     "v": 236.2,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 59.2,
     "x": 1.298,
     "z": 2.846,
     "u": 795.8,
     "v": 241.5,
     "goal": "ntable",
     "moving": true
    },
    {
     "t": 59.6,
     "x": 1.603,
     "z": 2.796,
     "u": 827.8,
     "v": 246.9,
     "goal": "ntable",
     "moving": true
    }
   ]
  },
  {
   "job": "wash",
   "role": "far",
   "points": [
    {
     "t": 0,
     "x": 4.524,
     "z": -2.601,
     "u": 1134.1,
     "v": 812.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 0.4,
     "x": 4.424,
     "z": -2.752,
     "u": 1123.6,
     "v": 828.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 0.8,
     "x": 4.324,
     "z": -2.902,
     "u": 1113.1,
     "v": 844.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 1.2,
     "x": 4.223,
     "z": -3.052,
     "u": 1102.6,
     "v": 860,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 1.6,
     "x": 4.123,
     "z": -3.203,
     "u": 1092,
     "v": 875.8,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 2,
     "x": 4.023,
     "z": -3.353,
     "u": 1081.5,
     "v": 891.6,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 2.4,
     "x": 3.923,
     "z": -3.504,
     "u": 1071,
     "v": 907.4,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 2.8,
     "x": 3.822,
     "z": -3.654,
     "u": 1060.5,
     "v": 923.1,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 3.2,
     "x": 3.722,
     "z": -3.804,
     "u": 1050,
     "v": 938.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 3.6,
     "x": 3.622,
     "z": -3.955,
     "u": 1039.5,
     "v": 954.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 4,
     "x": 3.522,
     "z": -4.105,
     "u": 1029,
     "v": 970.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 4.4,
     "x": 3.424,
     "z": -4.23,
     "u": 1018.7,
     "v": 983.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 4.8,
     "x": 3.248,
     "z": -4.272,
     "u": 1000.3,
     "v": 987.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 5.2,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 5.6,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 6,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 6.4,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 6.8,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 7.2,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 7.6,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 8,
     "x": 3.125,
     "z": -4.301,
     "u": 987.3,
     "v": 991,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 8.4,
     "x": 3.049,
     "z": -4.312,
     "u": 979.4,
     "v": 992.2,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 8.8,
     "x": 2.796,
     "z": -4.349,
     "u": 952.9,
     "v": 996,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 9.2,
     "x": 2.544,
     "z": -4.386,
     "u": 926.4,
     "v": 999.8,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 9.6,
     "x": 2.291,
     "z": -4.422,
     "u": 899.9,
     "v": 1003.7,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 10,
     "x": 2.215,
     "z": -4.433,
     "u": 892,
     "v": 1004.9,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 10.4,
     "x": 2.215,
     "z": -4.433,
     "u": 892,
     "v": 1004.9,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 10.8,
     "x": 2.215,
     "z": -4.433,
     "u": 892,
     "v": 1004.9,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 11.2,
     "x": 2.215,
     "z": -4.433,
     "u": 892,
     "v": 1004.9,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 11.6,
     "x": 2.39,
     "z": -4.41,
     "u": 910.3,
     "v": 1002.4,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 12,
     "x": 2.608,
     "z": -4.381,
     "u": 933.2,
     "v": 999.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 12.4,
     "x": 2.827,
     "z": -4.351,
     "u": 956.1,
     "v": 996.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 12.8,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 13.2,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 13.6,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 14,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 14.4,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 14.8,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 15.2,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 15.6,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 16,
     "x": 2.914,
     "z": -4.34,
     "u": 965.3,
     "v": 995,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 16.4,
     "x": 3.115,
     "z": -4.156,
     "u": 986.3,
     "v": 975.8,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 16.8,
     "x": 3.315,
     "z": -3.972,
     "u": 1007.3,
     "v": 956.5,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 17.2,
     "x": 3.521,
     "z": -3.813,
     "u": 1028.9,
     "v": 939.8,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 17.6,
     "x": 3.716,
     "z": -3.624,
     "u": 1049.3,
     "v": 920,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 18,
     "x": 3.911,
     "z": -3.434,
     "u": 1069.8,
     "v": 900.1,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 18.4,
     "x": 4.066,
     "z": -3.247,
     "u": 1086.1,
     "v": 880.4,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 18.8,
     "x": 4.221,
     "z": -3.023,
     "u": 1102.3,
     "v": 857,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 19.2,
     "x": 4.376,
     "z": -2.8,
     "u": 1118.6,
     "v": 833.6,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 19.6,
     "x": 4.531,
     "z": -2.577,
     "u": 1134.8,
     "v": 810.2,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 20,
     "x": 4.686,
     "z": -2.353,
     "u": 1151.1,
     "v": 786.7,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 20.4,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 20.8,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 21.2,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 21.6,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 22,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 22.4,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 22.8,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 23.2,
     "x": 4.717,
     "z": -2.308,
     "u": 1154.3,
     "v": 782.1,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 23.6,
     "x": 4.674,
     "z": -2.36,
     "u": 1149.8,
     "v": 787.4,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 24,
     "x": 4.531,
     "z": -2.53,
     "u": 1134.8,
     "v": 805.3,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 24.4,
     "x": 4.388,
     "z": -2.701,
     "u": 1119.8,
     "v": 823.2,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 24.8,
     "x": 4.245,
     "z": -2.871,
     "u": 1104.8,
     "v": 841,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 25.2,
     "x": 4.102,
     "z": -3.042,
     "u": 1089.8,
     "v": 858.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 25.6,
     "x": 3.959,
     "z": -3.212,
     "u": 1074.8,
     "v": 876.8,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 26,
     "x": 3.816,
     "z": -3.383,
     "u": 1059.8,
     "v": 894.7,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 26.4,
     "x": 3.673,
     "z": -3.553,
     "u": 1044.8,
     "v": 912.6,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 26.8,
     "x": 3.53,
     "z": -3.724,
     "u": 1029.8,
     "v": 930.4,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 27.2,
     "x": 3.443,
     "z": -3.896,
     "u": 1020.8,
     "v": 948.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 27.6,
     "x": 3.288,
     "z": -4.055,
     "u": 1004.5,
     "v": 965.2,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 28,
     "x": 3.133,
     "z": -4.214,
     "u": 988.2,
     "v": 981.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 28.4,
     "x": 3.102,
     "z": -4.246,
     "u": 984.9,
     "v": 985.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 28.8,
     "x": 3.102,
     "z": -4.246,
     "u": 984.9,
     "v": 985.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 29.2,
     "x": 3.102,
     "z": -4.246,
     "u": 984.9,
     "v": 985.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 29.6,
     "x": 3.102,
     "z": -4.246,
     "u": 984.9,
     "v": 985.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 30,
     "x": 3.102,
     "z": -4.246,
     "u": 984.9,
     "v": 985.2,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 30.4,
     "x": 2.889,
     "z": -4.29,
     "u": 962.6,
     "v": 989.8,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 30.8,
     "x": 2.652,
     "z": -4.338,
     "u": 937.7,
     "v": 994.8,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 31.2,
     "x": 2.415,
     "z": -4.386,
     "u": 912.9,
     "v": 999.9,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 31.6,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 32,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 32.4,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 32.8,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 33.2,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 33.6,
     "x": 2.201,
     "z": -4.429,
     "u": 890.5,
     "v": 1004.4,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 34,
     "x": 2.277,
     "z": -4.42,
     "u": 898.4,
     "v": 1003.4,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 34.4,
     "x": 2.528,
     "z": -4.388,
     "u": 924.8,
     "v": 1000.1,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 34.8,
     "x": 2.78,
     "z": -4.356,
     "u": 951.2,
     "v": 996.8,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 35.2,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 35.6,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 36,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 36.4,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 36.8,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 37.2,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 37.6,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 38,
     "x": 2.93,
     "z": -4.337,
     "u": 967,
     "v": 994.8,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 38.4,
     "x": 2.765,
     "z": -4.359,
     "u": 949.7,
     "v": 997.1,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 38.8,
     "x": 2.529,
     "z": -4.392,
     "u": 924.9,
     "v": 1000.5,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 39.2,
     "x": 2.293,
     "z": -4.424,
     "u": 900.2,
     "v": 1003.8,
     "goal": "isle2",
     "moving": true
    },
    {
     "t": 39.6,
     "x": 2.199,
     "z": -4.436,
     "u": 890.3,
     "v": 1005.2,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 40,
     "x": 2.199,
     "z": -4.436,
     "u": 890.3,
     "v": 1005.2,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 40.4,
     "x": 2.199,
     "z": -4.436,
     "u": 890.3,
     "v": 1005.2,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 40.8,
     "x": 2.199,
     "z": -4.436,
     "u": 890.3,
     "v": 1005.2,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 41.2,
     "x": 2.199,
     "z": -4.436,
     "u": 890.3,
     "v": 1005.2,
     "goal": "isle2",
     "moving": false
    },
    {
     "t": 41.6,
     "x": 2.273,
     "z": -4.426,
     "u": 898.1,
     "v": 1004.1,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 42,
     "x": 2.459,
     "z": -4.401,
     "u": 917.6,
     "v": 1001.5,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 42.4,
     "x": 2.645,
     "z": -4.376,
     "u": 937.1,
     "v": 998.9,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 42.8,
     "x": 2.831,
     "z": -4.351,
     "u": 956.6,
     "v": 996.2,
     "goal": "nrack",
     "moving": true
    },
    {
     "t": 43.2,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 43.6,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 44,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 44.4,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 44.8,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 45.2,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 45.6,
     "x": 2.924,
     "z": -4.339,
     "u": 966.3,
     "v": 994.9,
     "goal": "nrack",
     "moving": false
    },
    {
     "t": 46,
     "x": 3.059,
     "z": -4.214,
     "u": 980.5,
     "v": 981.9,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 46.4,
     "x": 3.209,
     "z": -4.075,
     "u": 996.2,
     "v": 967.3,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 46.8,
     "x": 3.359,
     "z": -3.937,
     "u": 1011.9,
     "v": 952.8,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 47.2,
     "x": 3.521,
     "z": -3.818,
     "u": 1028.9,
     "v": 940.3,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 47.6,
     "x": 3.667,
     "z": -3.675,
     "u": 1044.2,
     "v": 925.3,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 48,
     "x": 3.813,
     "z": -3.532,
     "u": 1059.5,
     "v": 910.4,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 48.4,
     "x": 3.958,
     "z": -3.389,
     "u": 1074.7,
     "v": 895.4,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 48.8,
     "x": 4.071,
     "z": -3.246,
     "u": 1086.6,
     "v": 880.3,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 49.2,
     "x": 4.187,
     "z": -3.078,
     "u": 1098.8,
     "v": 862.7,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 49.6,
     "x": 4.303,
     "z": -2.91,
     "u": 1110.9,
     "v": 845.1,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 50,
     "x": 4.419,
     "z": -2.742,
     "u": 1123,
     "v": 827.5,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 50.4,
     "x": 4.535,
     "z": -2.574,
     "u": 1135.2,
     "v": 809.9,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 50.8,
     "x": 4.65,
     "z": -2.406,
     "u": 1147.3,
     "v": 792.3,
     "goal": "sink",
     "moving": true
    },
    {
     "t": 51.2,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 51.6,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 52,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 52.4,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 52.8,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 53.2,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 53.6,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 54,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 54.4,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 54.8,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 55.2,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 55.6,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 56,
     "x": 4.708,
     "z": -2.322,
     "u": 1153.4,
     "v": 783.5,
     "goal": "sink",
     "moving": false
    },
    {
     "t": 56.4,
     "x": 4.55,
     "z": -2.293,
     "u": 1136.8,
     "v": 780.4,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 56.8,
     "x": 4.352,
     "z": -2.256,
     "u": 1116,
     "v": 776.6,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 57.2,
     "x": 4.154,
     "z": -2.22,
     "u": 1095.2,
     "v": 772.8,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 57.6,
     "x": 3.956,
     "z": -2.183,
     "u": 1074.5,
     "v": 768.9,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 58,
     "x": 3.758,
     "z": -2.147,
     "u": 1053.7,
     "v": 765.1,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 58.4,
     "x": 3.56,
     "z": -2.111,
     "u": 1032.9,
     "v": 761.3,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 58.8,
     "x": 3.362,
     "z": -2.074,
     "u": 1012.2,
     "v": 757.5,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 59.2,
     "x": 3.164,
     "z": -2.038,
     "u": 991.4,
     "v": 753.7,
     "goal": "wrack",
     "moving": true
    },
    {
     "t": 59.6,
     "x": 2.966,
     "z": -2.001,
     "u": 970.7,
     "v": 749.8,
     "goal": "wrack",
     "moving": true
    }
   ]
  }
 ]
}
```
