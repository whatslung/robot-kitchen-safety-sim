# 급식 조리로봇 안전 시뮬레이터 (Web / Babylon.js)

급식 조리 교반솥(국솥) 로봇의 **안전(끼임·충돌 회피)** 연구용 웹 3D 시뮬레이터.
학습 데이터는 이 시뮬레이터에서 합성하고, 실사 촬영본은 검증용으로만 쓴다.

빌드 과정 없이 정적 파일만으로 동작한다. 실행 중 외부 서버에 접속하지 않는다.

---

## 실행

```bash
python -m http.server 5173
```

→ <http://localhost:5173/sim.html?layout=island>

더블클릭 실행이 필요하면 `launch/` 안의 스크립트를 쓴다 (macOS: `실행하기_Mac.command`,
Windows: `실행하기_Windows.bat`). 빈 포트를 찾아 서버를 띄우고 Chrome/Edge로 열어준다.
`index.html`과 이 스크립트들은 아래 플래그를 이미 붙여 연다.

> **`?layout=island`를 반드시 붙일 것.** 이게 팀에서 확정한 배치다
> (방 11.5×11.5 m · 천장 3.9 m · 구역담당 CCTV 15대 · 안전링 3단계 · 로봇 팔 1.70 m).
> 플래그 없이 `sim.html`만 열면 **확정 이전의 옛 배치**가 뜬다
> (방 9 m · 천장 3.3 m · 카메라 10대 · 팔 1.30 m). 옛 배치는 비교용으로만 남겨 둔 것이라
> 여기서 뽑은 데이터셋은 팀 기준과 맞지 않는다. 화면 우측 정보줄의 `방 11.5m`로 확인할 수 있다.

> **`sim.html`을 파일로 직접 열지 말 것.** `file://`로 열면 3D 화면은 뜨지만
> ONNX 런타임이 wasm/mjs를 ES 모듈로 로드하지 못해 모델 검증이 동작하지 않는다.
> 그렇게 열면 화면 위에 경고가 뜬다.

**브라우저**: Chrome 또는 Edge 113+ 권장 (WebGPU). Safari는 그림자·MSAA에서 느리고,
WebGPU가 없으면 wasm으로 자동 폴백된다. 브라우저 탭을 앞에 두어야 한다 —
백그라운드로 내리면 렌더 루프가 멈춰 불·연기가 자라지 않고 캡처도 느려진다.

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
