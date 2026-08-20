# 칼만 예측 시각화 구현 계획 (σ 음영 + 조감 패널)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 또는 superpowers:executing-plans로 태스크 단위 구현. 단계는 `- [ ]` 체크박스.
> **검증 방식**: 이 저장소는 테스트 프레임워크가 없다(단일 `sim.html` + Babylon). 각 태스크는 **브라우저 기능 확인**으로 검증한다 — 서버 `uv run python backend/detect_server.py --port 8001` 기동 후 `http://127.0.0.1:8001/sim.html?person=1`에서 `window.__sim` JS로 상태를 읽어 확인.

**Goal:** 라이브 (x,z) 트랙에 칼만 예측 + σ 불확실성을 2D 조감 패널과 3D 씬 밴드로 시각화한다.

**Architecture:** 전부 `sim.html`(JS). 예측(`PRED`/`kfStep`)·σ·링·소스선택은 이미 존재 — `predictionUpdate`가 산출한 예측점을 `PRED.pathPts`로 저장해, (1) 2D 조감 패널(`drawOverlay` 내 `drawBirdseye`)과 (2) 3D 리본 밴드가 공유한다. Python은 변경 없음.

**Tech Stack:** Babylon.js, Canvas 2D 오버레이. 좌표 변환 `SCALE.m()/SCALE.au()`, 로봇 기준 `LAYOUT.robot.base`.

## Global Constraints

- `sim.html` **단일 파일만** 수정. Python/의존성 변경 없음.
- 정지=빨강 `#f85149`, 감속=노랑 `#e3b341`, 정상=초록 `#2ea043` (기존 `PRED.line` 색 규칙과 일치).
- 화면 세로축=월드 +Z, 가로축=월드 +X (orthotop 계약과 동일).
- 이동 중(속도>0.08㎧)일 때만 예측선·밴드 표시(기존 규칙 유지).
- 토글 기본 on, `window.__sim`에 노출.

---

### Task 1: 예측점 저장 (`PRED.pathPts`) — 2D/3D 공유 데이터

**Files:** Modify `sim.html` (함수 `predictionUpdate`, ~2954–3025)

**Interfaces:**
- Produces: `PRED.pathPts` = `[{x,z}, …]` (AU, 지금→지평선), `PRED.level` = 0 정상 / 1 감속 / 2 정지, `PRED.moving` = bool.

- [ ] **Step 1: `predictionUpdate`에서 예측점·레벨 저장**

`predictionUpdate` 안, `pts.push(new V3(px, 0.06, pz));`(~3007) 직후 같은 루프에서 AU 좌표도 모으도록, 루프 앞에 `const flat = [];` 추가하고 루프 안에 `flat.push({x:px, z:pz});` 추가. 루프 종료 후(예측선 갱신 직전, ~3014) 추가:

```javascript
  PRED.pathPts = flat;
  PRED.moving = SCALE.m(speed) > 0.08;
  PRED.level = PRED.tStop < Infinity ? 2 : PRED.tSlow < Infinity ? 1 : 0;
```

- [ ] **Step 2: 브라우저에서 확인**

서버 기동 후 콘솔:
```javascript
const S=__sim; S.setView('orthotop'); S.MIL.mode='http'; S.MIL.url='/detect';
await S.setExtraCount(2); S.MIL.safety=true; S.MIL.on=true;
for(let i=0;i<8;i++){ for(let j=0;j<2;j++) S.scene.render(); await S.milCapture(); await new Promise(r=>setTimeout(r,120)); }
JSON.stringify({pts:S.PRED.pathPts&&S.PRED.pathPts.length, level:S.PRED.level, moving:S.PRED.moving, sigma:S.PRED.sigma})
```
Expected: `pts`=13(steps+1), `level`은 0~2, `sigma`≥0 (숫자).

- [ ] **Step 3: 커밋**

```bash
git add sim.html
git commit -m "feat(sim): 예측점 PRED.pathPts 저장 — 2D/3D σ 시각화 공유용"
```

---

### Task 2: 2D 조감 패널 — 로봇·정지/감속 원·눈금 + 토글

**Files:** Modify `sim.html` (신규 함수 `drawBirdseye`; `drawOverlay` 끝 ~4948에 호출 추가; `BE` 설정 객체 추가; `__sim` export ~6658)

**Interfaces:**
- Consumes: `overlayCv`, 2D `ctx`(drawOverlay 스코프), `SAFE.stopR/slowR`(m), `LAYOUT.robot.base`, `SCALE`.
- Produces: `BE` = `{on:true, size:220, viewR:3.5}`; `drawBirdseye(ctx, W, H)`; `window.__sim.BE`.

- [ ] **Step 1: `BE` 설정 + `drawBirdseye` 함수 추가**

`drawMilBoxes` 함수 정의 위(~4950)에 추가:

```javascript
/* 조감(bird's-eye) 예측 패널 — 위에서 내려다본 미니맵. 로봇 중심, 정지/감속 원,
   추적 사람 점, 위협 트랙의 예측 경로 + σ 음영. 월드(x,z)[AU]→패널px 선형 매핑. */
const BE = { on:true, size:220, viewR:3.5 };   // viewR = 패널 반폭이 담는 반경(m)
const BE_IDCOL = ["#2ecc71","#3498db","#e67e22","#9b59b6","#1abc9c","#f39c12"];
function beMap(wx, wz, cx, cy, half) {          // 월드 AU → 패널 px
  const dxm = SCALE.m(wx - LAYOUT.robot.base.x), dzm = SCALE.m(wz - LAYOUT.robot.base.z);
  return [cx + dxm / BE.viewR * half, cy - dzm / BE.viewR * half];  // 위=+Z
}
function drawBirdseye(ctx, W, H) {
  if (!BE.on) return;
  const sz = BE.size, pad = 12, x0 = W - sz - pad, y0 = H - sz - pad;
  const cx = x0 + sz/2, cy = y0 + sz/2, half = sz/2, pxPerM = half / BE.viewR;
  ctx.save();
  ctx.fillStyle = "rgba(13,17,23,0.82)"; ctx.fillRect(x0, y0, sz, sz);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, sz, sz);
  // 1m 격자
  ctx.strokeStyle = "rgba(120,130,140,0.18)";
  for (let m = -Math.floor(BE.viewR); m <= BE.viewR; m++) {
    const gx = cx + m*pxPerM, gy = cy + m*pxPerM;
    ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y0+sz); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x0+sz, gy); ctx.stroke();
  }
  // 감속(노랑)·정지(빨강) 원
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#2ea043"; ctx.beginPath(); ctx.arc(cx, cy, SAFE.slowR*pxPerM, 0, 7); ctx.stroke();
  ctx.strokeStyle = "#e3b341"; ctx.beginPath(); ctx.arc(cx, cy, SAFE.stopR*pxPerM, 0, 7); ctx.stroke();
  // 로봇 중심
  ctx.fillStyle = "#e6edf3"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill();
  ctx.fillStyle = "#8b949e"; ctx.font = "10px ui-monospace,monospace"; ctx.textBaseline="top";
  ctx.fillText(BE.viewR.toFixed(1)+"m", x0+4, y0+4);
  ctx._beCtx = { cx, cy, half, pxPerM };   // Task3에서 재사용
  ctx.restore();
}
```

`drawOverlay`의 마지막 줄 `drawMilBoxes(...)`(~4948) **다음 줄**에 추가:

```javascript
  drawBirdseye(ctx, overlayCv.width, overlayCv.height);
```

- [ ] **Step 2: `__sim` export에 `BE` 추가**

`__sim = { … }`(~6641)의 `MIL, milCapture, …` 줄에 `BE,` 추가.

- [ ] **Step 3: 브라우저 확인**

```javascript
const S=__sim; S.setView('orthotop'); for(let j=0;j<3;j++) S.scene.render();
S.drawOverlay ? 'drawOverlay 존재' : '없음';   // 그리고 화면 우하단에 링·격자 패널이 보이는지 육안
JSON.stringify({on:S.BE.on, size:S.BE.size})
```
Expected: `{on:true,size:220}`, 우하단에 정지/감속 원 + 격자 패널 표시(픽셀 확인은 캡처로).

- [ ] **Step 4: 커밋**

```bash
git add sim.html
git commit -m "feat(sim): 2D 조감 패널 — 로봇·정지/감속 원·격자 + 토글(BE)"
```

---

### Task 3: 조감 패널에 트랙 점 + 예측 경로 + σ 음영

**Files:** Modify `sim.html` (`drawBirdseye`에 이어 그림)

**Interfaces:**
- Consumes: `MIL.on/tracks`, `person.node.position`(GT 폴백), `PRED.pathPts/level/moving/sigma`, `ctx._beCtx`(Task2).

- [ ] **Step 1: `drawBirdseye` 끝(ctx.restore() 앞)에 점·경로·σ 추가**

`ctx._beCtx = {…};` 다음, `ctx.restore();` 앞에 삽입:

```javascript
  const LVL = ["#2ea043","#e3b341","#f85149"];
  // 추적 사람 점 (MIL 켜짐이면 트랙, 아니면 GT 사람)
  if (MIL.on && MIL.tracks && Object.keys(MIL.tracks).length) {
    let i = 0;
    for (const id in MIL.tracks) {
      const p = MIL.tracks[id].pos; if (!p) continue;
      const [px, py] = beMap(p.x, p.z, cx, cy, half);
      ctx.fillStyle = BE_IDCOL[(+id) % BE_IDCOL.length] || "#e6edf3";
      ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill();
      ctx.font = "9px ui-monospace,monospace"; ctx.textBaseline="bottom";
      ctx.fillText("#"+id, px+5, py-2); i++;
    }
  } else {
    const p = person.node.position;
    const [px, py] = beMap(p.x, p.z, cx, cy, half);
    ctx.fillStyle = "#e6edf3"; ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill();
  }
  // 위협 트랙 예측 경로 + σ 음영 (이동 중일 때만)
  if (PRED.moving && PRED.pathPts && PRED.pathPts.length > 1) {
    const col = LVL[PRED.level || 0];
    const pp = PRED.pathPts.map(q => beMap(q.x, q.z, cx, cy, half));
    const sigPx = SAFE ? PRED.sigma * pxPerM : 0;   // σ[m]→px
    // σ 음영: 경로를 따라 좌우로 (t/horizon)·σ 만큼 벌린 반투명 폴리곤
    const n = pp.length, L = [], R = [];
    for (let k = 0; k < n; k++) {
      const a = pp[Math.max(0,k-1)], b = pp[Math.min(n-1,k+1)];
      let dx = b[0]-a[0], dy = b[1]-a[1]; const len = Math.hypot(dx,dy)||1;
      const nx = -dy/len, ny = dx/len, w = sigPx * (k/(n-1));
      L.push([pp[k][0]+nx*w, pp[k][1]+ny*w]); R.push([pp[k][0]-nx*w, pp[k][1]-ny*w]);
    }
    ctx.beginPath(); ctx.moveTo(L[0][0], L[0][1]);
    for (const q of L) ctx.lineTo(q[0], q[1]);
    for (let k = R.length-1; k >= 0; k--) ctx.lineTo(R[k][0], R[k][1]);
    ctx.closePath(); ctx.fillStyle = col + "33"; ctx.fill();   // 반투명
    // 예측 경로 선
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(pp[0][0], pp[0][1]); for (const q of pp) ctx.lineTo(q[0], q[1]); ctx.stroke();
  }
```

- [ ] **Step 2: 브라우저 확인**

```javascript
const S=__sim; S.setView('orthotop'); S.MIL.mode='http'; S.MIL.url='/detect';
await S.setExtraCount(2); S.MIL.safety=true; S.MIL.on=true;
for(let i=0;i<10;i++){ for(let j=0;j<2;j++) S.scene.render(); await S.milCapture(); await new Promise(r=>setTimeout(r,120)); }
JSON.stringify({tracks:Object.keys(S.MIL.tracks).length, pts:S.PRED.pathPts.length, moving:S.PRED.moving, level:S.PRED.level})
```
Expected: `tracks`≥1, `pts`=13, 패널에 색 점 + (이동 시)예측선·σ 음영. 캡처 이미지로 육안 확인(σ가 끝으로 갈수록 넓어짐).

- [ ] **Step 3: 커밋**

```bash
git add sim.html
git commit -m "feat(sim): 조감 패널에 트랙 점·예측 경로·σ 음영 추가"
```

---

### Task 4: 3D 씬 σ 밴드 (리본 메시)

**Files:** Modify `sim.html` (`predictionUpdate`, 예측선 갱신부 ~3014–3024; `PRED` 정의에 `band` 필드)

**Interfaces:**
- Consumes: `PRED.pathPts`(AU), `PRED.sigma`(m), `PRED.level`, `PRED.moving`, `SCALE.au`.
- Produces: `PRED.band` (BABYLON ribbon mesh).

- [ ] **Step 1: `PRED` 정의(~2899)에 `band:null,` 추가**

`line:null,` 옆에 `band:null,`.

- [ ] **Step 2: 예측선 갱신 직후(~3024, `PRED.line.color = …` 다음)에 밴드 갱신 추가**

```javascript
  // 3D σ 밴드 — 예측 경로를 따라 (t/horizon)·σ 만큼 좌우로 벌린 반투명 리본
  const bandCol = PRED.tStop < Infinity ? C3.FromHexString("#f85149")
                : PRED.tSlow < Infinity ? C3.FromHexString("#e3b341")
                : C3.FromHexString("#2ea043");
  const sAU = SCALE.au(PRED.sigma), np = pts.length;
  const left = [], right = [];
  for (let k = 0; k < np; k++) {
    const a = pts[Math.max(0,k-1)], b = pts[Math.min(np-1,k+1)];
    let dx = b.x-a.x, dz = b.z-a.z; const len = Math.hypot(dx,dz)||1;
    const nx = -dz/len, nz = dx/len, w = sAU * (k/(np-1));
    left.push(new V3(pts[k].x+nx*w, 0.05, pts[k].z+nz*w));
    right.push(new V3(pts[k].x-nx*w, 0.05, pts[k].z-nz*w));
  }
  PRED.band = BABYLON.MeshBuilder.CreateRibbon("predBand",
      { pathArray:[left, right], updatable:true, instance: PRED.band || undefined }, scene);
  if (!PRED.band.material) {
    const bm = new BABYLON.StandardMaterial("predBandMat", scene);
    bm.backFaceCulling = false; bm.alpha = 0.22; bm.disableLighting = true;
    PRED.band.material = bm; PRED.band.isPickable = false;
  }
  PRED.band.material.emissiveColor = bandCol; PRED.band.material.diffuseColor = bandCol;
  PRED.band.setEnabled(PRED.moving);
```

`else PRED.line.setEnabled(false);` 계열의 정지 분기(safetyUpdate ~3141, `else if (PRED.line) PRED.line.setEnabled(false);`)에 밴드도 끄기:
```javascript
  } else { if (PRED.line) PRED.line.setEnabled(false); if (PRED.band) PRED.band.setEnabled(false); }
```

- [ ] **Step 3: 브라우저 확인**

```javascript
const S=__sim; S.setView('orthotop'); S.MIL.mode='http'; S.MIL.url='/detect';
await S.setExtraCount(2); S.MIL.safety=true; S.MIL.on=true;
for(let i=0;i<10;i++){ for(let j=0;j<2;j++) S.scene.render(); await S.milCapture(); await new Promise(r=>setTimeout(r,120)); }
JSON.stringify({band: !!S.PRED.band, enabled: S.PRED.band && S.PRED.band.isEnabled(), verts: S.PRED.band && S.PRED.band.getTotalVertices()})
```
Expected: `band`=true, 이동 시 `enabled`=true, `verts`>0. orthotop 캡처에서 예측선을 감싼 반투명 밴드가 위협도 색으로 보임.

- [ ] **Step 4: 커밋**

```bash
git add sim.html
git commit -m "feat(sim): 3D σ 밴드 리본 — 예측 경로 불확실성 음영"
```

---

### Task 5: 토글 노출·마무리·핸드오프 갱신

**Files:** Modify `sim.html`(`__sim` export 확인), `docs/chanwoo/detection-eval.md` 또는 `HANDOFF` 링크

- [ ] **Step 1: 토글 동작 확인**

콘솔에서 `__sim.BE.on=false`(패널 숨김), `true`(복귀); `__sim.PRED.band.setEnabled(false)` 동작 확인.

- [ ] **Step 2: 조감 패널+3D 밴드가 함께 뜬 orthotop 캡처 1장 확보**

`groundTruth`는 오버레이를 안 그리므로, 화면 캔버스 `toDataURL`로 패널 포함 캡처(수동). 데모 스냅으로 저장.

- [ ] **Step 3: 핸드오프에 결과 한 줄 추가 + 커밋**

`docs/chanwoo/HANDOFF.md`에 "예측 시각화(σ 음영+조감 패널) 완료 — 라이브 트랙 기반" 한 줄. 커밋:
```bash
git add sim.html docs/chanwoo/HANDOFF.md
git commit -m "feat(sim): 예측 시각화 토글 정리 + 핸드오프 갱신"
```

---

## Self-Review

- **스펙 커버리지**: ①2D 패널(Task2·3) ②3D 밴드(Task4) ③σ 음영(Task3·4) ④라이브 소스(기존, Task3에서 사용) ⑤토글(Task2·5) — 모두 태스크 존재.
- **플레이스홀더**: 없음(코드 전부 기재).
- **타입 일관성**: `PRED.pathPts`(AU {x,z})는 Task1 생성 → Task3(2D) 소비. 3D 밴드는 `pts`(V3, predictionUpdate 지역)에서 직접 생성. `beMap`·`BE`·`ctx._beCtx` 일관.
- **주의**: `pts`는 `predictionUpdate` 지역 배열 — Task4는 같은 함수 안이라 접근 가능. Task3(drawBirdseye)는 `PRED.pathPts`(저장본) 사용.
