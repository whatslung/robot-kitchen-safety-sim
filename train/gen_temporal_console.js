/* 시간축(temporal) 클립 생성 — 한 대의 존 카메라로 '연속 프레임'을 찍는다.
   목적: 공간 겹침(카메라 여러 대) 융합이 ~0.9에서 막히는 이유가 '놓침의 상관성'이었다.
        다른 축 = 시간. 한 사람이 프레임 t에서 놓쳐도 t±1에서 잡히면 추적이 메운다.
        그걸 검증하려면 사람이 '움직이는 하나의 연속 클립'이 필요하다(사람 id 고정).

   기존 데이터셋과의 차이:
   - 기존: 매 프레임 randomizeScene → 서로 무관한 독립 장면(추적 불가).
   - 여기: randomizeScene 은 처음 한 번. 그 뒤엔 사람이 라이브 렌더 루프로 걸어다니는
           사이사이를 잠깐씩 얼려(gtFreeze) 캡처 → 스트로보 방식의 연속 클립.
   - 사람 id 는 person_0(주 조리원)·person_1..N(추가 인원)로 프레임 간 고정(sim.html:11982).

   저장: HTTP POST → gtwriter(8178) → dataset/temporal-clip/{images,labels,meta}.
   meta.persons = [{id, cx, cy, w, h}]  (프레임별 사람 위치, id로 시간축 매칭).

   실행: sim.html 페이지 콘솔에 붙여넣기. window.__temporal = {N:150, gapMs:120} 로 조절. */
(async () => {
  const s = window.__sim, B = BABYLON, scene = s.scene;
  window.__customPredictor = null;
  const POST = "http://127.0.0.1:8178/";
  const cfg = Object.assign({ N: 150, gapMs: 120, cx: 1.375, cz: 1.85, span: 5.8, spanZ: 4.3 }, window.__temporal || {});

  // 연기 완화(라벨 유효성 유지) — 학습 데이터와 동일 조건
  const lowSmoke = () => { FIRE_FOG.max = 0.08; if (fsmoke.color1) fsmoke.color1.a = 0.09; if (fsmoke.color2) fsmoke.color2.a = 0.05; };

  // 단일 존 카메라(span 5.8×4.3, noCeil) — 학습 모델과 같은 배율
  const hw = cfg.span / 2, hh = cfg.spanZ / 2;
  const cam = new B.UniversalCamera("cam_T", new B.Vector3(cfg.cx, 5, cfg.cz), scene);
  cam.setTarget(new B.Vector3(cfg.cx, 0, cfg.cz + 0.001)); cam.mode = B.Camera.ORTHOGRAPHIC_CAMERA;
  cam.orthoLeft = -hw; cam.orthoRight = hw; cam.orthoBottom = -hh; cam.orthoTop = hh; cam.minZ = 0.01; cam.maxZ = 20;
  cam.layerMask = 0x0FFFFFFF & ~0x8000;   // noCeil
  s.SURV["T"] = { cam, label: "temporalT", def: { pos: new B.Vector3(cfg.cx, 5, cfg.cz), tgt: new B.Vector3(cfg.cx, 0, cfg.cz + 0.001) } };

  const raf = () => new Promise(r => requestAnimationFrame(r));
  const stepLive = async (ms) => { gtUnfreeze(); const t0 = performance.now(); while (performance.now() - t0 < ms) await raf(); };
  const pngSize = (u) => { const b = atob(u.slice(u.indexOf(",") + 1, u.indexOf(",") + 60)), c = i => b.charCodeAt(i); return [(c(16) << 24) | (c(17) << 16) | (c(18) << 8) | c(19), (c(20) << 24) | (c(21) << 16) | (c(22) << 8) | c(23)]; };
  const post = async (base, gt, frame) => {
    const persons = (gt.labels || []).filter(l => l.instance && l.instance.startsWith("person"))
      .map(l => ({ id: l.instance, cx: l.cx, cy: l.cy, w: l.w, h: l.h }));
    const meta = JSON.stringify({ clip: "temporal", frame, cam: "T",
      center: [cfg.cx, cfg.cz], span: [cfg.span, cfg.spanZ], persons });
    await fetch(POST, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base, rgb: gt.rgb, label: gt.labelText || "", meta }) });
    return persons.length;
  };

  // 클립 시작: 장면 1회 구성(고정 외형) — 이후 재랜덤 금지(연속성 유지)
  gtUnfreeze(); s.randomizeScene(); await s.setExtraCount(6);
  lowSmoke(); datasetDiversify(); SENSOR.distortion = 0; SENSOR.chroma = 0; sensorApply();
  for (let k = 0; k < 6; k++) scene.render();

  window.__tstate = { i: 0, N: cfg.N, done: false };
  for (let idx = 0; idx < cfg.N; idx++) {
    await stepLive(cfg.gapMs);              // 사람이 gapMs 동안 움직인다
    let gt, tries = 0;
    do { gtFreeze(); for (let k = 0; k < 3; k++) scene.render();
         gt = await s.groundTruth("T", { noDepth: true });
         const [w, h] = pngSize(gt.rgb); if (w === 960 && h === 720) break; } while (++tries < 3);
    const base = "t_" + String(idx).padStart(4, "0");
    const np = await post(base, gt, idx);
    window.__tstate.i = idx + 1;
    if (idx % 10 === 0) console.log(`[temporal] ${idx}/${cfg.N}  사람 ${np}명`);
  }
  gtUnfreeze();
  window.__tstate.done = true;
  console.log("✅ temporal 완료");
})();
