/* 나디르 4구역(zone) person 데이터셋 — 브라우저 콘솔 생성 스니펫.
   수정된 sim.html(datasetDiversify 현실틴트 + gtFreeze + 연결요소 bbox)에서 실행.
   폴더 선택 창에 빈 폴더를 고르면 images/ · labels/ 에 저장.

   왜 4구역:
   - 단일 나디르(8.9×6.7m)는 사람이 프레임의 0.6%(~59px)로 너무 작아 검출이 약했다.
   - 조리실을 2×2로 나눠 각 카메라가 좁은 구역(5.8×4.3m)만 담으면 사람이 1.45%(~107px)로
     커져 검출·위치정밀도가 대폭 향상(실측: zone recall 0.86 vs wide 0.81, mAP50-95 0.53 vs 0.30).
   - 구역은 x·z로 겹침(1.8m·1.3m) → 경계 넘는 사람도 옆 구역이 잡아 추적 유지. 융합의 토대.
   ⚠️ 조리라인 후드(x[-1.6,4.1]·z[-1.1,1.0], 2.2~2.6m)는 남쪽 두 구역의 로봇 작업공간을
      가린다 — 순수 나디르의 물리적 한계. 여기선 noCeil로 후드를 빼 개방구역 검출을 먼저 검증.

   연기 완화: 화재 프레임이 FIRE_FOG(전체 어둡게)로 너무 어두워 사람이 저대비로 묻혔다.
   FIRE_FOG.max 0.22→0.08, fsmoke α 0.18→0.09 로 낮춰 화재 중에도 사람이 보이게 한다
   (기본값은 데모 현실감을 위해 그대로 두고, 데이터 생성에서만 낮춘다). */
(async () => {
  const s = window.__sim, B = BABYLON, scene = s.scene;
  window.__customPredictor = null;
  const lowSmoke = () => { FIRE_FOG.max = 0.08; if (fsmoke.color1) fsmoke.color1.a = 0.09; if (fsmoke.color2) fsmoke.color2.a = 0.05; };
  lowSmoke();
  // 2×2 구역 카메라 (겹침 포함), span 5.8×4.3, noCeil
  const hw = 5.8 / 2, hh = 4.3 / 2;
  const centers = [[-0.625, 0.35], [3.375, 0.35], [-0.625, 3.35], [3.375, 3.35]];
  centers.forEach(([cx, cz], i) => {
    const cam = new B.UniversalCamera("cam_z" + i, new B.Vector3(cx, 5, cz), scene);
    cam.setTarget(new B.Vector3(cx, 0, cz + 0.001)); cam.mode = B.Camera.ORTHOGRAPHIC_CAMERA;
    cam.orthoLeft = -hw; cam.orthoRight = hw; cam.orthoBottom = -hh; cam.orthoTop = hh; cam.minZ = 0.01; cam.maxZ = 20;
    cam.layerMask = 0x0FFFFFFF & ~0x8000;   // noCeil
    s.SURV["z" + i] = { cam, label: "zone" + i, def: { pos: new B.Vector3(cx, 5, cz), tgt: new B.Vector3(cx, 0, cz + 0.001) } };
  });

  const dir = await showDirectoryPicker({ mode: "readwrite" });
  const imgs = await dir.getDirectoryHandle("images", { create: true });
  const lbls = await dir.getDirectoryHandle("labels", { create: true });
  const wPNG = async (n, u) => { const b = atob(u.split(",", 2)[1]); const a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); const f = await imgs.getFileHandle(n, { create: true }); const w = await f.createWritable(); await w.write(a); await w.close(); };
  const wTXT = async (n, t) => { const f = await lbls.getFileHandle(n, { create: true }); const w = await f.createWritable(); await w.write(t || ""); await w.close(); };
  const pngSize = (u) => { const b = atob(u.slice(u.indexOf(",") + 1, u.indexOf(",") + 60)), c = i => b.charCodeAt(i); return [(c(16) << 24) | (c(17) << 16) | (c(18) << 8) | c(19), (c(20) << 24) | (c(21) << 16) | (c(22) << 8) | c(23)]; };
  const prep = () => { gtUnfreeze(); s.randomizeScene(); lowSmoke(); datasetDiversify(); SENSOR.distortion = 0; SENSOR.chroma = 0; sensorApply(); for (let k = 0; k < 5; k++) scene.render(); gtFreeze(); };

  const N = 340;
  await s.setExtraCount(5);
  for (let idx = 0, bad = 0; idx < N; idx++) {
    let gt, tries = 0;
    do { prep(); gt = await s.groundTruth("z" + (idx % 4), { noDepth: true }); const [w, h] = pngSize(gt.rgb); if (w === 960 && h === 720) break; bad++; } while (++tries < 4);
    const base = "zone" + (idx % 4) + "_" + String(idx).padStart(4, "0");
    await wPNG(base + ".png", gt.rgb); await wTXT(base + ".txt", gt.labelText);
    if (idx % 20 === 0) console.log(`[zone] ${idx}/${N} (재시도 ${bad})`);
  }
  gtUnfreeze();
  console.log("✅ 완료");
})();
