/* 나디르(orthotop) person 데이터셋 — 브라우저 네이티브 대량 생성 스니펫 (v3 다양성).
   수정된 sim.html(tol + 연결요소 bbox + datasetDiversify/gtFreeze)에서 실행. 콘솔에 붙여넣고 Enter.
   폴더 선택 창이 뜨면 빈 폴더를 하나 고르면 images/ · labels/ 에 N장이 쓰인다.

   v3 개선점:
   - datasetDiversify(): 사람 옷·모자 색 + 바닥 밝기 랜덤화 → 저대비(흰옷↔밝은 바닥) 하드미스 해소.
     GT 라벨은 인스턴스 마스크 기준이라 색 변경에 무영향(라벨 정확 유지).
   - gtFreeze/gtUnfreeze: 캡처 순간 이동·포즈 고정 → 걷는 사람 박스 어긋남 제거.
   - noCeil layerMask: 후드가 위에서 사람을 가리지 않게.
   - distortion=0: 라벨-RGB 정렬. 960×720 크기 가드로 리사이즈 레이스 재시도.

   ★ 생성 중 그 탭을 화면에 그대로 두세요(백그라운드로 가면 렌더가 멈춥니다). */
(async () => {
  const s = window.__sim, B = BABYLON, scene = s.scene;
  window.__customPredictor = null;                         // /predict 501 도배 중단(생성 무관)
  if (!s.SURV.orthotop) {                                   // orthotop 직교 나디르 카메라
    const cam = new B.UniversalCamera("cam_orthotop", new B.Vector3(1.375, 5, 1.85), scene);
    cam.setTarget(new B.Vector3(1.375, 0, 1.851));
    cam.mode = B.Camera.ORTHOGRAPHIC_CAMERA;
    cam.orthoLeft = -4.45; cam.orthoRight = 4.45; cam.orthoBottom = -3.3375; cam.orthoTop = 3.3375;
    cam.minZ = 0.01; cam.maxZ = 20;
    s.SURV.orthotop = { cam, label: "orthotop",
      def: { pos: new B.Vector3(1.375, 5, 1.85), tgt: new B.Vector3(1.375, 0, 1.851) } };
  }
  s.SURV.orthotop.cam.layerMask = 0x0FFFFFFF & ~0x8000;    // noCeil — 후드·천장 제외

  const dir = await showDirectoryPicker({ mode: "readwrite" });
  const imgs = await dir.getDirectoryHandle("images", { create: true });
  const lbls = await dir.getDirectoryHandle("labels", { create: true });
  const wPNG = async (n, u) => { const b = atob(u.split(",", 2)[1]); const a = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i);
    const f = await imgs.getFileHandle(n, { create: true }); const w = await f.createWritable(); await w.write(a); await w.close(); };
  const wTXT = async (n, t) => { const f = await lbls.getFileHandle(n, { create: true });
    const w = await f.createWritable(); await w.write(t || ""); await w.close(); };
  const pngSize = (u) => { const b = atob(u.slice(u.indexOf(",") + 1, u.indexOf(",") + 60)), c = i => b.charCodeAt(i);
    return [(c(16)<<24)|(c(17)<<16)|(c(18)<<8)|c(19), (c(20)<<24)|(c(21)<<16)|(c(22)<<8)|c(23)]; };

  const prep = () => { gtUnfreeze(); s.randomizeScene(); datasetDiversify();
    SENSOR.distortion = 0; SENSOR.chroma = 0; sensorApply();
    for (let k = 0; k < 5; k++) scene.render(); gtFreeze(); };

  const plan = [[2, 120], [3, 180], [4, 120], [1, 50], [0, 30]];   // 사람 3·4·5·2·1명 (총 500) — 밀도↑
  let idx = 0, t0 = performance.now(), bad = 0;
  for (const [cnt, num] of plan) {
    await s.setExtraCount(cnt);
    for (let j = 0; j < num; j++) {
      let gt, tries = 0;
      do { prep(); gt = await s.groundTruth("orthotop", { noDepth: true });
        const [w, h] = pngSize(gt.rgb); if (w === 960 && h === 720) break; bad++;
      } while (++tries < 4);
      const base = "orthotop_" + String(idx).padStart(4, "0");
      await wPNG(base + ".png", gt.rgb); await wTXT(base + ".txt", gt.labelText);
      if (idx % 10 === 0) console.log(`[nadir] ${idx}/500  (${Math.round((performance.now()-t0)/1000)}s, 재시도 ${bad})`);
      idx++;
    }
  }
  gtUnfreeze();
  if (s.restoreAfterDataset) try { s.restoreAfterDataset(); } catch (e) {}
  console.log(`✅ 완료 — ${idx}장`);
})();
