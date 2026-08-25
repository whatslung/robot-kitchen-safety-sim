/* 나디르(orthotop) person 데이터셋 — 브라우저 네이티브 생성 스니펫.
   수정된 sim.html(tol + 연결요소 bbox)에서 실행한다. 콘솔에 통째로 붙여넣고 Enter.
   폴더 선택 창이 뜨면 빈 폴더를 하나 고르면 images/ · labels/ 에 200장이 쓰인다.
   (SENSOR 렌즈 왜곡을 0으로 둬 라벨-RGB 정렬을 보장한다 — 왜곡은 RGB에만 걸리고
    마스크엔 안 걸려 라벨이 어긋나던 문제.) */
(async () => {
  const s = window.__sim, B = BABYLON, scene = s.scene;
  window.__customPredictor = null;                         // /predict 서버로 매 틱 POST(501 도배) 중단 — 생성엔 무관
  if (!s.SURV.orthotop) {                                   // orthotop 직교 나디르 카메라 등록
    const cam = new B.UniversalCamera("cam_orthotop", new B.Vector3(1.375, 5, 1.85), scene);
    cam.setTarget(new B.Vector3(1.375, 0, 1.851));
    cam.mode = B.Camera.ORTHOGRAPHIC_CAMERA;
    cam.orthoLeft = -4.45; cam.orthoRight = 4.45; cam.orthoBottom = -3.3375; cam.orthoTop = 3.3375;
    cam.minZ = 0.01; cam.maxZ = 20;
    s.SURV.orthotop = { cam, label: "orthotop",
      def: { pos: new B.Vector3(1.375, 5, 1.85), tgt: new B.Vector3(1.375, 0, 1.851) } };
  }
  // noCeil — 천장·후드(CEIL_BIT=0x8000)를 빼서 위에서 사람을 가리지 않게 한다(원본 orthotop과 동일).
  s.SURV.orthotop.cam.layerMask = 0x0FFFFFFF & ~0x8000;
  const dir = await showDirectoryPicker({ mode: "readwrite" });   // 빈 폴더 선택
  const imgs = await dir.getDirectoryHandle("images", { create: true });
  const lbls = await dir.getDirectoryHandle("labels", { create: true });
  const writePNG = async (name, dataURL) => {
    const bin = atob(dataURL.split(",", 2)[1]);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const fh = await imgs.getFileHandle(name, { create: true });
    const w = await fh.createWritable(); await w.write(u8); await w.close();
  };
  const writeTXT = async (name, text) => {
    const fh = await lbls.getFileHandle(name, { create: true });
    const w = await fh.createWritable(); await w.write(text || ""); await w.close();
  };
  // PNG 폭·높이(IHDR)를 dataURL에서 바로 읽는다 — 리사이즈 레이스로 크기가 틀어진 프레임 걸러내기.
  const pngSize = (durl) => {
    const b64 = durl.slice(durl.indexOf(",") + 1, durl.indexOf(",") + 60);
    const b = atob(b64), u = i => b.charCodeAt(i);
    return [(u(16)<<24)|(u(17)<<16)|(u(18)<<8)|u(19), (u(20)<<24)|(u(21)<<16)|(u(22)<<8)|u(23)];
  };
  const plan = [[1, 50], [2, 70], [3, 50], [0, 30]];       // 사람 2·3·4·1명 (총 200)
  let idx = 0, t0 = performance.now(), bad = 0;
  for (const [cnt, num] of plan) {
    await s.setExtraCount(cnt);
    for (let j = 0; j < num; j++) {
      let gt, tries = 0;
      do {                                                    // 960×720 정상 캡처가 나올 때까지 재시도(최대 4회)
        s.randomizeScene();
        SENSOR.distortion = 0; SENSOR.chroma = 0; sensorApply();  // 기하 왜곡 제거 → 라벨 정렬
        for (let k = 0; k < 5; k++) scene.render();               // 포즈·파티클 안정화
        gt = await s.groundTruth("orthotop", { noDepth: true });
        const [w, h] = pngSize(gt.rgb);
        if (w === 960 && h === 720) break;
        bad++;
      } while (++tries < 4);
      const base = "orthotop_" + String(idx).padStart(4, "0");
      await writePNG(base + ".png", gt.rgb);
      await writeTXT(base + ".txt", gt.labelText);
      if (idx % 10 === 0) console.log(`[nadir] ${idx}/200  (${Math.round((performance.now()-t0)/1000)}s, 재시도 ${bad})`);
      idx++;
    }
  }
  if (s.restoreAfterDataset) try { s.restoreAfterDataset(); } catch (e) {}
  console.log(`✅ 완료 — ${idx}장. 폴더의 images/ · labels/ 확인.`);
})();
