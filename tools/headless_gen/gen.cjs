// 헤드리스 데이터셋 생성 — 창 없이 sim.html을 띄워 GT 캡처, 파일은 Node가 직접 저장.
// 사용: node gen.cjs <out_dir> <samples>
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = 'http://127.0.0.1:8123/sim.html';
const CAMS = ['cvNW', 'cvNE', 'cvSE', 'cvSW', 'cvN', 'cvS'];
const OUT = process.argv[2] || 'C:/Users/chanwoo/workspace/robot-kitchen-safety-sim/dataset/sim-oblique-6cam-headless';
const SAMPLES = parseInt(process.argv[3] || '20', 10);

function writeDataURL(file, dataURL) {
  const b64 = dataURL.split(',', 2)[1];
  fs.writeFileSync(file, Buffer.from(b64, 'base64'));
}

(async () => {
  fs.mkdirSync(path.join(OUT, 'images'), { recursive: true });
  fs.mkdirSync(path.join(OUT, 'labels'), { recursive: true });
  // headful + 화면 밖 창 + 스로틀 해제 — 헤드리스는 WebGL 백버퍼가 0이라 실패.
  // 창은 -2400,-2400에 떠서 사용자에게 안 보이고, 스로틀 해제로 포커스 없어도 렌더된다.
  const browser = await chromium.launch({ executablePath: CHROME, headless: false,
    args: ['--disable-background-timer-throttling','--disable-renderer-backgrounding',
           '--disable-backgrounding-occluded-windows','--window-position=-2400,-2400'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', e => console.log('  [pageerr]', String(e).slice(0, 160)));
  await page.goto(URL, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction('window.__simReady === true', { timeout: 90000 });
  await page.waitForFunction("eval('typeof SURV')!=='undefined' && eval('SURV').cvN && eval('scene').meshes.length>800", { timeout: 90000 });
  console.log('sim ready. generating', SAMPLES, 'samples x', CAMS.length, 'cams ->', OUT);

  const t0 = Date.now();
  let saved = 0, minColors = 1e9;
  for (let i = 0; i < SAMPLES; i++) {
    // 장면 랜덤화 (인원 수도 흔든다)
    await page.evaluate(async () => {
      const n = [0, 1, 1, 1, 2, 2][(Math.random() * 6) | 0];
      await setExtraCount(n); randomizeScene();
      for (let f = 0; f < 4; f++) scene.render();
    });
    for (const cam of CAMS) {
      const r = await page.evaluate(async (cam) => {
        const g = await groundTruth(cam, { noDepth: true });
        // 색 가짓수(단색=렌더실패) 검사
        const img = new Image(); img.src = g.rgb; await img.decode();
        const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
        const cx = cv.getContext('2d'); cx.drawImage(img, 0, 0);
        const d = cx.getImageData(0, 0, cv.width, cv.height).data;
        const set = new Set(); for (let k = 0; k < d.length; k += 64) set.add((d[k] << 16) | (d[k + 1] << 8) | d[k + 2]);
        return { rgb: g.rgb, labelText: g.labelText, colors: set.size };
      }, cam);
      const base = `${cam}_${String(i).padStart(4, '0')}`;
      writeDataURL(path.join(OUT, 'images', base + '.png'), r.rgb);
      fs.writeFileSync(path.join(OUT, 'labels', base + '.txt'), r.labelText || '');
      saved++; minColors = Math.min(minColors, r.colors);
    }
    const per = (Date.now() - t0) / (i + 1) / 1000;
    console.log(`  sample ${i + 1}/${SAMPLES} done · saved ${saved} · ~${per.toFixed(1)}s/sample · minColors ${minColors}`);
  }
  // data.yaml
  fs.writeFileSync(path.join(OUT, 'data.yaml'),
    `path: ${OUT}\ntrain: images\nval: images\ntest: images\nnc: 6\nnames: ['person','fire','smoke','robot','kettle','equipment']\n`);
  console.log(`DONE: ${saved} images in ${((Date.now() - t0) / 60000).toFixed(1)} min. minColors=${minColors} (${minColors > 500 ? '렌더 정상' : '단색 의심!'})`);
  await browser.close();
})().catch(e => { console.error('ERR', e); process.exit(1); });
