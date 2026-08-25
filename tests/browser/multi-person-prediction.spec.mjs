// 다인원 예측 + 미래 위험 중재 회귀 테스트 (감사 P0-5, 스펙 §9).
//
// 검증(2026-08-24 실제 브라우저에서 통과 확인):
//   1단계 — 활성 전원이 실제 LSTM 경로선(K개)을 갖는다(휴리스틱 아님).
//   2단계 — 최고위험(worst)이 로컬 칼만보다 이르면 로봇 제어를 당긴다(effStop=min(local,worst)),
//           worst의 최빈 경로선이 위험색(빨강)으로 그려진다.
//
// /predict는 페이지 내 mock으로 대체해 실제 LSTM 모델 없이 시뮬 배선만 검증한다(백엔드 위험·중재
// 로직은 pytest tests/test_risk.py·test_predict_batch.py가 담당). P0-4로 Playwright+CI가 들어오면 활성.
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5199';

const INSTALL_MOCK = (tEntryStop) => `
  (function(){
    const real = window.fetch.bind(window);
    window.fetch = function(url, opts){
      if (typeof url==="string" && url.indexOf("/predict")>=0 && opts && opts.body){
        const b = JSON.parse(opts.body);
        if (b.tracks){
          const rx=b.robot.x, rz=b.robot.z;
          const tracks=b.tracks.map(t=>{
            const last=t.hist[t.hist.length-1];
            const dx=rx-last[0], dz=rz-last[1], d=Math.hypot(dx,dz)||1e-6;
            const path=[],sigma=[];
            for(let i=0;i<12;i++){const f=(i+1)/12; path.push([last[0]+dx*f,last[1]+dz*f]); sigma.push(0.1);}
            return {id:t.id, modes:[{path,w:1,sigma}], risk:{tEntryStop:${tEntryStop},tEntrySlow:${tEntryStop}*0.5,riskMass:1,dMin:0.3}};
          });
          const worst = tracks.length ? {id:tracks[0].id,tEntryStop:${tEntryStop},tEntrySlow:${tEntryStop}*0.5,riskMass:1,dMin:0.3} : null;
          return Promise.resolve(new Response(JSON.stringify({tracks,worst}),{status:200,headers:{'Content-Type':'application/json'}}));
        }
      }
      return real(url,opts);
    };
  })();
`;

test('stage 1 — every active worker gets a real LSTM path polyline', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });
  await page.evaluate(INSTALL_MOCK(0.8));

  const r = await page.evaluate(async () => {
    PMODE.on = true;
    for (let k = 0; k < 4; k++) {
      const now = performance.now();
      mpredObserve(now); mpredTick(now, 200);
      await new Promise(res => setTimeout(res, 120));
      safetyUpdate(); safetyUpdate();   // 2회: PVEL 초기화 후 실제 렌더
    }
    const people = peopleList().length;
    const paths = Object.keys(PMODE.mesh).filter(k => k.endsWith('_path') && PMODE.mesh[k].isEnabled());
    const perPerson = [...Array(people).keys()].map(i => paths.some(k => k.startsWith(i + '_')));
    return { people, predSize: MPRED.pred.size, everyoneHasPath: perPerson.every(Boolean) };
  });
  expect(r.predSize).toBe(r.people);         // 전원 예측
  expect(r.everyoneHasPath).toBe(true);      // 전원 경로선
});

test('stage 2 — worst tightens control beyond a non-threatening nearest, drawn red', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });
  await page.evaluate(INSTALL_MOCK(0.6));

  const r = await page.evaluate(async () => {
    PMODE.on = true;
    const bx = LAYOUT.robot.base.x, bz = LAYOUT.robot.base.z;
    const nodes = peopleList();
    nodes.forEach((n, i) => { n.position.x = bx + 8 + i * 0.5; n.position.z = bz + 8; });  // 멀리(위협 아님)
    const now = performance.now();
    mpredObserve(now); mpredTick(now, 200);
    await new Promise(res => setTimeout(res, 150));
    safetyUpdate(); safetyUpdate();
    const wid = MPRED.worst && MPRED.worst.id;
    const m = wid ? PMODE.mesh[wid.split(':')[1] + '_0_path'] : null;
    const c = m && m.greasedLineMaterial ? m.greasedLineMaterial.color : null;
    return {
      localTStop: PRED.tStop === Infinity ? 'Inf' : PRED.tStop,
      effStop: PRED.effStop,
      worstTStop: MPRED.worst ? MPRED.worst.tEntryStop : null,
      red: !!(c && c.r > 0.9 && c.g < 0.45 && c.b < 0.4),
    };
  });
  expect(r.worstTStop).toBe(0.6);
  expect(r.localTStop).toBe('Inf');          // 최근접은 위협 아님
  expect(Math.abs(r.effStop - 0.6)).toBeLessThan(1e-6);  // worst가 제어를 당김
  expect(r.red).toBe(true);                  // worst 경로 = 위험색
});
