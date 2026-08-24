// 예측 관측 이력의 대상 분리 회귀 테스트 (감사 P0 — track ID별 이력 분리).
//
// 버그: PRED.hist 는 전역 단일 배열이라, 안전 판정 대상이 A→B로 바뀌면 A의 좌표가 남은
//       관측창에 B의 좌표가 이어 붙어 예측이 '순간이동'으로 오염된다.
// 수정: predictionUpdate(srcPos, now, srcId) 가 srcId 변화 시 관측 상태(hist·칼만·속도·
//       예측경로)를 통째로 초기화한다. 이 테스트는 실제 sim.html 함수를 브라우저에서 그대로
//       호출해 서로 다른 사람의 좌표가 한 관측창에 섞이지 않음을 고정한다.
//
// 실행(감사 P0-4로 Playwright+CI가 들어오면 활성화):
//   1) 정적 서버:  uv run python -m http.server 5199   (또는 detect_server)
//   2) BASE_URL=http://localhost:5199 npx playwright test tests/browser
//
// 현재 저장소엔 아직 Playwright devDependency/러너가 없다(P0-4에서 추가). 그전까지는
// 문서화된 회귀 명세로 둔다 — 위 로직은 2026-08-24 실제 브라우저에서 통과를 확인했다.
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5199';

test('predictionUpdate resets observation history on target switch', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(() => {
    const P = (x, z) => new BABYLON.Vector3(x, 0, z);
    PRED.srcId = null; PRED.hist = []; PRED.lastPos = null;
    PRED.vel = new BABYLON.Vector3(0, 0, 0);
    let t = 100000;
    for (let i = 0; i < 10; i++) { predictionUpdate(P(1 + 0.1 * i, 2), t, 'gt:0'); t += 100; }
    const aLen = PRED.hist.length;
    predictionUpdate(P(2.0, 2), t, 'gt:0'); t += 100;   // 같은 대상 → append
    const sameLen = PRED.hist.length;
    predictionUpdate(P(-5, -5), t, 'gt:1');             // 대상 전환 → reset
    return {
      aLen, sameLen,
      bLen: PRED.hist.length,
      bFirstX: PRED.hist[0].x,
      bSrcId: PRED.srcId,
      bVelMag: Math.hypot(PRED.vel.x, PRED.vel.z),
    };
  });

  expect(r.aLen).toBe(10);          // A 좌표가 정상 누적
  expect(r.sameLen).toBe(11);       // 같은 id면 리셋 없이 append
  expect(r.bLen).toBe(1);           // 전환 시 A의 11점이 비워지고 B 1점만
  expect(r.bFirstX).toBe(-5);       // 관측창 첫 점이 B 좌표(= A 좌표 섞이지 않음)
  expect(r.bSrcId).toBe('gt:1');    // 대상 식별자 갱신
  expect(r.bVelMag).toBeLessThan(0.01);  // A→B 점프에서 유령 속도가 생기지 않음
});
