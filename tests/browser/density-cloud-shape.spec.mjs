// 밀도 구름 형상 회귀 테스트 (PR #40 — "사각형 구름" 버그 수정).
//
// 버그: 학습형(LSTM/Transformer) 예측은 진행 방향이 없는 등방 블롭이라 ux=uz=0 으로 들어온다.
//       옛 코드는 이를 회전 투영(al=따라, ac=가로)에 그대로 넣어 al=ac=0 → q=0 이 됐고,
//       그 결과 3σ 바운딩 박스 전체가 균일하게 채워져 방향성 없는 '사각형 구름'이 그려졌다.
// 수정: 방향이 없으면 회전 없이 반경 가우시안(sAlong==sAcross → 원)으로 평가한다.
//
// 이 테스트는 densTexelQ(순수 함수)를 실제 sim.html 로부터 브라우저에서 직접 호출해,
//   (1) 등방 블롭은 같은 거리면 방향에 무관하게 동일 q(=원)이고 0이 아니며(버그면 0),
//   (2) 방향성 블롭은 기존 회전 투영식과 정확히 일치(무손상)함을 고정한다.
//
// 실행(Playwright 러너가 들어오면 활성화 — 현재 저장소엔 아직 devDependency/러너가 없다):
//   1) 정적 서버:  uv run python -m http.server 5199   (또는 detect_server)
//   2) BASE_URL=http://localhost:5199 npx playwright test tests/browser
// 그전까지는 문서화된 회귀 명세로 둔다.
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5199';

test('densTexelQ: 등방 블롭은 원(방향 무관·비영), 방향성 블롭은 회전식과 일치', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(() => {
    // (1) 등방 블롭: 방향 없음(ux=uz=0), sAlong==sAcross.
    const iso = { x: 0, z: 0, ux: 0, uz: 0, sAlong: 1, sAcross: 1, w: 1 };
    const qEast = densTexelQ(iso, 2, 0);            // +x 로 2
    const qNorth = densTexelQ(iso, 0, 2);           // +z 로 2 (같은 거리)
    const qDiag = densTexelQ(iso, Math.SQRT2, Math.SQRT2);   // 대각으로 같은 거리

    // (2) 방향성 블롭: 단위 heading +x, 비등방(sAlong≠sAcross).
    const dir = { x: 0, z: 0, ux: 1, uz: 0, sAlong: 2, sAcross: 1, w: 1 };
    const dx = 1.5, dz = 0.7;
    const al = dx * dir.ux + dz * dir.uz, ac = -dx * dir.uz + dz * dir.ux;
    const expectedDir = (al / dir.sAlong) ** 2 + (ac / dir.sAcross) ** 2;

    return { qEast, qNorth, qDiag, qDir: densTexelQ(dir, dx, dz), expectedDir };
  });

  // 등방: 같은 거리(=2)면 방향에 무관하게 q=(2/1)^2=4 — 원이다.
  expect(r.qEast).toBeCloseTo(4, 9);
  expect(r.qNorth).toBeCloseTo(4, 9);
  expect(r.qDiag).toBeCloseTo(4, 9);
  expect(r.qEast).toBeGreaterThan(0);          // 버그(q≡0, 사각형 균일 채움)가 아님을 고정
  // 방향성: 기존 회전 투영식과 정확히 일치(휴리스틱 경로 무손상).
  expect(r.qDir).toBeCloseTo(r.expectedDir, 12);
});
