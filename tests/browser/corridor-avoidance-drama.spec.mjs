// 통로 회피 데모(발표 모드) 회귀 명세 (2026-08-25).
//
// 설계: docs/chanwoo/specs/2026-08-25-corridor-avoidance-drama-design.md
//
// 검증 포인트
//   1) 발표 모드 OFF(기본)이면 stepUpdate 동작이 기존과 동일하고 고스트·인과선은 숨는다.
//   2) presentArc(t)가 통로 횡단 큰 아크의 끝점(카트 쪽 도달점)을 돌려준다.
//   3) corrRun("before") — 회피 OFF: 팔이 사람을 쳐서 쓰러뜨린다(PERSON.acting === "down").
//   4) corrRun("after")  — 회피 ON: 예측 침범에 hold/retract로 대응하고 고스트가 표시된다.
//
// 실행(감사 P0-4로 Playwright+CI가 들어오면 활성화):
//   1) 정적 서버:  uv run python -m http.server 5199
//   2) BASE_URL=http://localhost:5199 npx playwright test tests/browser
//
// 현재 저장소엔 아직 Playwright devDependency/러너가 없다(P0-4에서 추가). 그전까지는
// 문서화된 회귀 명세로 둔다 — 아래 로직은 2026-08-25 실제 브라우저에서 통과를 확인했다.
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5199';

test('발표 모드 OFF면 stepUpdate 동작 불변 · 고스트/인과선 숨김', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(() => {
    // 기본 상태
    const presentOff = PRESENT.on === false;
    // 발표 모드 OFF에서 로봇 정상 사이클을 몇 프레임 돌린다
    state.auto = true; state.autoLoop = true; startStep(0);
    const seen = new Set();
    for (let k = 0; k < 30; k++) { for (let i = 0; i < 12; i++) scene.render(); seen.add(state.seqIdx); }
    return {
      presentOff,
      corrOff: CORR.on === false,
      ghostHidden: !(GHOST.line && GHOST.line.isEnabled()),
      causalHidden: !(CAUSAL.line && CAUSAL.line.isEnabled()),
      advanced: seen.size >= 2,     // 단계가 진행됐다(멈춰 있지 않다)
    };
  });

  expect(r.presentOff).toBe(true);
  expect(r.corrOff).toBe(true);
  expect(r.ghostHidden).toBe(true);
  expect(r.causalHidden).toBe(true);
  expect(r.advanced).toBe(true);
});

test('presentArc가 통로 횡단 큰 아크의 끝점을 돌려준다', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(() => {
    const p0 = presentArc(0), p1 = presentArc(1);
    return { p0: [p0.x, p0.y, p0.z], p1: [p1.x, p1.y, p1.z], crossIdx: SEQ.findIndex(s => s.cross) };
  });

  expect(r.p0[1]).toBeCloseTo(0.80, 1);     // 시작은 솥 수면 높이
  expect(r.p1[1]).toBeGreaterThan(0.85);    // 끝은 들어올린 이송 높이
  expect(Math.abs(r.p1[2])).toBeGreaterThan(1.5);  // 통로를 가로질러 z로 크게 나간다
  expect(r.crossIdx).toBeGreaterThanOrEqual(0);    // 통로 횡단 단계가 존재
});

test('corrRun("before") — 회피 OFF면 팔이 사람을 쳐서 쓰러뜨린다', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(async () => {
    corrRun('before');
    let hit = false;
    for (let k = 0; k < 30 && !hit; k++) { for (let i = 0; i < 8; i++) scene.render(); hit = CORR.hit; }
    return { avoidOff: AVOID.on === false, hit, acting: PERSON.acting };
  });

  expect(r.avoidOff).toBe(true);
  expect(r.hit).toBe(true);
  expect(r.acting).toBe('down');            // 맞고 쓰러진다
});

test('corrRun("after") — 회피 ON이면 hold/retract로 대응하고 고스트가 뜬다', async ({ page }) => {
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });

  const r = await page.evaluate(async () => {
    corrRun('after');
    const modes = new Set();
    let ghostSeen = false;
    for (let k = 0; k < 30; k++) {
      for (let i = 0; i < 8; i++) scene.render();
      modes.add(AVOID.mode);
      if (GHOST.line && GHOST.line.isEnabled()) ghostSeen = true;
    }
    return {
      avoidOn: AVOID.on === true,
      present: PRESENT.on === true,
      avoided: modes.has('hold') || modes.has('retract'),   // 회피 기동이 나왔다
      ghostSeen,
    };
  });

  expect(r.avoidOn).toBe(true);
  expect(r.present).toBe(true);
  expect(r.avoided).toBe(true);
  expect(r.ghostSeen).toBe(true);
});
