import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:8001';

test('A-to-B collision sends the walking person into the arm physics response', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true && PHYS.ready, null, { timeout: 60000 });
  await page.evaluate(() => {
    QUALITY.apply('low');
    SEQ.forEach(step => { step.ms = Math.min(step.ms, 300); });
    engine.stopRenderLoop();
    document.querySelector('#collisionScenarioBtn').click();
    window.__collisionDriver = setInterval(() => {
      const dt = 20;
      stepUpdate(dt); updateCooking(dt); personUpdate(dt); extrasUpdate(dt);
      pushUpdate(dt); armContactUpdate(dt); safetyUpdate();
    }, 20);
  });
  await page.waitForFunction(() => PHYS.estop || (person.armStruckT || 0) > 0,
    null, { timeout: 40000 });
  const result = await page.evaluate(() => {
    clearInterval(window.__collisionDriver);
    return {
      estop: PHYS.estop,
      struck: person.armStruckT,
      mode: person.mode,
      blind: SAFE.blind,
    };
  });
  expect(result.estop || result.struck > 0, JSON.stringify(result)).toBe(true);
  expect(result.mode, JSON.stringify(result)).toBe('idle');
  expect(result.blind).toBe(true);
});
