import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:5199';

test('route-blocking environment cart is parked and its physics state is restored', async ({ page }) => {
  test.setTimeout(90000);
  await page.goto(`${BASE_URL}/sim.html?person=0&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });
  await page.waitForFunction(() => PHYS.ready && PUSH.defs.some(item =>
    ['env_pancart.glb', 'env_basketcart.glb'].includes(ENV_PROPS[item.idx]?.f)),
    null, { timeout: 60000 });
  await page.evaluate(() => buildPushables());
  await page.waitForFunction(() => PUSH.active && PUSH.bodies.length, null, { timeout: 10000 });
  const result = await page.evaluate(() => {
    engine.stopRenderLoop();
    const cartFiles = new Set(['env_pancart.glb', 'env_basketcart.glb']);
    const body = PUSH.bodies.find(item => cartFiles.has(ENV_PROPS[item.idx]?.f));
    if (!body) return { error:'environment cart unavailable' };
    const route = lstmYieldDemoRoute();
    const crossingCenter = V3.Lerp(route.fixedBend, route.fixedCross, 0.5);
    const cartY = body.box.position.y;
    body.box.position.copyFrom(crossingCenter);
    body.box.position.y = cartY;
    body.box.computeWorldMatrix(true);
    body.ag.body.disablePreStep = false;
    body.restore = 2; body.restorePreStep = true;
    pushUpdate(16); pushUpdate(16);
    body.ag.body.setLinearVelocity(new V3(0.12, 0, -0.04));
    body.ag.body.setAngularVelocity(new V3(0, 0.08, 0));
    const before = {
      position:body.box.position.clone(),
      rotation:body.box.rotation.clone(),
      quaternion:body.box.rotationQuaternion ? body.box.rotationQuaternion.clone() : null,
      linear:body.ag.body.getLinearVelocity().clone(),
      angular:body.ag.body.getAngularVelocity().clone(),
      disablePreStep:body.ag.body.disablePreStep,
    };
    buildPersonColliders(); buildNavGrid();
    LSTM_YIELD_DEMO.route = route;
    const intendedRoute = { start:route.gate, gate:route.fixedBend,
      bend:route.fixedCross, cross:route.fixedExit, exit:route.fixedExit };
    const blockedBefore = demoRouteBlocked(intendedRoute);
    const crossingPath = prepareLstmDemoCrossingPath(route.gate.clone());
    const removed = LSTM_YIELD_DEMO.blockers.some(item => item.kind === 'cart'
      && item.body === body);
    const parked = body.box.position.y < -2;
    restoreDemoRouteBlockers();
    pushUpdate(16); pushUpdate(16);
    const delta = (a, b) => Math.hypot(a.x-b.x, a.y-b.y, a.z-b.z);
    return {
      blockedBefore, pathReady:Array.isArray(crossingPath) && crossingPath.length > 1,
      removed, parked, blockers:LSTM_YIELD_DEMO.blockers.length,
      positionDelta:delta(body.box.position, before.position),
      rotationDelta:delta(body.box.rotation, before.rotation),
      quaternionDelta:before.quaternion && body.box.rotationQuaternion
        ? Math.hypot(body.box.rotationQuaternion.x-before.quaternion.x,
          body.box.rotationQuaternion.y-before.quaternion.y,
          body.box.rotationQuaternion.z-before.quaternion.z,
          body.box.rotationQuaternion.w-before.quaternion.w) : 0,
      linearDelta:delta(body.ag.body.getLinearVelocity(), before.linear),
      angularDelta:delta(body.ag.body.getAngularVelocity(), before.angular),
      disablePreStep:body.ag.body.disablePreStep,
      expectedDisablePreStep:before.disablePreStep,
    };
  });
  expect(result.error, JSON.stringify(result)).toBeUndefined();
  expect(result.blockedBefore, JSON.stringify(result)).toBe(true);
  expect(result.pathReady, JSON.stringify(result)).toBe(true);
  expect(result.removed, JSON.stringify(result)).toBe(true);
  expect(result.parked, JSON.stringify(result)).toBe(true);
  expect(result.blockers).toBe(0);
  expect(result.positionDelta).toBeLessThan(1e-5);
  expect(result.rotationDelta).toBeLessThan(1e-5);
  expect(result.quaternionDelta).toBeLessThan(1e-5);
  expect(result.linearDelta).toBeLessThan(1e-5);
  expect(result.angularDelta).toBeLessThan(1e-5);
  expect(result.disablePreStep).toBe(result.expectedDisablePreStep);
});
