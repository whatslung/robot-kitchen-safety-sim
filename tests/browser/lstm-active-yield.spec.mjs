import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:5199';

test('fixed-route LSTM demo moves first, actively yields, and finishes safely', async ({ page }) => {
  test.setTimeout(150000);
  const consoleErrors = [];
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', error => consoleErrors.push(error.message));

  await page.route('**/health', route => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }),
  }));
  await page.route('**/predict', async route => {
    const request = route.request();
    const body = request.postDataJSON();
    if (body.hist) {
      const last = body.hist[body.hist.length - 1];
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ modes: [{
          path: Array.from({ length: 12 }, (_, i) => [last[0] - 0.2 * (i + 1), last[1]]),
          w: 1, sigma: Array(12).fill(0.40),
        }] }),
      });
      return;
    }
    const tracks = body.tracks.map(track => {
      const hist = track.hist, last = hist[hist.length - 1];
      const previous = hist[Math.max(0, hist.length - 2)];
      let vx = (last[0] - previous[0]) / 0.4;
      let vz = (last[1] - previous[1]) / 0.4;
      const clear = track.id === 'gt:0' && Math.abs(last[1] - 1.015) > 0.30;
      if (track.id === 'gt:0') {
        vx = clear ? 0 : Math.min(vx, -0.50);
        vz = clear ? (last[1] >= body.robot.z ? 0.50 : -0.50) : 0;
      }
      const path = [], sigma = [];
      for (let i = 0; i < 12; i++) {
        const t = 0.4 * (i + 1);
        path.push([last[0] + vx * t, last[1] + vz * t]);
        sigma.push(0.40);
      }
      return { id: track.id, modes: [{ path, w: 1, sigma }], risk: {
        tEntryStop: clear ? null : 1.2, tEntrySlow: clear ? null : 0.6,
        riskMass: clear ? 0 : 1, dMin: clear ? 3 : 0.2,
      } };
    });
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ tracks, worst: tracks.length ? {
        id: tracks[0].id, tEntryStop: 1.2, tEntrySlow: 0.6, riskMass: 1, dMin: 0.2,
      } : null }),
    });
  });

  await page.goto(`${BASE_URL}/sim.html?person=1&scenario=none`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });
  await page.waitForFunction(() => window.__sim && window.__sim.LSTM_YIELD_DEMO,
    null, { timeout: 20000 });
  await page.waitForFunction(() => PHYS.ready, null, { timeout: 30000 });
  await page.evaluate(() => {
    QUALITY.apply('low');
    SEQ.forEach(step => { step.ms = Math.min(step.ms, 350); });
    engine.stopRenderLoop();
  });
  await page.evaluate(() => document.querySelector('#lstmYieldBtn').click());
  await page.evaluate(() => {
    stepUpdate(16);
    lstmYieldDemoUpdate(performance.now(), 16);
  });

  await page.waitForTimeout(450);
  await page.evaluate(() => lstmYieldDemoUpdate(performance.now(), 16));
  await page.waitForFunction(() => window.__sim.LSTM_YIELD_DEMO.personStartMs !== null
      || window.__sim.LSTM_YIELD_DEMO.phase === 'failed',
    null, { timeout: 5000 });
  const lead = await page.evaluate(() => ({
    active: window.__sim.LSTM_YIELD_DEMO.active,
    phase: window.__sim.LSTM_YIELD_DEMO.phase,
    failure: window.__sim.LSTM_YIELD_DEMO.failure,
    status: window.__sim.state.msg,
    robotFirstMotionMs: window.__sim.LSTM_YIELD_DEMO.robotFirstMotionMs,
    robotMotionDelta: window.__sim.LSTM_YIELD_DEMO.saved ? Math.max(...JOINTS.map(
      (joint, index) => Math.abs(
        joint.value-window.__sim.LSTM_YIELD_DEMO.saved.startJoints[index]))) : null,
    personStartMs: window.__sim.LSTM_YIELD_DEMO.personStartMs,
    scheduledPersonDelayMs: window.__sim.LSTM_YIELD_DEMO.personStartDueAt
      - window.__sim.LSTM_YIELD_DEMO.startedAt,
    blocked: (() => {
      const route = lstmYieldDemoRoute();
      for (const point of demoRouteSamples(route)) {
        const hits = COLL_FOOTPRINTS.concat(COLL_AUTO, COLL_PUSH).filter(f =>
          Math.abs(point.x-f[0]) < f[2]+PERSON_R && Math.abs(point.z-f[1]) < f[3]+PERSON_R);
        if (hits.length) return { point:{x:point.x,z:point.z}, hits };
      }
      return null;
    })(),
  }));
  expect(lead.active, JSON.stringify(lead)).toBe(true);
  expect(lead.robotMotionDelta).toBeGreaterThan(1e-4);
  expect(lead.robotFirstMotionMs).toBeLessThanOrEqual(200);
  expect(lead.scheduledPersonDelayMs).toBe(400);
  expect(lead.personStartMs).toBeGreaterThanOrEqual(390);

  await page.evaluate(() => {
    AVOID.samples = 4;
    window.__demoTestDriver = setInterval(() => {
      const now = performance.now();
      const dt = LSTM_YIELD_DEMO.phase === 'clearing' ? 100 : 20;
      stepUpdate(dt);
      updateCooking(dt);
      personUpdate(dt);
      armContactUpdate(dt);
      safetyUpdate();
      mpredObserve(now);
      mpredTick(now, dt);
      lstmYieldDemoUpdate(now, dt);
    }, 20);
  });

  try {
    await page.waitForFunction(() => {
      const demo = window.__sim.LSTM_YIELD_DEMO;
      return demo.phase === 'complete' || demo.phase === 'failed';
    }, null, { timeout: 90000 });
  } catch (error) {
    const snapshot = await page.evaluate(() => ({
      demo: (() => { const d = window.__sim.LSTM_YIELD_DEMO; return {
        active:d.active, phase:d.phase, failure:d.failure, observationCount:d.observationCount,
        predictionUsed:d.predictionUsed, maneuvers:d.maneuvers,
        enteredStopRadius:d.enteredStopRadius, escapeMotionInsideStop:d.escapeMotionInsideStop,
        escapeJointDelta:d.escapeJointDelta, minDistance:d.minDistance,
        maxAvoidStationaryMs:d.maxAvoidStationaryMs, stationarySnapshot:d.stationarySnapshot,
      }; })(),
      state: { auto:window.__sim.state.auto, seqIdx:window.__sim.state.seqIdx,
        seqT:window.__sim.state.seqT, msg:window.__sim.state.msg,
        basketHeld:window.__sim.state.basketHeld },
      avoid: { mode:AVOID.mode, reason:AVOID.reason, escapeVerified:AVOID.escapeVerified,
        targetFactor:AVOID.targetFactor, appliedFactor:AVOID.appliedFactor,
        predictionReady:AVOID.predictionReady },
      mpred: { ids:[...window.__sim.MPRED.pred.keys()], at:window.__sim.MPRED.at,
        err:window.__sim.MPRED.err, lastErr:window.__sim.MPRED.lastErr,
        busy:window.__sim.MPRED.busy, hist:[...window.__sim.MPRED.hist.keys()],
        record:window.__sim.MPRED.pred.get('gt:0'),
        learned:window.__sim.learnedPeopleOccupancy([0, 0.4, 0.8], performance.now()) },
    }));
    throw new Error(`demo did not finish: ${JSON.stringify(snapshot)}`, { cause:error });
  }

  const result = await page.evaluate(() => {
    clearInterval(window.__demoTestDriver);
    const demo = window.__sim.LSTM_YIELD_DEMO;
    return {
      phase: demo.phase,
      failure: demo.failure,
      predictionUsed: demo.predictionUsed,
      observationCount: demo.observationCount,
      maneuvers: demo.maneuvers,
      maxAvoidStationaryMs: demo.maxAvoidStationaryMs,
      stationarySnapshot: demo.stationarySnapshot,
      resumeDelayMs: demo.resumeDelayMs,
      enteredStopRadius: demo.enteredStopRadius,
      escapeMotionInsideStop: demo.escapeMotionInsideStop,
      escapeJointDelta: demo.escapeJointDelta,
      minDistance: demo.minDistance,
      finalDistance: Math.hypot(person.node.position.x-LAYOUT.robot.base.x,
        person.node.position.z-LAYOUT.robot.base.z),
      nominalSlowRadius: SAFE.NOM_SLOW,
      contact: demo.contact,
      basketDelivered: demo.basketDelivered,
      homeReached: demo.homeReached,
    };
  });
  expect(result.phase, JSON.stringify(result)).toBe('complete');
  expect(result.predictionUsed).toBe(true);
  expect(result.observationCount).toBe(8);
  expect(result.maneuvers.some(mode => mode === 'RETRACT' || mode === 'SAFE_LIFT'),
    JSON.stringify(result)).toBe(true);
  expect(result.maxAvoidStationaryMs, JSON.stringify(result)).toBeLessThanOrEqual(800);
  expect(result.enteredStopRadius, JSON.stringify(result)).toBe(true);
  expect(result.minDistance, JSON.stringify(result)).toBeLessThan(3.0);
  expect(result.escapeMotionInsideStop, JSON.stringify(result)).toBe(true);
  expect(result.escapeJointDelta, JSON.stringify(result)).toBeGreaterThan(1e-5);
  expect(result.finalDistance, JSON.stringify(result)).toBeGreaterThan(result.nominalSlowRadius);
  expect(result.resumeDelayMs).toBeGreaterThanOrEqual(300);
  // 300 ms policy hysteresis plus rendering/timer scheduling must still resume promptly.
  expect(result.resumeDelayMs).toBeLessThanOrEqual(600);
  expect(result.contact).toBe(false);
  expect(result.basketDelivered).toBe(true);
  expect(result.homeReached).toBe(true);

  const cancelled = await page.evaluate(() => {
    const before = {
      person:[person.node.position.x, person.node.position.y, person.node.position.z],
      basket:[basketNode.position.x, basketNode.position.y, basketNode.position.z],
      basketHeld:state.basketHeld,
    };
    const route = lstmYieldDemoRoute();
    const allColliders = COLL_FOOTPRINTS.concat(COLL_AUTO, COLL_PUSH);
    const routeBlocker = demoRouteSamples(route).map(point => ({ point, hits:allColliders
      .map((box, index) => ({ box, index }))
      .filter(item => Math.abs(point.x-item.box[0]) < item.box[2]+PERSON_R
        && Math.abs(point.z-item.box[1]) < item.box[3]+PERSON_R) }))
      .find(item => item.hits.length) || null;
    const started = startLstmYieldDemo();
    const startStatus = state.msg;
    document.querySelector('#stopBtn').click();
    const stopped = {
      started, startStatus, routeBlocker, active:LSTM_YIELD_DEMO.active,
      saved:LSTM_YIELD_DEMO.saved,
      blockers:LSTM_YIELD_DEMO.blockers.length, mode:AVOID.mode, lift:AVOID.lift,
      person:[person.node.position.x, person.node.position.y, person.node.position.z],
      basket:[basketNode.position.x, basketNode.position.y, basketNode.position.z],
      basketHeld:state.basketHeld,
    };
    const restarted = startLstmYieldDemo();
    document.querySelector('#homeBtn').click();
    return { before, stopped, home:{ restarted, active:LSTM_YIELD_DEMO.active,
      saved:LSTM_YIELD_DEMO.saved, blockers:LSTM_YIELD_DEMO.blockers.length,
      mode:AVOID.mode, lift:AVOID.lift } };
  });
  expect(cancelled.stopped.started, JSON.stringify(cancelled)).toBe(true);
  expect(cancelled.stopped.active).toBe(false);
  expect(cancelled.stopped.saved).toBe(null);
  expect(cancelled.stopped.blockers).toBe(0);
  expect(cancelled.stopped.mode).toBe('PROCEED');
  expect(cancelled.stopped.lift).toBe(null);
  expect(cancelled.stopped.person).toEqual(cancelled.before.person);
  expect(cancelled.stopped.basket).toEqual(cancelled.before.basket);
  expect(cancelled.stopped.basketHeld).toBe(cancelled.before.basketHeld);
  expect(cancelled.home.restarted).toBe(true);
  expect(cancelled.home.active).toBe(false);
  expect(cancelled.home.saved).toBe(null);
  expect(cancelled.home.blockers).toBe(0);
  expect(cancelled.home.mode).toBe('PROCEED');
  expect(cancelled.home.lift).toBe(null);

  // The repository intentionally omits some optional GLBs; their loader 404s are unrelated
  // to this scenario. Keep all other console and uncaught runtime errors fatal.
  const unexpectedErrors = consoleErrors.filter(message =>
    !message.startsWith('Failed to load resource: the server responded with a status of 404')
    && !message.startsWith('[GLB] load failed:'));
  expect(unexpectedErrors).toEqual([]);
});
