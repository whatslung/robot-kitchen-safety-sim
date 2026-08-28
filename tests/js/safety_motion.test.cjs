const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../safety_motion.js');

test('samplePlannedPath follows remaining waypoints at walking speed', () => {
  const out = S.samplePlannedPath(
    { x: 0, y: 0, z: 0 },
    [{ x: 1, y: 0, z: 0 }, { x: 1, y: 0, z: 1 }],
    0,
    1,
    [0, 0.5, 1, 1.5, 2],
  );
  assert.deepEqual(out.map(p => [p.x, p.z]), [
    [0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1],
  ]);
});

test('samplePlannedPath treats invalid input as stationary', () => {
  const out = S.samplePlannedPath(
    { x: 2, y: 0, z: 3 },
    null,
    0,
    Number.NaN,
    [0, 1],
  );
  assert.deepEqual(out.map(p => [p.x, p.z]), [[2, 3], [2, 3]]);
});

test('sweptSegmentCapsuleContact catches contact between rendered frames', () => {
  const result = S.sweptSegmentCapsuleContact({
    previous: {
      a: { x: -1, y: 1, z: 0 },
      b: { x: -0.5, y: 1, z: 0 },
    },
    current: {
      a: { x: 0.5, y: 1, z: 0 },
      b: { x: 1, y: 1, z: 0 },
    },
    linkRadius: 0.05,
    capsule: {
      center: { x: 0, y: 1, z: 0 },
      radius: 0.2,
      halfHeight: 0.8,
    },
    maxStep: 0.04,
  });

  assert.equal(result.hit, true);
  assert.equal(result.invalid, false);
  assert.ok(result.time > 0 && result.time < 1);
});

test('sweptSegmentCapsuleContact handles an oblique 3D arm link', () => {
  const segment = {
    a: { x: -0.5, y: 0.4, z: -0.5 },
    b: { x: 0.5, y: 1.6, z: 0.5 },
  };
  const result = S.sweptSegmentCapsuleContact({
    previous: segment,
    current: segment,
    linkRadius: 0.04,
    capsule: {
      center: { x: 0, y: 1, z: 0 },
      radius: 0.2,
      halfHeight: 0.85,
    },
  });

  assert.equal(result.hit, true);
  assert.ok(result.clearance <= 0);
});

test('sweptSegmentCapsuleContact ignores horizontal overlap at a different height', () => {
  const segment = {
    a: { x: -0.5, y: 3, z: 0 },
    b: { x: 0.5, y: 3, z: 0 },
  };
  const result = S.sweptSegmentCapsuleContact({
    previous: segment,
    current: segment,
    linkRadius: 0.05,
    capsule: {
      center: { x: 0, y: 1, z: 0 },
      radius: 0.2,
      halfHeight: 0.8,
    },
  });

  assert.equal(result.hit, false);
  assert.ok(result.clearance > 0);
});

test('sweptSegmentCapsuleContact is invariant to horizontal approach angle', () => {
  const capsule = {
    center: { x: 0, y: 1, z: 0 }, radius: 0.2, halfHeight: 0.8,
  };
  const xSweep = S.sweptSegmentCapsuleContact({
    previous: { a:{x:-1,y:1,z:0}, b:{x:-0.5,y:1,z:0} },
    current: { a:{x:0.5,y:1,z:0}, b:{x:1,y:1,z:0} },
    linkRadius: 0.05, capsule, maxStep: 0.04,
  });
  const zSweep = S.sweptSegmentCapsuleContact({
    previous: { a:{x:0,y:1,z:-1}, b:{x:0,y:1,z:-0.5} },
    current: { a:{x:0,y:1,z:0.5}, b:{x:0,y:1,z:1} },
    linkRadius: 0.05, capsule, maxStep: 0.04,
  });

  assert.equal(xSweep.hit, zSweep.hit);
  assert.ok(Math.abs(xSweep.clearance - zSweep.clearance) < 1e-12);
});

test('sweptSegmentCapsuleContact fails closed for malformed geometry', () => {
  const result = S.sweptSegmentCapsuleContact({
    previous: null,
    current: null,
    linkRadius: 0.04,
    capsule: null,
  });

  assert.equal(result.hit, true);
  assert.equal(result.invalid, true);
});

test('approachFactor limits braking and acceleration slopes', () => {
  assert.equal(S.approachFactor(1, 0, 0.1, 2.5, 0.8, false), 0.75);
  assert.ok(Math.abs(S.approachFactor(0.5, 1, 0.1, 2.5, 0.8, false) - 0.58) < 1e-12);
});

test('approachFactor applies an immediate safety stop and fails closed', () => {
  assert.equal(S.approachFactor(0.8, 0, 0.01, 2.5, 0.8, true), 0);
  assert.equal(S.approachFactor(Number.NaN, 1, 0.1, 2.5, 0.8, false), 0);
});

test('chooseManeuver uses the safety candidate priority', () => {
  const safe = clearance => ({ safe: true, clearance });
  const blocked = clearance => ({ safe: false, clearance });
  const base = {
    currentMode: 'PROCEED', holdMs: 1000, minHoldMs: 300,
    proceed: safe(0.3), retract: safe(0.25), safeLift: safe(0.2),
  };

  assert.equal(S.chooseManeuver({ ...base, danger: false }).mode, 'PROCEED');
  assert.equal(S.chooseManeuver({ ...base, danger: true, beforeCross: true }).mode, 'SAFE_LIFT');
  assert.equal(S.chooseManeuver({ ...base, danger: true, beforeCross: false }).mode, 'RETRACT');
  assert.equal(S.chooseManeuver({
    ...base, danger: true, beforeCross: false, retract: blocked(-0.1),
  }).mode, 'SAFE_LIFT');
  assert.equal(S.chooseManeuver({
    ...base, danger: true, beforeCross: false,
    retract: blocked(-0.1), safeLift: blocked(-0.2),
  }).mode, 'STOP');
});

test('chooseManeuver actively yields before crossing when a lift is verified safe', () => {
  const result = S.chooseManeuver({
    danger: true,
    beforeCross: true,
    currentMode: 'PROCEED',
    holdMs: 1000,
    safeLift: { safe: true, clearance: 0.42 },
  });

  assert.equal(result.mode, 'SAFE_LIFT');
  assert.equal(result.reason, 'safe-lift-before-cross');
});

test('chooseManeuver never overrides an emergency stop with active motion', () => {
  const result = S.chooseManeuver({
    emergencyStop: true,
    danger: true,
    beforeCross: false,
    currentMode: 'PROCEED',
    holdMs: 1000,
    retract: { safe: true, clearance: 0.4 },
    safeLift: { safe: true, clearance: 0.5 },
  });

  assert.equal(result.mode, 'STOP');
  assert.equal(result.reason, 'emergency-stop');
});

test('predictionFresh accepts only finite prediction timestamps younger than the limit', () => {
  assert.equal(S.predictionFresh({ at: 4001 }, 5000), true);
  assert.equal(S.predictionFresh({ at: 4000 }, 5000), false);
  assert.equal(S.predictionFresh({ at: Number.NaN }, 5000), false);
  assert.equal(S.predictionFresh(null, 5000), false);
});

test('chooseManeuver holds a conservative mode for at least 300ms', () => {
  const options = {
    danger: false,
    currentMode: 'STOP',
    minHoldMs: 300,
    proceed: { safe: true, clearance: 0.4 },
  };

  assert.equal(S.chooseManeuver({ ...options, holdMs: 299 }).mode, 'STOP');
  assert.equal(S.chooseManeuver({ ...options, holdMs: 300 }).mode, 'PROCEED');
});

test('chooseManeuver requires release margin to remain clear for 300ms', () => {
  const options = {
    danger: false,
    currentMode: 'HOLD',
    holdMs: 1000,
    minHoldMs: 300,
    releaseClearance: 0.3,
  };
  assert.equal(S.chooseManeuver({
    ...options, clearMs: 1000, proceed: { safe: true, clearance: 0.29 },
  }).mode, 'HOLD');
  assert.equal(S.chooseManeuver({
    ...options, clearMs: 299, proceed: { safe: true, clearance: 0.31 },
  }).mode, 'HOLD');
  assert.equal(S.chooseManeuver({
    ...options, clearMs: 300, proceed: { safe: true, clearance: 0.31 },
  }).mode, 'PROCEED');
});

test('chooseManeuver applies release hysteresis to stop-to-motion transitions', () => {
  const options = {
    danger: true,
    beforeCross: false,
    currentMode: 'STOP',
    holdMs: 1000,
    minHoldMs: 300,
    releaseClearance: 0.3,
    retract: { safe: true, clearance: 0.31 },
    safeLift: { safe: true, clearance: 0.4 },
  };
  assert.equal(S.chooseManeuver({
    ...options, clearMsByMode: { RETRACT: 299 },
  }).mode, 'STOP');
  assert.equal(S.chooseManeuver({
    ...options, clearMsByMode: { RETRACT: 300 },
  }).mode, 'RETRACT');
  assert.equal(S.chooseManeuver({
    ...options,
    retract: { safe: true, clearance: 0.29 },
    clearMsByMode: { RETRACT: 1000 },
  }).mode, 'STOP');
});

test('chooseManeuver fails closed on invalid candidates', () => {
  assert.equal(S.chooseManeuver({ danger: false }).mode, 'STOP');
});

test('chooseManeuver accepts infinite clearance when no people are present', () => {
  const result = S.chooseManeuver({
    danger: false,
    proceed: { safe: true, clearance: Number.POSITIVE_INFINITY },
  });
  assert.equal(result.mode, 'PROCEED');
});

test('trajectoryClearance scores a candidate against every planned person path', () => {
  const robot = [
    { x: 0, y: 1, z: 0 },
    { x: 1, y: 1, z: 0 },
  ];
  const people = [
    { points: [{ x: 0, y: 0, z: 2 }, { x: 1, y: 0, z: 0.5 }] },
    { points: [{ x: 3, y: 0, z: 3 }, { x: 3, y: 0, z: 3 }] },
  ];

  assert.equal(S.trajectoryClearance(robot, people, 0.1, 0.2), 0.2);
  assert.equal(S.trajectoryClearance([], people, 0.1, 0.2), Number.NEGATIVE_INFINITY);
});

test('armTrajectoryClearance scores every 3D link against planned capsules', () => {
  const armSamples = [[{
    a: { x: 0, y: 1, z: 0 },
    b: { x: 1, y: 1, z: 0 },
  }]];
  const people = [{
    radius: 0.2,
    halfHeight: 0.8,
    points: [{ x: 0.5, y: 0.9, z: 0.5 }],
  }];

  assert.ok(Math.abs(S.armTrajectoryClearance(armSamples, people, 0.1) - 0.2) < 1e-12);
});

test('armTrajectoryClearance honors a payload segment radius larger than the arm', () => {
  const armSamples = [[{
    a: { x: 0, y: 1, z: 0 },
    b: { x: 1, y: 1, z: 0 },
    radius: 0.4,
  }]];
  const people = [{
    radius: 0.2,
    halfHeight: 0.8,
    points: [{ x: 0.5, y: 0.9, z: 0.5 }],
  }];

  assert.ok(Math.abs(S.armTrajectoryClearance(armSamples, people, 0.1) + 0.1) < 1e-12);
});
