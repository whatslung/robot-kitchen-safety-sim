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
  assert.equal(S.chooseManeuver({ ...base, danger: true, beforeCross: true }).mode, 'HOLD');
  assert.equal(S.chooseManeuver({ ...base, danger: true, beforeCross: false }).mode, 'RETRACT');
  assert.equal(S.chooseManeuver({
    ...base, danger: true, beforeCross: false, retract: blocked(-0.1),
  }).mode, 'SAFE_LIFT');
  assert.equal(S.chooseManeuver({
    ...base, danger: true, beforeCross: false,
    retract: blocked(-0.1), safeLift: blocked(-0.2),
  }).mode, 'STOP');
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

test('chooseManeuver fails closed on invalid candidates', () => {
  assert.equal(S.chooseManeuver({ danger: false }).mode, 'STOP');
});
