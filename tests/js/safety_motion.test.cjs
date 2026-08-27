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
