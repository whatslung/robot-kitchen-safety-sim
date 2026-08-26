import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let helpers = {};
try {
  helpers = require('../../tools/headless_gen/lib.cjs');
} catch (error) {
  if (error.code !== 'MODULE_NOT_FOUND') throw error;
}

test('sceneSeed는 장면마다 재현 가능한 uint32 seed를 만든다', () => {
  assert.equal(typeof helpers.sceneSeed, 'function');
  assert.equal(helpers.sceneSeed(20260826, 0), 20260826);
  assert.equal(helpers.sceneSeed(20260826, 3), 20260829);
  assert.equal(helpers.sceneSeed(0xffffffff, 1), 0);
});

test('seededRandom은 같은 seed에 같은 수열을 만들고 다른 seed를 구분한다', () => {
  assert.equal(typeof helpers.seededRandom, 'function');
  const sequence = seed => {
    const random = helpers.seededRandom(seed);
    return [random(), random(), random(), random()];
  };
  const first = sequence(17);
  assert.deepEqual(first, sequence(17));
  assert.notDeepEqual(first, sequence(18));
  assert.ok(first.every(value => value >= 0 && value < 1));
});

test('단색 의심 캡처는 어떤 파일도 쓰기 전에 거부한다', () => {
  assert.equal(typeof helpers.writeCaptureAtomic, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-reject-'));
  try {
    assert.throws(() => helpers.writeCaptureAtomic({
      outDir,
      base: 'cvN_0000',
      rgbDataUrl: 'data:image/png;base64,YWJj',
      labelText: '0 0.5 0.5 0.1 0.2\n',
      colors: 499,
      minColors: 500,
    }), /색상 수 499/);
    assert.equal(fs.existsSync(path.join(outDir, 'images', 'cvN_0000.png')), false);
    assert.equal(fs.existsSync(path.join(outDir, 'labels', 'cvN_0000.txt')), false);
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('정상 캡처는 원자적으로 저장하고 파일 hash를 반환한다', () => {
  assert.equal(typeof helpers.writeCaptureAtomic, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-write-'));
  try {
    const result = helpers.writeCaptureAtomic({
      outDir,
      base: 'cvS_0001',
      rgbDataUrl: 'data:image/png;base64,YWJj',
      labelText: '0 0.5 0.5 0.1 0.2\n',
      colors: 700,
      minColors: 500,
    });
    assert.equal(fs.readFileSync(path.join(outDir, result.image), 'utf8'), 'abc');
    assert.equal(fs.readFileSync(path.join(outDir, result.label), 'utf8'), '0 0.5 0.5 0.1 0.2\n');
    assert.equal(result.imageSha256, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    assert.match(result.labelSha256, /^[0-9a-f]{64}$/);
    assert.deepEqual(fs.readdirSync(path.join(outDir, 'images')), ['cvS_0001.png']);
    assert.deepEqual(fs.readdirSync(path.join(outDir, 'labels')), ['cvS_0001.txt']);
    assert.deepEqual(fs.readdirSync(outDir).sort(), ['images', 'labels'],
      '정상 완료 뒤 incomplete 표식이 남으면 안 된다');
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('기존 산출물이 있는 출력 폴더는 manifest를 덮어쓰기 전에 거부한다', () => {
  assert.equal(typeof helpers.assertEmptyOutputDir, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-existing-'));
  try {
    helpers.assertEmptyOutputDir(outDir);
    fs.writeFileSync(path.join(outDir, 'manifest.json'), '{"status":"complete"}\n');
    assert.throws(() => helpers.assertEmptyOutputDir(outDir), /비어 있지 않은 출력 폴더/);
    assert.equal(fs.readFileSync(path.join(outDir, 'manifest.json'), 'utf8'),
      '{"status":"complete"}\n', '이전 manifest를 보존해야 한다');
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('manifest는 임시 파일을 남기지 않고 JSON으로 교체한다', () => {
  assert.equal(typeof helpers.writeJsonAtomic, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-manifest-'));
  try {
    const file = path.join(outDir, 'manifest.json');
    helpers.writeJsonAtomic(file, { schemaVersion: 1, samples: [{ seed: 3 }] });
    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')),
      { schemaVersion: 1, samples: [{ seed: 3 }] });
    assert.deepEqual(fs.readdirSync(outDir), ['manifest.json']);
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('단색 캡처는 제한 횟수 안에서 재시도하고 정상 결과만 반환한다', async () => {
  assert.equal(typeof helpers.captureWithRetry, 'function');
  let calls = 0;
  const result = await helpers.captureWithRetry(async () => {
    calls++;
    return { colors: calls < 3 ? 120 : 720, value: calls };
  }, { attempts: 3, minColors: 500 });
  assert.equal(calls, 3);
  assert.deepEqual(result, { colors: 720, value: 3 });
});

test('단색 캡처를 다시 찍기 전에 복구 동작을 실행한다', async () => {
  const events = [];
  await helpers.captureWithRetry(async attempt => {
    events.push(`capture-${attempt}`);
    return { colors: attempt === 1 ? 20 : 700 };
  }, {
    attempts: 2,
    minColors: 500,
    onRetry: async attempt => events.push(`retry-${attempt}`),
  });
  assert.deepEqual(events, ['capture-1', 'retry-1', 'capture-2']);
});

test('모든 렌더 재시도가 단색이면 오류로 종료한다', async () => {
  assert.equal(typeof helpers.captureWithRetry, 'function');
  let calls = 0;
  await assert.rejects(() => helpers.captureWithRetry(async () => {
    calls++;
    return { colors: 20 };
  }, { attempts: 2, minColors: 500 }), /2회 재시도/);
  assert.equal(calls, 2);
});

test('입력 파일 fingerprint는 상대경로와 SHA-256을 정렬해 기록한다', () => {
  assert.equal(typeof helpers.fingerprintFiles, 'function');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-fingerprint-'));
  try {
    fs.mkdirSync(path.join(root, 'assets'));
    fs.writeFileSync(path.join(root, 'sim.html'), 'sim');
    fs.writeFileSync(path.join(root, 'assets', 'b.glb'), 'b');
    fs.writeFileSync(path.join(root, 'assets', 'a.glb'), 'a');
    assert.deepEqual(helpers.fingerprintFiles(root, [
      path.join(root, 'assets', 'b.glb'),
      path.join(root, 'sim.html'),
      path.join(root, 'assets', 'a.glb'),
    ]), [
      { path: 'assets/a.glb', sha256: 'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb' },
      { path: 'assets/b.glb', sha256: '3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d' },
      { path: 'sim.html', sha256: '507a9a8be3d145a86daa9644b28bf42a8dc0720d8baeabdf0406c393692bf082' },
    ]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
