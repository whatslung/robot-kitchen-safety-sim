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

test('장면은 임시 디렉터리에서 완성한 뒤 한 번에 공개한다', () => {
  assert.equal(typeof helpers.beginSceneTransaction, 'function');
  assert.equal(typeof helpers.publishSceneTransaction, 'function');
  assert.equal(typeof helpers.finishSceneTransaction, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-scene-'));
  try {
    const transaction = helpers.beginSceneTransaction(outDir, '0007');
    helpers.writeCaptureAtomic({
      outDir: transaction.stageDir,
      base: 'cvN_0007',
      rgbDataUrl: 'data:image/png;base64,YWJj',
      labelText: '0 0.5 0.5 0.1 0.2\n',
      colors: 700,
      minColors: 500,
    });
    assert.equal(fs.existsSync(path.join(outDir, 'scenes', '0007')), false,
      '장면 완성 전에는 최종 경로에 보이면 안 된다');
    const prefix = helpers.publishSceneTransaction(transaction);
    assert.equal(prefix, 'scenes/0007');
    assert.equal(fs.existsSync(path.join(outDir, 'scenes', '0007', 'images', 'cvN_0007.png')), true);
    assert.equal(fs.existsSync(transaction.markerFile), true,
      'manifest 확정 전까지 중단 표식이 남아야 한다');
    helpers.finishSceneTransaction(transaction);
    assert.equal(fs.existsSync(transaction.markerFile), false);
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('실패한 장면 rollback은 임시·공개 파일과 중단 표식을 모두 제거한다', () => {
  assert.equal(typeof helpers.rollbackSceneTransaction, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-scene-fail-'));
  try {
    const transaction = helpers.beginSceneTransaction(outDir, '0002');
    fs.writeFileSync(path.join(transaction.stageDir, 'partial.txt'), 'partial');
    helpers.publishSceneTransaction(transaction);
    helpers.rollbackSceneTransaction(transaction);
    assert.equal(fs.existsSync(transaction.stageDir), false);
    assert.equal(fs.existsSync(transaction.finalDir), false);
    assert.equal(fs.existsSync(transaction.markerFile), false);
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('실제 장면 절차는 중간 카메라 실패 시 manifest와 모든 장면 파일을 되돌린다', async () => {
  assert.equal(typeof helpers.runSceneTransaction, 'function');
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-scene-flow-fail-'));
  const manifest = { scenes: [] };
  try {
    await assert.rejects(() => helpers.runSceneTransaction({
      outDir,
      sceneName: '0003',
      build: async stageDir => {
        for (const camera of ['cvNW', 'cvNE', 'cvSE']) {
          if (camera === 'cvSE') throw new Error('third camera failed');
          fs.mkdirSync(path.join(stageDir, 'images'), { recursive: true });
          fs.writeFileSync(path.join(stageDir, 'images', `${camera}.png`), camera);
        }
        return { index: 3 };
      },
      commit: record => { manifest.scenes.push(record); },
    }), /third camera failed/);
    assert.deepEqual(manifest.scenes, []);
    assert.equal(fs.existsSync(path.join(outDir, 'scenes', '0003')), false);
    assert.equal(fs.existsSync(path.join(outDir, '.incomplete-scene-0003')), false);
    assert.equal(fs.existsSync(path.join(outDir, '.incomplete-scene-0003.json')), false);
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

test('필수 렌더 입력이 빠지면 조용히 제외하지 않고 파일명을 알려준다', () => {
  assert.equal(typeof helpers.collectInputFiles, 'function');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-inputs-'));
  const requiredExceptWasm = [
    'sim.html', 'gtboxes.js', 'babylon.js', 'babylonjs.loaders.min.js',
    'HavokPhysics_umd.js', 'character-manifest.json',
  ];
  try {
    for (const file of requiredExceptWasm) fs.writeFileSync(path.join(root, file), file);
    fs.mkdirSync(path.join(root, 'assets'));
    assert.throws(() => helpers.collectInputFiles(root), /HavokPhysics\.wasm\.js/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('내장 정적 서버는 지정한 저장소 파일을 query와 무관하게 그대로 제공한다', async () => {
  assert.equal(typeof helpers.startStaticServer, 'function');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-server-'));
  fs.writeFileSync(path.join(root, 'sim.html'), '<!doctype html><p>repo copy</p>');
  const server = await helpers.startStaticServer(root);
  try {
    const response = await fetch(`${server.baseUrl}/sim.html?v=verified`);
    assert.equal(response.status, 200);
    assert.equal(await response.text(), '<!doctype html><p>repo copy</p>');
    const missing = await fetch(`${server.baseUrl}/missing.js`);
    assert.equal(missing.status, 404);
    const traversal = await fetch(`${server.baseUrl}/%2e%2e%2foutside.txt`);
    assert.equal(traversal.status, 403);
    const nul = await fetch(`${server.baseUrl}/bad%00name`);
    assert.equal(nul.status, 400);
  } finally {
    await server.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
