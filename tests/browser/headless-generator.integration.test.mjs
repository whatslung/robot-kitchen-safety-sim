import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createRequire } from 'node:module';

const execFileAsync = promisify(execFile);
const require = createRequire(import.meta.url);
const repoRoot = path.resolve(import.meta.dirname, '..', '..');
const generator = path.join(repoRoot, 'tools', 'headless_gen', 'gen.cjs');
const generatorRequire = createRequire(generator);
const defaultChrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const chrome = process.env.CHROME_PATH || defaultChrome;
const enabled = process.env.RUN_HEADLESS_GEN_INTEGRATION === '1';
const { chromium } = generatorRequire('playwright-core');
const { startStaticServer } = require('../../tools/headless_gen/lib.cjs');

function rgbSignatureDifference(left, right) {
  const a = Buffer.from(left, 'hex');
  const b = Buffer.from(right, 'hex');
  assert.equal(a.length, 16 * 16 * 3);
  assert.equal(a.length, b.length);
  const differences = Array.from(a, (value, index) => Math.abs(value - b[index])).sort((x, y) => x - y);
  return {
    mean: differences.reduce((sum, value) => sum + value, 0) / differences.length,
    p95: differences[Math.floor(differences.length * 0.95)],
  };
}

function readLabels(root, capture) {
  const text = fs.readFileSync(path.join(root, ...capture.label.split('/')), 'utf8').trim();
  return text ? text.split(/\r?\n/).map(line => line.trim().split(/\s+/).map(Number)) : [];
}

function compareInstanceMetadata(leftCapture, rightCapture, context) {
  const ignored = new Set(['fire', 'smoke']);
  const left = new Map(leftCapture.meta.instances
    .filter(item => !ignored.has(item.class)).map(item => [item.instance, item]));
  const right = new Map(rightCapture.meta.instances
    .filter(item => !ignored.has(item.class)).map(item => [item.instance, item]));
  const classes = new Set([...left.values(), ...right.values()].map(item => item.class));
  for (const className of classes) {
    const leftCount = [...left.values()].filter(item => item.class === className).length;
    const rightCount = [...right.values()].filter(item => item.class === className).length;
    const countTolerance = className === 'person' ? 0 : 1;
    assert.ok(Math.abs(leftCount - rightCount) <= countTolerance,
      `${context} ${className} 인스턴스 수 차이가 ${countTolerance}개를 넘는다`);
  }
  for (const [instance, box] of left) {
    const matched = right.get(instance);
    if (!matched) {
      assert.notEqual(box.class, 'person', `${context} ${instance}가 한 실행에서 사라졌다`);
      continue;
    }
    const tolerance = box.class === 'person' ? 0.0075 : 0.03;
    for (const field of ['cx', 'cy', 'w', 'h']) {
      const delta = Math.abs(box[field] - matched[field]);
      assert.ok(delta <= tolerance,
        `${context} ${instance}.${field} 차이 ${delta.toFixed(6)} > ${tolerance}`);
    }
  }
}

test('같은 복합 seed를 두 번 생성하면 6카메라 장면과 비입자 라벨이 허용 오차 안에서 같다', {
  skip: !enabled || !fs.existsSync(chrome),
  timeout: 180_000,
}, async () => {
  const first = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-repro-a-'));
  const second = fs.mkdtempSync(path.join(os.tmpdir(), 'headless-gen-repro-b-'));
  try {
    const run = async outDir => {
      await execFileAsync(process.execPath, [generator, outDir, '2', '424242'], {
        cwd: repoRoot,
        env: { ...process.env, CHROME_PATH: chrome },
        timeout: 150_000,
        maxBuffer: 1024 * 1024,
      });
      return JSON.parse(fs.readFileSync(path.join(outDir, 'manifest.json'), 'utf8'));
    };
    const a = await run(first);
    const recorded = a.scenes[0].conditions;
    assert.deepEqual(recorded.identity, { staticSeed: 424242, sceneSeed: 424242 });
    assert.ok(Array.isArray(recorded.robot.joints) && recorded.robot.joints.length > 0,
      '무작위 로봇 관절값을 기록해야 한다');
    assert.ok(Array.isArray(recorded.animations), '활성 애니메이션 상태를 기록해야 한다');
    assert.equal(typeof recorded.scene.environment.upperWallColor, 'string');
    assert.equal(typeof recorded.scene.environment.floorTint, 'string');
    assert.equal(typeof recorded.scene.environment.lighting.keyIntensity, 'number');
    assert.equal(typeof recorded.scene.environment.materials.steelRoughness, 'number');
    assert.deepEqual({
      enabled: recorded.sensor.enabled,
      lowResolution: recorded.sensor.lowResolution,
      grain: recorded.sensor.grain,
      distortion: recorded.sensor.distortion,
      chroma: recorded.sensor.chroma,
      blur: recorded.sensor.blur,
      vignette: recorded.sensor.vignette,
      exposureJitter: recorded.sensor.exposureJitter,
    }, {
      enabled: false,
      lowResolution: 0,
      grain: 0,
      distortion: 0,
      chroma: 0,
      blur: 0,
      vignette: 0,
      exposureJitter: 0,
    },
    '원본 학습 이미지에는 RGB와 GT 좌표계를 다르게 만드는 센서 후처리를 적용하면 안 된다');
    const b = await run(second);
    for (const scene of a.scenes) {
      for (const capture of scene.captures) {
        assert.equal(capture.meta.scene.seed, scene.seed,
          `${capture.camera} GT 메타에는 현재 장면 seed가 기록돼야 한다`);
      }
    }
    const personClass = a.readiness.classContract.find(item => item.key === 'person').id;
    assert.equal(typeof personClass, 'number');
    const particleClasses = new Set(a.readiness.classContract
      .filter(item => item.key === 'fire' || item.key === 'smoke').map(item => item.id));
    assert.equal(a.scenes.length, 2);
    assert.equal(b.scenes.length, 2);
    for (let sceneIndex = 0; sceneIndex < a.scenes.length; sceneIndex += 1) {
      const leftScene = a.scenes[sceneIndex];
      const rightScene = b.scenes[sceneIndex];
      assert.deepEqual(leftScene.conditions, rightScene.conditions,
        `${sceneIndex}번 장면의 기록된 조건부터 같아야 한다`);
      assert.equal(leftScene.captures.length, 6);
      assert.equal(rightScene.captures.length, 6);
      for (let index = 0; index < leftScene.captures.length; index += 1) {
        const left = leftScene.captures[index];
        const right = rightScene.captures[index];
        assert.equal(left.camera, right.camera);
        assert.equal(left.visualSignatureAlgorithm, 'rgb16x16-v1');
        const visualDifference = rgbSignatureDifference(left.visualSignature, right.visualSignature);
        assert.ok(visualDifference.mean <= 8 && visualDifference.p95 <= 32,
          `${sceneIndex}/${left.camera} RGB 차이가 너무 크다: mean=${visualDifference.mean.toFixed(2)}, p95=${visualDifference.p95}`);
        const leftLabels = readLabels(first, left)
          .filter(row => !particleClasses.has(row[0]));
        const rightLabels = readLabels(second, right)
          .filter(row => !particleClasses.has(row[0]));
        assert.equal(leftLabels.length,
          left.meta.instances.filter(item => item.class !== 'fire' && item.class !== 'smoke').length);
        assert.equal(rightLabels.length,
          right.meta.instances.filter(item => item.class !== 'fire' && item.class !== 'smoke').length);
        compareInstanceMetadata(left, right, `${sceneIndex}/${left.camera}`);
      }
    }
  } finally {
    fs.rmSync(first, { recursive: true, force: true });
    fs.rmSync(second, { recursive: true, force: true });
  }
});

test('추가 인원을 반복해서 만들고 제거해도 Babylon 자원이 누적되지 않는다', {
  skip: !enabled || !fs.existsSync(chrome),
  timeout: 120_000,
}, async () => {
  const server = await startStaticServer(repoRoot);
  let browser;
  try {
    browser = await chromium.launch({ executablePath: chrome, headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.addInitScript(() => {
      let state = 424242;
      Math.random = () => {
        state = (state + 0x6D2B79F5) | 0;
        let value = Math.imul(state ^ (state >>> 15), 1 | state);
        value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
      };
    });
    await page.goto(`${server.baseUrl}/sim.html?seed=424242`, { waitUntil: 'load', timeout: 60_000 });
    await page.waitForFunction(() => window.__simReady === true && ENV_READY === true,
      null, { timeout: 90_000 });
    await page.evaluate(() => engine.stopRenderLoop());
    const result = await page.evaluate(async () => {
      await setExtraCount(0);
      const baseline = {
        animationGroups: scene.animationGroups.length,
        skeletons: scene.skeletons.length,
        extraRoots: scene.transformNodes.filter(node => /^extraRoot/.test(node.name)).length,
      };
      const snapshots = [];
      for (const count of [2, 1, 0, 2, 0]) {
        await setExtraCount(count);
        snapshots.push({
          count,
          animationGroups: scene.animationGroups.length,
          skeletons: scene.skeletons.length,
          extraRoots: scene.transformNodes.filter(node => /^extraRoot/.test(node.name)).length,
          expectedAnimationGroups: baseline.animationGroups
            + EXTRAS.reduce((sum, person) => sum + new Set(person.animationGroups).size, 0),
          expectedSkeletons: baseline.skeletons + count,
          expectedExtraRoots: baseline.extraRoots + count,
        });
      }
      await groundTruth('cvSE', { noDepth: true });
      const materialBaseline = scene.materials.length;
      const smokeMaterialBaseline = scene.materials.filter(material => material.name === 'inst_smoke_0').length;
      await groundTruth('cvSE', { noDepth: true });
      await groundTruth('cvSE', { noDepth: true });
      return {
        baseline,
        snapshots,
        materialBaseline,
        materialFinal: scene.materials.length,
        smokeMaterialBaseline,
        smokeMaterialFinal: scene.materials.filter(material => material.name === 'inst_smoke_0').length,
      };
    });
    for (const snapshot of result.snapshots) {
      assert.equal(snapshot.skeletons, snapshot.expectedSkeletons,
        `${snapshot.count}명 상태에서 제거된 골격이 남으면 안 된다`);
      assert.equal(snapshot.animationGroups, snapshot.expectedAnimationGroups,
        `${snapshot.count}명 상태에서 제거된 애니메이션이 남으면 안 된다`);
      assert.equal(snapshot.extraRoots, snapshot.expectedExtraRoots,
        `${snapshot.count}명 상태에서 제거된 최상위 노드가 남으면 안 된다`);
    }
    assert.equal(result.materialFinal, result.materialBaseline,
      '반복 GT 캡처가 임시 인스턴스 재질을 남기면 안 된다');
    assert.equal(result.smokeMaterialFinal, result.smokeMaterialBaseline,
      '반복 GT 캡처가 inst_smoke_0 재질을 누적하면 안 된다');
  } finally {
    if (browser) await browser.close();
    await server.close();
  }
});
