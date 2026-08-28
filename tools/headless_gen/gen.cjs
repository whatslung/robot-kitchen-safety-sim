// 재현 가능한 원본 시뮬레이터 캡처 생성기.
// 사용: node gen.cjs <out_dir> <samples> [base_seed]
// 이 도구는 train/val/test 분할을 만들지 않는다. 장면 단위 분할은 후속 단계에서 한다.
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');
const {
  captureWithRetry,
  collectInputFiles,
  fingerprintFiles,
  runSceneTransaction,
  assertEmptyOutputDir,
  sceneSeed,
  startStaticServer,
  writeCaptureAtomic,
  writeJsonAtomic,
} = require('./lib.cjs');

const REPO = path.resolve(__dirname, '..', '..');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const CAMERAS = ['cvNW', 'cvNE', 'cvSE', 'cvSW', 'cvN', 'cvS'];
const OUTPUT = path.resolve(process.argv[2] || path.join(REPO, 'dataset', 'sim-oblique-6cam-raw'));
const SAMPLE_COUNT = Number(process.argv[3] || 20);
const BASE_SEED = Number(process.argv[4] || process.env.DATASET_SEED || 20260826);
const MIN_COLORS = Number(process.env.MIN_CAPTURE_COLORS || 500);
const MAX_ATTEMPTS = Number(process.env.CAPTURE_ATTEMPTS || 3);
const MANIFEST_PATH = path.join(OUTPUT, 'manifest.json');

function requireInteger(name, value, minimum) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}; got ${value}`);
  }
}

function git(...args) {
  return execFileSync('git', args, { cwd: REPO, encoding: 'utf8' }).trim();
}

async function captureCamera(page, camera) {
  return page.evaluate(async selectedCamera => {
    const groundTruthResult = await groundTruth(selectedCamera, { noDepth: true });
    const image = new Image();
    image.src = groundTruthResult.rgb;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const colors = new Set();
    for (let offset = 0; offset < pixels.length; offset += 64) {
      colors.add((pixels[offset] << 16) | (pixels[offset + 1] << 8) | pixels[offset + 2]);
    }
    const signatureCanvas = document.createElement('canvas');
    signatureCanvas.width = 16;
    signatureCanvas.height = 16;
    const signatureContext = signatureCanvas.getContext('2d');
    signatureContext.drawImage(image, 0, 0, 16, 16);
    const signaturePixels = signatureContext.getImageData(0, 0, 16, 16).data;
    const visualSignature = Array.from(signaturePixels)
      .filter((_, index) => index % 4 !== 3)
      .map(value => value.toString(16).padStart(2, '0')).join('');
    return {
      rgb: groundTruthResult.rgb,
      labelText: groundTruthResult.labelText || '',
      colors: colors.size,
      meta: groundTruthResult.meta,
      visualSignature,
    };
  }, camera);
}

async function randomizeSceneWithSeed(page, seed) {
  return page.evaluate(async selectedSeed => {
    const originalRandom = Math.random;
    const originalAnimationRatio = scene.getAnimationRatio;
    datasetCaptureFreeze(false);
    const authoredRandom = makeRNG(selectedSeed);
    const rendererRandom = makeRNG((selectedSeed ^ 0xA5A5A5A5) >>> 0);
    window.__datasetRandom = authoredRandom.next;
    window.__datasetSceneSeed = selectedSeed >>> 0;
    window.__datasetPreparationTimeMs = selectedSeed % 10_000_000;
    window.__datasetCaptureTimeMs = selectedSeed % 10_000_000;
    Math.random = rendererRandom.next;
    scene.getAnimationRatio = () => 1;
    try {
      const peopleCount = [0, 1, 1, 1, 2, 2][(sceneRandom() * 6) | 0];
      TRAJ.seedBase = selectedSeed >>> 0;
      EXTRA_GEN = 0;
      RNG.reseed(selectedSeed);
      await setExtraCount(0);
      await setExtraCount(peopleCount);
      for (const system of scene.particleSystems) {
        if (!Array.isArray(system._stockParticles)
            || typeof system._newPartsExcess !== 'number'
            || typeof system._actualFrame !== 'number') {
          throw new Error(`지원하지 않는 Babylon ParticleSystem 내부 구조: ${system.name}`);
        }
        system.reset();
        system._stockParticles.length = 0;
        system._newPartsExcess = 0;
        system._actualFrame = 0;
      }
      ENV_RAND.propJitter = 0;
      randomizeScene();
      // RGB와 GT 마스크가 캡처 내내 같은 렌더 경로와 픽셀 좌표계를 쓰게 한다.
      // 일반 시뮬레이터의 CCTV 효과는 유지하되 원본 학습 캡처에서는 센서를 완전히 끈다.
      Object.assign(SENSOR, {
        on: false,
        grain: 0,
        distortion: 0,
        chroma: 0,
        blur: 0,
        vignette: 0,
        expJit: 0,
        lowres: 0,
      });
      sensorApply();
      const trimParticles = (system, desired) => {
        while (system.particles.length > desired) {
          system.recycleParticle(system.particles[system.particles.length - 1]);
        }
      };
      const steadyCount = (system, ratio) => Math.max(0, Math.floor(
        system.emitRate * (system.minLifeTime + system.maxLifeTime) / 2 * ratio,
      ));
      trimParticles(fsmoke, state.fire ? steadyCount(fsmoke, 0.9) : 0);
      trimParticles(firePS, state.fire ? steadyCount(firePS, 0.9) : 0);
      trimParticles(steam, steadyCount(steam, 0.8));
      const animationRandom = makeRNG((selectedSeed ^ 0x5A5A5A5A) >>> 0);
      window.__datasetAnimationState = [];
      for (const group of scene.animationGroups) {
        if (!group.isStarted) continue;
        const frame = group.from + animationRandom.next() * Math.max(1, group.to - group.from);
        group.pause();
        group.goToFrame(frame);
        window.__datasetAnimationState.push({
          name: group.name,
          frame: +frame.toFixed(6),
          speedRatio: +group.speedRatio.toFixed(6),
        });
      }
      const captureTimeMs = selectedSeed % 10_000_000;
      sensorTick(captureTimeMs);
      datasetCaptureFreeze(true);
      for (let frame = 0; frame < 4; frame += 1) scene.render();
      const vector = value => value.asArray().map(item => +item.toFixed(6));
      return {
        identity: {
          staticSeed: window.__datasetStaticSeed,
          sceneSeed: selectedSeed >>> 0,
        },
        peopleCount,
        resources: {
          animationGroups: scene.animationGroups.length,
          skeletons: scene.skeletons.length,
        },
        captureTimeMs,
        sensor: {
          grain: SENSOR.grain,
          distortion: SENSOR.distortion,
          chroma: SENSOR.chroma,
          blur: SENSOR.blur,
          vignette: SENSOR.vignette,
          exposureJitter: SENSOR.expJit,
          lowResolution: SENSOR.lowres,
          enabled: SENSOR.on,
        },
        scene: {
          fire: state.fire,
          fireProgress: state.fire ? +Math.min(1, FIRE_FOG.t / FIRE_FOG.full).toFixed(6) : 0,
          fogDensity: +scene.fogDensity.toFixed(6),
          smokeParticles: fsmoke.particles.length,
          steamScale: +STEAM.k.toFixed(6),
          environment: JSON.parse(JSON.stringify(ENV_RAND.last || {})),
        },
        people: [
          { id: 'person_0', position: vector(person.node.position), rotationY: +person.node.rotation.y.toFixed(6), mode: person.mode, crouch: person.crouch },
          ...EXTRAS.map((item, index) => ({
            id: `person_${index + 1}`,
            position: vector(item.root.position),
            rotationY: +item.root.rotation.y.toFixed(6),
            mode: item.mode,
            animations: Object.keys(item.ACT || {}).sort(),
          })),
        ],
        robot: {
          joints: JOINTS.map(joint => ({ label: joint.label, value: +joint.value.toFixed(8) })),
        },
        animations: JSON.parse(JSON.stringify(window.__datasetAnimationState)),
        cameras: Object.fromEntries(Object.entries(SURV).map(([id, item]) => [id, {
          position: vector(item.cam.position),
          worldMatrix: Array.from(item.cam.getWorldMatrix().m).map(value => +value.toFixed(8)),
        }])),
      };
    } finally {
      delete window.__datasetRandom;
      delete window.__datasetPreparationTimeMs;
      scene.getAnimationRatio = originalAnimationRatio;
      Math.random = originalRandom;
    }
  }, seed);
}

async function main() {
  requireInteger('samples', SAMPLE_COUNT, 1);
  requireInteger('base_seed', BASE_SEED, 0);
  requireInteger('MIN_CAPTURE_COLORS', MIN_COLORS, 1);
  requireInteger('CAPTURE_ATTEMPTS', MAX_ATTEMPTS, 1);
  if (!fs.existsSync(CHROME)) throw new Error(`Chrome executable not found: ${CHROME}`);
  assertEmptyOutputDir(OUTPUT);
  fs.mkdirSync(OUTPUT, { recursive: true });

  const inputs = collectInputFiles(REPO);
  const manifest = {
    schemaVersion: 1,
    kind: 'raw-simulator-captures',
    split: null,
    status: 'starting',
    repository: {
      commit: git('rev-parse', 'HEAD'),
      workingTreeDirty: git('status', '--porcelain').length > 0,
    },
    runtime: {
      simulatorUrl: null,
      chromeExecutable: path.resolve(CHROME),
      playwrightVersion: require('playwright-core/package.json').version,
    },
    generation: {
      baseSeed: BASE_SEED >>> 0,
      sampleCount: SAMPLE_COUNT,
      cameras: CAMERAS,
      minimumSampledColors: MIN_COLORS,
      captureAttempts: MAX_ATTEMPTS,
    },
    inputs: fingerprintFiles(REPO, inputs),
    startedAt: new Date().toISOString(),
    scenes: [],
  };
  writeJsonAtomic(MANIFEST_PATH, manifest);

  let browser;
  let staticServer;
  try {
    staticServer = await startStaticServer(REPO);
    const simulatorUrl = `${staticServer.baseUrl}/sim.html?seed=${BASE_SEED >>> 0}`;
    manifest.runtime.simulatorUrl = simulatorUrl;
    writeJsonAtomic(MANIFEST_PATH, manifest);
    console.log(`launching Chrome: ${CHROME}`);
    browser = await chromium.launch({
      executablePath: CHROME,
      headless: false,
      args: [
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--window-position=-2400,-2400',
      ],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.addInitScript(initialSeed => {
      let state = (initialSeed >>> 0) || 1;
      window.__datasetStaticSeed = initialSeed >>> 0;
      Math.random = () => {
        state = (state + 0x6D2B79F5) | 0;
        let value = Math.imul(state ^ (state >>> 15), 1 | state);
        value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
      };
    }, BASE_SEED >>> 0);
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    console.log(`loading verified simulator: ${simulatorUrl}`);
    await page.goto(simulatorUrl, { waitUntil: 'load', timeout: 60000 });
    console.log('waiting for simulator initialization');
    await page.waitForFunction(() => window.__simReady === true, null, { timeout: 120000 });
    console.log('waiting for all environment props');
    await page.waitForFunction(() => {
      const holders = scene.transformNodes.filter(node => /^envProp_/.test(node.name));
      return ENV_READY === true && ENV_PROPS.length > 0 && holders.length === ENV_PROPS.length;
    }, null, { timeout: 120000 });
    if (pageErrors.length) throw new Error(`Simulator page error: ${pageErrors.join(' | ')}`);
    await page.evaluate(() => engine.stopRenderLoop());
    // 시뮬레이터가 데모용으로 미리 만든 추가 인원을 제거한 뒤 기준 자원 수를 잡는다.
    await page.evaluate(() => setExtraCount(0));

    manifest.readiness = await page.evaluate(() => ({
      meshCount: scene.meshes.length,
      animationGroupCount: scene.animationGroups.length,
      skeletonCount: scene.skeletons.length,
      expectedEnvironmentProps: ENV_PROPS.length,
      loadedEnvironmentProps: scene.transformNodes.filter(node => /^envProp_/.test(node.name)).length,
      classContract: GT_CLASSES.map(item => ({ id: item.id, key: item.key, label: item.label })),
    }));
    manifest.runtime.chromeVersion = await browser.version();
    manifest.status = 'running';
    writeJsonAtomic(MANIFEST_PATH, manifest);
    console.log(`sim ready. generating ${SAMPLE_COUNT} scenes x ${CAMERAS.length} cameras -> ${OUTPUT}`);

    const started = Date.now();
    for (let index = 0; index < SAMPLE_COUNT; index += 1) {
      const seed = sceneSeed(BASE_SEED, index);
      const conditions = await randomizeSceneWithSeed(page, seed);
      const sceneRecord = { index, seed, conditions, captures: [] };
      const sceneName = String(index).padStart(4, '0');
      await runSceneTransaction({
        outDir: OUTPUT,
        sceneName,
        build: async stageDir => {
          for (const camera of CAMERAS) {
            const capture = await captureWithRetry(
              () => captureCamera(page, camera),
              {
                attempts: MAX_ATTEMPTS,
                minColors: MIN_COLORS,
                onRetry: () => page.evaluate(() => { scene.render(); scene.render(); }),
              },
            );
            const baseName = `${camera}_${sceneName}`;
            const files = writeCaptureAtomic({
              outDir: stageDir,
              base: baseName,
              rgbDataUrl: capture.rgb,
              labelText: capture.labelText,
              colors: capture.colors,
              minColors: MIN_COLORS,
            });
            sceneRecord.captures.push({
              camera, ...files, meta: capture.meta,
              visualSignature: capture.visualSignature,
              visualSignatureAlgorithm: 'rgb16x16-v1',
            });
          }
          return sceneRecord;
        },
        commit: (record, prefix) => {
          for (const capture of record.captures) {
            capture.image = path.posix.join(prefix, capture.image);
            capture.label = path.posix.join(prefix, capture.label);
          }
          manifest.scenes.push(record);
          try {
            writeJsonAtomic(MANIFEST_PATH, manifest);
          } catch (error) {
            manifest.scenes.pop();
            throw error;
          }
        },
      });
      const secondsPerScene = (Date.now() - started) / (index + 1) / 1000;
      console.log(`  scene ${index + 1}/${SAMPLE_COUNT} done · ~${secondsPerScene.toFixed(1)}s/scene`);
    }

    manifest.status = 'complete';
    manifest.completedAt = new Date().toISOString();
    manifest.savedCaptures = manifest.scenes.reduce((sum, item) => sum + item.captures.length, 0);
    writeJsonAtomic(MANIFEST_PATH, manifest);
    console.log(`DONE: ${manifest.savedCaptures} raw captures. No train/val/test split was created.`);
  } catch (error) {
    manifest.status = 'failed';
    manifest.failedAt = new Date().toISOString();
    manifest.error = error && error.stack ? error.stack : String(error);
    writeJsonAtomic(MANIFEST_PATH, manifest);
    throw error;
  } finally {
    if (browser) await browser.close();
    if (staticServer) await staticServer.close();
  }
}

main().catch(error => {
  console.error('ERR', error);
  process.exitCode = 1;
});
