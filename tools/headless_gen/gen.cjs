// 재현 가능한 원본 시뮬레이터 캡처 생성기.
// 사용: node gen.cjs <out_dir> <samples> [base_seed]
// 이 도구는 train/val/test 분할을 만들지 않는다. 장면 단위 분할은 후속 단계에서 한다.
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');
const {
  captureWithRetry,
  fingerprintFiles,
  assertEmptyOutputDir,
  sceneSeed,
  writeCaptureAtomic,
  writeJsonAtomic,
} = require('./lib.cjs');

const REPO = path.resolve(__dirname, '..', '..');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = process.env.SIM_URL || 'http://127.0.0.1:8123/sim.html';
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

function walkFiles(root) {
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root, { withFileTypes: true }).flatMap(entry => {
    const absolute = path.join(root, entry.name);
    return entry.isDirectory() ? walkFiles(absolute) : [absolute];
  });
}

function inputFiles() {
  const fixed = [
    'sim.html', 'gtboxes.js', 'babylon.js', 'babylonjs.loaders.min.js',
    'HavokPhysics_umd.js', 'HavokPhysics.wasm', 'character-manifest.json',
  ].map(file => path.join(REPO, file));
  return [...fixed, ...walkFiles(path.join(REPO, 'assets'))].filter(file => fs.existsSync(file));
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
    return {
      rgb: groundTruthResult.rgb,
      labelText: groundTruthResult.labelText || '',
      colors: colors.size,
    };
  }, camera);
}

async function randomizeSceneWithSeed(page, seed) {
  return page.evaluate(async selectedSeed => {
    const originalRandom = Math.random;
    let state = selectedSeed >>> 0;
    Math.random = () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
    try {
      const peopleCount = [0, 1, 1, 1, 2, 2][(Math.random() * 6) | 0];
      await setExtraCount(peopleCount);
      randomizeScene();
      for (let frame = 0; frame < 4; frame += 1) scene.render();
      return {
        peopleCount,
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
      };
    } finally {
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
      simulatorUrl: URL,
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
    inputs: fingerprintFiles(REPO, inputFiles()),
    startedAt: new Date().toISOString(),
    scenes: [],
  };
  writeJsonAtomic(MANIFEST_PATH, manifest);

  let browser;
  try {
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
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    console.log(`loading simulator: ${URL}`);
    await page.goto(URL, { waitUntil: 'load', timeout: 60000 });
    console.log('waiting for simulator initialization');
    await page.waitForFunction(() => window.__simReady === true, null, { timeout: 120000 });
    console.log('waiting for all environment props');
    await page.waitForFunction(() => {
      const holders = scene.transformNodes.filter(node => /^envProp_/.test(node.name));
      return ENV_READY === true && ENV_PROPS.length > 0 && holders.length === ENV_PROPS.length;
    }, null, { timeout: 120000 });
    if (pageErrors.length) throw new Error(`Simulator page error: ${pageErrors.join(' | ')}`);

    manifest.readiness = await page.evaluate(() => ({
      meshCount: scene.meshes.length,
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
      for (const camera of CAMERAS) {
        const capture = await captureWithRetry(
          () => captureCamera(page, camera),
          {
            attempts: MAX_ATTEMPTS,
            minColors: MIN_COLORS,
            onRetry: () => page.evaluate(() => { scene.render(); scene.render(); }),
          },
        );
        const baseName = `${camera}_${String(index).padStart(4, '0')}`;
        const files = writeCaptureAtomic({
          outDir: OUTPUT,
          base: baseName,
          rgbDataUrl: capture.rgb,
          labelText: capture.labelText,
          colors: capture.colors,
          minColors: MIN_COLORS,
        });
        sceneRecord.captures.push({ camera, ...files });
      }
      manifest.scenes.push(sceneRecord);
      writeJsonAtomic(MANIFEST_PATH, manifest);
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
  }
}

main().catch(error => {
  console.error('ERR', error);
  process.exitCode = 1;
});
