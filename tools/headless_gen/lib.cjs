const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const path = require('path');

const REQUIRED_RENDER_INPUTS = [
  'sim.html', 'gtboxes.js', 'babylon.js', 'babylonjs.loaders.min.js',
  'HavokPhysics_umd.js', 'HavokPhysics.wasm.js', 'character-manifest.json',
];

function sceneSeed(baseSeed, index) {
  if (!Number.isInteger(baseSeed) || !Number.isInteger(index) || index < 0) {
    throw new TypeError('baseSeed와 index는 음수가 아닌 정수여야 한다');
  }
  return (baseSeed + index) >>> 0;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function tempPath(file) {
  return `${file}.tmp-${process.pid}-${Date.now()}`;
}

function writeJsonAtomic(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = tempPath(file);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
}

function assertEmptyOutputDir(outDir) {
  if (!fs.existsSync(outDir)) return;
  const entries = fs.readdirSync(outDir);
  if (entries.length) {
    throw new Error(`비어 있지 않은 출력 폴더는 사용할 수 없다: ${outDir} (${entries.join(', ')})`);
  }
}

function beginSceneTransaction(outDir, sceneName) {
  if (!/^[A-Za-z0-9_-]+$/.test(sceneName || '')) {
    throw new Error(`잘못된 장면 이름: ${sceneName}`);
  }
  const stageDir = path.join(outDir, `.incomplete-scene-${sceneName}`);
  const finalDir = path.join(outDir, 'scenes', sceneName);
  const markerFile = path.join(outDir, `.incomplete-scene-${sceneName}.json`);
  if (fs.existsSync(stageDir) || fs.existsSync(finalDir) || fs.existsSync(markerFile)) {
    throw new Error(`기존 장면 트랜잭션과 충돌한다: ${sceneName}`);
  }
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(markerFile, `${JSON.stringify({ scene: sceneName, status: 'in_progress' })}\n`, { flag: 'wx' });
  try {
    fs.mkdirSync(stageDir);
  } catch (error) {
    fs.rmSync(markerFile, { force: true });
    throw error;
  }
  return { outDir, sceneName, stageDir, finalDir, markerFile, published: false };
}

function publishSceneTransaction(transaction) {
  fs.mkdirSync(path.dirname(transaction.finalDir), { recursive: true });
  fs.renameSync(transaction.stageDir, transaction.finalDir);
  transaction.published = true;
  return path.posix.join('scenes', transaction.sceneName);
}

function finishSceneTransaction(transaction) {
  if (!transaction.published || !fs.existsSync(transaction.finalDir)) {
    throw new Error(`공개되지 않은 장면을 완료할 수 없다: ${transaction.sceneName}`);
  }
  fs.rmSync(transaction.markerFile);
}

function rollbackSceneTransaction(transaction) {
  fs.rmSync(transaction.stageDir, { recursive: true, force: true });
  fs.rmSync(transaction.finalDir, { recursive: true, force: true });
  fs.rmSync(transaction.markerFile, { force: true });
}

async function runSceneTransaction({ outDir, sceneName, build, commit }) {
  if (typeof build !== 'function' || typeof commit !== 'function') {
    throw new TypeError('build와 commit 함수가 필요하다');
  }
  const transaction = beginSceneTransaction(outDir, sceneName);
  let committed = false;
  try {
    const record = await build(transaction.stageDir);
    const prefix = publishSceneTransaction(transaction);
    await commit(record, prefix);
    committed = true;
    finishSceneTransaction(transaction);
    return record;
  } catch (error) {
    if (!committed) rollbackSceneTransaction(transaction);
    throw error;
  }
}

function decodeDataUrl(dataUrl) {
  const match = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(dataUrl || '');
  if (!match) throw new Error('PNG data URL 형식이 아니다');
  return Buffer.from(match[1], 'base64');
}

function writeCaptureAtomic({ outDir, base, rgbDataUrl, labelText, colors, minColors = 500 }) {
  if (!Number.isFinite(colors) || colors < minColors) {
    throw new Error(`렌더 색상 수 ${colors} < ${minColors}: 단색 캡처 의심`);
  }
  const imageData = decodeDataUrl(rgbDataUrl);
  const labelData = Buffer.from(labelText || '', 'utf8');
  const imageDir = path.join(outDir, 'images'), labelDir = path.join(outDir, 'labels');
  const imageFile = path.join(imageDir, `${base}.png`), labelFile = path.join(labelDir, `${base}.txt`);
  const incompleteFile = path.join(outDir, `.incomplete-${base}.json`);
  if (fs.existsSync(imageFile) || fs.existsSync(labelFile)) {
    throw new Error(`기존 캡처를 덮어쓸 수 없다: ${base}`);
  }
  fs.mkdirSync(imageDir, { recursive: true });
  fs.mkdirSync(labelDir, { recursive: true });
  const imageTemporary = tempPath(imageFile), labelTemporary = tempPath(labelFile);
  let imageCommitted = false;
  try {
    fs.writeFileSync(incompleteFile, `${JSON.stringify({ image: path.basename(imageFile), label: path.basename(labelFile) })}\n`, { flag: 'wx' });
    fs.writeFileSync(imageTemporary, imageData, { flag: 'wx' });
    fs.writeFileSync(labelTemporary, labelData, { flag: 'wx' });
    fs.renameSync(imageTemporary, imageFile); imageCommitted = true;
    fs.renameSync(labelTemporary, labelFile);
    fs.rmSync(incompleteFile);
  } catch (error) {
    if (imageCommitted) fs.rmSync(imageFile, { force: true });
    fs.rmSync(incompleteFile, { force: true });
    throw error;
  } finally {
    fs.rmSync(imageTemporary, { force: true });
    fs.rmSync(labelTemporary, { force: true });
  }
  return {
    image: path.posix.join('images', `${base}.png`),
    label: path.posix.join('labels', `${base}.txt`),
    colors,
    imageSha256: sha256(imageData),
    labelSha256: sha256(labelData),
  };
}

async function captureWithRetry(capture, { attempts = 3, minColors = 500, onRetry } = {}) {
  if (!Number.isInteger(attempts) || attempts < 1) throw new TypeError('attempts는 1 이상의 정수여야 한다');
  for (let attempt = 1; attempt <= attempts; attempt++) {
    const result = await capture(attempt);
    if (Number.isFinite(result && result.colors) && result.colors >= minColors) return result;
    if (attempt < attempts && onRetry) await onRetry(attempt, result);
  }
  throw new Error(`렌더 색상 검증 실패: ${attempts}회 재시도 후에도 ${minColors}색 미만`);
}

function fingerprintFiles(root, files) {
  const resolvedRoot = path.resolve(root);
  return files.map(file => {
    const resolvedFile = path.resolve(file);
    const relative = path.relative(resolvedRoot, resolvedFile);
    if (!relative || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error(`fingerprint 대상이 저장소 밖이거나 루트다: ${file}`);
    }
    return { path: relative.split(path.sep).join('/'), sha256: sha256(fs.readFileSync(resolvedFile)) };
  }).sort((a, b) => a.path.localeCompare(b.path));
}

function collectFiles(root) {
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root, { withFileTypes: true }).flatMap(entry => {
    const absolute = path.join(root, entry.name);
    return entry.isDirectory() ? collectFiles(absolute) : [absolute];
  });
}

function collectInputFiles(repoRoot) {
  const required = REQUIRED_RENDER_INPUTS.map(file => path.join(repoRoot, file));
  const missing = required.filter(file => !fs.existsSync(file));
  if (missing.length) {
    throw new Error(`필수 렌더 입력이 없다: ${missing.map(file => path.basename(file)).join(', ')}`);
  }
  return [...required, ...collectFiles(path.join(repoRoot, 'assets'))];
}

function contentType(file) {
  switch (path.extname(file).toLowerCase()) {
    case '.html': return 'text/html; charset=utf-8';
    case '.js': case '.cjs': case '.mjs': return 'text/javascript; charset=utf-8';
    case '.json': return 'application/json; charset=utf-8';
    case '.wasm': return 'application/wasm';
    case '.png': return 'image/png';
    case '.jpg': case '.jpeg': return 'image/jpeg';
    case '.glb': return 'model/gltf-binary';
    default: return 'application/octet-stream';
  }
}

async function startStaticServer(root) {
  const resolvedRoot = path.resolve(root);
  const server = http.createServer((request, response) => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    } catch {
      response.writeHead(400).end('bad request');
      return;
    }
    const relative = (pathname === '/' ? 'sim.html' : pathname.replace(/^\/+/, ''));
    if (relative.includes('\0')) {
      response.writeHead(400).end('bad request');
      return;
    }
    const file = path.resolve(resolvedRoot, relative);
    if (file !== resolvedRoot && !file.startsWith(`${resolvedRoot}${path.sep}`)) {
      response.writeHead(403).end('forbidden');
      return;
    }
    fs.stat(file, (error, stat) => {
      if (error || !stat.isFile()) {
        response.writeHead(404).end('not found');
        return;
      }
      response.writeHead(200, { 'Content-Type': contentType(file), 'Content-Length': stat.size });
      fs.createReadStream(file).pipe(response);
    });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve())),
  };
}

module.exports = {
  sceneSeed,
  seededRandom,
  sha256,
  writeCaptureAtomic,
  writeJsonAtomic,
  assertEmptyOutputDir,
  beginSceneTransaction,
  publishSceneTransaction,
  finishSceneTransaction,
  rollbackSceneTransaction,
  runSceneTransaction,
  captureWithRetry,
  fingerprintFiles,
  collectInputFiles,
  startStaticServer,
};
