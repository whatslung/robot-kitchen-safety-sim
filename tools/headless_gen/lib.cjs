const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

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

module.exports = {
  sceneSeed,
  seededRandom,
  sha256,
  writeCaptureAtomic,
  writeJsonAtomic,
  assertEmptyOutputDir,
  captureWithRetry,
  fingerprintFiles,
};
