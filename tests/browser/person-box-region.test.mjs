// person 라벨 = "정확색(±tol) 분류 → 연결성분 필터" — 순수 로직 회귀 테스트.
//
// 배경(2026-08-26): 마스크 person 박스를 classifyByNearest(최근접색·상한 없음)로 뽑으면
//   화재/연기 혼합 픽셀이 person 색으로 오분류되고, CC 필터가 그 먼 성분까지 union 해
//   박스가 프레임 가로 전체로 부풀었다(눈으로 확인). person 인스턴스 색은 서로·배경과
//   ≥MINSEP 떨어지도록 base 예약돼 있으므로, **정확색 ±tol 로만** 모으면 화재·연기·설비
//   픽셀(색거리 먼)은 -1 로 떨어져 부풀림이 원천 불가능해진다.
//   (계측: 사람별 정확색 픽셀 수는 0 아니면 실제 실루엣뿐 — 먼 stray 없음 확인.)
//
// 실행:  node --test tests/browser/person-box-region.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import mod from '../../gtboxes.js';
const { classifyExact, labelBoxesFiltered } = mod;

const P0 = [230, 25, 75], P1 = [60, 180, 75];   // person_0, person_1 색

test('classifyExact: 정확색 픽셀만 해당 index, 나머지는 -1', () => {
  const w = 4, h = 1, px = new Uint8Array(w * h * 4);
  const set = (x, c) => { const i = x * 4; px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; px[i + 3] = 255; };
  set(0, P0); set(1, P1); set(2, [255, 140, 20]); set(3, [231, 27, 74]);   // 화재색·근사색
  const cls = classifyExact(px, w, h, [['a', P0], ['b', P1]], 12);
  assert.deepEqual([...cls], [0, 1, -1, 0], '0=P0,1=P1,2=화재(먼색)→-1,3=P0근사→0');
});

test('classifyExact: tol 밖이면 가장 가까워도 -1 (부풀림 차단)', () => {
  const w = 2, h = 1, px = new Uint8Array(w * h * 4);
  px.set([200, 60, 90, 255], 0);   // P0와 거리 ~49 > tol
  px.set([230, 25, 75, 255], 4);   // 정확 P0
  const cls = classifyExact(px, w, h, [['a', P0]], 20);
  assert.deepEqual([...cls], [-1, 0], '먼 색은 -1, 정확색만 0');
});

test('classifyExact: tol을 생략하면 기본값 26을 사용한다', () => {
  const px = new Uint8ClampedArray([120, 100, 100, 255]);
  const idx = classifyExact(px, 1, 1, [[0, [100, 100, 100]]]);
  assert.equal(idx[0], 0);
});

// w×h RGBA 버퍼에 사각형을 색으로 칠하는 헬퍼
function buf(w, h) {
  const px = new Uint8Array(w * h * 4);
  return { px, fill(c, x0, y0, x1, y1) {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = (y * w + x) * 4; px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; px[i + 3] = 255;
    } return this; } };
}

test('정확색 분류 + CC필터: 화재색 stray 는 박스를 안 부풀린다 (통합)', () => {
  const w = 100, h = 100, b = buf(w, h);
  b.fill(P0, 15, 20, 30, 55);          // 진짜 사람 실루엣 (16×36)
  b.fill([255, 140, 20], 80, 5, 95, 20);  // 프레임 반대편 화재색 덩어리(먼 색)
  const cls = classifyExact(b.px, w, h, [['person_0', P0]], 26);
  const boxes = labelBoxesFiltered(cls, w, h, { minPixels: 80 });
  const bx = boxes.get(0);
  assert.ok(bx, 'person 박스가 있어야 한다');
  assert.deepEqual([bx.minX, bx.minY, bx.maxX, bx.maxY], [15, 20, 30, 55],
    '화재색은 -1 로 떨어져 박스는 사람만 (부풀림 없음)');
});

test('정확색 분류 + CC필터: 가림에 쪼개진 사람의 두 조각은 유지', () => {
  const w = 100, h = 100, b = buf(w, h);
  b.fill(P0, 40, 10, 59, 29);          // 상단 조각 400px
  b.fill(P0, 40, 35, 59, 54);          // 하단 조각 400px (파이프에 가려 갈림)
  const cls = classifyExact(b.px, w, h, [['person_0', P0]], 26);
  const boxes = labelBoxesFiltered(cls, w, h, { minPixels: 80 });
  const bx = boxes.get(0);
  assert.deepEqual([bx.minX, bx.minY, bx.maxX, bx.maxY], [40, 10, 59, 54], '두 조각 모두 감싼다');
});
