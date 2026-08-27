// GT 인스턴스 마스크 → bbox 필터의 순수 로직 회귀 테스트.
//
// 버그(2026-08-24, 사선 카메라): 인스턴스 박스를 그 색으로 분류된 **모든** 픽셀의
//   순수 min/max로 만들었다. 최근접색 판정(절대 상한 없음)이 MSAA 경계·어두운 픽셀
//   한 줌을 프레임 아무 곳의 인스턴스 색으로 오분류하면, 단 1~4개의 흩어진 stray 픽셀이
//   박스를 프레임 전체로 늘렸다(person 박스 area 최대 0.67, 대형 43%).
// 수정: 색별로 연결성분을 구해 작은/고립 성분을 버리고(성분 ≥ max(MIN_COMP, FRAC×최대성분)),
//   유지 픽셀합 ≥ minPixels 인 것만 박스로 만든다. 가림에 쪼개진 실제 사람의 큰 조각은
//   모두 살아남고, 멀리 흩어진 오분류 stray는 사라진다.
//
// 실행:  node --test tests/browser/instance-box-filter.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import mod from '../../gtboxes.js';
const { labelBoxesFiltered } = mod;

// w×h 라벨 배열을 만들고 사각형 영역을 key 로 칠하는 헬퍼 (idx: >=0 key, -1 배경)
function grid(w, h) {
  const idx = new Int32Array(w * h).fill(-1);
  return {
    idx,
    fill(key, x0, y0, x1, y1) {
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) idx[y * w + x] = key;
      return this;
    },
    px(key, x, y) { idx[y * w + x] = key; return this; },
  };
}

test('main blob 하나 + 멀리 흩어진 stray 픽셀 → 박스는 blob 에만 타이트', () => {
  const w = 100, h = 100, g = grid(w, h);
  g.fill(0, 10, 10, 19, 19);          // 10×10 = 100px 실제 물체 (좌상단)
  g.px(0, 90, 90).px(0, 95, 5).px(0, 5, 95);  // 프레임 반대편에 흩어진 stray 3개
  const boxes = labelBoxesFiltered(g.idx, w, h, { minPixels: 50 });
  const b = boxes.get(0);
  assert.ok(b, 'key 0 박스가 있어야 한다');
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [10, 10, 19, 19],
    'stray 를 무시하고 blob(10,10)-(19,19) 로 타이트해야 한다');
  assert.equal(b.n, 100, '유지 픽셀은 blob 100개뿐');
});

test('가림에 쪼개진 한 물체의 큰 조각들은 모두 유지 (박스가 조각 전체를 감싼다)', () => {
  const w = 100, h = 100, g = grid(w, h);
  // 세로로 쌓인 두 큰 조각 (사람이 파이프에 가려 상·하로 갈림). 사이 5px 간격.
  g.fill(0, 40, 10, 59, 29);   // 상단 20×20 = 400px
  g.fill(0, 40, 35, 59, 54);   // 하단 20×20 = 400px
  const boxes = labelBoxesFiltered(g.idx, w, h, { minPixels: 80, relativeMin: 0 });
  const b = boxes.get(0);
  assert.ok(b);
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [40, 10, 59, 54],
    '두 조각을 모두 감싸야 한다');
  assert.equal(b.n, 800);
});

test('기본 정책은 큰 물체에서 멀리 떨어진 30px 이상 stray도 제외한다', () => {
  const w = 100, h = 40, g = grid(w, h);
  g.fill(0, 5, 5, 54, 24);     // 주 성분 1000px
  g.fill(0, 80, 5, 87, 9);     // 먼 오분류 40px: minComp는 넘지만 주 성분의 15% 미만
  const b = labelBoxesFiltered(g.idx, w, h, { minPixels: 80 }).get(0);
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [5, 5, 54, 24]);
  assert.equal(b.n, 1000);
});

test('가림으로 크기가 달라진 두 사람 조각도 모두 유지한다', () => {
  const w = 100, h = 40, g = grid(w, h);
  g.fill(0, 5, 5, 54, 24);     // 몸통 50×20 = 1000px
  g.fill(0, 70, 5, 79, 14);    // 가림 뒤 보이는 팔 10×10 = 100px
  const boxes = labelBoxesFiltered(g.idx, w, h, { minPixels: 80, relativeMin: 0 });
  const b = boxes.get(0);
  assert.ok(b);
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [5, 5, 79, 24],
    '절대 크기 기준을 넘는 작은 조각도 사람 박스에 포함해야 한다');
  assert.equal(b.n, 1100);
});

test('실제 blob 없이 흩어진 노이즈뿐인 인스턴스는 드롭된다', () => {
  const w = 100, h = 100, g = grid(w, h);
  // 최대 성분이 4px 인 작은 얼룩 수십 개 (near-bg 팔레트색이 어두운 픽셀을 훔친 경우)
  for (let k = 0; k < 30; k++) {
    const x = (k * 7) % 90 + 2, y = (k * 11) % 90 + 2;
    g.px(0, x, y).px(0, x + 1, y).px(0, x, y + 1).px(0, x + 1, y + 1);  // 2×2 = 4px
  }
  const boxes = labelBoxesFiltered(g.idx, w, h, { minComp: 30, minPixels: 80 });
  assert.equal(boxes.get(0), undefined, '최대 성분이 MIN_COMP 미만이면 인스턴스가 통째로 드롭');
});

test('필터 후 유지 픽셀이 minPixels 미만이면 드롭 (거의 가려진 물체)', () => {
  const w = 100, h = 100, g = grid(w, h);
  g.fill(0, 10, 10, 16, 16);   // 7×7 = 49px 하나짜리 blob
  const boxes = labelBoxesFiltered(g.idx, w, h, { minComp: 30, minPixels: 80 });
  assert.equal(boxes.get(0), undefined, '49px < 80 이면 드롭');
});

test('maxGap: 멀리 떨어진 같은-key 큰 성분은 합치지 않는다 (person 부풀림 차단)', () => {
  const w = 200, h = 100, g = grid(w, h);
  g.fill(0, 20, 20, 45, 70);    // 주 blob 26×51 ≈ 1326px (한 사람)
  g.fill(0, 150, 20, 170, 60);  // 멀리 떨어진 같은 색 큰 성분 21×41 ≈ 861px (다른 곳)
  // maxGap 없으면 둘을 union → 폭 20~170 으로 부풀음
  const wide = labelBoxesFiltered(g.idx, w, h, {});
  assert.equal(wide.get(0).maxX, 170, 'maxGap 없으면 먼 성분까지 union(부풀림)');
  // maxGap=30 이면 주 blob 만 (150-45=105px 간격 > 30)
  const b = labelBoxesFiltered(g.idx, w, h, { maxGap: 30 }).get(0);
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [20, 20, 45, 70],
    '멀리 떨어진 성분은 제외 → 주 blob 만 타이트');
});

test('maxGap: 가림에 쪼개진 가까운 조각은 여전히 합친다', () => {
  const w = 100, h = 100, g = grid(w, h);
  g.fill(0, 40, 10, 59, 29);    // 상단 조각
  g.fill(0, 40, 35, 59, 54);    // 하단 조각 (사이 5px 간격 < maxGap)
  const b = labelBoxesFiltered(g.idx, w, h, { maxGap: 30 }).get(0);
  assert.deepEqual([b.minX, b.minY, b.maxX, b.maxY], [40, 10, 59, 54],
    '가까운 두 조각(간격5)은 maxGap30 안 → 합쳐진다');
});

test('서로 다른 key 는 독립적으로 필터·박스된다', () => {
  const w = 100, h = 100, g = grid(w, h);
  g.fill(0, 5, 5, 24, 24);     // key 0: 20×20
  g.fill(1, 70, 70, 89, 89);   // key 1: 20×20
  g.px(1, 2, 98);              // key 1 의 stray 하나
  const boxes = labelBoxesFiltered(g.idx, w, h, { minPixels: 80 });
  assert.deepEqual([boxes.get(0).minX, boxes.get(0).maxX], [5, 24]);
  assert.deepEqual([boxes.get(1).minX, boxes.get(1).minY, boxes.get(1).maxX, boxes.get(1).maxY],
    [70, 70, 89, 89], 'key 1 은 자기 stray 를 무시');
});
