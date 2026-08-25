/* GT 인스턴스/클래스 마스크 라벨 → bbox — stray 픽셀에 강건한 박스 산출.
 *
 * 배경(2026-08-24, 사선 카메라 회귀): 인스턴스 박스를 "그 색으로 분류된 모든 픽셀의
 * min/max" 로 만들면, 최근접색 판정(절대 상한 없음)이 오분류한 소수의 흩어진 픽셀이
 * 박스를 프레임 전체로 늘린다. 나디르(직교)에선 겹침·전경 가림이 없어 stray 가 없었고
 * 사선에서만 터졌다.
 *
 * 해결: 색별 4-연결 성분을 구해 작고 고립된 성분을 버린다.
 *   - 성분 크기 ≥ max(MIN_COMP, FRAC × 그 색의 최대 성분)  인 것만 유지
 *     · FRAC: 가림에 쪼개진 실제 물체의 큰 조각들은 (최대 성분 대비 충분히 크므로) 모두 살린다
 *     · MIN_COMP: 최대 성분 자체가 이보다 작으면 = 실제 물체 없이 노이즈뿐 → 전부 버려 드롭
 *   - 유지 픽셀합 ≥ minPixels 인 색만 박스로 낸다(거의 가려진 물체 제외 — 종전 n<80 규칙 계승)
 *
 * 브라우저(sim.html)와 Node 테스트 양쪽에서 같은 코드를 쓴다(<script src> + require/import 하이브리드).
 */
(function (root) {
  /* idx: Int32Array(w*h). 값 = key(>=0) 또는 -1(배경).
   * opt.minComp(기본 30) · opt.frac(기본 0.15) · opt.minPixels(기본 80).
   * 반환: Map<key, {minX,maxX,minY,maxY,n}>  (필터 후 유지 픽셀 기준). */
  function labelBoxesFiltered(idx, w, h, opt) {
    const MIN_COMP = opt && opt.minComp != null ? opt.minComp : 30;
    const FRAC = opt && opt.frac != null ? opt.frac : 0.15;
    const NMIN = opt && opt.minPixels != null ? opt.minPixels : 80;
    const N = w * h;

    // ── 4-연결 성분 라벨링 (union-find). 같은 key 이고 상/좌 이웃과 붙으면 합친다.
    const parent = new Int32Array(N);
    for (let i = 0; i < N; i++) parent[i] = i;
    const find = a => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
    for (let y = 0, i = 0; y < h; y++) {
      for (let x = 0; x < w; x++, i++) {
        const k = idx[i];
        if (k < 0) continue;
        if (x > 0 && idx[i - 1] === k) union(i, i - 1);
        if (y > 0 && idx[i - w] === k) union(i, i - w);
      }
    }

    // ── 성분별 집계: 픽셀 수 + bbox + 소속 key.
    const comp = new Map();   // root → {k, n, minX, maxX, minY, maxY}
    for (let y = 0, i = 0; y < h; y++) {
      for (let x = 0; x < w; x++, i++) {
        const k = idx[i];
        if (k < 0) continue;
        const r = find(i);
        let c = comp.get(r);
        if (!c) { c = { k, n: 0, minX: x, maxX: x, minY: y, maxY: y }; comp.set(r, c); }
        c.n++;
        if (x < c.minX) c.minX = x; if (x > c.maxX) c.maxX = x;
        if (y < c.minY) c.minY = y; if (y > c.maxY) c.maxY = y;
      }
    }

    // ── key 별로 성분을 모은다.
    const byKey = new Map();  // k → [components]
    for (const c of comp.values()) {
      let arr = byKey.get(c.k);
      if (!arr) { arr = []; byKey.set(c.k, arr); }
      arr.push(c);
    }

    // ── key 별 필터 + 박스 합치기.
    const out = new Map();
    for (const [k, comps] of byKey) {
      let mx = 0;
      for (const c of comps) if (c.n > mx) mx = c.n;
      const thr = Math.max(MIN_COMP, FRAC * mx);
      let box = null, n = 0;
      for (const c of comps) {
        if (c.n < thr) continue;                    // 작거나 고립된 stray 성분 제거
        n += c.n;
        if (!box) box = { minX: c.minX, maxX: c.maxX, minY: c.minY, maxY: c.maxY };
        else {
          if (c.minX < box.minX) box.minX = c.minX; if (c.maxX > box.maxX) box.maxX = c.maxX;
          if (c.minY < box.minY) box.minY = c.minY; if (c.maxY > box.maxY) box.maxY = c.maxY;
        }
      }
      if (box && n >= NMIN) { box.n = n; out.set(k, box); }
    }
    return out;
  }

  const api = { labelBoxesFiltered };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;   // Node (require/import)
  if (root) root.labelBoxesFiltered = labelBoxesFiltered;                      // 브라우저 전역
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : null));
