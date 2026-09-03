// 방향 회피(옆비킴) 오프셋 계산 — 순수 함수. sim.html 에 같은 본문을 심는다(빌드 없음).
//   ⚠️ 이 함수 본문은 tests/browser/lateral-offset.test.mjs 로 검증된 canonical 버전이다.
//      sim.html 쪽 복사본을 바꾸면 여기도 같이 바꾸고 node --test 로 다시 통과시킬 것.
//
// person/base/tcp = {x,z} 수평(m). base=로봇 베이스, tcp=현재 팔끝(TCP).
// opts = { actR, dangerR, maxOffset, reachR }
//   actR=발동 시작 반경(밖이면 0) · dangerR=빨간 원(여기서 최대) · maxOffset=최대 옆비킴(m)
//   reachR=팔 도달 한계(tcp+offset 이 이 밖으로 못 나감)
// 반환: TCP 목표에 더할 수평 오프셋 {dx,dz}. 사람 반대쪽으로 밀어 팔을 옆으로 피하게 한다.
export function computeLateralOffset(person, base, tcp, opts) {
  const { actR, dangerR, maxOffset, reachR } = opts;
  const px = person.x - base.x, pz = person.z - base.z;   // 사람(베이스 기준)
  const d = Math.hypot(px, pz);
  if (d >= actR || d < 1e-6) return { dx: 0, dz: 0 };      // 너무 멀거나 겹침 → 발동 안 함

  // 램프: actR 에서 0 → dangerR(빨간 원)에서 1, 안쪽은 1 로 상한.
  const ramp = Math.max(0, Math.min(1, (actR - d) / (actR - dangerR)));
  const mag = maxOffset * ramp;
  if (mag <= 0) return { dx: 0, dz: 0 };

  // 방향: 사람 반대쪽(=베이스−사람) 단위벡터. 사람이 왼쪽이면 +x(오른쪽).
  const dirx = -px / d, dirz = -pz / d;
  let dx = dirx * mag, dz = dirz * mag;

  // 도달 클램프: tcp+offset 이 reachR 밖으로 나가면, 나가기 직전까지만 민다(ray↔원 교점).
  const tx = tcp.x - base.x, tz = tcp.z - base.z;
  if (Math.hypot(tx + dx, tz + dz) > reachR) {
    const b = 2 * (tx * dirx + tz * dirz);
    const c = tx * tx + tz * tz - reachR * reachR;
    const disc = b * b - 4 * c;                            // a=1 (dir 단위)
    if (disc < 0) return { dx: 0, dz: 0 };                 // 이미 밖 → 밀지 않음
    const t = Math.max(0, Math.min(mag, (-b + Math.sqrt(disc)) / 2));
    dx = dirx * t; dz = dirz * t;
  }
  return { dx, dz };
}
