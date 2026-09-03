// 방향 회피(옆비킴) 오프셋 계산 — 순수 함수 단위 테스트 (node:test).
//   실행:  node --test tests/browser/lateral-offset.test.mjs
//
// computeLateralOffset(person, base, tcp, opts) → {dx, dz}
//   · person/base/tcp = {x, z} 수평 좌표(m). base=로봇 베이스, tcp=현재 팔끝.
//   · opts = { actR, dangerR, maxOffset, reachR }
//       actR     = 발동 시작 반경(이보다 멀면 0) — "너무 먼 반경부터 작동 안 함"
//       dangerR  = 팔 반경(빨간 원). 여기서 오프셋 최대. "빨간 원 즈음 작동"
//       maxOffset= 최대 옆비킴(m)
//       reachR   = 팔 도달 한계 — tcp+offset 이 이보다 밖으로 못 나가게
//   반환: TCP 목표에 더할 수평 오프셋. 사람 반대쪽으로 밀어 팔을 옆으로 피하게 한다.
import { test } from 'node:test';
import assert from 'node:assert';
import { computeLateralOffset } from './lateral-offset.mjs';

const OPTS = { actR: 2.2, dangerR: 1.87, maxOffset: 0.30, reachR: 1.87 };
const base = { x: 0, z: 0 };
const tcp = { x: 0.5, z: 0.5 };
const mag = (o) => Math.hypot(o.dx, o.dz);

test('발동 반경 밖(사람 멀다) → 오프셋 0', () => {
  const o = computeLateralOffset({ x: 3.0, z: 0 }, base, tcp, OPTS);   // 3.0m > actR 2.2
  assert.deepStrictEqual(o, { dx: 0, dz: 0 });
});

test('사람이 빨간 원(dangerR)에 있으면 → 크기 = maxOffset', () => {
  const o = computeLateralOffset({ x: -1.87, z: 0 }, base, tcp, OPTS);
  assert.ok(Math.abs(mag(o) - 0.30) < 1e-9, `mag=${mag(o)}`);
});

test('빨간 원 안쪽이어도 → maxOffset 로 상한', () => {
  const o = computeLateralOffset({ x: -1.0, z: 0 }, base, tcp, OPTS);
  assert.ok(Math.abs(mag(o) - 0.30) < 1e-9, `mag=${mag(o)}`);
});

test('사람이 왼쪽(-x) → 오프셋 오른쪽(+x)', () => {
  const o = computeLateralOffset({ x: -1.5, z: 0 }, base, tcp, OPTS);
  assert.ok(o.dx > 0 && Math.abs(o.dz) < 1e-9, `o=${JSON.stringify(o)}`);
});

test('사람이 앞(+z) → 오프셋 뒤(-z)', () => {
  const o = computeLateralOffset({ x: 0, z: 1.5 }, base, tcp, OPTS);
  assert.ok(o.dz < 0 && Math.abs(o.dx) < 1e-9, `o=${JSON.stringify(o)}`);
});

test('발동대 안에서 멀수록(빨간 원에서 멀수록) 크기 작다', () => {
  const near = computeLateralOffset({ x: -1.90, z: 0 }, base, tcp, OPTS);  // 빨간 원 근처
  const far  = computeLateralOffset({ x: -2.05, z: 0 }, base, tcp, OPTS);  // 발동대 바깥쪽
  assert.ok(mag(far) > 0 && mag(far) < mag(near), `near=${mag(near)} far=${mag(far)}`);
});

test('사람이 베이스와 겹침(퇴화) → 0 (0나눗셈 없음)', () => {
  const o = computeLateralOffset({ x: 0, z: 0 }, base, tcp, OPTS);
  assert.deepStrictEqual(o, { dx: 0, dz: 0 });
});

test('도달 클램프: 오프셋이 팔 반경(reachR) 밖으로 TCP를 밀지 않는다', () => {
  const tcpEdge = { x: 1.8, z: 0 };                                  // 도달 한계 근처
  const o = computeLateralOffset({ x: -1.5, z: 0 }, base, tcpEdge, OPTS);  // +x 로 밀림
  const d = Math.hypot(tcpEdge.x + o.dx, tcpEdge.z + o.dz);
  assert.ok(d <= OPTS.reachR + 1e-9, `resulting TCP dist ${d} > reachR ${OPTS.reachR}`);
});
