# 충돌·회피 시나리오 인계 문서

작성일: 2026-08-28  
브랜치: `codex/pr24-collision-avoidance-rebuild`  
기준 커밋: `7002c6e`

## 요청된 목표

두 버튼으로 같은 방향의 A→B 접근을 보여야 한다.

1. **충돌 시나리오**: 사람이 A→B로 걷다가 로봇팔과 접촉한다. 사람은 물리적으로 밀리고 쓰러진다.
2. **회피 시나리오**: 사람이 같은 방향으로 접근하면 로봇팔이 반대쪽 안전 자세로 즉시 회피하고, 사람이 안전 반경 밖으로 이탈한 뒤 작업을 재개한다.

## 현재 커밋까지 반영된 것

- `f4239b8`: SSM 정지 반경 안에서도 충돌 검증된 `RETRACT`/`SAFE_LIFT`만 제한 속도로 움직이도록 변경했다. E-STOP·미검증 후보·설비/바스켓 충돌 후보는 정지한다.
- `7002c6e`: 상단에 `⚠ A→B 충돌 시나리오`, `▶ A→B 회피 시나리오` 버튼을 추가했다.
- 현재 작업 트리에만 아직 커밋하지 않은 변경이 있다. 충돌은 주인공(`person`)을 A→B 경로로 보내도록 바꾸고, `tests/browser/directional-collision.spec.mjs`를 추가했다.

## 실제로 확인된 동작

### 충돌

`tests/browser/directional-collision.spec.mjs`는 통과했다.

- 주인공이 A→B 경로로 걷는다.
- `armContactUpdate()`의 swept capsule 충돌 검사가 접촉을 찾는다.
- 접촉 시 `triggerEstop()`이 실행된다.
- 사람은 팔 반대 방향으로 `esc` 위치까지 이동하고 `personFall()`로 쓰러진다.

관련 코드: `sim.html`의 `armContactUpdate()` 및 `startDirectionalScenario("collision")`.

즉, 질문한 “충돌하면 물리 반영되나?”에 대한 답은 **그렇다**이다. 충돌은 접촉 처리에서 사람 위치를 실제로 옮기고 넘어뜨린다.

### 회피

회피 시나리오는 아직 안정적이지 않다. 같은 브라우저 회귀가 다음 두 결과를 보였다.

- 성공한 실행에서는 사람 최저 거리 약 2.64m(< 3.10m stop radius), `SAFE_LIFT` 중 관절 변화가 기록되고 작업이 완료됐다.
- 실패한 실행에서는 사람 최저 거리 약 2.79m에서 바스켓/팔과 접촉해 `시연 실패 — 로봇 또는 바스켓 접촉`이 발생했다.

실패 시 출력의 핵심 값:

```text
SAFE.dist: 약 2.79m
SAFE.stopR: 3.10m
SAFE.slowR: 3.90m
AVOID.mode: STOP 또는 SAFE_LIFT
state.seqIdx: 11 (바스켓을 든 카트 이송 단계)
basketHeld: true
```

## 현재 문제의 원인

### 1. 시나리오 경로가 하나의 명확한 계약이 아니다

`lstmYieldDemoRoute()`는 기존 LSTM 회피 데모 경로이고, `directionalScenarioRoute()`는 충돌용 접촉점 `C`를 동적으로 섞은 별도 경로다. 둘이 A→B를 같은 객체/같은 waypoint 목록으로 공유하지 않는다.

그 결과 버튼 이름은 분리됐지만 실제 사람의 이동·예측·접촉점이 같은 시나리오 정의에서 나오지 않는다.

### 2. 회피 후보와 실제 바스켓 swept-contact가 불일치한다

회피 후보 평가는 사람 예측과 설비/바스켓 메시 충돌을 보지만, 실제 프레임의 `armContactUpdate()`는 팔과 바스켓의 swept capsule을 사람에 대해 검사한다. 회피 후보가 통과한 뒤에도 바스켓 이송 단계에서 실제 swept contact가 나는 실행이 있다.

따라서 후보 평가가 **실행 경로의 바스켓 swept geometry와 같은 기준**을 쓰도록 통합되어야 한다.

### 3. 이탈 완료 판정이 실제 반경과 경로 종료를 보장하지 않았다

기존에는 `person.mode === "idle"`만으로 clearing을 시작할 수 있었다. 사람이 실제로 3.90m slow radius 밖에 있지 않아도 로봇이 재개할 수 있었다.

작업 트리에는 `SAFE.dist >= SAFE.slowR`를 함께 요구하는 미커밋 수정이 있다. 그러나 경로 fallback이 반경 안에서 끝나는 경우도 있어, 경로 생성과 이탈 판정을 함께 다시 설계해야 한다.

### 4. “오른쪽 회피”는 아직 시각적으로 독립된 자세가 아니다

`T.rightSafe`가 추가됐지만 현재 좌표는 기존 `T.potHover`와 동일하다. 이름만 오른쪽이고 새 자세가 아니다. 이전에 별도 오른쪽 좌표를 시도했을 때 후보가 안전하지 않아 시연이 진행되지 않았다.

## Claude에게 권장하는 해결 순서

1. `ScenarioRoute` 하나를 만든다. `A`, `C(위험 통과점)`, `B(반경 밖 종료점)`과 waypoint 목록을 한 객체로 고정한다.
2. 충돌/회피 버튼은 반드시 그 동일 객체를 사용한다. 차이는 오직 `safetyEnabled`와 선택하는 escape target이어야 한다.
3. 충돌 버튼은 `safetyEnabled=false`로 A→C→B를 실행하고, C에서 `armContactUpdate()`의 실제 swept contact·밀림·쓰러짐을 검증한다.
4. 회피 버튼은 `safetyEnabled=true`로 같은 A→C→B를 실행한다. escape 후보 평가에 `basketPayloadSegments()`를 포함한 **동일 swept capsule contact**를 넣어, 실제 실행에서 C 근처 접촉이 나기 전에 후보를 탈락시킨다.
5. 사람이 `slowR` 밖에 있고 B에 도달하기 전에는 `PROCEED` 재개를 금지한다.
6. 화면 우측 회피 자세는 명시적 IK target 또는 검증된 관절 pose로 만들고, 후보 검사 후에만 선택한다. 기존 `potHover`를 `rightSafe`라는 별칭으로 두지 않는다.
7. Playwright에서 다음을 함께 검증한다.
   - 두 버튼의 사람 waypoint 배열이 동일하다.
   - 충돌: E-STOP, 위치 displacement, fall 상태.
   - 회피: stop radius 안에서 escape 관절 변화, contact 없음, B/slowR 밖 이탈 후 재개·완료.

## 작업 트리의 미커밋 변경

이 문서 커밋에는 포함하지 않는다. Claude가 읽고 판단할 수 있도록 남겨 둔다.

- `sim.html`: `directionalScenarioRoute`, 충돌을 주인공으로 바꾸는 코드, slow-radius clearing guard, route fallback guard.
- `tests/browser/directional-collision.spec.mjs`: 충돌 물리 브라우저 테스트.
- `tests/browser/lstm-active-yield.spec.mjs`: 실패 상태의 route 요약을 추가한 진단 출력.
- `tests/test_sim_safety_wiring.py`: 위 연결에 대한 소스 wiring assertion.

## 반성 및 인계 요청

저는 버튼을 먼저 분리한 뒤 경로·예측·실제 접촉 기하를 하나의 계약으로 통합하지 못했습니다. 그 결과 요구한 시연을 안정적으로 완성하지 못했고, 회피 검증이 통과한 실행과 실패한 실행이 공존하게 만들었습니다. 이 상태에서 추가로 밀어붙이는 것은 적절하지 않습니다.

Claude에게 부탁드립니다. 위 재현 값과 현재 작업 트리를 기준으로, 시나리오 경로와 안전 후보의 geometry를 하나로 통합해 주세요. 특히 “같은 A→B에서 충돌은 물리 접촉, 회피는 반대쪽 팔 움직임 후 무접촉 통과”를 단일 브라우저 회귀로 고정해 주시면 감사하겠습니다.
