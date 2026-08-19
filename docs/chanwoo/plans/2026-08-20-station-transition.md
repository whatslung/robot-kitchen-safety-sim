# 스테이션 전이 모델 구현 계획

> **에이전트 작업자용:** 필수 하위 스킬 — `superpowers:executing-plans` 또는
> `superpowers:subagent-driven-development`로 태스크 단위로 구현하라.

**목표:** `sim.html`에 세 번째 이동 모드(직무 사이클 기반 확률 전이)를 추가해,
학습형 궤적 예측기가 잡아낼 **불균등·문맥의존 구조**를 갖는 사람 동선을 만든다.

**접근:** 기존 `wanderTarget`(균등 랜덤)과 `jobNextTarget`(결정적 순회)을 **건드리지 않고**
새 함수 `workflowTarget()`을 추가한다. 전역 토글 `WF.on`이 켜져 있을 때만
주인공(`wanderStep`)과 추가 인원(`extraNextTarget`)이 새 경로를 탄다.
꺼져 있으면 기존과 동일하게 동작한다 — 기존 데이터셋 재현성 보존.

**기술 스택:** 바닐라 JS + Babylon.js 단일 파일(`sim.html`). 테스트 러너가 없으므로
검증은 브라우저 콘솔에서 `__sim.*`를 직접 호출하는 **실측 스크립트**로 한다.

**스펙:** `docs/chanwoo/specs/2026-08-20-station-transition-design.md`

## 전역 제약

- `wanderTarget()` · `jobNextTarget()` · `JOB` · `ROUTE` · `ROLE` 는 **수정 금지**.
- 난수는 반드시 전달받은 `rng`(사람별 `makeRNG`)를 쓴다. `Math.random()` 금지 —
  `?seed=N` 재현이 깨진다.
- 전이 확률·τ는 상수 객체 `WF`에 모아 둔다(하드코딩 금지) — 3단계 베이스라인이
  같은 값을 읽어야 한다.
- 스테이션 키 존재 여부는 런타임에 확인한다. `nkettle`·`ntable`·`nrack`은
  `ISLAND` 배치에서만 `STATION`에 들어간다.
- 새 시각화를 만들면 `vizMeshList()`에 등록해야 한다(이번 계획에는 시각화 없음).

## 파일 구조

`sim.html` 한 파일만 수정한다. 삽입 위치는 책임별로 나눈다:

| 무엇 | 어디 |
|---|---|
| `WORKFLOW` 테이블 + `WF` 상수 | `JOB` 정의 직후(현 3065행 부근) — 직무 어휘가 모여 있는 곳 |
| `wfPlan` / `wfStepIndex` / `wfWeightedPick` / `workflowTarget` | 위 테이블 바로 아래 |
| `wanderStep` 분기 · `personWorkflow` | 현 `wanderStep`(3187행) · `personWander`(3297행) 부근 |
| `extraNextTarget` 분기 | 현 10455행 `if (P.job)` 줄 |
| 버튼 | 현 439행 `scWander` 옆 · 8825행 리스너 옆 |
| `__sim` export | 현 9734~9740행 |

---

### Task 1: `WORKFLOW` 테이블과 전이 함수

**파일:** 수정 `sim.html` (`JOB` 정의 직후 · 현 3065행 부근)

**인터페이스:**
- 사용: `STATION`, `RNG`, `makeRNG`, `NAV`, `navReachable`
- 제공:
  - `WORKFLOW` — `{ [job]: { cycle: string[][] } }`
  - `WF` — `{ on:boolean, P_NEXT:0.65, P_NEAR:0.30, TAU:2.0 }`
  - `wfPlan(job) -> { steps:string[][], set:string[], outside:string[] } | null`
  - `workflowTarget(job, curKey, from, rng) -> { key, pos, act, dwell, label } | null`

- [ ] **1단계: 테이블과 상수 추가**

`JOB` 정의가 끝나는 `};` 다음, `jobRole` 앞에 삽입:

```js
/* ══ 스테이션 전이 모델 (WORKFLOW) ══════════════════════════════════════════
   `wanderTarget`은 21개 스테이션을 균등 랜덤으로 뽑는다 — 엔트로피가 최대라
   **학습할 구조가 없다**(실측 평균 이동 5.51m vs 실제 동선 0.98~2.8m).
   `JOB.route`는 반대쪽 극단이다 — 완전히 결정적이라 예측할 게 없다.

   그 중간을 만든다: 직무 사이클을 **기준선**으로 두고 확률적으로 벗어난다.
   같은 스테이션에 서 있어도 직무와 사이클 진행도에 따라 다음 목표의 확률이
   달라지는 것 — 그게 학습형 예측기가 휴리스틱을 이길 수 있는 유일한 여지다.

   `JOB.route`와 따로 두는 이유: `JOB`은 추가 인원의 **재현 가능한 결정적 순회**이고
   기존 데이터셋 생성이 그것에 의존한다. 여기에 확률을 섞으면 그 회차들이 깨진다. */
const WF = { on:false, P_NEXT:0.65, P_NEAR:0.30, TAU:2.0 };
/* 사이클의 한 단계는 **키 배열**이다. 길이가 2 이상이면 그 단계에서 택일한다
   (전처리 담당이 중앙 준비대 두 자리 중 가까운 쪽으로 가는 것 — 실제로 그렇게 일한다). */
const WORKFLOW = {
  prep : { cycle:[["fridge"], ["wrack"], ["island"], ["isle1","isle2"], ["prep"]] },
  cook : { cycle:[["prep"], ["kettle"], ["panel"], ["nkettle"], ["ntable"]] },
  wash : { cycle:[["sink"], ["store"], ["nrack"]] },
  carry: { cycle:[["serve"], ["isle3"], ["etable"], ["cartPick"], ["door"]] },
  lead : { cycle:[["panel"], ["robotside"], ["aisle"], ["prep"]] },
};
```

- [ ] **2단계: `wfPlan` — 직무별 계획 해석(캐시)**

`WORKFLOW` 바로 아래:

```js
/* 직무 계획 해석 — 없는 스테이션은 버린다.
   `nkettle`·`ntable`·`nrack`은 ISLAND 배치에서만 STATION에 들어오므로
   테이블을 그대로 믿으면 조리·세정 담당이 undefined를 향해 걷는다. */
const WF_PLAN = {};
function wfPlan(job) {
  if (WF_PLAN[job]) return WF_PLAN[job];
  const W = WORKFLOW[job];
  if (!W) return null;
  const steps = W.cycle.map(s => s.filter(k => STATION[k])).filter(s => s.length);
  if (!steps.length) return null;
  const set = steps.flat(), inSet = new Set(set);
  const outside = Object.keys(STATION).filter(k => !inSet.has(k));
  return (WF_PLAN[job] = { steps, set, outside });
}
/* 길이 막힌 스테이션은 후보에서 뺀다 — roleStations와 같은 규약.
   고르고 나서 실패하면 그 턴을 버리게 되고, 사람이 제자리에 서 있는 프레임이 남는다. */
const wfReach = k => !NAV.ready || navReachable(STATION[k].pos);
```

- [ ] **3단계: 거리 가중 선택과 현재 단계 판정**

```js
/* 거리 가중 — w(s) = exp(-d/τ), τ=2m. 가까운 곳을 선호하되 먼 곳도 가끔 간다.
   최근접 하나로 고정하면 결정적이 되고, 균등으로 두면 구조가 사라진다. */
function wfWeightedPick(keys, from, rng) {
  if (!keys.length) return null;
  if (keys.length === 1) return keys[0];
  const w = keys.map(k => Math.exp(
    -Math.hypot(STATION[k].pos.x - from.x, STATION[k].pos.z - from.z) / WF.TAU));
  const sum = w.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) return rng.pick(keys);
  let r = rng.next() * sum;
  for (let i = 0; i < keys.length; i++) { r -= w[i]; if (r <= 0) return keys[i]; }
  return keys[keys.length - 1];
}
/* 현재 사이클 단계 — 사이클 밖에 서 있으면(모드 시작 직후·이탈 후) **가장 가까운 단계**로
   붙인다. -1로 두면 65% 분기가 죽어 구조 없는 근거리 랜덤으로 퇴화한다. */
function wfStepIndex(plan, curKey, from) {
  for (let i = 0; i < plan.steps.length; i++)
    if (plan.steps[i].indexOf(curKey) >= 0) return i;
  let best = 0, bd = Infinity;
  for (let i = 0; i < plan.steps.length; i++) for (const k of plan.steps[i]) {
    const d = Math.hypot(STATION[k].pos.x - from.x, STATION[k].pos.z - from.z);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
```

- [ ] **4단계: `workflowTarget`**

```js
/* 다음 목표 — 반환 형태를 wanderTarget과 같게 맞춘다(+ 추적용 key).
     p=0.65  사이클의 다음 단계          ← 구조(학습 가능)
     p=0.30  직무 집합 안 거리 가중       ← 현실성
     p=0.05  직무 밖 이탈                ← 예외
   확률은 WF에 있다 — 3단계 베이스라인이 같은 값을 읽어야 한다. */
function workflowTarget(job, curKey, from, rng) {
  const plan = wfPlan(job);
  if (!plan) return null;
  const R = rng || RNG;
  const ok = pool => pool.filter(k => k !== curKey && wfReach(k));
  const i = wfStepIndex(plan, curKey, from);
  const u = R.next();
  let key = null;
  if (u < WF.P_NEXT) {
    key = wfWeightedPick(ok(plan.steps[(i + 1) % plan.steps.length]), from, R);
  } else if (u < WF.P_NEXT + WF.P_NEAR) {
    key = wfWeightedPick(ok(plan.set), from, R);
  } else {
    key = wfWeightedPick(ok(plan.outside), from, R);
  }
  if (!key) key = wfWeightedPick(ok(plan.set), from, R);   // 세 분기 다 비면 집합 전체
  if (!key) return null;
  const s = STATION[key];
  return { key, pos:s.pos.clone(), act:s.act,
           dwell:s.dwell * R.range(0.6, 1.5), label:s.label };
}
```

- [ ] **5단계: `__sim`에 export** (현 9739행 `JOB, SCENARIO, ...` 줄에 추가)

```js
  WF, WORKFLOW, wfPlan, workflowTarget,
```

- [ ] **6단계: 실측 — 히스토그램이 불균등한지**

브라우저에서 시뮬을 열고(`?seed=7`) 콘솔에 붙여넣는다:

```js
const S = __sim, from = S.STATION.island.pos, r = S.makeRNG(11), h = {};
for (let i = 0; i < 4000; i++) { const t = S.workflowTarget("prep", "island", from, r); h[t.key] = (h[t.key]||0)+1; }
console.table(Object.entries(h).sort((a,b)=>b[1]-a[1]).map(([k,n])=>({키:k, 비율:+(n/4000).toFixed(3)})));
```

기대: `isle1`+`isle2`(다음 단계) 합이 **0.60~0.70**, 최상위와 최하위 비율 차가 10배 이상.
모든 키가 0.04~0.06 근처면 **실패**(균등) — 분기 조건을 다시 본다.

- [ ] **7단계: 커밋**

```bash
git add sim.html docs/chanwoo && git commit -m "sim: 직무 사이클 기반 스테이션 전이 모델 — 균등 랜덤에는 학습할 구조가 없다"
```

---

### Task 2: 주인공(`wanderStep`)에 모드 연결

**파일:** 수정 `sim.html` (현 3187행 `wanderStep` · 3297행 `personWander` · 439행 버튼 · 8825행 리스너)

**인터페이스:**
- 사용: Task 1의 `WF`, `workflowTarget`; 기존 `WANDER`, `person`, `personGo`, `RNG`
- 제공: `personWorkflow(on, job)` · `WANDER.job` 필드

- [ ] **1단계: `WANDER`에 `job` 필드 + `wanderStep`에 분기**

```js
const WANDER = { on:false, last:null, count:0, job:null };
```

```js
function wanderStep() {
  /* WF 모드가 꺼져 있으면 기존 경로 그대로다 — 기존 데이터셋 회차가 재현돼야 한다. */
  let t = null;
  if (WF.on && WANDER.job)
    t = workflowTarget(WANDER.job, WANDER.last, person.node.position, RNG);
  if (t) WANDER.last = t.key;
  else t = wanderTarget();
  WANDER.count++;
  person.stations = [t];
  personGo([t.pos], { speed: RNG.range(0.42, 0.88) });
}
```

- [ ] **2단계: `personWorkflow` 추가** (`personWander` 정의 바로 아래)

```js
/* 직무 사이클 모드 — personWander와 같은 뼈대를 쓰고 목표 선택만 바꾼다.
   직무 배정이 없으면 WORKFLOW 첫 직무를 준다 — 조용히 균등 랜덤으로 폴백하면
   "켰는데 아무것도 안 달라졌다"가 된다. */
function personWorkflow(on, job) {
  WF.on = on;
  WANDER.job = on ? (job || WANDER.job || Object.keys(WORKFLOW)[0]) : null;
  personWander(on);
  if (on) setStatus("직무 전이 모드 — " + ((JOB[WANDER.job] || {}).label || WANDER.job)
                    + " · seed " + RNG.seed + " (?seed=" + RNG.seed + " 로 재현)");
}
```

`__sim` export에 `personWorkflow` 추가.

- [ ] **3단계: 버튼** — 439행 `scWander` 다음 줄, 8825행 리스너 다음 줄

```html
    <button id="scWorkflow">🧭 직무 전이 모드 시작/중지</button>
```

```js
$("scWorkflow").addEventListener("click", () => personWorkflow(!WF.on));
```

- [ ] **4단계: 실측 — 평균 이동거리 · 직무별 집합 · 재현성**

```js
const S = __sim;
const run = job => { const r = S.makeRNG(7); let cur = S.wfPlan(job).set[0], d = 0, keys = [];
  for (let i = 0; i < 600; i++) { const from = S.STATION[cur].pos;
    const t = S.workflowTarget(job, cur, from, r);
    d += Math.hypot(t.pos.x - from.x, t.pos.z - from.z); cur = t.key; keys.push(t.key); }
  return { 직무:job, 평균이동m:+(d/600).toFixed(2), 방문스테이션수:new Set(keys).size,
           첫10:keys.slice(0,10).join(">") }; };
console.table(["prep","cook","wash","carry","lead"].map(run));
// 균등 랜덤 대조군 — 5.51m가 재현되는지 같은 방식으로 재다
const ks = Object.keys(S.STATION), r0 = S.makeRNG(7); let cur0 = "island", d0 = 0;
for (let i = 0; i < 4000; i++) { const c = ks.filter(k => k !== cur0);
  const nk = c[(r0.next()*c.length)|0]; const a = S.STATION[cur0].pos, b = S.STATION[nk].pos;
  d0 += Math.hypot(b.x-a.x, b.z-a.z); cur0 = nk; }
console.log("균등 랜덤 평균 이동", (d0/4000).toFixed(2), "m");
```

기대:
- 각 직무의 `평균이동m`이 **2~3 m**, 균등 랜덤 대조군은 5.5 m 근처
- 직무마다 `첫10`이 **서로 다른 스테이션 집합**
- 같은 스크립트를 두 번 돌리면 `첫10`이 **글자 단위로 동일**(재현)

3 m를 넘으면 τ 또는 사이클 구성을 다시 본다 — 추정으로 넘기지 말고 재측정한다.

- [ ] **5단계: 브라우저에서 실제로 걸려 보기 (끼임·관통 확인)**

버튼을 눌러 90초 돌린 뒤:

```js
const P = __sim.person; console.log(P.mode, __sim.WANDER.count, P.node.position);
```

기대: `WANDER.count`가 계속 오르고 사람이 설비를 **돌아서** 간다.
`count`가 멈추면 도달 판정이 안 되는 스테이션이 있다는 뜻 — `verifyRoute()`로 확인한다.

- [ ] **6단계: 커밋**

```bash
git add sim.html && git commit -m "sim: 주인공에 직무 전이 모드 연결 — 버튼·상태표시·시드 재현"
```

---

### Task 3: 추가 인원(`extraNextTarget`)에 모드 연결

**파일:** 수정 `sim.html` (현 10455행)

**인터페이스:**
- 사용: Task 1의 `WF`, `workflowTarget`; 기존 `P.job`, `P.rng`, `P.station`, `setPathTo`, `EXTRAS`
- 제공: 없음(기존 진입점 재사용)

궤적 데이터의 양은 추가 인원에서 나온다. 주인공만 바꾸면 1인분만 얻는다.

- [ ] **1단계: `jobNextTarget` 호출 앞에 분기 삽입**

현 10455행 `if (P.job) { const t = jobNextTarget(P, rng); if (t) return t; }` 를 아래로 바꾼다:

```js
  /* WF 모드면 결정적 순회 대신 확률 전이를 쓴다. 꺼져 있으면 기존 경로 그대로 —
     기존 데이터셋 회차가 jobNextTarget의 결정적 순서에 의존한다. */
  if (P.job && WF.on) {
    const here = P.station && P.station.key;
    for (let tries = 0; tries < 4; tries++) {
      const t = workflowTarget(P.job, here, P.root.position, rng);
      if (!t) break;
      /* 한 자리에 두 명은 못 선다 — jobNextTarget과 같은 규약. 다시 뽑는다. */
      if (EXTRAS.some(o => o !== P && o.station && o.station.key === t.key)) continue;
      if (setPathTo(P, t.pos)) {
        P.station = Object.assign({ dist:0 }, STATION[t.key], { key:t.key, dwell:t.dwell });
        return P.station;
      }
    }
  }
  if (P.job) { const t = jobNextTarget(P, rng); if (t) return t; }
```

- [ ] **2단계: 실측 — 인원별로 다른 집합을 도는가**

```js
__sim.runScenario("prepWash"); __sim.WF.on = true;
const seen = {}; setInterval(() => __sim.EXTRAS.forEach(P => {
  const k = P.station && P.station.key; if (!k) return;
  (seen[P.job] = seen[P.job] || new Set()).add(k); }), 500);
// 120초 뒤
console.table(Object.entries(seen).map(([j,s]) => ({ 직무:j, 방문:[...s].join(",") })));
```

기대: 직무마다 방문 집합이 **겹치지 않게 갈린다**(공용 스테이션 몇 개는 겹쳐도 된다).

- [ ] **3단계: 회귀 — WF를 끄면 기존과 같은가**

```js
__sim.WF.on = false; __sim.EXTRAS.forEach(P => { P.routeIdx = -1; });
```
120초 뒤 각 인원이 `JOB[P.job].route` **순서대로** 도는지 확인한다(무작위면 회귀).

- [ ] **4단계: 커밋**

```bash
git add sim.html && git commit -m "sim: 추가 인원도 직무 전이 모드를 쓴다 — 궤적 데이터는 여기서 나온다"
```

---

### Task 4: 검증 결과 문서화 + PR

- [ ] **1단계:** 스펙의 검증 체크리스트 5항목에 **실측값**을 적는다
      (`docs/chanwoo/specs/2026-08-20-station-transition-design.md` 하단).
      "확인했다"가 아니라 숫자를 적는다.
- [ ] **2단계:** `docs/chanwoo/HANDOFF.md` 갱신 — 1단계 완료, 다음은 2단계(궤적 수집).
- [ ] **3단계:** 커밋 후 PR. **base는 `main`이 아니라 `chanwoo/camera-strategy-2026-08-17`**
      (PR #1이 아직 열려 있다). PR #1이 먼저 병합되면 그때 `main`으로.

```bash
gh pr create --base chanwoo/camera-strategy-2026-08-17 --title "sim: 직무 사이클 기반 스테이션 전이 모델 (이슈 #2 1단계)"
```

---

## 자기 점검

- **스펙 커버리지**: ① 직무별 집합·사이클 → Task 1-1. ② 전이 규칙 0.65/0.30/0.05 → Task 1-4.
  ③ 체류·동작 재사용 → Task 1-4(`dwell`·`act`), A\* → Task 3-1(`setPathTo`) ·
  Task 2-1(`personGo`), 사람별 시드 → Task 1-4(`rng` 인자) · Task 3-1(`P.rng`).
  진입점·버튼 → Task 2-2/2-3. 검증 5항목 → Task 1-6(히스토그램) ·
  Task 2-4(이동거리·직무별·재현) · Task 2-5(끼임).
- **자리표시자**: 없음. 모든 코드 단계에 실제 코드가 들어 있다.
- **타입 일관성**: `workflowTarget(job, curKey, from, rng)` → `{key, pos, act, dwell, label}`,
  `wfPlan(job)` → `{steps, set, outside}`. `WF.on`·`WANDER.job` 이름이 Task 2·3에서 동일하다.
