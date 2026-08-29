# A4 Safety AI Overview PPT Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the current A4 portrait PowerPoint into a focused one-page overview that features the supplied motion and fire pipeline images and adds CCTV monitoring, dense-crowd prediction, and steam/smoke fire applications.

**Architecture:** Import the current PPTX with artifact-tool, duplicate its only slide, preserve the inherited header artwork, and explicitly remove/rewrite the inspected body elements. Recompose the body into a thesis block, a large motion visual, a supporting fire visual, and compact design/application/technology sections. Crop risky claims from the supplied images inside PowerPoint rather than editing the image files.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation template-following scripts, `slides_test.py` for overflow QA.

---

## File Structure

- Source: `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_작성본.pptx`
- Source asset: `E:/다운로드/motion.jpg`
- Source asset: `E:/다운로드/fire.jpg`
- Create: `.codex-pptx-final-20260823/template-audit.txt` — source layout and edit contract
- Create: `.codex-pptx-final-20260823/template-frame-map.json` — exact inherited element actions
- Create: `.codex-pptx-final-20260823/deviation-log.txt` — explicit redesign deviations
- Create: `.codex-pptx-final-20260823/build-final.mjs` — artifact-tool authoring module
- Output: `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_최종본.pptx`

### Task 1: Inspect the Current Deck and Assets

**Files:**
- Read: `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_작성본.pptx`
- Read: `E:/다운로드/motion.jpg`
- Read: `E:/다운로드/fire.jpg`
- Create: `.codex-pptx-final-20260823/template-audit.txt`
- Create: `.codex-pptx-final-20260823/template-frame-map.json`
- Create: `.codex-pptx-final-20260823/deviation-log.txt`

- [ ] **Step 1: Create a fresh build directory and copy the source deck**

Create `.codex-pptx-final-20260823` under the repository and copy the source as `source-current.pptx`. Do not overwrite either source file or image.

- [ ] **Step 2: Inspect and render the only source slide**

Run `inspect_template_deck.mjs` with the bundled Node runtime and review the PNG, layout JSON, and element inventory. Record every slide-local element below the header by exact artifact-tool ID.

- [ ] **Step 3: Write the exact edit map**

Map output slide 1 to source slide 1 with `reuseMode: "duplicate-slide"`. Mark the title and subtitle as `rewrite-and-reposition`; mark every inspected element below y=220 as `delete`; allow bounded new primitives only in these zones:

```json
[
  { "left": 57, "top": 224, "width": 680, "height": 112, "role": "thesis" },
  { "left": 57, "top": 346, "width": 680, "height": 222, "role": "motion visual" },
  { "left": 57, "top": 580, "width": 680, "height": 190, "role": "fire visual" },
  { "left": 57, "top": 786, "width": 326, "height": 142, "role": "design priorities" },
  { "left": 411, "top": 786, "width": 326, "height": 142, "role": "applications" },
  { "left": 57, "top": 946, "width": 680, "height": 116, "role": "technology strip" }
]
```

- [ ] **Step 4: Build and verify the starter deck**

Run `prepare_template_starter_deck.mjs` and confirm the starter contains one duplicated portrait slide with the inherited header preserved.

### Task 2: Author the Focused A4 Layout

**Files:**
- Create: `.codex-pptx-final-20260823/build-final.mjs`
- Output: `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_최종본.pptx`

- [ ] **Step 1: Start the artifact edit operation exactly once**

Run:

```powershell
& $env:RUNTIME_NODE "$SKILL_DIR/container_tools/mark_artifact_operation_started.mjs" --operation-kind edit --expected-output-count 1 --output-format pptx
```

Expected: exit code 0 before the first authoring command.

- [ ] **Step 2: Import the starter and remove only mapped body elements**

The authoring module must import `template-starter.pptx`, resolve the exact IDs recorded in the map, and call `delete()` only for mapped body elements. It must preserve the inherited header image and title/subtitle objects.

- [ ] **Step 3: Rewrite and reposition the header**

Use this exact visible copy:

```js
const copy = {
  title: "가상에서 화재를 배우고,\n사람의 미래를 읽는 조리로봇 안전 AI",
  slogan: "불은 번지기 전에, 사람은 부딪히기 전에",
  subtitle: "합성 위험 학습 × 다중 미래경로 예측 × 로컬 선제 제어",
};
```

Make the title an intentional two-line block inside the header. Keep the title white and bold; keep the slogan and subtitle secondary.

- [ ] **Step 4: Add the thesis section**

Add the following copy inside the bounded thesis zone:

```js
const thesis = {
  lead: "사고 데이터는 없지만, 로봇은 기다릴 수 없습니다.",
  body: "없는 위험은 가상에서 만들고, 작업자의 다음 경로를 읽어 위험이 닿기 전에 조리로봇을 감속·정지시킵니다.",
};
```

The lead is the first black text after the header and must visually dominate the body.

- [ ] **Step 5: Add the motion image as the primary visual**

Add `motion.jpg` at `{ left: 57, top: 372, width: 680, height: 188 }` with a section label at y=346:

```js
"01 사람 미래예측 — 움직임을 읽어 로봇을 먼저 멈춥니다"
```

Use a bottom crop of at least `0.23` so the image does not display `ADE 1.11→0.43` or `zero-shot sim-to-real 성공`. The visible crop must retain YOLO11s, ByteTrack, LSTM/MTP, K-path prediction, and SSM response.

- [ ] **Step 6: Add the fire image as supporting evidence**

Add `fire.jpg` at `{ left: 57, top: 606, width: 680, height: 154 }` with a section label at y=580:

```js
"02 화재 합성 연구 — 수증기·연기 환경의 조기감지로 확장합니다"
```

Use `crop: { left: 0, top: 0.08, right: 0, bottom: 0.45 }` with `fit: "contain"` so `수증기 오탐 88%→1.3%` is not visible. Retain the real kitchen background, flame compositing, noise augmentation, YOLO, and fire detection flow.

- [ ] **Step 7: Add design priorities and applications**

Use these exact lists:

```js
const designPriorities = [
  "반응보다 예측 — 다음 경로로 위험 판단",
  "촬영 대신 합성 — 희귀 사고를 가상 학습",
  "성과와 한계 — 검증 수치와 연구과제 구분",
];

const applications = [
  "CCTV 기반 산업현장 안전관제",
  "밀집 군중의 다중 이동경로 예측",
  "수증기·연기 환경 화재 조기감지·예측 연구",
];
```

Use separate headings `설계의 주안점` and `활용 분야`. Do not present crowd prediction or fire prediction as completed current performance; retain the application/research wording.

- [ ] **Step 8: Add the technology strip and source notes**

Use this compact stack:

```js
"YOLO11s · ByteTrack · 멀티모달 LSTM(K=3) · ISO 기반 SSM · Babylon.js/Havok · ONNX Runtime Web · 로컬 PyTorch API"
```

Add speaker notes containing:

```text
[Sources]
https://github.com/K-H-MOON/kitchen-fire-noise-poc/blob/main/docs/SUMMARY_meeting.md
https://github.com/whatslung/robot-kitchen-safety-sim
E:/다운로드/fire.jpg
E:/다운로드/motion.jpg
```

- [ ] **Step 9: Export the revised copy**

Export through `PresentationFile.exportPptx` to `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_최종본.pptx`. The original and previous draft must remain unchanged.

### Task 3: Render and Verify the Final Page

**Files:**
- Read: `E:/다운로드/2024_SACP_4팀_조리로봇_안전AI_개요서_최종본.pptx`
- Create: `.codex-pptx-final-20260823/final-render/final-slide-01.png`
- Create: `.codex-pptx-final-20260823/final-layout/final-slide-01.layout.json`

- [ ] **Step 1: Render the final slide at high resolution**

Render the only slide and inspect it at full size. Confirm both images remain legible and the motion image has greater visual weight.

- [ ] **Step 2: Verify claim-safe crops**

Check the rendered page for these forbidden strings and visual regions:

```text
88% → 1.3%
ADE 1.11 → 0.43
zero-shot sim-to-real 성공
```

Expected: none is visible in the final page.

- [ ] **Step 3: Run template fidelity QA**

Run `check_template_fidelity.mjs` against the starter and final decks. Expected: `status: pass` and `issueCount: 0`.

- [ ] **Step 4: Run overflow QA**

Run `slides_test.py` with the bundled Python runtime and all three required runtime environment variables. Expected:

```text
Test passed. No overflow detected.
```

- [ ] **Step 5: Inspect structural placeholders**

Inspect `ppt/slides/slide1.xml` in the final PPTX. Expected: no empty `<p:ph>` slide placeholders.

- [ ] **Step 6: Final visual checklist**

Confirm:

- Title, thesis, motion visual, and applications read in that order.
- `CCTV 기반 산업현장 안전관제` is visible.
- `밀집 군중의 다중 이동경로 예측` is visible.
- `수증기·연기 환경 화재 조기감지·예측 연구` is visible.
- No text overlaps, clips, or wraps unexpectedly.
- The final output is the only deliverable outside the build directory.
