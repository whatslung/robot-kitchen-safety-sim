# Person-only Ground Truth Design

## Goal

The simulator currently needs to train and validate only person detection. Ground-truth images and YOLO labels must therefore expose one class: `0 person`.

## Design

- Keep RGB rendering unchanged so fire, smoke, robots, kettles, and equipment remain useful background and occlusion variation.
- Change `GT_CLASSES` to the single `person` class.
- During instance-mask rendering, assign palette colors only to meshes classified as `person`; render every other mesh and all particles as the background color.
- Preserve a distinct instance color for each visible person so multiple workers still produce separate YOLO boxes.
- Decode instance colors by exact palette color rather than nearest color. Full-color interior person pixels determine the box; MSAA-blended edge pixels are ignored instead of being reassigned to another person.
- Keep the minimum 80-pixel visibility threshold.

## Output contract

- `classes.txt`: `person`
- `dataset.json.classes`: `["person"]`
- Every YOLO label row starts with class id `0`.
- `meta.classes` contains only `person`.
- Non-person objects never appear in `labels`, `labelText`, or `instances`.

## Verification

- A browser regression test checks the one-class contract and person-only labels.
- A synthetic browser assertion checks that an MSAA-like blended color is rejected by the exact decoder.
- The full Python suite must remain green.
- RGB, instance mask, and drawn YOLO boxes are visually checked on at least two simulator cameras.
