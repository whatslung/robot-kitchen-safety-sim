# 서드파티 출처

이 저장소에 함께 배포되는 라이브러리와 3D 에셋의 출처.

## 라이브러리

| 파일 | 이름 / 버전 | 라이선스 |
|---|---|---|
| `babylon.js`, `babylonjs.loaders.min.js`, `babylon.inspector.bundle.js` | Babylon.js 9.20.0 | Apache-2.0 |
| `HavokPhysics_umd.js`, `HavokPhysics.wasm.js` | Havok Physics for Babylon.js (`@babylonjs/havok`) | MIT |
| `vendor/ort/*` | ONNX Runtime Web 1.20.1 (Microsoft) | MIT |

## 3D 에셋

`assets/`의 GLB 대부분은 `tools/blender_assets.py`, `tools/blender_env.py`로
절차 생성한 자체 제작물이다.

```
kitchen_static  lid  basket
robot_j1  robot_upper  robot_fore  robot_hand  robot_pedestal
env_table  env_rack  env_sink  env_bin  env_pancart  env_wallrack
env_kettle_nb  env_kettle_nb_closed  env_fridge  env_serve  env_basketcart
```

사람 캐릭터는 외부 소스 기반 파생 에셋이다.

| 파일 | 용도 |
|---|---|
| `person_cook_v5.glb` | 기본 조리원 — 본 67개, 애니메이션 45종 |
| `person_base_v1.glb` | `?cook=base` 비교용 원본 리그 |
| `cook_rig_v47.glb` | `?cook=old` 이전 캐릭터 |
