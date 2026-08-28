# 서드파티 출처

이 저장소에 함께 배포되는 라이브러리와 3D 에셋의 출처.

## 라이브러리

| 파일 | 이름 / 버전 | 라이선스 |
|---|---|---|
| `babylon.js`, `babylonjs.loaders.min.js`, `babylon.inspector.bundle.js` | Babylon.js 9.20.0 | Apache-2.0 |
| `HavokPhysics_umd.js`, `HavokPhysics.wasm.js` | Havok Physics for Babylon.js (`@babylonjs/havok`) | MIT |
| `vendor/ort/*` | ONNX Runtime Web 1.20.1 (Microsoft) | MIT |
| `assets/person_cook_v5.glb` 의 베이스 리그·애니메이션 | [BJS Character Controller V2](https://github.com/crazyramirez/BJS_Character_Controller_V2) (crazyramirez) | MIT |

## 3D 에셋

`assets/`의 GLB 대부분은 `tools/blender_assets.py`, `tools/blender_env.py`로
절차 생성한 자체 제작물이다.

```
kitchen_static  lid  basket
robot_j1  robot_upper  robot_fore  robot_hand  robot_pedestal
env_table  env_rack  env_sink  env_bin  env_pancart  env_wallrack
env_kettle_nb  env_kettle_nb_closed  env_fridge  env_serve  env_basketcart
```

`assets/food_heaps.glb` (선반·바트·소쿠리에 담기는 식자재 더미 4종)는 아래 CC-BY 에셋에서
파생했다. **저작자 표기가 라이선스 의무이므로 이 문단을 지우고 배포하면 안 된다.**

> This work is based on "FREE | Fruits and Vegetables Pack (CS2)"
> (https://sketchfab.com/3d-models/free-fruits-and-vegetables-pack-cs2-6706dd502e214389adc3245a3b019258)
> by 6lucius (https://sketchfab.com/6lucius)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

원본 채소 9종 중 **감자·양파(노랑)·양배추·애호박·피망**을 골라 썼다.

**당근**은 아래 별도 스캔에서 가져왔다:

> This work is based on "Carrot(3d Scan)"
> (https://sketchfab.com/3d-models/896181392c1f4e2d9316bf2eeafc3042)
> by tanophotogrammetry (https://sketchfab.com/tanophotogrammetry)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

여섯 종을 각각 더미로 쌓고, 복셀 리메시로 연속 표면화한 1,000삼각형 저폴리에
BaseColor·Normal·AO 를 구웠다(6종이 아틀라스 1장·머티리얼 1개 공유).
원본 메시·텍스처는 배포물에 포함되지 않는다.

사람 캐릭터 `person_cook_v5.glb` 는 위 BJS Character Controller V2(MIT)의 베이스에서
파생했다. 지오메트리와 mixamorig 67본 스켈레톤·애니메이션 45종은 그대로 두고,
재질을 위생복(상의·바지·신발·위생 두건)으로 바꾸고 마스크 메시를 추가했다.
