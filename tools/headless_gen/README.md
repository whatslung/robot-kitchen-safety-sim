# 분할 전 원본 시뮬레이터 캡처 생성

`gen.cjs`는 `sim.html`을 화면 밖 Chrome 창에서 실행하고, 6개 카메라의 이미지와 YOLO 라벨을 저장한다.
이 단계에서는 `train`, `val`, `test`를 나누지 않으며 `data.yaml`도 만들지 않는다. 같은 장면의 서로 다른
카메라 이미지가 학습과 평가에 섞이지 않도록, 장면 단위 분할은 후속 데이터 준비 단계에서 수행한다.

Chrome 창은 화면 밖에 두므로 작업 중 포그라운드를 차지하지 않는다. Playwright가 매 실행마다 임시
사용자 프로필을 만든다. 이미 열려 있는 Chrome과 그 로그인·탭·확장 프로그램은 사용하거나 종료하지
않는다. 완전한 headless 모드는 이 시뮬레이터의 WebGL 캡처가 단색으로 나오는 문제가 있어 사용하지 않는다.

## 준비

Node 의존성을 설치한다. 생성기가 현재 작업트리를 검증한 뒤 전용 로컬 서버를 임시 포트에 직접 연다.
별도의 `python -m http.server`나 `SIM_URL`은 사용하지 않는다.

```powershell
cd .\tools\headless_gen
npm ci
```

기본 Chrome 위치가 다르면 `CHROME_PATH`를 지정한다.

```powershell
$env:CHROME_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
```

## 실행

```powershell
# node gen.cjs <출력 폴더> <장면 수> [기준 seed]
node .\gen.cjs 'C:\dataset\sim-oblique-6cam-raw' 20 20260826
```

출력 폴더는 없거나 완전히 비어 있어야 한다. 기존 `manifest.json`, 이미지 또는 라벨이 있으면 이전
생성 기록을 보호하기 위해 시작 전에 중단한다. 이어서 생성하는 resume 모드는 아직 지원하지 않는다.

단색 판정 기준과 재시도 횟수는 각각 `MIN_CAPTURE_COLORS`, `CAPTURE_ATTEMPTS` 환경 변수로 바꿀 수 있다.

## 출력과 실패 처리

- `scenes/<scene>/images/<camera>_<scene>.png`: 원본 이미지
- `scenes/<scene>/labels/<camera>_<scene>.txt`: 해당 이미지의 YOLO 라벨
- `manifest.json`: Git 커밋, 작업트리 변경 여부, 입력 파일 SHA-256, Chrome과 Playwright 버전,
  기준 seed와 장면 seed, 센서 조건, 각 결과 파일 SHA-256, 실행 상태

각 장면은 6개 카메라를 같은 고정 시점에서 함께 생성한다. 장면 식별자는
`(generation.baseSeed, scene.seed)`의 복합값이다. 같은 입력 fingerprint, Chrome·Playwright 버전과
이 복합 seed를 사용하면 기록된 장면 조건이 같아야 한다. WebGL 픽셀 경계 차이를 고려해 영상은
`rgb16x16-v1` RGB 축소 서명(평균 8/255·95백분위 32/255 이하), fire·smoke를 제외한 라벨은
person 0.75%·나머지 클래스 3% 이내로 실제 Chrome 통합 테스트한다. 연기 경계에서 최소 픽셀 수를
넘나드는 비-person 성분은 클래스별 1개 차이까지 허용한다. 파일 SHA-256은 재생성 간 동일성 판정이 아니라 각 파일의
무결성 확인에 쓴다.

단색으로 의심되는 캡처는 파일을 쓰기 전에 거부하고 다시 렌더한다. 6개 카메라는 임시 장면 폴더에서
모두 완성된 뒤 한 번에 공개된다. 어느 카메라에서든 실패하면 해당 장면의 임시·공개 파일을 전부
되돌리므로 manifest 밖의 부분 장면이 남지 않는다. 강제 종료 중에는 `.incomplete-scene-*` 표식으로
불완전 장면을 식별할 수 있다. 기존 결과 파일과 manifest는 덮어쓰지 않는다.

완료된 `manifest.json`의 `split` 값은 `null`이다. 후속 단계에서는 이 장면 번호를 기준으로
학습·검증·평가를 분리한 뒤 별도의 `data.yaml`을 만들어야 한다.
