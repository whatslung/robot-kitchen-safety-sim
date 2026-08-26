# 분할 전 원본 시뮬레이터 캡처 생성

`gen.cjs`는 `sim.html`을 화면 밖 Chrome 창에서 실행하고, 6개 카메라의 이미지와 YOLO 라벨을 저장한다.
이 단계에서는 `train`, `val`, `test`를 나누지 않으며 `data.yaml`도 만들지 않는다. 같은 장면의 서로 다른
카메라 이미지가 학습과 평가에 섞이지 않도록, 장면 단위 분할은 후속 데이터 준비 단계에서 수행한다.

Chrome 창은 화면 밖에 두므로 작업 중 포그라운드를 차지하지 않는다. 완전한 headless 모드는 이
시뮬레이터의 WebGL 캡처가 단색으로 나오는 문제가 있어 사용하지 않는다.

## 준비

PowerShell 창 두 개를 사용한다. 첫 번째 창에서 저장소 루트를 정적 서버로 연다.

```powershell
python -m http.server 8123 --bind 127.0.0.1
```

두 번째 창에서 Node 의존성을 설치한다.

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

서버 주소가 다르면 `SIM_URL`을 지정할 수 있다. 단색 판정 기준과 재시도 횟수는 각각
`MIN_CAPTURE_COLORS`, `CAPTURE_ATTEMPTS` 환경 변수로 바꿀 수 있다.

## 출력과 실패 처리

- `images/<camera>_<scene>.png`: 원본 이미지
- `labels/<camera>_<scene>.txt`: 해당 이미지의 YOLO 라벨
- `manifest.json`: Git 커밋, 작업트리 변경 여부, 입력 파일 SHA-256, Chrome과 Playwright 버전,
  기준 seed와 장면 seed, 센서 조건, 각 결과 파일 SHA-256, 실행 상태

각 장면은 하나의 seed로 6개 카메라를 함께 생성한다. 단색으로 의심되는 캡처는 파일을 쓰기 전에
거부하고 다시 렌더한다. 제한 횟수까지 실패하면 실행 상태가 `failed`로 기록되며, 이미지와 라벨은
한 쌍이 모두 준비된 경우에만 저장된다. 저장 중에는 `.incomplete-*` 표식을 두므로 강제 종료 뒤 남은
불완전 결과를 식별할 수 있다. 기존 결과 파일과 manifest는 덮어쓰지 않는다.

완료된 `manifest.json`의 `split` 값은 `null`이다. 후속 단계에서는 이 장면 번호를 기준으로
학습·검증·평가를 분리한 뒤 별도의 `data.yaml`을 만들어야 한다.
