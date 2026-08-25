# 헤드리스 데이터셋 생성 (백그라운드, 창 불필요)

브라우저 창을 띄워 두지 않고도(포그라운드 유지 불필요) sim.html에서 YOLO 데이터셋을
자동 생성한다. Node+Playwright가 **화면 밖 headful 창**에서 sim을 돌려 GT를 캡처하고,
파일(images/*.png · labels/*.txt · data.yaml)을 **직접 저장**한다(폴더 선택창 불필요).

> 헤드리스(headless) Chromium은 WebGL 백버퍼가 0이라 실패한다. 그래서 headful로 띄우되
> `--window-position=-2400,-2400`으로 화면 밖에 두고, 스로틀 해제 플래그로 포커스 없이도
> 렌더되게 한다.

## 준비
```bash
# 1) 정적 서버로 sim.html 서빙 (저장소 루트에서)
python -m http.server 8123
# 2) playwright-core 설치 (이미 받은 브라우저 재사용)
cd tools/headless_gen
PLAYWRIGHT_BROWSERS_PATH="$HOME/AppData/Local/ms-playwright" npm i playwright-core
```

## 실행
```bash
# gen.cjs <출력폴더> <샘플수>   (샘플당 6대 = 코너4 + 변중앙 N/S)
PLAYWRIGHT_BROWSERS_PATH="$HOME/AppData/Local/ms-playwright" \
  node gen.cjs /abs/out/dir 20
```
출력: `<out>/images/<cam>_<idx>.png`, `<out>/labels/<cam>_<idx>.txt`, `<out>/data.yaml`.
`minColors`가 500 미만이면 렌더 실패(단색) 신호. 카메라 목록·시스템 Chrome 경로는 gen.cjs 상단 상수.
