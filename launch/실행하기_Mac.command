#!/bin/bash
# 급식 조리로봇 안전 시뮬레이터 — 실행 (macOS)
# 이 파일을 더블클릭하세요. 터미널이 열리고 브라우저가 자동으로 뜹니다.
# 끝낼 때는 이 터미널 창에서 Ctrl+C 를 누르거나 창을 닫으면 됩니다.

cd "$(dirname "$0")/.." || { echo "  시뮬레이터 폴더를 찾지 못했습니다."; read -r -p "  엔터..." _; exit 1; }

# 파이썬 찾기 — python3 우선
PY=""
for c in python3 python /usr/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo ""
  echo "  파이썬을 찾지 못했습니다."
  echo "  터미널에서 다음을 실행해 설치하세요:  xcode-select --install"
  echo ""
  read -r -p "  엔터를 누르면 닫힙니다..." _
  exit 1
fi

# 비어 있는 포트 찾기 (5173부터)
PORT=5173
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
  PORT=$((PORT+1))
  [ $PORT -gt 5199 ] && break
done

# 확정 배치(섬 배치)로 연다 — 플래그를 빼면 옛 배치가 뜬다. README "실행" 절 참조.
URL="http://localhost:$PORT/sim.html?layout=island"
echo ""
echo "  ┌────────────────────────────────────────────┐"
echo "  │  급식 조리로봇 안전 시뮬레이터              │"
echo "  └────────────────────────────────────────────┘"
echo ""
echo "  주소 : $URL"
echo "  종료 : 이 창에서 Ctrl+C"
echo ""
echo "  ※ 브라우저 탭을 앞에 두세요. 뒤로 내리면 화면이 멈춥니다."
echo ""

# 서버가 뜬 뒤에 브라우저를 연다.
# Chrome/Edge를 먼저 찾는다 — Safari는 WebGL이 느리고(그림자·MSAA에 특히 약하다)
# ONNX 모델 검증도 WebGPU가 있는 Chromium 계열이 훨씬 빠르다.
open_browser() {
  for app in "Google Chrome" "Microsoft Edge" "Brave Browser" "Chromium"; do
    if [ -d "/Applications/$app.app" ] || [ -d "$HOME/Applications/$app.app" ]; then
      open -a "$app" "$URL" 2>/dev/null && { echo "  브라우저: $app"; return 0; }
    fi
  done
  echo "  브라우저: 기본 브라우저 (Chrome을 설치하면 더 부드럽습니다)"
  open "$URL"
}
( sleep 1.2; open_browser ) &
"$PY" -m http.server "$PORT" --bind 127.0.0.1
