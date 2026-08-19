@echo off
chcp 65001 >nul
REM 급식 조리로봇 안전 시뮬레이터 — 실행 (Windows)
REM 이 파일을 더블클릭하세요. 검은 창이 뜨고 브라우저가 자동으로 열립니다.
REM 끝낼 때는 검은 창을 닫으면 됩니다.

cd /d "%~dp0\.."

REM 파이썬 찾기
set PY=
where python >nul 2>&1 && set PY=python
if "%PY%"=="" ( where py >nul 2>&1 && set PY=py )
if "%PY%"=="" (
  echo.
  echo   파이썬을 찾지 못했습니다.
  echo   https://www.python.org/downloads/ 에서 설치하세요.
  echo   설치할 때 "Add Python to PATH" 를 반드시 체크하세요.
  echo.
  pause
  exit /b 1
)

set PORT=5173
set URL=http://localhost:%PORT%/sim.html

echo.
echo   +--------------------------------------------+
echo   ^|  급식 조리로봇 안전 시뮬레이터              ^|
echo   +--------------------------------------------+
echo.
echo   주소 : %URL%
echo   종료 : 이 창을 닫으세요
echo.
echo   ※ 브라우저 탭을 앞에 두세요. 뒤로 내리면 화면이 멈춥니다.
echo.

REM Chrome/Edge를 먼저 찾는다 — Safari와 마찬가지로 기본 브라우저가 느릴 수 있고,
REM ONNX 모델 검증은 WebGPU가 있는 Chromium 계열이 훨씬 빠르다.
set "BROWSER="
for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
  "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
) do if not defined BROWSER if exist %%P set "BROWSER=%%~P"

if defined BROWSER (
  echo   브라우저 : %BROWSER%
  start "" "%BROWSER%" "%URL%"
) else (
  echo   브라우저 : 기본 브라우저 ^(Chrome을 설치하면 더 부드럽습니다^)
  start "" "%URL%"
)

%PY% -m http.server %PORT% --bind 127.0.0.1
pause
