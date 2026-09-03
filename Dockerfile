# ─────────────────────────────────────────────────────────────────────────────
# Cloud Run 배포용 이미지 — detect_server 를 "예측 전용(경량)"으로 띄운다.
#   · DETECT_MODEL=none  → YOLO 검출을 끔(ultralytics/torchvision/supervision/trackers 불필요)
#   · torch 는 CPU 전용 휠만 설치(GPU 커널 없음 → 용량↓, GPU 인스턴스 불필요)
#   · 예측기 가중치(LSTM)를 빌드 시 이미지에 구워둠 → 런타임 네트워크 의존 0, 콜드스타트↓
#   · 앱 코드는 예측 경로에 필요한 backend/ 와 trajectory/ 만 복사
# 로컬 검증:  docker build -t detect-server . && docker run --rm -p 8080:8080 -e PORT=8080 detect-server
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# 1) 파이썬 의존성 — 설치기는 uv(프로젝트 기본)로 통일, 목록은 requirements-serve.txt 유지.
#    · uv pip install --system : 컨테이너의 시스템 파이썬에 바로 설치(가상환경 불필요)
#    · torch 는 CPU 전용 인덱스에서 따로 설치(2.4.x: torch.load 가 state_dict 를 그대로
#      로드 — 2.6+ 의 weights_only 기본값 변경에 걸리지 않아 가중치 로딩이 안전하다).
#    ※ pyproject 의 uv sync 는 base 의존성(ultralytics·CUDA torch·onnx)을 다 끌고 와
#      GPU용 대용량 이미지가 되므로, 배포 이미지는 CPU 예측 경로만 담은 이 목록을 쓴다.
RUN pip install --no-cache-dir uv
COPY backend/requirements-serve.txt ./backend/requirements-serve.txt
RUN uv pip install --system --no-cache torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system --no-cache -r backend/requirements-serve.txt

# 2) 예측기 가중치를 이미지에 굽는다(chanubc/human-move-lstm/model.pt → /app/models/).
#    런타임엔 PREDICT_MODEL 로컬 경로를 써서 HuggingFace 를 건드리지 않는다.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='chanubc/human-move-lstm', filename='model.pt', local_dir='/app/models')"

# 3) 앱 코드 — 예측 경로에 필요한 것만.
COPY backend/ ./backend/
COPY trajectory/ ./trajectory/

# 검출 끄기(예측 전용) · 예측 아키텍처(LSTM) · 로컬 가중치 경로 · 로그 즉시 출력
#   API_ONLY=1 : 공개 배포 하드닝 — 정적 파일 마운트와 dev 쓰기 엔드포인트(/dataset·
#   /shot·/traj)를 노출하지 않는다(임의 파일 읽기·tmpfs 메모리 고갈 차단).
ENV DETECT_MODEL=none \
    PREDICT_NET=lstm \
    PREDICT_MODEL=/app/models/model.pt \
    API_ONLY=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Cloud Run 은 컨테이너에 $PORT(기본 8080)로 요청을 보낸다 → 0.0.0.0 바인딩 필수.
# uvicorn 이 app 객체를 직접 로드하므로 detect_server.py 는 수정할 필요가 없다.
CMD ["sh", "-c", "uvicorn backend.detect_server:app --host 0.0.0.0 --port ${PORT}"]
