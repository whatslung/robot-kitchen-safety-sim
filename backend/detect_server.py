#!/usr/bin/env python3
"""
detect_server.py — 시뮬레이터 '모델 검증(http 모드)'용 어댑터 서버 (FastAPI + trackers).

검출(YOLO) + 다중객체 추적(ByteTrack)까지 온디바이스(localhost)에서 돌려,
프레임별 박스에 **track id 와 속도(vx, vy)** 를 얹어 돌려준다.
영상은 브라우저↔이 서버(127.0.0.1) 사이에서만 오가고 인터넷으로 나가지 않는다.

팀원 모델을 꽂는 자리는 run_detect() 하나다. 추적·속도·서버 배선은 건드릴 필요 없음.

실행 (이 서버 하나가 시뮬 정적 파일 + 검출 API를 모두 서빙):
    uv sync --group serve
    uv run python backend/detect_server.py --port 8001
    # → 시뮬:  http://127.0.0.1:8001/sim.html?person=1
    # → 검출:  http://127.0.0.1:8001/detect   (시뮬과 동일 출처라 CORS 불필요)
    # 모델 교체:  DETECT_MODEL=training/real_sim/weights/best.pt uv run python backend/detect_server.py --port 8001

시뮬 쪽 계약 (하위호환 — 기존 필드는 그대로, id·vx·vy 만 추가):
    요청  POST /detect  {"camera": "corner", "image": "data:image/png;base64,...", "t": 1699...}
          - t(ms)는 선택. 없으면 서버 도착시각(단조시계)으로 속도를 계산한다.
    응답  {"boxes": [{
              "label":"person", "conf":0.93,
              "cx":0.5, "cy":0.5, "w":0.2, "h":0.6,   # 0~1 정규화, 원점 좌상단 (YOLO 동일)
              "id": 7,                                  # 트랙 id (-1 = 미할당)
              "vx": 0.04, "vy": -0.01                   # 정규화 이미지좌표/초 (dcx/dt, dcy/dt)
          }], "mode": "yolo+bytetrack", "camera": "corner"}

    ※ vx,vy 는 '이미지 평면'에서의 속도다(월드 속도가 아니다). 시뮬은 박스 중심을
      카메라 내·외부 파라미터로 월드에 역투영하므로, 월드 속도가 필요하면 연속 위치에서
      직접 구해도 된다. vx,vy 는 서버가 트랙 연속성으로 이미 평활해 둔 편의값이다.

    ※ 시뮬(sim.html)의 현재 파서는 label/conf/cx/cy/w/h 만 읽고 id·vx·vy 는 무시한다
      (알 수 없는 필드는 안전하게 무시됨). 이 값들을 실제로 쓰려면 시뮬 쪽에서
      __customModel 응답을 트랙 단위로 받도록 확장하는 후속 작업이 필요하다.
"""
import base64
import io
import os
import time
import argparse
from pathlib import Path

import numpy as np

# ── 의존성 로드 — 없으면 DUMMY 모드로 떠서 배선을 먼저 검증한다 ──────────────
MODEL = None
TRACKER_CLS = None
sv = None
MODE = "dummy"

# 모델 경로: env DETECT_MODEL 로 교체 가능. 기본 = 시뮬 파인튜닝(orthotop) best.pt.
#   real+sim 모델로 스왑:  DETECT_MODEL=training/real_sim/weights/best.pt
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = os.environ.get(
    "DETECT_MODEL", str(_ROOT / "training" / "yolo11s_orthotop" / "weights" / "best.pt"))
try:
    from ultralytics import YOLO          # pip install ultralytics
    MODEL = YOLO(_MODEL_PATH)
    MODE = "yolo"
    print(f"[detect_server] 모델 로드: {_MODEL_PATH} · 클래스 {list(MODEL.names.values())}")
except Exception as e:                    # noqa: BLE001
    print(f"[detect_server] 모델 로드 실패 → 검출 DUMMY ({_MODEL_PATH}: {e})")

try:
    import supervision as sv              # pip install supervision
    from trackers import ByteTrackTracker  # pip install trackers
    TRACKER_CLS = ByteTrackTracker
except Exception as e:                    # noqa: BLE001
    print(f"[detect_server] trackers/supervision 없음 → 추적 비활성 ({e})")

if MODE == "yolo" and TRACKER_CLS is not None:
    MODE = "yolo+bytetrack"

# COCO 클래스 → 시뮬 라벨 매핑 (팀원 클래스 체계에 맞게 수정).
# None 이면 전부 통과. {"person": "person"} 이면 person 만 남긴다.
LABEL_MAP = {"person": "person"}

# 속도 평활 계수(EMA) 와 트랙 만료 시간
VEL_EMA = 0.4          # 0=매우 매끈(느림), 1=순간속도(노이즈)
TRACK_TTL_S = 5.0      # 이 시간 이상 안 보인 트랙의 속도 이력은 폐기


# ── 모델 추론 — 팀원 모델을 꽂는 지점 ─────────────────────────────────────
def run_detect(pil_image):
    """PIL 이미지를 받아 검출 리스트를 돌려준다(추적 전, id 없음).

    반환: [{"label","conf","cx","cy","w","h"}]  (cx,cy,w,h 는 0~1 정규화)
    ultralytics 대신 다른 검출기를 쓰려면 이 함수만 바꾸면 된다.
    """
    if MODE.startswith("yolo"):
        w, h = pil_image.size
        out = []
        for r in MODEL(pil_image, verbose=False):
            for b in r.boxes:
                name = MODEL.names[int(b.cls)]
                if LABEL_MAP and name not in LABEL_MAP:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append({
                    "label": LABEL_MAP.get(name, name) if LABEL_MAP else name,
                    "conf": float(b.conf),
                    "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
                    "w": (x2 - x1) / w, "h": (y2 - y1) / h,
                })
        return out
    # DUMMY — 배선 확인용. 살짝 좌우로 흔들리는 박스 1개(속도가 0이 아니게 나오도록)
    t = time.monotonic()
    return [{"label": "person", "conf": 0.50,
             "cx": 0.42 + 0.05 * np.sin(t), "cy": 0.55,
             "w": 0.18, "h": 0.55}]


# ── 카메라별 추적 상태 ────────────────────────────────────────────────────
class CamState:
    """카메라 한 대의 트래커 + 트랙별 속도 이력."""
    def __init__(self):
        self.tracker = _new_tracker()
        self.hist = {}   # id -> {"t","cx","cy","vx","vy"}

    def sweep(self, now):
        stale = [i for i, s in self.hist.items() if now - s["t"] > TRACK_TTL_S]
        for i in stale:
            del self.hist[i]


CAMS = {}


def _new_tracker():
    if TRACKER_CLS is None:
        return None
    try:
        return TRACKER_CLS()
    except Exception as e:                # noqa: BLE001
        print(f"[detect_server] ByteTrackTracker 생성 실패 → 추적 비활성 ({e})")
        return None


def _detections_from(dets, img_w, img_h):
    """run_detect() 출력(정규화 박스) → sv.Detections(픽셀 xyxy)."""
    if not dets:
        return sv.Detections.empty()
    xyxy, conf, cls = [], [], []
    label_to_id, id_to_label = {}, {}
    for d in dets:
        cx, cy, w, h = d["cx"] * img_w, d["cy"] * img_h, d["w"] * img_w, d["h"] * img_h
        xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        conf.append(d["conf"])
        lbl = d["label"]
        if lbl not in label_to_id:
            cid = len(label_to_id)
            label_to_id[lbl] = cid
            id_to_label[cid] = lbl
        cls.append(label_to_id[lbl])
    det = sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        confidence=np.array(conf, dtype=float),
        class_id=np.array(cls, dtype=int),
    )
    det.data["class_name"] = np.array([id_to_label[c] for c in cls])
    return det


def track_and_measure(dets, img_w, img_h, camera_id, t_ms):
    """검출에 track id 와 속도(정규화/초)를 붙여 돌려준다."""
    now = (t_ms / 1000.0) if t_ms is not None else time.monotonic()
    cam = CAMS.setdefault(camera_id, CamState())

    # 추적 불가(라이브러리 없음/생성 실패) → id -1, 속도 0 으로 그대로 통과
    if cam.tracker is None or sv is None:
        return [{**d, "id": -1, "vx": 0.0, "vy": 0.0} for d in dets]

    det = _detections_from(dets, img_w, img_h)
    try:
        tracked = cam.tracker.update(det)
    except Exception as e:               # noqa: BLE001
        print(f"[detect_server] tracker.update 실패({camera_id}) → id 미할당 ({e})")
        return [{**d, "id": -1, "vx": 0.0, "vy": 0.0} for d in dets]

    out = []
    seen = set()
    n = len(tracked)
    names = tracked.data.get("class_name") if tracked.data else None
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        if names is not None:
            label = str(names[i])
        elif tracked.class_id is not None and MODE.startswith("yolo"):
            label = MODEL.names.get(int(tracked.class_id[i]), str(tracked.class_id[i]))
        else:
            label = "person"

        vx = vy = 0.0
        if tid >= 0:
            seen.add(tid)
            prev = cam.hist.get(tid)
            if prev is not None:
                dt = now - prev["t"]
                if 1e-3 < dt < 1.5:
                    ivx = (cx - prev["cx"]) / dt
                    ivy = (cy - prev["cy"]) / dt
                    vx = (1 - VEL_EMA) * prev["vx"] + VEL_EMA * ivx
                    vy = (1 - VEL_EMA) * prev["vy"] + VEL_EMA * ivy
                else:                     # dt 비정상 → 직전 속도 유지
                    vx, vy = prev["vx"], prev["vy"]
            cam.hist[tid] = {"t": now, "cx": cx, "cy": cy, "vx": vx, "vy": vy}

        out.append({
            "label": label, "conf": conf,
            "cx": cx, "cy": cy, "w": bw, "h": bh,
            "id": tid, "vx": vx, "vy": vy,
        })

    cam.sweep(now)
    return out


# ── FastAPI 앱 ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except Exception as e:                    # noqa: BLE001
    raise SystemExit(
        "FastAPI 가 필요합니다. 설치: pip install fastapi uvicorn\n"
        f"(원인: {e})"
    )

app = FastAPI(title="detect_server", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                  # 로컬 시뮬(다른 포트)에서 POST 하므로 허용
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _decode_image(data_url):
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "cameras": list(CAMS.keys())}


@app.post("/detect")
async def detect(req: Request):
    try:
        body = await req.json()
        camera_id = body.get("camera", "?")
        t_ms = body.get("t")
        if MODE.startswith("yolo"):
            img = _decode_image(body["image"])
            img_w, img_h = img.size
        else:                             # DUMMY — 이미지 디코드 없이 고정 해상도 가정
            img, img_w, img_h = None, 1280, 720
        dets = run_detect(img)
        boxes = track_and_measure(dets, img_w, img_h, camera_id, t_ms)
        return {"boxes": boxes, "mode": MODE, "camera": camera_id}
    except Exception as e:                # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/dataset")
async def dataset(req: Request):
    """시뮬이 만든 한 샘플(이미지+YOLO 라벨)을 dataset/<set>/{images,labels}/ 에 쓴다.

    시뮬의 generateDataset은 showDirectoryPicker(사용자 제스처) 또는 <a download>를 쓴다 —
    둘 다 사람이 클릭해야 해서 자동 생성이 안 된다. 서버가 받아 쓰면 브라우저를 띄워두기만
    하면 되고, 라벨/이미지가 항상 짝으로 저장된다.
    요청: {"set":"sim-person-island","name":"orthotop_0000","image":"data:image/png;base64,...","label":"0 .. 
"}
    """
    try:
        body = await req.json()
        safe = lambda v, d: "".join(c for c in str(v) if c.isalnum() or c in "-_")[:80] or d
        st = safe(body.get("set", "ds"), "ds")
        nm = safe(body.get("name", "sample"), "sample")
        raw = body["image"]
        b64 = raw.split(",", 1)[1] if "," in raw else raw
        root = _ROOT / "dataset" / st
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "images" / (nm + ".png")).write_bytes(base64.b64decode(b64))
        (root / "labels" / (nm + ".txt")).write_text(body.get("label", ""), encoding="utf-8")
        n = len(list((root / "images").glob("*.png")))
        return {"ok": True, "count": n, "dir": str(root)}
    except Exception as e:                # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/shot")
async def shot(req: Request):
    """브라우저가 만든 이미지를 그대로 디스크에 저장한다 (captures/, gitignore 대상).

    종전에는 시뮬의 capture()가 <a download>로 브라우저 다운로드를 띄워 **사람이 매번 저장
    버튼을 눌러야** 했다. 데모/문서용 스냅을 자동으로 남기려면 서버가 받아 쓰는 쪽이 맞다.
    요청: {"name": "demo", "image": "data:image/png;base64,..."}
    """
    try:
        body = await req.json()
        raw = body["image"]
        b64 = raw.split(",", 1)[1] if "," in raw else raw
        name = "".join(c for c in str(body.get("name", "shot")) if c.isalnum() or c in "-_")[:60] or "shot"
        d = _ROOT / "captures"
        d.mkdir(exist_ok=True)
        f = d / (name + ".png")
        f.write_bytes(base64.b64decode(b64))
        return {"saved": str(f), "bytes": f.stat().st_size}
    except Exception as e:                # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


# 시뮬 정적 파일(sim.html·assets·vendor…)도 같은 FastAPI가 서빙한다 →
# 서버 하나로 시뮬 + 검출을 모두 처리(별도 http.server 불필요, 동일 출처라 CORS도 불필요).
# 라우트(/detect·/health)를 먼저 등록한 뒤 마운트하므로 그 경로들이 우선한다.
try:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_ROOT), html=True), name="sim")
except Exception as e:                    # noqa: BLE001
    print(f"[detect_server] 정적 서빙 비활성 ({e}) — 시뮬은 별도 http.server로 여세요")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    print(f"[detect_server] {MODE.upper()} — http://{args.host}:{args.port}/detect")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
