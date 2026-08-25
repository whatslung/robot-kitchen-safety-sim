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
    # 가중치는 로컬 training/ 에 있으면 그걸 쓰고, 없으면 허깅페이스에서 자동으로 받는다
    #   → 팀원은 별도 파일 전달 없이 위 두 줄만 실행하면 된다 (chanubc/robot-kitchen-nadir-yolo11s)
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
import json
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

# 모델 경로: env DETECT_MODEL 로 교체 가능. 기본 = 섬 배치 파인튜닝 best.pt.
#   real+sim 모델로 스왑:  DETECT_MODEL=training/real_sim/weights/best.pt
# 학습 산출물(training/)은 용량이 커서 git에 없다. 로컬에 없으면 허깅페이스에서 받아온다
# → 팀원은 아무것도 내려받지 않고 서버만 켜면 된다.
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = os.environ.get(
    "DETECT_MODEL", str(_ROOT / "training" / "island_yolo11s" / "weights" / "best.pt"))
_HUB_REPO = os.environ.get("DETECT_MODEL_REPO", "chanubc/robot-kitchen-nadir-yolo11s")
_HUB_FILE = os.environ.get("DETECT_MODEL_FILE", "best.pt")


def _resolve_model_path(path: str) -> str:
    """로컬 가중치가 있으면 그대로, 없으면 허깅페이스 허브에서 받아 캐시 경로를 준다."""
    if Path(path).exists():
        return path
    print(f"[detect_server] 로컬 가중치 없음 ({path}) → 허브에서 받는다: {_HUB_REPO}/{_HUB_FILE}")
    from huggingface_hub import hf_hub_download   # pip install huggingface_hub
    got = hf_hub_download(repo_id=_HUB_REPO, filename=_HUB_FILE)
    print(f"[detect_server] 허브 캐시: {got}")
    return got


if os.environ.get("DETECT_DISABLE_MODEL", "").strip().lower() in {"1", "true", "yes"}:
    print("[detect_server] DETECT_DISABLE_MODEL=1 → 검출 DUMMY")
else:
    try:
        from ultralytics import YOLO          # pip install ultralytics
        _MODEL_PATH = _resolve_model_path(_MODEL_PATH)
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
_CAMERA_UPDATED_AT = {}

try:
    from backend.multiview import CalibrationError, CameraCalibration, MultiViewFusion
except ImportError:  # backend/detect_server.py를 스크립트로 직접 실행할 때
    from multiview import CalibrationError, CameraCalibration, MultiViewFusion

FUSION = MultiViewFusion(
    gate=0.8,
    fusion_window_ms=250,
    coast_ms=750,
    remove_ms=1500,
)


# ByteTrack 활성 임계값. 라이브러리 기본값은 0.7인데, 우리 시뮬 모델의 라이브 confidence는
# 0.25~0.58에 몰린다(도메인 랜덤화된 노이즈·지터 프레임으로 학습했는데 라이브 화면은 깨끗해서
# 절대 confidence가 낮게 나온다 — recall 0.87인데도 그렇다). 기본값이면 **트랙이 하나도
# 활성화되지 않아 id가 영영 -1**이 된다(실측). 안전 시스템에서 낮은 confidence 검출을 버리는 건
# 위험한 쪽 오류라 문턱을 낮춘다. env로 조정 가능.
TRACK_ACT = float(os.environ.get("TRACK_ACT", "0.35"))
TRACK_HIGH = float(os.environ.get("TRACK_HIGH", "0.35"))


def _new_tracker():
    if TRACKER_CLS is None:
        return None
    try:
        return TRACKER_CLS(track_activation_threshold=TRACK_ACT,
                           high_conf_det_threshold=TRACK_HIGH,
                           minimum_consecutive_frames=2)
    except TypeError:                     # 다른 버전이면 인자 없이
        try:
            return TRACKER_CLS()
        except Exception as e:            # noqa: BLE001
            print(f"[detect_server] 트래커 생성 실패 → 추적 비활성 ({e})")
            return None
    except Exception as e:                # noqa: BLE001
        print(f"[detect_server] 트래커 생성 실패 → 추적 비활성 ({e})")
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _decode_image(data_url):
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.get("/health")
def health():
    now = time.monotonic()
    return {
        "status": "ok",
        "mode": MODE,
        "cameras": sorted(CAMS),
        "calibrated_cameras": sorted(FUSION.calibrations),
        "camera_update_age_ms": {
            camera: max(0, int(round((now - updated_at) * 1000)))
            for camera, updated_at in sorted(_CAMERA_UPDATED_AT.items())
        },
        "global_track_count": len(FUSION.tracks),
    }


@app.post("/calibrate")
async def calibrate(req: Request):
    try:
        body = await req.json()
        camera_id = str(body["camera"]).strip()
        points = body["points"]
        image = [point["image"] for point in points]
        world = [point["world"] for point in points]
        calibration = CameraCalibration.from_points(
            image=image,
            world=world,
            valid_world_polygon=body["valid_world_polygon"],
        )
        FUSION.calibrate(camera_id, calibration)
        return {
            "camera": camera_id,
            "reprojection_rms": calibration.reprojection_rms,
            "point_count": len(points),
        }
    except (KeyError, TypeError, ValueError, CalibrationError):
        return JSONResponse(status_code=422, content={"error": "invalid calibration"})


@app.post("/tracks/reset")
async def tracks_reset():
    FUSION.reset_tracks()
    CAMS.clear()
    _CAMERA_UPDATED_AT.clear()
    return {"ok": True, "calibrated_cameras": sorted(FUSION.calibrations)}


@app.post("/detect")
async def detect(req: Request):
    try:
        body = await req.json()
        camera_id = body.get("camera", "?")
        t_ms = body.get("t")
        fusion_t_ms = int(t_ms) if t_ms is not None else int(round(time.monotonic() * 1000))
        if MODE.startswith("yolo"):
            img = _decode_image(body["image"])
            img_w, img_h = img.size
        else:                             # DUMMY — 이미지 디코드 없이 고정 해상도 가정
            img, img_w, img_h = None, 1280, 720
        dets = run_detect(img)
        boxes = track_and_measure(dets, img_w, img_h, camera_id, t_ms)
        boxes = FUSION.update(camera_id, boxes, fusion_t_ms)
        _CAMERA_UPDATED_AT[camera_id] = time.monotonic()
        return {
            "boxes": boxes,
            "mode": MODE,
            "camera": camera_id,
            "seq": body.get("seq"),
            "global_tracks": FUSION.snapshot(fusion_t_ms),
        }
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


@app.post("/traj")
async def traj(req: Request):
    """시뮬이 만든 궤적 scene 하나를 dataset/trajectories/<scene_id>.json 에 쓴다.

    /dataset·/shot과 같은 규약 — 시뮬이 scene 완결 시 scene JSON 전체를 POST하면 서버가
    받아 쓴다(폴더 선택창 없음, 클릭 0회). dataset/이 gitignore라 궤적도 자동 제외된다.
    요청: {"scene_id":"island_seed7_0000", "schema":1, "seed":7, ..., "nodes":[...]}
    """
    try:
        body = await req.json()
        safe = lambda v, d: "".join(c for c in str(v) if c.isalnum() or c in "-_")[:80] or d
        sid = safe(body.get("scene_id", "scene"), "scene")
        root = _ROOT / "dataset" / "trajectories"
        root.mkdir(parents=True, exist_ok=True)
        (root / (sid + ".json")).write_text(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        n = len(list(root.glob("*.json")))
        return {"ok": True, "count": n, "dir": str(root)}
    except Exception as e:                # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


_PREDICTOR = None
_PREDICTOR_ERR = None


def _get_predictor():
    """학습형 궤적 예측기를 1회 로드(지연). 없으면 _PREDICTOR_ERR에 이유."""
    global _PREDICTOR, _PREDICTOR_ERR
    if _PREDICTOR is not None or _PREDICTOR_ERR is not None:
        return _PREDICTOR
    try:
        import sys
        sys.path.insert(0, str(_ROOT))
        from trajectory.learned_predictor import LearnedPredictor
        # YOLO와 동일: 로컬 가중치 있으면 그대로, 없으면 허깅페이스 허브에서 받아 캐시.
        # → 팀원은 재학습 없이 서버만 켜면 된다. 재학습본 스왑: PREDICT_MODEL=경로/model.pt
        w = Path(os.environ.get("PREDICT_MODEL", str(_ROOT / "training" / "traj_predictor" / "model.pt")))
        if w.exists():
            wpath = str(w)
        else:
            repo = os.environ.get("PREDICT_MODEL_REPO", "chanubc/human-move-lstm")
            file = os.environ.get("PREDICT_MODEL_FILE", "model.pt")
            print(f"[detect_server] 예측기 가중치 로컬 없음 ({w}) → 허브에서 받는다: {repo}/{file}")
            from huggingface_hub import hf_hub_download
            wpath = hf_hub_download(repo_id=repo, filename=file)
            print(f"[detect_server] 허브 캐시: {wpath}")
        _PREDICTOR = LearnedPredictor(weights_path=wpath, device="cpu")
        print(f"[detect_server] 궤적 예측기 로드: {wpath}")
    except Exception as e:                # noqa: BLE001
        _PREDICTOR_ERR = str(e)
        print(f"[detect_server] 궤적 예측기 로드 실패 → /predict 비활성 ({e})")
    return _PREDICTOR


@app.post("/predict")
async def predict(req: Request):
    """학습형 멀티모달 궤적 예측. 이슈 #2 5단계.

    시뮬의 window.__customPredictor가 관측 8점(씬 AU, 0.4s 간격으로 리샘플)을 보내면
    K개 모드(각 경로+가중치+스텝별 σ)를 돌려준다. 좌표는 모델 학습 단위(AU)와 동일.
    요청: {"hist": [[x,z], … 8개]}   응답: {"modes": [{"path":[[x,z]…], "w":.., "sigma":[…]}]}
    """
    p = _get_predictor()
    if p is None:
        return JSONResponse(status_code=503, content={"error": _PREDICTOR_ERR or "predictor 미로드"})
    try:
        body = await req.json()
        hist = [(float(x), float(z)) for x, z in body["hist"]]
        modes = p.predict_modes(hist)
        return {"modes": [{"path": [[round(x, 4), round(z, 4)] for (x, z) in m["path"]],
                           "w": round(m["w"], 4),
                           "sigma": [round(s, 4) for s in m["sigma"]]} for m in modes]}
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
