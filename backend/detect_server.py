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
import math
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


def _is_detect_off(path) -> bool:
    """DETECT_MODEL 이 검출 비활성(예측 전용)을 뜻하는 값인가.

    `none`/`off`/빈 값이면 검출을 명시적으로 끈다 — 모델을 로드하지도, 허브에서 받지도
    않는다. 과거엔 `Path("none").exists()`가 False라 이 값이 그대로 허브 다운로드로 빠져,
    문서(=GT 좌표만 쓰는 예측 전용)와 어긋나고 오프라인에서 부팅이 실패했다.
    """
    return str(path).strip().lower() in ("", "none", "off")


def _resolve_model_path(path: str) -> str:
    """로컬 가중치가 있으면 그대로, 없으면 허깅페이스 허브에서 받아 캐시 경로를 준다."""
    if Path(path).exists():
        return path
    print(f"[detect_server] 로컬 가중치 없음 ({path}) → 허브에서 받는다: {_HUB_REPO}/{_HUB_FILE}")
    from huggingface_hub import hf_hub_download   # pip install huggingface_hub
    got = hf_hub_download(repo_id=_HUB_REPO, filename=_HUB_FILE)
    print(f"[detect_server] 허브 캐시: {got}")
    return got


if _is_detect_off(_MODEL_PATH):
    # 검출 비활성 — 예측 전용(GT 좌표). 모델/허브를 건드리지 않는다.
    MODE = "off"
    print(f"[detect_server] DETECT_MODEL={_MODEL_PATH!r} → 검출 비활성(예측 전용). "
          "모델 로드·허브 다운로드 안 함. /detect 는 빈 박스를 돌려준다.")
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
    if MODE == "off":                     # 검출 비활성(예측 전용) — 박스 없음
        return []
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


def _round(v):
    """JSON 직렬화용 반올림. None·비유한 float는 None으로(무한대 JSON 오염 방지)."""
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else None
    return v


def _mode_json(m):
    return {"path": [[round(x, 4), round(z, 4)] for (x, z) in m["path"]],
            "w": round(m["w"], 4),
            "sigma": [round(s, 4) for s in m["sigma"]]}


def _predict_response(body, p):
    """/predict 본체 — 순수 함수(HTTP 분리, 테스트 가능). 스펙 §3·§4-2.

    배치: body에 tracks 있으면 predict_batch(forward 1회) → 트랙별 위험 → 중재.
    하위호환: 옛 단일 hist면 {"modes": …} 그대로.
    """
    from trajectory.risk import track_risk, arbitrate

    if "tracks" in body:
        robot = (float(body["robot"]["x"]), float(body["robot"]["z"]))
        stopR = float(body["stopR"])
        slowR = float(body["slowR"])
        horizon = float(body.get("horizon", 1.6))
        ksig = float(body.get("safeKsig", 1.0))
        tau = float(body.get("safeTau", 0.1))
        tracks_in = body["tracks"]
        hists = [[(float(x), float(z)) for x, z in t["hist"]] for t in tracks_in]
        modes_all = p.predict_batch(hists)

        out_tracks, risks = [], []
        for tin, modes in zip(tracks_in, modes_all):
            r = track_risk(modes, robot, stopR, slowR, horizon, ksig, tau)
            risks.append({"id": tin["id"], **r})
            out_tracks.append({
                "id": tin["id"],
                "modes": [_mode_json(m) for m in modes],
                "risk": {k: _round(v) for k, v in r.items()},
            })
        worst = arbitrate(risks)
        if worst is not None:
            worst = {k: _round(v) for k, v in worst.items()}
        return {"tracks": out_tracks, "worst": worst}

    # 하위호환 — 단일 hist
    hist = [(float(x), float(z)) for x, z in body["hist"]]
    return {"modes": [_mode_json(m) for m in p.predict_modes(hist)]}


@app.post("/predict")
async def predict(req: Request):
    """학습형 멀티모달 궤적 예측 (다인원 배치 + 위험 중재). 이슈 #2 5단계 · 감사 P0-5.

    배치 요청(스펙 §3):
      {"tracks":[{"id","hist":[[x,z]…8]}], "robot":{x,z}, "stopR","slowR",
       "horizon","safeKsig","safeTau"}
      → {"tracks":[{"id","modes":[{path,w,sigma}],"risk":{tEntryStop,tEntrySlow,riskMass,dMin}}],
         "worst":{id,…}|null}
    하위호환: {"hist":[[x,z]…8]} → {"modes":[…]}. 좌표는 씬 AU(모델 학습 단위와 동일).
    """
    p = _get_predictor()
    if p is None:
        return JSONResponse(status_code=503, content={"error": _PREDICTOR_ERR or "predictor 미로드"})
    try:
        body = await req.json()
        return _predict_response(body, p)
    except Exception as e:                # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── 나디르 다중카메라 월드 융합 파이프라인 (/nadir) ──────────────────────────
# /detect(카메라별 이미지 검출)와 /predict(월드 궤적 예측) 사이의 빠진 층:
# 여러 나디르 카메라의 검출을 월드좌표로 올려 융합·추적해 사람별 월드 트랙을 만들고,
# 그 히스토리로 예측·위험까지 한 번에 돌린다(문서 5-9·5-15의 서버 이식).
NADIR_DT = 0.4                 # 예측기 학습 간격(초). hist를 이 간격으로 샘플링해 LSTM에 맞춘다.
NADIR_MERGE = 0.4              # 이보다 가까운 검출은 같은 사람(카메라 간 중복) → 병합(m)
NADIR_GATE = 0.8              # 트랙↔검출 연관 거리 게이트(m)
NADIR_MAXAGE = 5              # 이만큼 연속으로 안 보이면 트랙 폐기(coast 상한)
NADIR_OBS = 8                 # 예측 입력 관측 길이


class _WorldTrack:
    __slots__ = ("id", "pos", "vel", "last_t", "hist", "hist_t", "misses", "px")

    def __init__(self, tid, pos, t):
        self.id = tid
        self.pos = pos                 # (x, z) 현재 월드 위치
        self.vel = (0.0, 0.0)
        self.last_t = t
        self.px = pos                  # 예측(등속) 위치 — 매 프레임 갱신
        self.hist = [pos]              # 0.4s 간격 월드 위치 이력(예측 입력)
        self.hist_t = t
        self.misses = 0


class _NadirGroup:
    """나디르 카메라 그룹 하나의 월드 트랙 상태(프레임 간 유지)."""
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, world_pts, t):
        # 1) 예측(등속)
        for tr in self.tracks:
            dt = t - tr.last_t
            tr.px = (tr.pos[0] + tr.vel[0] * dt, tr.pos[1] + tr.vel[1] * dt)
        # 2) 거리 게이트 그리디 연관
        import numpy as _np
        matched_t, matched_d = set(), set()
        if self.tracks and world_pts:
            pairs = []
            for di, d in enumerate(world_pts):
                for ti, tr in enumerate(self.tracks):
                    dd = math.hypot(d[0] - tr.px[0], d[1] - tr.px[1])
                    if dd <= NADIR_GATE:
                        pairs.append((dd, di, ti))
            pairs.sort()
            for dd, di, ti in pairs:
                if di in matched_d or ti in matched_t:
                    continue
                matched_d.add(di); matched_t.add(ti)
                tr = self.tracks[ti]; d = world_pts[di]
                gdt = max(1e-3, t - tr.last_t)
                tr.vel = ((d[0] - tr.pos[0]) / gdt, (d[1] - tr.pos[1]) / gdt)
                tr.pos = d; tr.last_t = t; tr.misses = 0
        # 3) 미매칭 검출 → 새 트랙
        for di, d in enumerate(world_pts):
            if di in matched_d:
                continue
            self.tracks.append(_WorldTrack(self.next_id, d, t)); self.next_id += 1
        # 4) 미매칭 트랙 → coast(예측 위치 유지) + 만료
        for ti, tr in enumerate(self.tracks):
            if ti in matched_t:
                continue
            tr.misses += 1
            tr.pos = tr.px            # 등속 예측으로 위치 유지
        self.tracks = [tr for tr in self.tracks if tr.misses <= NADIR_MAXAGE]
        # 5) hist를 NADIR_DT 간격으로 샘플(예측기 dt에 맞춤)
        for tr in self.tracks:
            if t - tr.hist_t >= NADIR_DT - 1e-6:
                tr.hist.append(tr.pos); tr.hist_t = t
                if len(tr.hist) > NADIR_OBS:
                    tr.hist = tr.hist[-NADIR_OBS:]
        return self.tracks


NADIR_GROUPS = {}


def _map_world(cx, cy, aff):
    """이미지 정규화(cx,cy) → 월드(X,Z). aff = [[a,b],[c,d],[e,f]] (X=a·cx+c·cy+e, Z=b·cx+d·cy+f)."""
    return (aff[0][0]*cx + aff[1][0]*cy + aff[2][0],
            aff[0][1]*cx + aff[1][1]*cy + aff[2][1])


def _nadir_response(body):
    """/nadir 본체 — 순수 함수(테스트 가능). 4대 프레임 → 검출→월드융합→추적→예측→위험."""
    group_id = body.get("group", "nadir")
    t = (body["t"] / 1000.0) if body.get("t") is not None else time.monotonic()
    grp = NADIR_GROUPS.setdefault(group_id, _NadirGroup())

    # 1) 카메라별 검출 → 월드점 pool
    pooled = []
    per_cam = []
    for cam in body["cameras"]:
        aff = cam["affine"]
        img = _decode_image(cam["image"]) if MODE.startswith("yolo") else None
        dets = run_detect(img)
        cam_out = []
        for d in dets:
            wx, wz = _map_world(d["cx"], d["cy"], aff)
            pooled.append((wx, wz)); cam_out.append({"cam": cam.get("id"), "cx": d["cx"], "cy": d["cy"], "wx": round(wx, 3), "wz": round(wz, 3), "conf": d["conf"]})
        per_cam.append({"id": cam.get("id"), "n": len(cam_out)})

    # 2) 근접 병합(카메라 간 같은 사람 중복 제거)
    merged = []
    for p in pooled:
        hit = next((i for i, m in enumerate(merged) if math.hypot(p[0]-m[0], p[1]-m[1]) < NADIR_MERGE), None)
        if hit is None:
            merged.append([p[0], p[1]])
        else:
            merged[hit] = [(merged[hit][0]+p[0])/2, (merged[hit][1]+p[1])/2]

    # 3) 월드 추적 업데이트
    tracks = grp.update([tuple(m) for m in merged], t)

    # 4) 관측 충분한 트랙만 예측+위험
    out_tracks = []
    ready = [tr for tr in tracks if len(tr.hist) >= NADIR_OBS]
    if ready and body.get("robot") is not None:
        p = _get_predictor()
        if p is not None:
            from trajectory.risk import track_risk, arbitrate
            robot = (float(body["robot"]["x"]), float(body["robot"]["z"]))
            stopR = float(body.get("stopR", 3.10)); slowR = float(body.get("slowR", 3.90))
            horizon = float(body.get("horizon", 4.8)); ksig = float(body.get("safeKsig", 1.0)); tau = float(body.get("safeTau", 0.1))
            hists = [tr.hist[-NADIR_OBS:] for tr in ready]
            modes_all = p.predict_batch(hists)
            risks = []
            for tr, modes in zip(ready, modes_all):
                r = track_risk(modes, robot, stopR, slowR, horizon, ksig, tau)
                risks.append({"id": tr.id, **r})
                out_tracks.append({"id": tr.id, "pos": [round(tr.pos[0], 3), round(tr.pos[1], 3)],
                                   "vel": [round(tr.vel[0], 3), round(tr.vel[1], 3)],
                                   "modes": [_mode_json(m) for m in modes],
                                   "risk": {k: _round(v) for k, v in r.items()}})
            worst = arbitrate(risks)
            if worst is not None:
                worst = {k: _round(v) for k, v in worst.items()}
            return {"tracks": out_tracks, "worst": worst, "coverage": len(tracks), "per_cam": per_cam}
    # 예측기 없거나 관측 부족 — 트랙 위치만
    for tr in tracks:
        out_tracks.append({"id": tr.id, "pos": [round(tr.pos[0], 3), round(tr.pos[1], 3)],
                           "vel": [round(tr.vel[0], 3), round(tr.vel[1], 3)], "obs": len(tr.hist)})
    return {"tracks": out_tracks, "worst": None, "coverage": len(tracks), "per_cam": per_cam}


@app.post("/nadir")
async def nadir(req: Request):
    """나디르 다중카메라 월드 융합 → 추적 → 예측 → 위험 (한 번에).

    요청: {"group":"nadir", "t":ms,
           "cameras":[{"id":"nzNW","image":"data:...","affine":[[a,b],[c,d],[e,f]]}, …],
           "robot":{x,z}, "stopR","slowR","horizon","safeKsig","safeTau"}
      affine = 이미지 정규화(cx,cy,1) → 월드(X,Z) 아핀. 시뮬이 카메라 캘리브레이션으로 1회 계산해 보낸다.
    응답: {"tracks":[{id,pos,vel,modes,risk}], "worst":{id,…}|null, "coverage":N, "per_cam":[…]}
      좌표는 씬 AU(=m, 모델 학습 단위). 트랙은 프레임 간 유지(그룹별 상태).
    """
    try:
        body = await req.json()
        return _nadir_response(body)
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
