#!/usr/bin/env python3
"""
detect_server.py — 시뮬레이터 '모델 검증(http 모드)'용 어댑터 서버.

팀원 모델을 꽂는 자리는 아래 run_model() 하나다. 나머지는 건드릴 필요 없음.

실행:
    python detect_server.py            # http://localhost:8000/detect

시뮬 쪽 계약:
    요청  POST /detect  {"camera": "corner", "image": "data:image/png;base64,..."}
    응답  {"boxes": [{"label":"person","conf":0.93,"cx":0.5,"cy":0.5,"w":0.2,"h":0.6}]}
          - cx,cy,w,h는 0~1 정규화, 원점은 좌상단 (YOLO와 동일)
          - label은 자유 문자열 — person / robot / kettle / equipment면 색이 자동 매칭

ultralytics(YOLO)가 설치돼 있으면 자동으로 사용하고,
없으면 DUMMY 모드(화면 중앙에 가짜 person 박스 1개)로 떠서 연동 배선을 먼저 검증할 수 있다.
"""
import base64, io, json
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000

# ── 모델 로드 — 팀원 모델로 교체하는 지점 ① ──────────────────────────────
MODEL = None
MODE = "dummy"
try:
    from ultralytics import YOLO          # pip install ultralytics
    from PIL import Image
    MODEL = YOLO("yolov8n.pt")            # ← 팀원 가중치 경로로 교체 (예: "best.pt")
    MODE = "yolo"
except Exception as e:
    print(f"[detect_server] ultralytics 없음 → DUMMY 모드로 실행 ({e})")

# COCO 클래스 → 시뮬 라벨 매핑 (팀원 클래스 체계에 맞게 수정)
LABEL_MAP = {"person": "person"}


def run_model(pil_image, camera_id):
    """팀원 모델을 꽂는 지점 ② — PIL 이미지를 받아 박스 리스트를 돌려준다."""
    if MODE == "yolo":
        w, h = pil_image.size
        out = []
        for r in MODEL(pil_image, verbose=False):
            for b in r.boxes:
                name = MODEL.names[int(b.cls)]
                if LABEL_MAP and name not in LABEL_MAP:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append({
                    "label": LABEL_MAP.get(name, name),
                    "conf": float(b.conf),
                    "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
                    "w": (x2 - x1) / w, "h": (y2 - y1) / h,
                })
        return out
    # DUMMY — 배선 확인용 고정 박스
    return [{"label": "person", "conf": 0.50, "cx": 0.42, "cy": 0.55, "w": 0.18, "h": 0.55}]


# ── 이하 서버 배선 (수정 불필요) ──────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        if self.path != "/detect":
            self.send_response(404); self._cors(); self.end_headers(); return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            img = None
            if MODE == "yolo":
                raw = base64.b64decode(body["image"].split(",", 1)[1])
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            boxes = run_model(img, body.get("camera", "?"))
            payload = json.dumps({"boxes": boxes, "mode": MODE}).encode()
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(payload)
        except Exception as e:
            self.send_response(500); self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, fmt, *args):
        print(f"[detect_server] {args[0]} {args[1]}")


if __name__ == "__main__":
    print(f"[detect_server] {MODE.upper()} 모드 — http://localhost:{PORT}/detect")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
