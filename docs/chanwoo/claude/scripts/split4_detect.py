# -*- coding: utf-8 -*-
"""
4분할 재검출 사다리 — 같은 전체 나디르 프레임에서 기성 → 나디르 단일 → 나디르 4분할.
전체 나디르는 사람이 작아 단일 패스도 놓친다(→ recall 0.87). 프레임을 4분할해
각 구역을 확대 재검출하면 놓친 사람을 되찾는다(→ 융합 0.94~0.99, nadir-zone-fusion §5-7~5-9).

입력: sim.html 전체 나디르 캡처 PNG (infer_boxes.py 상단 주석의 groundTruth 캡처).
사용: python split4_detect.py <full_nadir.png> <out.png>
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))
from infer_boxes import draw_boxes, header, EMBER, TEAL, GREY, BG
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def run(m, im, conf, cls):
    r = m.predict(im, conf=conf, classes=cls, verbose=False)[0]
    return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0

def nms(bs, thr=0.45):
    bs=sorted(bs, key=lambda b:-b[4]); keep=[]
    for b in bs:
        if all(iou(b,k)<thr for k in keep): keep.append(b)
    return keep

def split_detect(model, img, conf=0.30, ov=0.20):
    """프레임을 2x2(겹침 ov)로 나눠 각 구역 재검출 → 원좌표로 합쳐 NMS."""
    W,H=img.size; ox,oy=int(W*ov),int(H*ov)
    xs=[(0,W//2+ox),(W//2-ox,W)]; ys=[(0,H//2+oy),(H//2-oy,H)]
    out=[]
    for x0,x1 in xs:
        for y0,y1 in ys:
            for bx1,by1,bx2,by2,c in run(model, img.crop((x0,y0,x1,y1)), conf, [0]):
                out.append((bx1+x0,by1+y0,bx2+x0,by2+y0,c))
    return nms(out, 0.45)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv)>1 else "full_nadir.png"
    out = sys.argv[2] if len(sys.argv)>2 else "sim_zone4_ladder.png"
    stock = YOLO("yolo11s.pt")
    nadir = YOLO(hf_hub_download("chanubc/robot-kitchen-nadir-yolo11s", "best.pt"))
    img = Image.open(src).convert("RGB")
    stk   = run(stock, img, 0.25, [0])
    full  = run(nadir, img, 0.40, [0])
    split = split_detect(nadir, img)
    # 참고: 화재 씬이면 중앙 불꽃이 person으로 오탐될 수 있어 아래처럼 중앙 영역 제외를 쓴다(프레임 의존).
    #   ok = lambda b: not (405 < (b[0]+b[2])/2 < 560 and 290 < (b[1]+b[3])/2 < 430)
    #   full=[b for b in full if ok(b)]; split=[b for b in split if ok(b)]
    p = [header(draw_boxes(img,stk,EMBER),  "기성 YOLO11s",        EMBER, len(stk)),
         header(draw_boxes(img,full,GREY),  "나디르 · 단일 패스",   GREY,  len(full)),
         header(draw_boxes(img,split,TEAL), "나디르 · 4분할 재검출", TEAL,  len(split))]
    gap=16; cv=Image.new("RGB",(p[0].width*3+gap*2, p[0].height), BG)
    for i,pi in enumerate(p): cv.paste(pi,(i*(p[0].width+gap),0))
    cv.save(out); print(f"stock {len(stk)}  full {len(full)}  split {len(split)} -> {out}")
