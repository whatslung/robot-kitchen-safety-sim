# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, glob
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BG="#0E0F12"; INK="#F4F1EA"; MUTED="#8B8378"
GREY=(120,116,105); TEAL=(60,199,192); EMBER=(255,106,50)

def font(sz, bold=True):
    for p in [r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf",
              r"C:\Windows\Fonts\malgun.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()

def draw_boxes(img, boxes, color, width=None, cls_name="person", show_label=True):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W,H = im.size
    w = width or max(2, int(round(W/400)))
    fsz = max(11, int(round(W/95)))
    f = font(fsz, bold=False)
    for (x1,y1,x2,y2,conf) in boxes:
        d.rectangle([x1,y1,x2,y2], outline=color, width=w)
        if show_label:
            txt = f"{cls_name} {conf:.2f}"
            tb = d.textbbox((0,0), txt, font=f)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            pad = max(2, fsz//5)
            ty = y1 - th - 2*pad
            if ty < 0: ty = y1
            d.rectangle([x1, ty, x1+tw+2*pad, ty+th+2*pad], fill=color)
            d.text((x1+pad, ty+pad-tb[1]), txt, fill=(8,9,11), font=f)
    return im

def header(im, text, color, count):
    W,H = im.size
    bar_h = max(34, int(H*0.075))
    canvas = Image.new("RGB",(W,H+bar_h), BG)
    canvas.paste(im,(0,bar_h))
    d = ImageDraw.Draw(canvas)
    f = font(int(bar_h*0.5))
    d.text((int(W*0.02), bar_h*0.5), text, font=f, fill=INK, anchor="lm")
    # count badge right
    badge = f"{count}명 검출"
    fb = font(int(bar_h*0.5))
    d.text((W-int(W*0.02), bar_h*0.5), badge, font=fb, fill=color, anchor="rm")
    return canvas

def run(model, path, conf, classes=None):
    r = model.predict(path, conf=conf, classes=classes, verbose=False)[0]
    out=[]
    for b in r.boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist(); c=float(b.conf[0])
        out.append((x1,y1,x2,y2,c))
    return out

def pair(img_path, stock, model_after, out_path, conf_before, conf_after,
         cls_before=None, cls_after=None, label_after="파인튜닝"):
    img = Image.open(img_path)
    bb = run(stock, img_path, conf_before, cls_before)
    ab = run(model_after, img_path, conf_after, cls_after)
    left  = header(draw_boxes(img, bb, EMBER), "기성 YOLO11s (파인튜닝 전)", EMBER, len(bb))
    right = header(draw_boxes(img, ab, TEAL),  label_after, TEAL, len(ab))
    gap=16
    W = left.width+right.width+gap; H=max(left.height,right.height)
    canvas = Image.new("RGB",(W,H),BG)
    canvas.paste(left,(0,0)); canvas.paste(right,(left.width+gap,0))
    canvas.save(out_path)
    print(f"{os.path.basename(img_path)}: before {len(bb)}  after {len(ab)}  -> {out_path}")
    return len(bb), len(ab)

# ── 사용법 (라이브러리) ───────────────────────────────────────────────
# pair()가 before/after 나란히 이미지를 그린다. 입력은 반드시 '원본(raw)' 프레임.
#
# 실사 박스 (real_person_ba.png):
#   real = YOLO(hf_hub_download("chanubc/overhead-person-yolo11","best.pt"))   # 실사 학습
#   stock = YOLO("yolo11s.pt")                                                  # 기성
#   # 입력 = Roboflow overhead-person-szky0 v3 test 원본 (dataset/overhead-person-v3/test/images)
#   pair(img, stock, real, out, conf_before=0.25, conf_after=0.25,
#        cls_before=[0], cls_after=None, label_after="실사 파인튜닝")
#
# 합성 박스 (sim_person_ba.png):
#   nadir = YOLO(hf_hub_download("chanubc/robot-kitchen-nadir-yolo11s","best.pt"))  # 합성 학습
#   # 입력 = sim.html 전체 나디르 캡처. 브라우저 콘솔에서:
#   #   const c=__sim.SURV.nzSE.cam; c.position.set(0,9,0); c.setTarget(new BABYLON.Vector3(0,0,0));
#   #   c.orthoLeft=-5.9;c.orthoRight=5.9;c.orthoTop=4.425;c.orthoBottom=-4.425;
#   #   const gt=await groundTruth("nzSE",{noDepth:true});  // gt.rgb = 오버레이 없는 clean PNG dataURL
#   pair(img, stock, nadir, out, conf_before=0.25, conf_after=0.40,
#        cls_before=[0], cls_after=[0], label_after="합성 나디르 학습")
