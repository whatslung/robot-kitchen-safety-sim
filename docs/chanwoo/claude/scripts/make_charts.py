# -*- coding: utf-8 -*-
"""
발표용 before/after 차트 생성 — 검증된 정본 수치만 사용.
출처: robot-kitchen-safety-sim/docs/chanwoo/{detection-eval, nadir-zone-fusion, prediction-eval, prediction-safety-eval}.md
      kitchen-fire-noise-poc/docs/{SUMMARY_meeting, TIMELINE, README}.md
스타일: 잡스 무대 톤 (검정 배경 · 엠버/틸)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---- palette (matches 미리 물러서다 artifact) ----
BG      = "#0E0F12"
INK     = "#F4F1EA"
MUTED   = "#8B8378"
GRID    = "#22242A"
GREY    = "#565149"   # before / baseline
EMBER   = "#FF6A32"   # 위험/정지/hero after
TEAL    = "#3CC7C0"   # 안전/검출 win
AMBERL  = "#F2B705"   # 참고(context)

OUT = os.path.join(os.path.dirname(__file__), "_generated")  # 생성 후 person/motion 폴더로 분류
os.makedirs(OUT, exist_ok=True)

def base_ax(figsize=(7.2, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax

def title(ax, t, sub=None):
    ax.set_title(t, color=INK, fontsize=15, fontweight="bold", pad=34, loc="left")
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, color=MUTED,
                fontsize=10.5, ha="left", va="bottom")

def source(fig, s):
    fig.text(0.012, 0.012, s, color=MUTED, fontsize=7.6, ha="left")

def save(fig, name):
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor=BG, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", p)

def barlabels(ax, bars, vals, fmt="{:.3f}", dy=0.012, color=INK, size=13):
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+dy, fmt.format(v),
                ha="center", va="bottom", color=color, fontsize=size, fontweight="bold")

# =====================================================================
# 1. 사람 검출 — 기성 vs 파인튜닝 (합성 val)  [recall + precision]
# =====================================================================
fig, ax = base_ax()
groups = ["Recall (재현율)", "Precision (정밀도)"]
stock  = [0.175, 0.374]
tuned  = [0.871, 0.872]
x = range(len(groups)); w = 0.36
b1 = ax.bar([i-w/2 for i in x], stock, w, color=GREY,  label="기성 YOLO11s (파인튜닝 전)")
b2 = ax.bar([i+w/2 for i in x], tuned, w, color=TEAL,  label="파인튜닝 후 (합성 나디르)")
barlabels(ax, b1, stock); barlabels(ax, b2, tuned)
ax.set_xticks(list(x)); ax.set_xticklabels(groups, color=INK, fontsize=12)
ax.set_ylim(0, 1.05); ax.set_ylabel("점수", color=MUTED)
title(ax, "천장 사람 검출 · 기성 vs 파인튜닝", "합성 val — 10명 중 8명 놓치던 것을 대부분 검출  ·  recall 0.175 → 0.871")
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=10, loc="upper left")
source(fig, "출처: detection-eval.md §1(기성 0.175) · §6 섬배치 재학습(파인튜닝 0.871) · SIM in-domain")
save(fig, "1_person_finetune.png")

# =====================================================================
# 2. 사람 검출 — 공간 4분할·월드융합·추적 사다리 (recall)
# =====================================================================
fig, ax = base_ax(figsize=(8.2, 4.6))
stages = ["단일 검출", "4대 커버리지", "월드좌표 융합", "+시간축 추적", "배포 추적기"]
vals   = [0.831, 0.900, 0.940, 0.960, 0.988]
xs = range(len(stages))
ax.plot(xs, vals, color=TEAL, lw=2.5, marker="o", ms=9, mfc=TEAL, mec=BG, zorder=3)
for i, v in zip(xs, vals):
    ax.text(i, v+0.006, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=12.5, fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels(stages, color=INK, fontsize=11)
ax.set_ylim(0.80, 1.0); ax.set_ylabel("Recall", color=MUTED)
title(ax, "공간 4분할 · 월드융합 · 추적 사다리", "합성/존 기준 — 나눠 보고, 하나로 합치고, 시간으로 잇는다  ·  0.831 → 0.988")
source(fig, "출처: nadir-zone-fusion.md §5-7~§5-9 · 잔차 5cm · 합성/존 기준")
save(fig, "2_person_fusion_ladder.png")

# =====================================================================
# 3. 사람 검출 — 실사 전이 (정직성)  recall @ 실사 test 137
# =====================================================================
fig, ax = base_ax(figsize=(7.6, 4.6))
labels = ["합성만\n(sim-only)", "실사+합성\n(현행 YOLO)", "RF-DETR\n(참고·실사500)"]
vals   = [0.270, 0.844, 0.917]
cols   = [GREY, TEAL, AMBERL]
bars = ax.bar(labels, vals, color=cols, width=0.6)
barlabels(ax, bars, vals)
ax.set_ylim(0, 1.0); ax.set_ylabel("Recall (실사 test 137장)", color=MUTED)
title(ax, "실사 전이 · 정직성", "합성만으론 붕괴(0.27) → 실사 소량을 얹어야 값을 한다  ·  실사 검출 회복이 남은 과제")
ax.text(0, 0.30, "합성 특징이\n실사에 전이 안 됨", ha="center", va="bottom", color=MUTED, fontsize=9)
source(fig, "출처: detection-eval.md §3 · §5-2 · 평가셋=실사 overhead test 137장(학습 0)")
save(fig, "3_person_real_transfer.png")

# =====================================================================
# 4. 예측 — 위험 진입 recall (풀 파이프라인 4대)  [HERO 57→93]
# =====================================================================
fig, ax = base_ax(figsize=(7.6, 4.8))
labels = ["등속(CV)\n직선 예측", "칼만", "학습형\n(LSTM)"]
vals   = [0.571, 0.696, 0.929]
cols   = [GREY, MUTED, EMBER]
bars = ax.bar(labels, vals, color=cols, width=0.58)
barlabels(ax, bars, vals, fmt="{:.1%}", size=14)
ax.set_ylim(0, 1.0); ax.set_ylabel("위험 진입 예측 recall", color=MUTED)
title(ax, "위험 진입, 절반 놓침 → 10에 9  ", "풀 파이프라인 4대 · 놓친 진입 = 충돌  ·  57% → 93%")
ax.annotate("FN 48명", xy=(0, 0.571), xytext=(0, 0.40), ha="center", color=MUTED, fontsize=9)
ax.annotate("FN 8명",  xy=(2, 0.929), xytext=(2, 0.78), ha="center", color=INK, fontsize=9)
source(fig, "출처: nadir-zone-fusion.md §5-15 풀 파이프라인(추적→예측→안전) · 로봇 원점 · 정지링 3.1m")
save(fig, "4_pred_entry_recall.png")

# =====================================================================
# 5. 예측 — ADE@1.6s (낮을수록 좋음)
# =====================================================================
fig, ax = base_ax(figsize=(7.6, 4.6))
labels = ["등속(CV)", "칼만", "학습형\nLSTM", "학습형\nTransformer"]
vals   = [0.372, 0.298, 0.242, 0.217]
cols   = [GREY, MUTED, TEAL, EMBER]
bars = ax.bar(labels, vals, color=cols, width=0.62)
barlabels(ax, bars, vals, fmt="{:.3f} m", dy=0.004, size=12)
ax.set_ylim(0, 0.45); ax.set_ylabel("ADE@1.6s (m · 낮을수록 좋음)", color=MUTED)
title(ax, "예측 오차 · 라이브 제어 지평선 1.6초", "안전 판단과 같은 지평선 · test 최종  ·  Transformer 0.217m")
source(fig, "출처: prediction-eval.md 1.6s test · P0-1 split · scene 단위 95%CI")
save(fig, "5_pred_ade.png")

print("charts 1-5 ->", OUT, "  (화재는 fire_chart.py — 실사 0.899 기준)")
