# -*- coding: utf-8 -*-
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
BG="#0E0F12";INK="#F4F1EA";MUTED="#8B8378";GRID="#22242A";GREY="#565149";EMBER="#FF6A32";TEAL="#3CC7C0"
OUT=r"E:\VsCodeProjects\robot-kitchen-safety-sim\docs\chanwoo\claude\fire\fire_before_after.png"
fig,ax=plt.subplots(figsize=(7.4,4.7),dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED,labelsize=11); ax.yaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
groups=["Recall (놓치지 않는 비율)","Precision (검출이 맞는 비율)"]
before=[0.237,0.842]; after=[0.899,0.979]
x=range(len(groups)); w=0.36
b1=ax.bar([i-w/2 for i in x],before,w,color=GREY,label="합성-only (실사에 약함)")
b2=ax.bar([i+w/2 for i in x],after,w,color=TEAL,label="실사 학습 (채택)")
for b,v in list(zip(b1,before))+list(zip(b2,after)):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.012,f"{v:.1%}",ha="center",va="bottom",color=INK,fontsize=13,fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(groups,color=INK,fontsize=11.5)
ax.set_ylim(0,1.08); ax.set_ylabel("점수",color=MUTED)
ax.set_title("화재 검출 · 합성만으론 부족, 실사로 해결",color=INK,fontsize=15,fontweight="bold",pad=34,loc="left")
ax.text(0.0,1.015,"같은 Indoor Fire Smoke grouped test · recall 23.7% → 89.9%",transform=ax.transAxes,color=MUTED,fontsize=10.5,ha="left",va="bottom")
ax.legend(facecolor=BG,edgecolor=GRID,labelcolor=INK,fontsize=10,loc="upper center")
fig.text(0.012,0.012,"출처: kitchen-fire-noise-poc b0c9d726 · AFTER_meeting.md §5B · YOLOv8s 60ep · conf 0.25 · frame-level · fire test 358장",color=MUTED,fontsize=7.4,ha="left")
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(OUT,facecolor=BG,dpi=200,bbox_inches="tight",pad_inches=0.25)
print("saved",OUT)
