"""Render an LSTM/Transformer trajectory training-sample into a clean BEV figure.

Reads one sim scene (3 people, 150 steps @0.4s, metre coords + goal stations)
and emits dataset_figure.html — two panels:
  A) whole scene: 3 goal-directed trajectories, stations, robot + safety rings
  B) one obs8 -> pred12 window (the actual LSTM training unit), curved segment
Style matches figures.html (same palette / fonts). Screenshot -> PNG afterwards.
"""
import json, math, sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim/.claude/worktrees/hohho-64cbc5/dataset/trajectories/island_h58_seed10_0009.json")
OUT = Path(r"C:/Users/chanwoo/Downloads/pipeline-figure-assets/dataset_figure.html")

sc = json.loads(SRC.read_text())
half = sc["half"]                       # room half-size (m)
robot = sc["robot"]
nodes = [n for n in sc["nodes"] if not n.get("discarded")]

PAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue / orange / aqua (validated categorical)
STOP, SLOW = 3.10, 3.90

# ---- panel A geometry: metre -> svg ----
A = 620                                  # plot square px
PAD = 16
def ax(x): return PAD + (x + half) / (2 * half) * A
def az(z): return PAD + (half - z) / (2 * half) * A   # +z up

# unique goal stations
goals = {}
for n in nodes:
    for f in n["frames"]:
        goals.setdefault(f["goal"], (f["gx"], f["gz"]))

def polyline(frames, col, w):
    pts = " ".join(f"{ax(f['x']):.1f},{az(f['z']):.1f}" for f in frames)
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>'

svgA = [f'<svg width="{A+2*PAD}" height="{A+2*PAD}" viewBox="0 0 {A+2*PAD} {A+2*PAD}">']
# room
svgA.append(f'<rect x="{PAD}" y="{PAD}" width="{A}" height="{A}" fill="#fbfbfa" stroke="#e1e0d9" stroke-width="1.5"/>')
# safety rings around robot
rx, rz = ax(robot["x"]), az(robot["z"])
for r, col in [(SLOW, "#eda100"), (STOP, "#e34948")]:
    rr = r / (2 * half) * A
    svgA.append(f'<circle cx="{rx:.1f}" cy="{rz:.1f}" r="{rr:.1f}" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.7"/>')
# goal stations
for name, (gx, gz) in goals.items():
    x, y = ax(gx), az(gz)
    svgA.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" rx="1.5" fill="#c3c2b7"/>')
    svgA.append(f'<text x="{x+7:.1f}" y="{y+4:.1f}" font-size="12" fill="#898781" font-family="system-ui,sans-serif">{name}</text>')
# trajectories
for i, n in enumerate(nodes):
    fr = n["frames"]
    svgA.append(polyline(fr, PAL[i], 2.5))
    sx, sy = ax(fr[0]["x"]), az(fr[0]["z"])
    ex, ey = ax(fr[-1]["x"]), az(fr[-1]["z"])
    svgA.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#fbfbfa" stroke="{PAL[i]}" stroke-width="2.5"/>')  # start hollow
    svgA.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" fill="{PAL[i]}"/>')                                       # end solid
# robot marker
svgA.append(f'<circle cx="{rx:.1f}" cy="{rz:.1f}" r="6" fill="#0b0b0b"/>')
svgA.append(f'<text x="{rx+9:.1f}" y="{rz-7:.1f}" font-size="12.5" font-weight="700" fill="#0b0b0b" font-family="system-ui,sans-serif">robot</text>')
svgA.append('</svg>')
svgA = "\n".join(svgA)

# ---- panel B: pick the most-curved obs8->pred12 window across all people ----
OBS, PREDN = 8, 12
best = None
for i, n in enumerate(nodes):
    fr = n["frames"]
    for s in range(0, len(fr) - (OBS + PREDN)):
        win = fr[s:s + OBS + PREDN]
        if sum(1 for f in win if not f["moving"]) > 3:   # skip mostly-standing windows
            continue
        # total heading change over the future part
        turn = 0.0
        for k in range(OBS, OBS + PREDN - 1):
            ax1, az1 = win[k]["x"] - win[k-1]["x"], win[k]["z"] - win[k-1]["z"]
            ax2, az2 = win[k+1]["x"] - win[k]["x"], win[k+1]["z"] - win[k]["z"]
            a1, a2 = math.atan2(az1, ax1), math.atan2(az2, ax2)
            d = abs(a2 - a1)
            turn += min(d, 2*math.pi - d)
        if best is None or turn > best[0]:
            best = (turn, i, win)
_, bi, win = best
obs, fut = win[:OBS], win[OBS-1:]        # share the junction point

# panel B geometry: local bbox with margin
xs = [f["x"] for f in win]; zs = [f["z"] for f in win]
minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
spanx, spanz = maxx - minx, maxz - minz
span = max(spanx, spanz, 1.0) * 1.25
cx, cz = (minx + maxx) / 2, (minz + maxz) / 2
B = 620
def bx(x): return PAD + (x - (cx - span/2)) / span * B
def bz(z): return PAD + ((cz + span/2) - z) / span * B

def bpoly(frames, col, w, dash=None):
    pts = " ".join(f"{bx(f['x']):.1f},{bz(f['z']):.1f}" for f in frames)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{d}/>'

svgB = [f'<svg width="{B+2*PAD}" height="{B+2*PAD}" viewBox="0 0 {B+2*PAD} {B+2*PAD}">']
svgB.append(f'<rect x="{PAD}" y="{PAD}" width="{B}" height="{B}" fill="#fbfbfa" stroke="#e1e0d9" stroke-width="1.5"/>')
svgB.append(bpoly(fut, "#898781", 2.5, dash="2 7"))      # future (ground truth) dashed grey under
svgB.append(bpoly(obs, "#2a78d6", 3.5))                   # observed solid blue
for f in obs:
    svgB.append(f'<circle cx="{bx(f["x"]):.1f}" cy="{bz(f["z"]):.1f}" r="5" fill="#2a78d6"/>')
for f in fut[1:]:
    svgB.append(f'<circle cx="{bx(f["x"]):.1f}" cy="{bz(f["z"]):.1f}" r="4.5" fill="#fbfbfa" stroke="#898781" stroke-width="2"/>')
# mark "now"
jx, jz = bx(obs[-1]["x"]), bz(obs[-1]["z"])
svgB.append(f'<circle cx="{jx:.1f}" cy="{jz:.1f}" r="7" fill="none" stroke="#0b0b0b" stroke-width="2"/>')
svgB.append(f'<text x="{jx+11:.1f}" y="{jz-6:.1f}" font-size="13" font-weight="700" fill="#0b0b0b" font-family="system-ui,sans-serif">now</text>')
svgB.append('</svg>')
svgB = "\n".join(svgB)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Dataset sample</title>
<style>
 :root{{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--font:system-ui,-apple-system,"Segoe UI",sans-serif}}
 *{{box-sizing:border-box}} body{{margin:0;background:#dfe1e4;font-family:var(--font);color:var(--ink)}}
 #fig4{{width:1400px;background:#fff;margin:32px auto;padding:34px 40px 30px}}
 .figtitle{{font-size:26px;font-weight:750;letter-spacing:-.01em;margin:0 0 4px}}
 .figsub{{font-size:16px;color:var(--ink2);margin:0 0 22px;font-weight:450}}
 .cols{{display:flex;gap:34px;align-items:flex-start}}
 .col{{flex:1 1 0;min-width:0}}
 .ptitle{{font-size:16px;font-weight:700;margin:0 0 4px}}
 .psub{{font-size:13px;color:var(--muted);margin:0 0 10px}}
 .legend{{display:flex;flex-wrap:wrap;gap:16px;font-size:13.5px;color:var(--ink2);margin:12px 2px 0}}
 .k{{display:inline-flex;align-items:center;gap:7px}} .sw{{width:22px;height:0;border-top-width:3px;border-top-style:solid;display:inline-block}}
 .dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
 .figcap{{font-size:14px;color:var(--ink2);line-height:1.5;margin:20px 2px 0;max-width:1280px}}
 .figcap b{{color:var(--ink);font-weight:650}}
 svg{{display:block;width:100%;height:auto}}
</style></head><body>
<section id="fig4">
 <h2 class="figtitle">Trajectory training sample (LSTM / Transformer)</h2>
 <p class="figsub">One simulated kitchen scene: goal-directed workers moving station&ndash;to&ndash;station &mdash; the multi-goal behaviour the learned predictor must model.</p>
 <div class="cols">
  <div class="col">
   <div class="ptitle">A &nbsp;Whole scene &middot; {len(nodes)} workers, 150 steps (0.4&nbsp;s)</div>
   <div class="psub">metre coordinates &middot; hollow = start, filled = end &middot; squares = goal stations</div>
   {svgA}
   <div class="legend">
    {" ".join(f'<span class="k"><span class="sw" style="border-top-color:{PAL[i]}"></span>worker&nbsp;{i+1} ({n["job"]})</span>' for i,n in enumerate(nodes))}
    <span class="k"><span class="dot" style="background:#0b0b0b"></span>robot</span>
    <span class="k"><span class="sw" style="border-top-color:#e34948;border-top-style:dashed"></span>stop 3.1&nbsp;m</span>
    <span class="k"><span class="sw" style="border-top-color:#eda100;border-top-style:dashed"></span>slow 3.9&nbsp;m</span>
   </div>
  </div>
  <div class="col">
   <div class="ptitle">B &nbsp;One training window &middot; observe 8 &rarr; predict 12</div>
   <div class="psub">the sliding-window unit fed to the model (3.2&nbsp;s in &rarr; 4.8&nbsp;s out)</div>
   {svgB}
   <div class="legend">
    <span class="k"><span class="sw" style="border-top-color:#2a78d6"></span>observed &middot; 8 steps (3.2&nbsp;s)</span>
    <span class="k"><span class="sw" style="border-top-color:#898781;border-top-style:dashed"></span>future ground truth &middot; 12 steps (4.8&nbsp;s)</span>
   </div>
  </div>
 </div>
 <p class="figcap"><b>Why this shapes the model.</b> Each worker chains goals (fridge&rarr;isle&rarr;kettle&hellip;), so a path bends
 at every station &mdash; exactly where straight-line prediction fails. The model slides an 8-step window over every track
 (32,488 train / 8,646 val windows) and learns to predict the next 12 steps. Sample scene: <code>{sc["scene_id"]}</code> (val split).</p>
</section>
<script>if(location.hash==='#2x'){{document.documentElement.style.zoom='2';}}</script>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print("wrote", OUT)
print("panelB person idx", bi, "of", len(nodes))
