# blender_assets.py — 실사 프레임 기준 급식 교반솥 + 보호커버 로봇팔 생성
#
# 실행:  /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_assets.py
# 출력:  kettle.glb, robot_j1.glb ~ robot_j5.glb  (이 파일과 같은 폴더)
#
# 좌표 계약 (sim.html 앵커와 1:1):
#   - 단위 m, Blender Z-up → glTF Y-up 자동 변환
#   - kettle.glb   : 원점 = 솥 바닥 중심. 림 높이 0.76, 반경 0.44 (sim LAYOUT.fryer와 동일)
#   - robot_jN.glb : 원점 = 관절 피벗, 팔 방향 = +X
#       j2 상완 길이 0.55 / j3 하완 0.45 / j4 손목 0.16  (LAYOUT.robot)
#   - 국 수면(soup)은 시뮬 절차적 메시를 그대로 쓰므로 만들지 않는다

import bpy, math, os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for x in list(coll):
            coll.remove(x)

def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def mat(name, hexcol, metallic=0.0, rough=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    c = hex2rgb(hexcol)
    b.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    return m

def paint(o, m):
    o.data.materials.append(m)
    return o

def cyl(r, d, loc, m, rot=(0,0,0), verts=28, fill='NGON'):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d,
        location=loc, rotation=rot, end_fill_type=fill)
    return paint(bpy.context.active_object, m)

def cone(r1, r2, d, loc, m, rot=(0,0,0), verts=28):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2,
        depth=d, location=loc, rotation=rot)
    return paint(bpy.context.active_object, m)

def sph(r, loc, m, seg=24, ring=16, scale=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=ring, radius=r, location=loc)
    o = bpy.context.active_object
    if scale: o.scale = scale
    return paint(o, m)

def tor(maj, mino, loc, m, rot=(0,0,0), seg=36):
    bpy.ops.mesh.primitive_torus_add(major_radius=maj, minor_radius=mino,
        location=loc, rotation=rot, major_segments=seg, minor_segments=12)
    return paint(bpy.context.active_object, m)

def box(sx, sy, sz, loc, m, bev=0.008):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.scale = (sx, sy, sz)
    if bev:
        md = o.modifiers.new("b", "BEVEL"); md.width = bev; md.segments = 2
    return paint(o, m)

def export(name):
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, name),
        export_format='GLB', export_apply=True)
    print("EXPORTED", name)

# 공통 팔레트
def palette():
    return dict(
        steel  = mat("steel",  "#c9d0d8", 0.95, 0.32),
        steel2 = mat("steel2", "#aab2bb", 0.90, 0.42),
        dark   = mat("dark",   "#3c434d", 0.60, 0.55),
        navy   = mat("navy",   "#17223f", 0.00, 0.93),   # 보호커버 천 (실사 남색 — 씬 노출 보정 포함)
        navy2  = mat("navy2",  "#101a33", 0.00, 0.95),   # 커버 주름링/스트랩
        red    = mat("red",    "#c62828", 0.10, 0.45),
        yellow = mat("yellow", "#e8b820", 0.10, 0.50),
        green  = mat("green",  "#2f9e44", 0.10, 0.45),
    )

# ── 1. 급식 교반솥 (틸팅형 국솥) ──────────────────────────────────────────
def build_kettle():
    clear(); P = palette()
    # 받침대 + 스커트
    box(0.62, 0.56, 0.30, (0, 0, 0.19), P["dark"])
    box(0.66, 0.60, 0.05, (0, 0, 0.025), P["steel2"])
    # 전면 제어판 (경사)
    box(0.20, 0.03, 0.12, (0.02, -0.295, 0.27), P["steel"])
    cyl(0.011, 0.022, (-0.04, -0.315, 0.27), P["dark"],  rot=(math.pi/2,0,0), verts=14)
    cyl(0.011, 0.022, (-0.005,-0.315, 0.27), P["green"], rot=(math.pi/2,0,0), verts=14)
    # E-STOP — 노란 베이스 + 빨간 버섯버튼
    cyl(0.028, 0.012, (0.07, -0.318, 0.27), P["yellow"], rot=(math.pi/2,0,0), verts=18)
    cyl(0.019, 0.030, (0.07, -0.330, 0.27), P["red"],    rot=(math.pi/2,0,0), verts=18)
    # 솥 몸통 — 테이퍼 원통 + 둥근 바닥 + 말린 림 + 내벽(개방)
    cone(0.335, 0.44, 0.42, (0, 0, 0.53), P["steel"], verts=44)
    sph(0.335, (0, 0, 0.335), P["steel"], scale=(1, 1, 0.55))
    tor(0.443, 0.030, (0, 0, 0.755), P["steel"], seg=44)
    cyl(0.40, 0.30, (0, 0, 0.60), P["dark"], verts=40, fill='NOTHING')
    # 틸팅 트러니언 + 핸드휠(+X측) + 힌지 브래킷
    cyl(0.048, 0.11, ( 0.475, 0, 0.52), P["steel2"], rot=(0, math.pi/2, 0), verts=18)
    cyl(0.048, 0.11, (-0.475, 0, 0.52), P["steel2"], rot=(0, math.pi/2, 0), verts=18)
    box(0.05, 0.10, 0.24, ( 0.50, 0, 0.32), P["dark"])
    tor(0.115, 0.014, (0.585, 0, 0.52), P["steel"], rot=(0, math.pi/2, 0), seg=30)
    cyl(0.008, 0.215, (0.585, 0, 0.52), P["steel2"], rot=(math.pi/2, 0, 0), verts=10)
    cyl(0.008, 0.215, (0.585, 0, 0.52), P["steel2"], rot=(0, 0, 0), verts=10)
    sph(0.022, (0.585, 0, 0.52), P["dark"], seg=14, ring=10)
    # 환기 후드 (4각 프러스텀) + 필터 그릴 + 덕트
    cone(0.62, 0.46, 0.30, (0, 0, 1.86), P["steel2"], rot=(0, 0, math.pi/4), verts=4)
    for i in range(3):
        box(0.26, 0.30, 0.008, (-0.28 + i*0.28, 0, 1.715), P["dark"], bev=0)
    cyl(0.13, 1.15, (0, 0, 2.58), P["steel2"], verts=22)
    export("kettle.glb")

# ── 2. 로봇팔 — Isaac Sim 기본 에셋(Franka류) 참조 디자인 ─────────────────
#    흰 유광 셸 + 차콜 관절 링. 관절별 분할 (원점=피벗, 팔=+X)
def smooth():
    bpy.ops.object.shade_smooth()

def s_sph(r, loc, m, seg=28, ring=18, scale=None):
    o = sph(r, loc, m, seg=seg, ring=ring, scale=scale); smooth(); return o

def s_cyl(r, d, loc, m, rot=(0,0,0), verts=32, fill='NGON'):
    o = cyl(r, d, loc, m, rot=rot, verts=verts, fill=fill); smooth(); return o

def s_cone(r1, r2, d, loc, m, rot=(0,0,0), verts=32):
    o = cone(r1, r2, d, loc, m, rot=rot, verts=verts); smooth(); return o

def franka_link(P, length, r0, r1, x0=0.06):
    """+X 테이퍼 캡슐 링크 — 흰 셸 + 양끝 차콜 밴드"""
    s_cone(r0, r1, length - x0, ((length + x0)/2, 0, 0), P["white"], rot=(0, -math.pi/2, 0))
    s_sph(r0*1.01, (x0, 0, 0), P["white"])
    s_sph(r1*1.01, (length, 0, 0), P["white"])
    tor(r0*1.02, 0.010, (x0 + (length-x0)*0.12, 0, 0), P["char"], rot=(0, math.pi/2, 0), seg=28)
    tor(r1*1.02, 0.010, (length - (length-x0)*0.10, 0, 0), P["char"], rot=(0, math.pi/2, 0), seg=28)

def franka_joint(P, r, w):
    """관절 하우징 — Y축(회전축) 흰 원통 + 둥근 캡 + 차콜 중앙 밴드"""
    s_cyl(r, w, (0, 0, 0), P["white"], rot=(math.pi/2, 0, 0))
    s_sph(r*1.005, (0,  w/2, 0), P["white"], scale=(1, 0.5, 1))
    s_sph(r*1.005, (0, -w/2, 0), P["white"], scale=(1, 0.5, 1))
    tor(r*1.02, 0.013, (0, 0, 0), P["char"], rot=(math.pi/2, 0, 0), seg=30)

def arm_palette():
    P = palette()
    P["white"] = mat("shellW", "#eef1f4", 0.05, 0.30)   # Franka류 유광 화이트 셸
    P["char"]  = mat("charB",  "#26282c", 0.20, 0.48)   # 차콜 관절 밴드
    return P

def build_j1():
    clear(); P = arm_palette()
    s_cone(0.145, 0.115, 0.07, (0, 0, 0.035), P["char"], verts=36)   # 차콜 베이스 스커트
    s_cone(0.115, 0.095, 0.09, (0, 0, 0.115), P["white"], verts=36)  # 흰 터릿
    tor(0.10, 0.011, (0, 0, 0.16), P["char"], seg=30)
    export("robot_j1.glb")

def build_j2():
    clear(); P = arm_palette()
    franka_joint(P, 0.105, 0.22)
    franka_link(P, 0.55, 0.085, 0.072)
    export("robot_j2.glb")

def build_j3():
    clear(); P = arm_palette()
    franka_joint(P, 0.090, 0.19)
    franka_link(P, 0.45, 0.070, 0.058)
    export("robot_j3.glb")

def build_j4():
    clear(); P = arm_palette()
    franka_joint(P, 0.075, 0.15)
    franka_link(P, 0.16, 0.054, 0.048, x0=0.035)
    export("robot_j4.glb")

def build_j5():
    clear(); P = arm_palette()
    s_sph(0.052, (0.01, 0, 0), P["white"])
    s_cyl(0.048, 0.070, (0.055, 0, 0), P["char"],  rot=(0, math.pi/2, 0))   # 차콜 플랜지
    s_cyl(0.052, 0.014, (0.093, 0, 0), P["steel2"], rot=(0, math.pi/2, 0))  # 툴 마운트
    export("robot_j5.glb")

build_kettle()
build_j1(); build_j2(); build_j3(); build_j4(); build_j5()
print("ALL DONE")
