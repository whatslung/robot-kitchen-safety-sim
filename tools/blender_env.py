# blender_env.py — 급식실 주변 설비(로봇 셀 밖) 생성
#
# 실행:  /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_env.py
# 출력:  assets/env_*.glb
#
# 근거 프레임 (조리데이터_영상에서 추출):
#   참조 CCTV  — 셀 전경. 이웃 국솥 2대, 바스켓 카트, 그레이팅, 마킹라인
#   참조 현장_국탕 — 통로 좌측 스테인리스 작업대·선반 라인, 재료 호텔팬
#   복수 참조 현장 — 호텔팬 3단 랙 카트, 벽 게시물, 소방 PULL
#   복수 참조 현장  — 벽걸이 조리도구(빨간 손잡이), 백스플래시
#
# 좌표 계약:
#   - 단위 m, Blender Z-up → glTF Y-up 자동 변환
#   - 각 프롭은 자기 원점(바닥 중심)에 세워 만든다. 월드 배치는 sim.html ENV_PROPS가 한다.
#   - 프롭 정면 = Blender -Y  (→ Babylon +Z). sim에서 rotation.y로 방향만 돌린다.
#   - 오브젝트 이름 접두어 Env* → sim classOfMesh가 GT 'equip'으로 잡는다

import bpy, math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

# ── 헬퍼 (blender_assets.py 규약 그대로) ──────────────────────────────────
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

def box(sx, sy, sz, loc, m, bev=0.006, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.scale = (sx, sy, sz)
    if bev:
        # 세그먼트 1 — 2로 두면 모서리마다 면이 2배가 된다. 6mm 베벨은 하이라이트를
        # 맺는 게 목적이라 1세그먼트로 충분하고, 상자당 삼각형이 약 절반으로 준다.
        md = o.modifiers.new("b", "BEVEL"); md.width = bev; md.segments = 1
    return paint(o, m)

def cyl(r, d, loc, m, rot=(0,0,0), verts=18, fill='NGON'):   # 기본 24 → 18. 큰 원통은 호출부에서 명시한다
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d,
        location=loc, rotation=rot, end_fill_type=fill)
    return paint(bpy.context.active_object, m)

def cone(r1, r2, d, loc, m, rot=(0,0,0), verts=28):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2,
        depth=d, location=loc, rotation=rot)
    return paint(bpy.context.active_object, m)

def tor(maj, mino, loc, m, rot=(0,0,0), seg=32):
    bpy.ops.mesh.primitive_torus_add(major_radius=maj, minor_radius=mino,
        location=loc, rotation=rot, major_segments=seg, minor_segments=10)
    return paint(bpy.context.active_object, m)

def sph(r, loc, m, seg=14, ring=10, scale=None, rot=None):   # 20/14 → 14/10 (작은 구가 대부분)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=ring, radius=r, location=loc)
    o = bpy.context.active_object
    if scale: o.scale = scale           # Blender TRS: 스케일이 먼저, 회전이 나중 → 납작하게 눌러서 기울인다
    if rot: o.rotation_euler = rot
    return paint(o, m)

# ── boolean 헬퍼 ────────────────────────────────────────────────────────────
#   프리미티브를 겹쳐 쌓기만 하면 **속이 꽉 찬 덩어리**가 된다. 그릇처럼 안이 비어야
#   하는 물건은 반드시 파내야 한다 — 안 그러면 뚜껑을 열어도 입구가 막혀 보인다.
def apply_scale(o):
    """오브젝트 스케일을 메시에 굽는다. boolean 은 스케일이 남아 있으면 결과가 틀어진다."""
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o

def bake(o):
    """스케일과 기존 모디파이어(베벨 등)를 메시에 굽는다.

    ⚠️ **boolean 전에 반드시 부른다.** 베벨이 스택에 남은 채 boolean 을 뒤에 얹어
       적용하면 Blender 가 '첫 모디파이어가 아니다'로 처리해 결과가 틀어진다 —
       실제로 상판이 바깥쪽까지 통째로 깎여 얇은 고리만 남았다."""
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for md in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=md.name)
    return o

def _boolean(a, b, op):
    """a 에 b 를 boolean 연산하고 b 를 지운다."""
    bake(a); bake(b)
    md = a.modifiers.new("bool", 'BOOLEAN')
    md.operation = op; md.object = b; md.solver = 'EXACT'
    bpy.ops.object.select_all(action='DESELECT')
    a.select_set(True); bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=md.name)
    bpy.ops.object.select_all(action='DESELECT')
    b.select_set(True); bpy.context.view_layer.objects.active = b
    bpy.ops.object.delete(use_global=False)
    return a

def fuse(objs):
    """겹친 프리미티브들을 **합집합**으로 하나의 매니폴드 덩어리로 만든다.

    ⚠️ `bpy.ops.object.join()`을 쓰면 안 된다. join 은 메시를 한 오브젝트에 담기만 할 뿐
       겹친 면을 정리하지 않아 **자기교차 비매니폴드**가 된다. 그 상태로 차집합을 걸면
       EXACT 솔버가 조용히 실패하고 결과가 두 덩어리의 합집합처럼 나온다 — 실제로
       커터(림 위 원통)가 볼에 그대로 붙어 z 1.50 까지 뻗은 메시가 나왔다.
       내보내기 로그의 'Mesh ... is not valid' 경고가 그 신호다."""
    a = objs[0]
    for b in objs[1:]:
        a = _boolean(a, b, 'UNION')
    return a

def carve(target, cutter):
    """target 에서 cutter 부피를 빼낸다 (차집합). 커터는 쓰고 나서 반드시 지운다.

    ⚠️ 커터를 안 지우면 **그대로 GLB에 실려 나간다.** 한 번 겪었다 — 입구를 여는
       원통 커터(림 위 0.90~1.50)가 남아, 뚜껑을 열어도 입구가 0.90에서 다시 막혔다.
       커터는 이름으로 찾아 지우고, export 직전에 남은 게 없는지 한 번 더 쓸어낸다."""
    cutter.name = "__cut"
    return _boolean(target, cutter, 'DIFFERENCE')

def P():
    return dict(
        steel  = mat("steel",  "#c9d0d8", 0.92, 0.34),
        steel2 = mat("steel2", "#aab2bb", 0.88, 0.44),
        dark   = mat("dark",   "#3c434d", 0.55, 0.58),
        black  = mat("black",  "#1c2126", 0.20, 0.70),
        red    = mat("red",    "#c62828", 0.10, 0.45),
        blue   = mat("blue",   "#2f6fd0", 0.10, 0.45),
        yellow = mat("yellow", "#e8b820", 0.10, 0.50),
        food   = mat("food",   "#c8a25a", 0.00, 0.80),
        white  = mat("white",  "#e8ebee", 0.05, 0.45),
    )

def export(fname, prefix):
    """씬의 모든 오브젝트를 Env 접두어로 개명 후 GLB 내보내기 (GT 분류가 이름을 본다)"""
    # boolean 커터 잔재 제거 — 남아 있으면 그대로 GLB에 실린다 (carve 주석 참조)
    for o in [x for x in bpy.context.scene.objects if x.name.startswith("__cut")]:
        print("  ⚠ 커터 잔재 제거:", o.name)
        bpy.data.objects.remove(o, do_unlink=True)
    for i, o in enumerate(list(bpy.context.scene.objects)):
        o.name = "%s_%02d" % (prefix, i)
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, fname),
        export_format='GLB', export_apply=True)
    print("EXPORTED", fname)

def casters(m, n, x0, dx, y0, dy):
    """이동 설비 캐스터 4개 (바퀴 φ100)"""
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(0.05, 0.026, (sx*x0, sy*y0, 0.05), m, rot=(0, math.pi/2, 0), verts=14)

# ── 1. 2조 싱크대 — 1800×750×850, 볼 2개 + 구스넥 수전 (복수 참조 현장) ──
def build_sink():
    clear(); p = P()
    W, D, H = 1.80, 0.75, 0.85
    box(W, D, 0.04, (0, 0, H-0.02), p["steel"])                 # 상판
    box(W, 0.04, 0.15, (0, D/2-0.02, H+0.075), p["steel"])      # 백가드
    for sx in (-0.42, 0.42):                                    # 볼 2조 (개구부 + 내벽)
        box(0.50, 0.45, 0.02, (sx, -0.02, H-0.27), p["steel2"])
        for ex, ey, sxx, syy in ((0.25,0,0.02,0.45), (-0.25,0,0.02,0.45),
                                 (0,0.225,0.50,0.02), (0,-0.225,0.50,0.02)):
            box(sxx, syy, 0.26, (sx+ex, ey, H-0.14), p["steel2"])
        cyl(0.022, 0.03, (sx, -0.02, H-0.27), p["dark"], verts=14)   # 배수구
    for sx in (-1, 1):                                          # 다리 + 하단 선반
        for sy in (-1, 1):
            box(0.04, 0.04, H-0.04, (sx*(W/2-0.06), sy*(D/2-0.06), (H-0.04)/2), p["steel2"])
    box(W-0.14, D-0.14, 0.02, (0, 0, 0.22), p["steel2"])
    # 구스넥 수전 — 벽쪽 중앙
    cyl(0.035, 0.06, (0, D/2-0.10, H+0.03), p["steel"], verts=16)
    cyl(0.020, 0.42, (0, D/2-0.10, H+0.24), p["steel"], verts=14)
    tor(0.13, 0.020, (0, D/2-0.23, H+0.45), p["steel"], rot=(0, math.pi/2, 0), seg=24)
    cyl(0.018, 0.10, (0, D/2-0.36, H+0.40), p["steel"], verts=14)
    for sx in (-0.10, 0.10):                                    # 밸브 레버
        box(0.11, 0.02, 0.02, (sx, D/2-0.10, H+0.08), p["blue" if sx < 0 else "red"])
    export("env_sink.glb", "EnvSink")

# ── 2. 스테인리스 작업대 — 1500×750×850, 하단 선반 1단 (참조 현장 통로 라인) ──
def build_table():
    clear(); p = P()
    W, D, H = 1.50, 0.75, 0.85
    box(W, D, 0.04, (0, 0, H-0.02), p["steel"])
    box(W, 0.04, 0.10, (0, D/2-0.02, H+0.05), p["steel"])       # 뒷턱
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(0.04, 0.04, H-0.04, (sx*(W/2-0.06), sy*(D/2-0.06), (H-0.04)/2), p["steel2"])
    box(W-0.14, D-0.14, 0.02, (0, 0, 0.24), p["steel2"])        # 하단 선반
    export("env_table.glb", "EnvTable")

# ── 3. 앵글 선반 4단 — 1200×600×1800 (참조 현장 좌측 벽면) ───────────────────
def build_rack():
    clear(); p = P()
    W, D, H = 1.20, 0.60, 1.80
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(0.04, 0.04, H, (sx*(W/2-0.04), sy*(D/2-0.04), H/2), p["steel2"])
    for k in range(4):
        z = 0.30 + k*0.48
        box(W-0.04, D-0.04, 0.025, (0, 0, z), p["steel"])
        for i in range(9):                                      # 와이어 선반 결
            box(0.012, D-0.06, 0.014, (-W/2+0.10+i*0.125, 0, z+0.02), p["steel2"], bev=0)
    export("env_rack.glb", "EnvRack")

# ── 4. 호텔팬 랙 카트 3단 — 700×500×1400 + GN1/1 팬 6개 (복수 참조 현장) ─
def build_pancart():
    clear(); p = P()
    W, D, H = 0.70, 0.50, 1.40
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(0.03, 0.03, H-0.10, (sx*(W/2-0.03), sy*(D/2-0.03), 0.10+(H-0.10)/2), p["steel2"])
    box(W, 0.03, 0.03, (0, -D/2+0.03, H+0.02), p["steel"])      # 밀대 손잡이
    box(W, 0.03, 0.03, (0,  D/2-0.03, H+0.02), p["steel"])
    for k in range(3):                                          # 3단 레일 + 호텔팬 2개씩
        z = 0.42 + k*0.34
        for sy in (-1, 1):
            box(W-0.06, 0.03, 0.02, (0, sy*(D/2-0.05), z), p["steel2"], bev=0)
        for sx in (-0.16, 0.16):
            box(0.30, 0.42, 0.02, (sx, 0, z+0.02), p["steel"])          # 팬 바닥
            for ex, ey, sxx, syy in ((0.15,0,0.02,0.42), (-0.15,0,0.02,0.42),
                                     (0,0.21,0.30,0.02), (0,-0.21,0.30,0.02)):
                box(sxx, syy, 0.10, (sx+ex, ey, z+0.07), p["steel"])    # 팬 벽
            box(0.26, 0.38, 0.05, (sx, 0, z+0.055), p["food"], bev=0)   # 담긴 재료
    casters(p["dark"], 4, W/2-0.06, 0, D/2-0.06, 0)
    export("env_pancart.glb", "EnvPanCart")

# ── 5. 이웃 국솥 — 급식 틸팅 국솥 (참조 현장_국탕 sg_5 / 참조 현장_국탕 실측) ────
#    구조: 사각 스테인리스 캐비닛 + 캐비닛보다 넓은 얕은 원형 볼 + 크롬 틸팅 핸드휠
#          + 상판 E-STOP + 앞면 배출 슈트 + 뒤 주름관 호스. (원통 몸통이 아니다)
#    GT는 'kettle'로 잡는다: 로봇 대상 솥과 형상이 같은데 라벨이 다르면 학습이 모순된다
def build_kettle_nb(lid_open=True, fname="env_kettle_nb.glb"):
    clear(); p = P()
    CW, CH = 0.84, 0.64          # 캐비닛 폭 / 높이 — 상판 밑까지만 (아래 ⚠️ 참조)
    TOP    = 0.735               # 상판 윗면
    RB, RIM = 0.51, 0.90         # 볼 외경 반지름 / 림 높이
    RBOT   = 0.39                # 볼 밑 반구 반지름 — 캐비닛(0.84) 안에 들어가야 옆구리가 안 터진다
    WALL   = 0.014               # 볼 벽 두께

    # ── 볼 안쪽 빈 공간 (커터) ────────────────────────────────────────────
    #   ⚠️ **이 부피는 볼뿐 아니라 캐비닛·상판에서도 빼야 한다.** 종전에는 프리미티브를
    #      그냥 겹쳐 놓기만 해서, ① 원뿔대의 윗면 캡이 입구를 통째로 덮고 ② 밑 반구가
    #      '온전한 구'라 윗절반이 국물 위로 솟고 ③ 캐비닛 박스(윗면 0.76)가 볼 안으로
    #      뚫고 올라왔다. 결과적으로 뚜껑을 열어도 **속이 꽉 막힌 솥**으로 보였다.
    #      sim 쪽에서 겹친 메시를 숨기는 우회로 덮지 말고, 여기서 실제로 파낸다.
    #   ⚠️ 커터 원뿔은 **림(RIM) 위로 뚫고 나가게** 0.10 더 높인다. 볼 윗면과 커터 윗면이
    #      z=RIM 으로 정확히 겹치면(동일 평면) EXACT 솔버가 그 캡을 못 지워 **입구가
    #      그대로 막힌 채 나온다** — 실제로 그렇게 나왔다. 테이퍼(0.6/z)는 그대로 두고
    #      높이만 늘려 림에서의 벽 두께가 WALL 로 유지되게 한다.
    def cavity():
        return fuse([
            cone(RBOT-WALL, RB-WALL + 0.06, 0.30, (0, 0, RIM-0.05), p["steel"], verts=48),
            apply_scale(sph(RBOT-WALL, (0, 0, RIM-0.20), p["steel"],
                            seg=40, ring=22, scale=(1, 1, 0.46))),
        ])

    # 캐비닛 — 브러시드 스테인리스 박스 + 상판 + 조절발
    cab   = box(CW, CW, CH, (0, 0, 0.06 + CH/2), p["steel"], bev=0.012)
    plate = box(CW+0.04, CW+0.04, 0.035, (0, 0, TOP-0.017), p["steel2"])
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(0.022, 0.06, (sx*(CW/2-0.07), sy*(CW/2-0.07), 0.03), p["dark"], verts=10)
    box(CW-0.10, 0.006, CH-0.16, (0, -CW/2-0.004, 0.06+CH/2), p["steel2"], bev=0)   # 도어 패널선

    # 얕고 넓은 볼 — 아래가 둥근 접시. 둥근 밑은 캐비닛 안에 잠겨 보이지 않는다
    bowl = fuse([
        cone(RBOT, RB, 0.20, (0, 0, RIM-0.10), p["steel"], verts=48),
        apply_scale(sph(RBOT, (0, 0, RIM-0.20), p["steel"], seg=40, ring=22, scale=(1, 1, 0.46))),
    ])
    carve(bowl,  cavity())        # 볼 속을 파낸다 → 벽 두께 WALL 짜리 그릇
    carve(cab,   cavity())        # 바닥지지대가 볼 안으로 솟지 않게 같은 부피를 뺀다
    carve(plate, cavity())        # 상판도 마찬가지 (볼 밑을 가로막고 있었다)
    tor(RB, 0.022, (0, 0, RIM), p["steel"], seg=48)                      # 말린 림
    cyl(RB-0.11, 0.012, (0, 0, RIM-0.16), p["food"], verts=44)           # 국물 수면 — 림에서 0.16 아래

    # 크롬 틸팅 핸드휠 — 좌측면 (sg_5: 캐비닛 옆에 크게 붙는다)
    hx, hz = -(CW/2 + 0.16), 0.44
    cyl(0.030, 0.20, (hx+0.10, 0, hz), p["steel2"], rot=(0, math.pi/2, 0), verts=16)
    tor(0.165, 0.016, (hx, 0, hz), p["steel"], rot=(0, math.pi/2, 0), seg=32)
    # 스포크 3개 — 휠은 YZ 평면에 있다. rot=(a,π/2,0)이면 오일러 XYZ 순서 탓에
    # 축이 XY 평면으로 눕는다. 축을 (0,cos a,sin a)로 두려면 X축 회전 (a-π/2)뿐이어야 한다.
    for a in (0, math.pi/3, 2*math.pi/3):
        cyl(0.010, 0.33, (hx, 0, hz), p["steel"], rot=(a - math.pi/2, 0, 0), verts=8)
    sph(0.036, (hx, 0, hz), p["steel2"], seg=16, ring=12)
    cyl(0.014, 0.075, (hx-0.03, 0.155, hz), p["black"], rot=(0, math.pi/2, 0), verts=10)  # 손잡이 노브

    # E-STOP — 노란 베이스 + 빨간 버섯버튼. **앞면**에 붙인다 (온도 컨트롤러 옆).
    #   ⚠️ 종전에는 상판(TOP) 위에 뒀는데, 이 솥은 볼(반경 RB=0.51)이 캐비닛(0.42)보다
    #      넓어 상판을 통째로 덮는다. 그래서 버튼이 **솥 안에 떠 있었다** — 반경 0.328
    #      지점인데 그 높이의 볼 내부 반경이 0.40~0.44다(실측). 볼 속을 파내기 전에는
    #      막힌 바닥에 가려 안 보였을 뿐, 원래부터 잘못된 자리였다.
    #      실물(sg_5)도 볼이 상판을 덮는 형식이라 정지버튼은 앞면·측면에 있다.
    EY = -CW/2                                    # 앞면 (프롭 정면 = -Y)
    cyl(0.046, 0.022, (-0.20, EY-0.011, 0.60), p["yellow"], rot=(math.pi/2, 0, 0), verts=20)
    cyl(0.032, 0.028, (-0.20, EY-0.036, 0.60), p["red"],    rot=(math.pi/2, 0, 0), verts=20)
    sph(0.032, (-0.20, EY-0.056, 0.60), p["red"], seg=18, ring=10,
        scale=(1, 1, 0.5), rot=(math.pi/2, 0, 0))

    # 디지털 온도 컨트롤러 — 앞면 상부 (참조 현장: 검정판 + 빨간 7세그)
    box(0.17, 0.012, 0.11, (0.20, -CW/2-0.008, 0.60), p["black"], bev=0)
    box(0.10, 0.008, 0.035, (0.20, -CW/2-0.016, 0.625), p["red"], bev=0)

    # 앞면 배출 슈트 + 손잡이 바 (sg_5)
    box(0.15, 0.13, 0.10, (0, -CW/2-0.06, 0.30), p["steel2"], rot=(0, 0, math.pi/4))
    cyl(0.016, 0.46, (0, -CW/2-0.03, 0.14), p["steel"], rot=(0, math.pi/2, 0), verts=12)
    for sx in (-1, 1):
        cyl(0.012, 0.07, (sx*0.23, -CW/2-0.005, 0.14), p["steel"], rot=(math.pi/2,0,0), verts=8)

    # 뒤 주름관 호스 — 스팀/급수. 캐비닛 뒤에서 바닥까지 내려가 끝난다 (sg_5)
    for i in range(9):
        cyl(0.030, 0.06, (0.30, CW/2 + 0.03 + i*0.010, 0.60 - i*0.070), p["steel2"],
            rot=(math.radians(14), 0, 0), verts=12)

    # ── 경첩 뚜껑 ────────────────────────────────────────────────────────
    # 힌지 기준 회전: 닫힘 상태 오프셋 (-R,0)을 X축으로 -θ 회전 →
    #   중심 (R - R·cosθ, RIM + R·sinθ),  판 회전 rot=(-θ,0,0)
    # 판·돔·손잡이가 전부 같은 θ로 돌아야 한다. 판만 돌리면 돔이 허공에 수평으로 뜬다.
    th = math.radians(68) if lid_open else 0.0
    RL, hy, hz2 = RB - 0.01, RB - 0.02, RIM + 0.03
    cy = hy - RL*math.cos(th)
    cz = hz2 + RL*math.sin(th)
    ny, nz = math.sin(th), math.cos(th)          # 뚜껑 바깥면 법선 (y,z)
    cyl(RL, 0.022, (0, cy, cz), p["steel"], rot=(-th, 0, 0), verts=44)
    sph(RL*0.90, (0, cy + 0.020*ny, cz + 0.020*nz), p["steel"],
        seg=36, ring=20, scale=(1, 1, 0.14), rot=(-th, 0, 0))             # 얕은 돔
    for sx in (-1, 1):                                                    # 힌지 브래킷
        box(0.05, 0.09, 0.05, (sx*0.20, hy + 0.02, hz2), p["steel2"])
    cyl(0.020, 0.46, (0, hy, hz2), p["steel2"], rot=(0, math.pi/2, 0), verts=14)  # 힌지 축
    # 손잡이 — 뚜껑 바깥면에서 법선 방향으로 띄우고, 지지대도 같은 방향으로 눕힌다
    cyl(0.014, 0.30, (0, cy + 0.085*ny, cz + 0.085*nz), p["black"],
        rot=(0, math.pi/2, 0), verts=12)
    for sx in (-1, 1):
        cyl(0.010, 0.075, (sx*0.13, cy + 0.043*ny, cz + 0.043*nz), p["steel2"],
            rot=(-th, 0, 0), verts=8)
    export(fname, "EnvKettleNb")

# ── 6. 벽걸이 조리도구 랙 — 국자·뒤집개 (참조 현장 빨간 손잡이) ──────────────
def build_wallrack():
    clear(); p = P()
    W = 0.90
    cyl(0.014, W, (0, 0, 0), p["steel"], rot=(0, math.pi/2, 0), verts=14)   # 행거 바
    for sx in (-1, 1):
        box(0.02, 0.06, 0.05, (sx*(W/2-0.02), 0.03, 0.0), p["steel2"])      # 벽 브래킷
    tools = [(-0.32, 0.42, "red"), (-0.11, 0.50, "black"), (0.11, 0.46, "red"), (0.32, 0.38, "black")]
    for x, L, col in tools:
        cyl(0.009, L, (x, 0, -L/2 - 0.02), p["steel"], verts=10)            # 자루
        cyl(0.014, 0.14, (x, 0, -0.09), p[col], verts=12)                   # 손잡이 그립
        if L > 0.44:                                                        # 국자 볼
            sph(0.055, (x, 0, -L - 0.05), p["steel"], scale=(1, 1, 0.6))
        else:                                                               # 뒤집개 날
            box(0.09, 0.02, 0.13, (x, 0, -L - 0.06), p["steel"])
    export("env_wallrack.glb", "EnvWallRack")

# ── 7. 재료통 / 쓰레기통 — φ450 h700 (전 현장) ────────────────────────────
def build_bin():
    clear(); p = P()
    cone(0.20, 0.235, 0.68, (0, 0, 0.34), p["steel2"], verts=28)
    tor(0.238, 0.016, (0, 0, 0.67), p["steel"], seg=28)
    cyl(0.22, 0.03, (0, 0, 0.015), p["dark"], verts=28)
    export("env_bin.glb", "EnvBin")

# ── 8. 업소용 냉장고/냉동고 — 2도어 직립형 (참조 현장 우측 후면) ─────────────
#    관찰: 바닥~약 1.95m 스테인리스 캐비닛, **긴 세로 손잡이 2개**,
#    상부에 제어/표시 패널, 하부 통풍 그릴, 낮은 받침 다리.
def build_fridge():
    clear(); p = P()
    W, D, H = 1.24, 0.78, 1.95
    box(W, D, H-0.10, (0, 0, 0.10 + (H-0.10)/2), p["steel"])          # 본체
    for sx in (-1, 1):                                                 # 도어 2짝 (분할선)
        box(W/2-0.012, 0.02, H-0.36, (sx*W/4, -D/2-0.006, 0.10+(H-0.36)/2+0.10), p["steel2"])
        # 긴 세로 손잡이 — 이 설비의 가장 눈에 띄는 특징
        cyl(0.016, H-0.62, (sx*(W/4) + (0.20 if sx<0 else -0.20), -D/2-0.055,
                            0.10+(H-0.36)/2+0.10), p["steel2"], verts=10)
        for dz in (-1, 1):                                             # 손잡이 브래킷
            cyl(0.012, 0.05, (sx*(W/4) + (0.20 if sx<0 else -0.20), -D/2-0.030,
                              0.10+(H-0.36)/2+0.10 + dz*((H-0.66)/2)), p["steel2"],
                rot=(math.pi/2, 0, 0), verts=8)
    box(W-0.06, 0.02, 0.20, (0, -D/2-0.006, H-0.10), p["dark"])        # 상부 제어 패널
    box(0.24, 0.012, 0.10, (0.30, -D/2-0.020, H-0.10), p["black"])     # 표시창
    box(W-0.20, 0.02, 0.07, (0, -D/2-0.006, 0.16), p["steel2"])        # 하부 통풍 그릴
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(0.028, 0.10, (sx*(W/2-0.09), sy*(D/2-0.09), 0.05), p["dark"], verts=10)
    export("env_fridge.glb", "EnvFridge")

# ── 9. 배식대 + 상부 선반 — 1800×700×850 + 기둥 위 선반 2단 (참조 현장 우측) ──
#    관찰: 스테인리스 카운터 위로 수직 기둥이 서고 그 위에 선반이 얹힌다.
#    선반 위에는 대야·팬이 올라가 있다.
def build_serve():
    clear(); p = P()
    W, D, H = 1.80, 0.70, 0.85
    box(W, D, 0.04, (0, 0, H-0.02), p["steel"])                        # 상판
    box(W, 0.04, 0.10, (0, D/2-0.02, H+0.05), p["steel"])              # 뒷턱
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(0.05, 0.05, H-0.04, (sx*(W/2-0.07), sy*(D/2-0.07), (H-0.04)/2), p["steel2"])
    box(W-0.16, D-0.16, 0.025, (0, 0, 0.26), p["steel2"])              # 하단 선반
    for sx in (-1, 1):                                                 # 상부 선반 기둥
        cyl(0.021, 0.86, (sx*(W/2-0.12), D/2-0.14, H + 0.43), p["steel2"], verts=12)
    for k, z in enumerate((H+0.42, H+0.80)):                           # 상부 선반 2단
        box(W-0.16, 0.36, 0.022, (0, D/2-0.14, z), p["steel"])
        for i in range(7):                                             # 와이어 결
            box(0.010, 0.34, 0.012, (-W/2+0.14+i*((W-0.28)/6), D/2-0.14, z+0.017),
                p["steel2"], bev=0)
    export("env_serve.glb", "EnvServe")

# ── 10. 와이어 바스켓 카트 — 망바구니 적재 (참조 현장 좌측 후면) ──────────────
def build_basketcart():
    clear(); p = P()
    W, D = 0.62, 0.44
    for sx in (-1, 1):                                                 # 프레임
        for sy in (-1, 1):
            box(0.026, 0.026, 0.30, (sx*(W/2-0.03), sy*(D/2-0.03), 0.24), p["steel2"])
    box(W, D, 0.02, (0, 0, 0.40), p["steel2"])
    for k in range(3):                                                 # 망바구니 3단
        z = 0.42 + k*0.135
        box(W-0.08, D-0.06, 0.012, (0, 0, z), p["steel"])              # 바닥
        for ex, ey, sw, sd in ((W/2-0.04,0,0.012,D-0.06), (-(W/2-0.04),0,0.012,D-0.06),
                               (0,D/2-0.03,W-0.08,0.012), (0,-(D/2-0.03),W-0.08,0.012)):
            box(sw, sd, 0.115, (ex, ey, z+0.058), p["steel"], bev=0)
        for i in range(6):                                             # 망 결
            box(0.008, D-0.08, 0.10, (-W/2+0.09+i*((W-0.18)/5), 0, z+0.055), p["steel2"], bev=0)
    casters(p["dark"], 4, W/2-0.05, 0, D/2-0.05, 0)
    export("env_basketcart.glb", "EnvBasketCart")

build_sink(); build_table(); build_rack(); build_pancart()
build_fridge(); build_serve(); build_basketcart()
# 국솥 2대는 뚜껑 상태를 다르게 — 참조 현장 CCTV도 한 대는 열고 한 대는 닫혀 있다
build_kettle_nb(lid_open=True,  fname="env_kettle_nb.glb")
build_kettle_nb(lid_open=False, fname="env_kettle_nb_closed.glb")
build_wallrack(); build_bin()
print("ALL DONE")
