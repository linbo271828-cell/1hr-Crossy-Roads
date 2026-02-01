# crossy_iso.py
# Crossy Road–inspired endless runner (isometric voxel-ish) in Pygame
#
# Fixes in this version:
# - Painting fixed via correct global depth sorting (depth = projected screen_y).
# - Removes seam/crack artifacts via round() (not int floor) + rounded camera offset.
# - Logs: per-river direction; logs spawn from the appropriate side consistently.
# - Train indicator: obvious WARNING overlay + flashing posts before train appears.
# - Eagle mechanics removed.

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

# ----------------------------
# Config
# ----------------------------
SCREEN_W, SCREEN_H = 1280, 720
FPS = 60

GRID_W = 13  # x=0..12

TILE_W = 96
TILE_H = 48
BLOCK_H = 42
Z_UNIT = 22  # px per model z-unit

ANCHOR_X = SCREEN_W // 2
ANCHOR_Y = int(SCREEN_H * 0.64)
CAM_LEAD = 4.5
CAM_SMOOTH = 0.12

AHEAD_BUFFER = 55
BEHIND_BUFFER = 45

INTRO_SAFE_LANES = 3
FIRST_HAZARD_Y = 3

# ----------------------------
# Colors
# ----------------------------
def clampi(v, lo, hi): return max(lo, min(hi, v))

def shade(color, factor: float) -> Tuple[int, int, int]:
    r, g, b = color
    return (clampi(int(r * factor), 0, 255),
            clampi(int(g * factor), 0, 255),
            clampi(int(b * factor), 0, 255))

COL_BG = (242, 242, 252)

COL_GRASS = (120, 190, 70)
COL_GRASS_ALT = (108, 178, 62)

COL_ROAD = (95, 95, 105)
COL_RIVER = (70, 160, 235)
COL_TRACK = (90, 90, 100)

COL_LANE_MARK = (165, 165, 175)

COL_LOG = (130, 92, 68)
COL_ROCK = (160, 160, 175)

COL_TREE_TRUNK = (120, 75, 45)
COL_TREE_TOP = (55, 155, 70)
COL_TREE_TOP_LIGHT = (75, 185, 90)

COL_COIN = (255, 215, 0)
COL_COIN_C = (200, 40, 40)

CAR_COLORS = [
    (70, 200, 90),
    (230, 100, 60),
    (85, 190, 210),
    (220, 200, 90),
]

COL_CHICKEN = (250, 250, 255)
COL_BEAK = (235, 70, 70)
COL_COMB = (230, 40, 40)
COL_EYE = (25, 25, 25)

COL_UI = (20, 20, 30)
COL_UI_BAD = (210, 40, 40)

# Train indicator colors
COL_WARN = (235, 70, 70)
COL_WARN_DARK = (150, 40, 40)

LaneType = str  # "grass"|"road"|"river"|"track"

# ----------------------------
# Voxel models (multi-part)
# part: (dx, dy, dz, color, w, d, h_scale)
# ----------------------------
MODELS = {
    "player": [
        (0.00, 0.00, 0.00, COL_CHICKEN, 0.62, 0.62, 0.80),  # body
        (0.18, 0.10, 0.45, COL_CHICKEN, 0.42, 0.42, 0.55),  # head
        (0.36, 0.10, 0.62, COL_BEAK,    0.22, 0.14, 0.25),  # beak
        (0.20, 0.12, 0.78, COL_COMB,    0.10, 0.22, 0.35),  # comb
        (0.26, 0.22, 0.60, COL_EYE,     0.06, 0.06, 0.15),  # eye R
        (0.26,-0.02, 0.60, COL_EYE,     0.06, 0.06, 0.15),  # eye L
    ],
    "coin": [
        (0.00, 0.00, 0.15, COL_COIN,   0.30, 0.30, 0.22),
        (0.02, 0.00, 0.25, COL_COIN_C, 0.10, 0.22, 0.18),
    ],
}

# ----------------------------
# Types
# ----------------------------
@dataclass
class Car:
    x: float
    length: int
    speed: float
    dir: int
    color: Tuple[int, int, int]

@dataclass
class Log:
    x: float
    length: int
    speed: float
    dir: int

@dataclass
class Train:
    x: float
    length: int
    speed: float
    dir: int

@dataclass
class Coin:
    x: int
    attached_to_log: Optional[int] = None
    offset: float = 0.0

@dataclass
class Lane:
    y: int
    kind: LaneType
    dir: int = 1
    speed: float = 0.0

    blockers: Dict[int, str] = field(default_factory=dict)
    coins: List[Coin] = field(default_factory=list)

    cars: List[Car] = field(default_factory=list)
    last_spawn_x: float = 0.0

    logs: List[Log] = field(default_factory=list)
    last_spawn_log_x: float = 0.0

    track_state: str = "IDLE"  # IDLE|WARNING|PASSING
    track_timer: float = 0.0
    train: Optional[Train] = None

@dataclass
class Player:
    x: int
    y: int
    xf: float = 0.0
    yf: float = 0.0

    hopping: bool = False
    hop_t: float = 0.0
    hop_dur: float = 0.12

    start_x: int = 0
    start_y: int = 0
    target_x: int = 0
    target_y: int = 0
    queued_move: Optional[Tuple[int, int]] = None

    on_log: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.xf = self.x + 0.5
        self.yf = float(self.y)

@dataclass
class Game:
    rng: random.Random
    lanes: Dict[int, Lane] = field(default_factory=dict)

    player: Player = field(default_factory=lambda: Player(6, 0))
    score: int = 0
    coins: int = 0
    max_forward_y: int = 0

    game_over: bool = False
    death_reason: str = ""

    camera_y: float = 0.0

    path_x: int = 6
    recent_kinds: List[LaneType] = field(default_factory=list)

# ----------------------------
# Projection: forward y+ goes UP screen
# ----------------------------
def project_iso(wx: float, wy: float) -> Tuple[float, float]:
    sx = (wx + wy) * (TILE_W / 2)
    sy = (wx - wy) * (TILE_H / 2)
    return sx, sy

def world_to_screen(wx: float, wy: float, cam_off: Tuple[float, float], z_px: float = 0.0) -> Tuple[int, int]:
    px, py = project_iso(wx, wy)
    cx, cy = cam_off
    # round instead of floor to prevent seams
    return int(round(px + cx)), int(round(py + cy - z_px))

def footprint_pts(x0: float, y0: float, x1: float, y1: float, cam: Tuple[float, float]) -> List[Tuple[int, int]]:
    A = world_to_screen(x0, y0, cam)
    B = world_to_screen(x1, y0, cam)
    C = world_to_screen(x1, y1, cam)
    D = world_to_screen(x0, y1, cam)
    return [A, B, C, D]

def depth_key_from_world(wx: float, wy: float) -> Tuple[float, float]:
    # Correct painter depth: projected screen_y primarily, then screen_x
    sx, sy = project_iso(wx, wy)
    return (sy, sx)

# ----------------------------
# Draw queue primitives
# ----------------------------
def draw_prism(surface: pygame.Surface,
               base_pts: List[Tuple[int, int]],
               height: int,
               top: Tuple[int, int, int],
               left: Tuple[int, int, int],
               right: Tuple[int, int, int]):
    A, B, C, D = base_pts
    A2 = (A[0], A[1] - height)
    B2 = (B[0], B[1] - height)
    C2 = (C[0], C[1] - height)
    D2 = (D[0], D[1] - height)

    pygame.draw.polygon(surface, left,  [D, C, C2, D2])
    pygame.draw.polygon(surface, right, [B, C, C2, B2])
    pygame.draw.polygon(surface, top,   [A2, B2, C2, D2])

# draw queue entries: (depth_tuple, kind, payload)
# kind: "tile", "poly", "prism_full"
def push_tile(q, depth, pts, color):
    q.append((depth, "tile", (pts, color)))

def push_poly(q, depth, pts, color):
    q.append((depth, "poly", (pts, color)))

def push_prism(q, depth, base_pts, height, top):
    q.append((depth, "prism_full", (base_pts, height, top, shade(top, 0.72), shade(top, 0.88))))

def push_prism_full(q, depth, base_pts, height, top, left, right):
    q.append((depth, "prism_full", (base_pts, height, top, left, right)))

def push_model(q, model_name: str, tile_x0: float, tile_y0: float, cam: Tuple[float, float], bounce_px: float = 0.0):
    base_cx = tile_x0 + 0.5
    base_cy = tile_y0 + 0.5
    parts = MODELS[model_name]

    for (dx, dy, dz, col, w, d, hs) in sorted(parts, key=lambda p: (p[2], p[1], p[0])):
        cx = base_cx + dx
        cy = base_cy + dy
        x0 = cx - w / 2
        y0 = cy - d / 2
        x1 = cx + w / 2
        y1 = cy + d / 2

        pts = footprint_pts(x0, y0, x1, y1, cam)
        lift = dz * Z_UNIT + bounce_px
        pts = [(sx, sy - int(round(lift))) for (sx, sy) in pts]

        h = int(BLOCK_H * hs)
        top = col
        left = shade(top, 0.72)
        right = shade(top, 0.88)

        depth = depth_key_from_world(cx, cy)
        push_prism_full(q, depth, pts, h, top, left, right)

# ----------------------------
# Lane generation
# ----------------------------
def lane_kind_choice(g: Game, y: int) -> LaneType:
    if y < FIRST_HAZARD_Y:
        return "grass"
    if y == FIRST_HAZARD_Y:
        return g.rng.choice(["road", "river"])

    s = g.score
    ramp = min(1.0, s / 90.0)

    w_grass = 0.35 - 0.12 * ramp
    w_road  = 0.30 + 0.08 * ramp
    w_river = 0.20 + 0.06 * ramp
    w_track = 0.15 + 0.08 * ramp

    recent = g.recent_kinds[-3:]
    if len(recent) >= 2 and recent[-1] == "river" and recent[-2] == "river":
        w_river = 0.0
    if len(recent) >= 1 and recent[-1] == "track":
        w_track *= (0.25 if s < 120 else 0.6)

    total = w_grass + w_road + w_river + w_track
    r = g.rng.random() * total
    if r < w_grass: return "grass"
    r -= w_grass
    if r < w_road: return "road"
    r -= w_road
    if r < w_river: return "river"
    return "track"

def generate_grass_blockers(g: Game, lane: Lane, density: float, force_open_x: int):
    blockers = {}
    for x in range(GRID_W):
        if x == force_open_x:
            continue
        if lane.y == 0 and x == 6:
            continue
        if g.rng.random() < density:
            blockers[x] = "tree" if g.rng.random() < 0.65 else "rock"

    px = force_open_x
    must_open = [clampi(px - 1, 0, GRID_W - 1), px, clampi(px + 1, 0, GRID_W - 1)]
    if all(x in blockers for x in must_open):
        del blockers[g.rng.choice(must_open)]

    while GRID_W - len(blockers) < 5:
        bx = g.rng.choice(list(blockers.keys()))
        del blockers[bx]

    lane.blockers = blockers

def spawn_grass_coins(g: Game, lane: Lane, chance: float):
    lane.coins.clear()
    if g.rng.random() > chance:
        return
    open_xs = [x for x in range(GRID_W) if x not in lane.blockers and not (lane.y == 0 and x == 6)]
    g.rng.shuffle(open_xs)
    count = 1 if g.rng.random() < 0.75 else 2
    for x in open_xs[:count]:
        lane.coins.append(Coin(x=x))

def spawn_car(g: Game, lane: Lane):
    length = 2 if g.rng.random() < 0.75 else 3
    col = g.rng.choice(CAR_COLORS)

    base = 2.4 + g.rng.random() * 1.2
    lane.speed = base * (1.0 + min(1.2, g.score / 120.0) * 0.6)

    gap = 1.8 + g.rng.random() * 2.8
    gap *= (0.85 if g.score > 80 else 1.0)
    gap *= (0.75 if g.score > 180 else 1.0)

    if lane.dir == 1:
        x = lane.last_spawn_x - (gap + length)
        lane.last_spawn_x = x
    else:
        x = lane.last_spawn_x + (gap + length)
        lane.last_spawn_x = x

    speed = lane.speed * (0.9 + 0.25 * g.rng.random())
    lane.cars.append(Car(x=x, length=length, speed=speed, dir=lane.dir, color=col))

def init_road_lane(g: Game, lane: Lane):
    lane.dir = g.rng.choice([-1, 1])
    lane.cars.clear()
    lane.last_spawn_x = (-2.0 if lane.dir == 1 else GRID_W + 2.0)
    for _ in range(3 + (1 if g.score > 40 else 0)):
        spawn_car(g, lane)
    lane.coins.clear()
    if g.rng.random() < 0.10:
        lane.coins.append(Coin(x=g.rng.randrange(GRID_W)))

def spawn_log(g: Game, lane: Lane):
    # per river lane, lane.dir is fixed; logs drift from that side only
    if g.score < 80:
        length = g.rng.choice([3, 4, 4, 5])
    else:
        length = g.rng.choice([2, 3, 3, 4])

    base = 1.6 + g.rng.random() * 1.1
    lane.speed = base * (1.0 + min(1.2, g.score / 120.0) * 0.5)

    gap = 2.0 + g.rng.random() * 3.0
    gap *= (0.9 if g.score > 80 else 1.0)

    if lane.dir == 1:
        # spawn left of screen, move right
        x = lane.last_spawn_log_x - (gap + length)
        lane.last_spawn_log_x = x
    else:
        # spawn right of screen, move left
        x = lane.last_spawn_log_x + (gap + length)
        lane.last_spawn_log_x = x

    speed = lane.speed * (0.9 + 0.2 * g.rng.random())
    lane.logs.append(Log(x=x, length=length, speed=speed, dir=lane.dir))

def init_river_lane(g: Game, lane: Lane):
    lane.dir = g.rng.choice([-1, 1])  # this decides which side logs drift from
    lane.logs.clear()
    lane.last_spawn_log_x = (-3.0 if lane.dir == 1 else GRID_W + 3.0)
    for _ in range(3 + (1 if g.score > 60 else 0)):
        spawn_log(g, lane)

    lane.coins.clear()
    if g.rng.random() < 0.22 and lane.logs:
        li = g.rng.randrange(len(lane.logs))
        lg = lane.logs[li]
        off = g.rng.uniform(0.4, max(0.4, lg.length - 0.6))
        lane.coins.append(Coin(x=0, attached_to_log=li, offset=off))

def init_track_lane(g: Game, lane: Lane):
    lane.dir = g.rng.choice([-1, 1])
    lane.speed = 8.0 * (1.0 + min(1.2, g.score / 140.0) * 0.35)
    lane.track_state = "IDLE"
    lane.train = None
    lane.track_timer = 2.6 + g.rng.random() * 2.6

def create_lane(g: Game, y: int) -> Lane:
    kind = lane_kind_choice(g, y)
    lane = Lane(y=y, kind=kind)

    if y >= FIRST_HAZARD_Y:
        g.recent_kinds.append(kind)
        if len(g.recent_kinds) > 8:
            g.recent_kinds.pop(0)

    if kind == "grass":
        density = 0.18 + min(0.14, g.score / 250.0 * 0.14)
        if y < FIRST_HAZARD_Y:
            density = 0.10
        generate_grass_blockers(g, lane, density, g.path_x)
        spawn_grass_coins(g, lane, chance=0.22 if y >= FIRST_HAZARD_Y else 0.12)

        step = g.rng.choice([-1, 0, 1])
        g.path_x = clampi(g.path_x + step, 0, GRID_W - 1)
        if g.path_x in lane.blockers:
            for dx in [1, -1, 2, -2, 0]:
                nx = clampi(g.path_x + dx, 0, GRID_W - 1)
                if nx not in lane.blockers:
                    g.path_x = nx
                    break

    elif kind == "road":
        init_road_lane(g, lane)

    elif kind == "river":
        init_river_lane(g, lane)

    elif kind == "track":
        init_track_lane(g, lane)

    return lane

def ensure_lanes(g: Game):
    p = g.player
    center = max(int(round(p.yf)), g.max_forward_y)
    lo = max(0, int(round(p.yf)) - BEHIND_BUFFER)
    hi = center + AHEAD_BUFFER

    for y in range(lo, hi + 1):
        if y not in g.lanes:
            g.lanes[y] = create_lane(g, y)

    prune_below = max(0, int(round(p.yf)) - (BEHIND_BUFFER + 14))
    for y in [yy for yy in g.lanes.keys() if yy < prune_below]:
        del g.lanes[y]

# ----------------------------
# Gameplay
# ----------------------------
def kill(g: Game, reason: str):
    if not g.game_over:
        g.game_over = True
        g.death_reason = reason

def pickup_coin_tile(g: Game, lane: Lane, px: int):
    if not lane.coins:
        return
    kept = []
    picked = False
    for c in lane.coins:
        if c.attached_to_log is None and c.x == px:
            picked = True
        else:
            kept.append(c)
    if picked:
        g.coins += 1
    lane.coins = kept

def pickup_coin_attached(g: Game, lane: Lane):
    if not lane.coins:
        return
    p = g.player
    kept = []
    picked = False
    for c in lane.coins:
        if c.attached_to_log is None:
            kept.append(c)
            continue
        if 0 <= c.attached_to_log < len(lane.logs):
            lg = lane.logs[c.attached_to_log]
            cx = lg.x + c.offset
            if abs(p.xf - cx) < 0.35 and abs(p.yf - lane.y) < 0.35:
                picked = True
            else:
                kept.append(c)
    if picked:
        g.coins += 1
    lane.coins = kept

def try_start_hop(g: Game, dx: int, dy: int):
    if g.game_over:
        return
    p = g.player
    if p.hopping:
        p.queued_move = (dx, dy)
        return

    p.x = clampi(int(round(p.xf - 0.5)), 0, GRID_W - 1)
    p.xf = p.x + 0.5
    p.y = max(0, int(round(p.yf)))
    p.yf = float(p.y)

    nx, ny = p.x + dx, p.y + dy
    if ny < 0 or nx < 0 or nx >= GRID_W:
        return

    ensure_lanes(g)
    lane = g.lanes.get(ny)
    if lane and lane.kind == "grass" and nx in lane.blockers:
        return

    p.hopping = True
    p.hop_t = 0.0
    p.start_x, p.start_y = p.x, p.y
    p.target_x, p.target_y = nx, ny
    p.on_log = None
    p.queued_move = None

def finish_hop(g: Game):
    p = g.player
    p.x, p.y = p.target_x, p.target_y
    p.xf = p.x + 0.5
    p.yf = float(p.y)
    p.hopping = False
    p.hop_t = 0.0

    if p.y > g.max_forward_y:
        g.score += (p.y - g.max_forward_y)
        g.max_forward_y = p.y

    lane = g.lanes.get(p.y)
    if lane and lane.kind in ("grass", "road"):
        pickup_coin_tile(g, lane, p.x)

def update_lanes(g: Game, dt: float):
    ensure_lanes(g)

    for lane in g.lanes.values():
        if lane.kind == "road":
            for car in lane.cars:
                car.x += car.speed * car.dir * dt
            margin = 6.0
            if lane.dir == 1:
                lane.cars = [c for c in lane.cars if c.x < GRID_W + margin]
            else:
                lane.cars = [c for c in lane.cars if c.x > -margin - c.length]
            target = 3 + (1 if g.score > 40 else 0) + (1 if g.score > 120 else 0)
            while len(lane.cars) < target:
                spawn_car(g, lane)

        elif lane.kind == "river":
            for lg in lane.logs:
                lg.x += lg.speed * lg.dir * dt
            margin = 7.0
            if lane.dir == 1:
                lane.logs = [l for l in lane.logs if l.x < GRID_W + margin]
            else:
                lane.logs = [l for l in lane.logs if l.x > -margin - l.length]
            target = 3 + (1 if g.score > 60 else 0)
            while len(lane.logs) < target:
                spawn_log(g, lane)
            lane.coins = [c for c in lane.coins if (c.attached_to_log is None) or (0 <= c.attached_to_log < len(lane.logs))]

        elif lane.kind == "track":
            if lane.track_state == "IDLE":
                lane.track_timer -= dt
                if lane.track_timer <= 0:
                    lane.track_state = "WARNING"
                    lane.track_timer = 0.90 + g.rng.random() * 0.35  # warning duration
            elif lane.track_state == "WARNING":
                lane.track_timer -= dt
                if lane.track_timer <= 0:
                    lane.track_state = "PASSING"
                    length = g.rng.randint(9, 14) if g.score < 120 else g.rng.randint(8, 16)
                    speed = lane.speed * (0.95 + 0.12 * g.rng.random())
                    x0 = (-length - 3.0) if lane.dir == 1 else (GRID_W + 3.0)
                    lane.train = Train(x=x0, length=length, speed=speed, dir=lane.dir)
            elif lane.track_state == "PASSING":
                if lane.train is None:
                    lane.track_state = "IDLE"
                    lane.track_timer = 2.6 + g.rng.random() * 2.6
                else:
                    tr = lane.train
                    tr.x += tr.speed * tr.dir * dt
                    if tr.dir == 1 and tr.x > GRID_W + 4.0:
                        lane.train = None
                        lane.track_state = "IDLE"
                        lane.track_timer = 2.6 + g.rng.random() * 2.6
                    elif tr.dir == -1 and tr.x < -tr.length - 4.0:
                        lane.train = None
                        lane.track_state = "IDLE"
                        lane.track_timer = 2.6 + g.rng.random() * 2.6

def update_player(g: Game, dt: float):
    p = g.player

    target_cam = max(0.0, p.yf - CAM_LEAD)
    g.camera_y += (target_cam - g.camera_y) * CAM_SMOOTH

    if p.hopping and not g.game_over:
        p.hop_t += dt / p.hop_dur
        t = min(1.0, p.hop_t)
        t2 = t * t * (3 - 2 * t)
        p.xf = (p.start_x + 0.5) * (1 - t2) + (p.target_x + 0.5) * t2
        p.yf = float(p.start_y) * (1 - t2) + float(p.target_y) * t2
        if t >= 1.0:
            finish_hop(g)
            if p.queued_move and not g.game_over:
                dx, dy = p.queued_move
                p.queued_move = None
                try_start_hop(g, dx, dy)

    if not g.game_over:
        y_int = int(round(p.yf))
        lane = g.lanes.get(y_int)

        if lane and lane.kind == "river":
            support = None
            for i, lg in enumerate(lane.logs):
                if p.xf >= lg.x and p.xf <= lg.x + lg.length:
                    support = i
                    break
            if support is None:
                if abs(p.yf - y_int) < 0.25 and not p.hopping:
                    kill(g, "You drowned!")
            else:
                p.on_log = (lane.y, support)
                if not p.hopping:
                    lg = lane.logs[support]
                    p.xf += lg.speed * lg.dir * dt
                    if p.xf < 0.0 or p.xf > GRID_W:
                        kill(g, "Fell off a log!")
                    pickup_coin_attached(g, lane)
        else:
            p.on_log = None
            if not p.hopping:
                p.x = clampi(int(round(p.xf - 0.5)), 0, GRID_W - 1)
                p.xf = p.x + 0.5
                p.y = int(round(p.yf))
                p.yf = float(p.y)

def check_collisions(g: Game):
    if g.game_over:
        return
    p = g.player
    y_int = int(round(p.yf))
    lane = g.lanes.get(y_int)
    if not lane:
        return

    if lane.kind == "road":
        for car in lane.cars:
            if p.xf >= car.x and p.xf <= car.x + car.length and abs(p.yf - lane.y) < 0.35:
                kill(g, "Hit by a car!")
                return

    # strict x-span train collision
    if lane.kind == "track":
        if lane.track_state == "PASSING" and lane.train is not None:
            tr = lane.train
            if p.xf >= tr.x and p.xf <= tr.x + tr.length and abs(p.yf - lane.y) < 0.35:
                kill(g, "Hit by a train!")
                return

# ----------------------------
# Rendering (global depth queue fixes left-side paint + overlaps)
# ----------------------------
def compute_cam_offset(g: Game) -> Tuple[float, float]:
    cam_x_world = 6.0
    cam_y_world = g.camera_y
    px, py = project_iso(cam_x_world, cam_y_world)
    # round to keep all tile corners aligned frame-to-frame (prevents seams)
    return (round(ANCHOR_X - px), round(ANCHOR_Y - py))

def render(g: Game, surface: pygame.Surface, font: pygame.font.Font, bigfont: pygame.font.Font):
    surface.fill(COL_BG)
    cam = compute_cam_offset(g)
    tms = pygame.time.get_ticks()

    # Build draw queue for visible region
    q = []

    start_row = max(0, int(g.camera_y) - 3)
    end_row = int(g.camera_y) + 28

    # 1) Floor tiles + markings
    for y in range(start_row, end_row + 1):
        lane = g.lanes.get(y)
        if lane is None:
            continue

        for x in range(GRID_W):
            pts = footprint_pts(x, y, x + 1, y + 1, cam)
            cx, cy = x + 0.5, y + 0.5
            d = depth_key_from_world(cx, cy)

            if lane.kind == "grass":
                col = COL_GRASS if (x + y) % 2 == 0 else COL_GRASS_ALT
            elif lane.kind == "road":
                col = COL_ROAD
            elif lane.kind == "river":
                col = COL_RIVER
            else:
                col = COL_TRACK

            push_tile(q, d, pts, col)

            if lane.kind == "road" and (x % 2 == 0):
                A, B, C, Dp = pts
                def lerp(P, Q, t):
                    return (int(round(P[0] + (Q[0] - P[0]) * t)), int(round(P[1] + (Q[1] - P[1]) * t)))
                p1 = lerp(A, C, 0.46)
                p2 = lerp(A, C, 0.54)
                q1 = lerp(Dp, B, 0.46)
                q2 = lerp(Dp, B, 0.54)
                stripe = [(sx, sy - 1) for (sx, sy) in [p1, q1, q2, p2]]
                push_poly(q, (d[0] + 0.01, d[1]), stripe, COL_LANE_MARK)

        # Track WARNING indicator overlay (big, obvious)
        if lane.kind == "track" and lane.track_state == "WARNING":
            blink = (tms // 170) % 2 == 0
            warn_col = COL_WARN if blink else COL_WARN_DARK

            # semi-wide overlay stripe across the lane
            overlay = footprint_pts(0.0, y + 0.42, GRID_W * 1.0, y + 0.58, cam)
            # push as poly slightly above floor depth
            center_d = depth_key_from_world(6.5, y + 0.5)
            push_poly(q, (center_d[0] + 0.02, center_d[1]), overlay, warn_col)

            # flashing posts at both edges
            for bx in [0, GRID_W - 1]:
                post = footprint_pts(bx + 0.10, y + 0.10, bx + 0.35, y + 0.35, cam)
                pd = depth_key_from_world(bx + 0.225, y + 0.225)
                push_prism(q, (pd[0] + 0.03, pd[1]), post, int(BLOCK_H * 0.25), warn_col)

    # 2) Static blockers + coins
    for y in range(start_row, end_row + 1):
        lane = g.lanes.get(y)
        if lane is None:
            continue

        if lane.kind == "grass":
            for bx, btype in lane.blockers.items():
                base = footprint_pts(bx, y, bx + 1, y + 1, cam)
                cx, cy = bx + 0.5, y + 0.5
                d = depth_key_from_world(cx, cy)

                if btype == "rock":
                    top = COL_ROCK
                    push_prism_full(q, (d[0] + 0.20, d[1]), base, int(BLOCK_H * 0.55),
                                    top, shade(top, 0.70), shade(top, 0.86))
                else:
                    top = COL_TREE_TRUNK
                    push_prism_full(q, (d[0] + 0.20, d[1]), base, int(BLOCK_H * 0.55),
                                    top, shade(top, 0.68), shade(top, 0.84))

                    canopy = footprint_pts(bx + 0.12, y + 0.12, bx + 0.88, y + 0.88, cam)
                    canopy = [(sx, sy - int(round(0.35 * Z_UNIT))) for (sx, sy) in canopy]
                    top2 = COL_TREE_TOP
                    push_prism_full(q, (d[0] + 0.22, d[1]), canopy, int(BLOCK_H * 0.95),
                                    top2, shade(top2, 0.70), shade(top2, 0.88))

                    tuft = footprint_pts(bx + 0.25, y + 0.25, bx + 0.75, y + 0.75, cam)
                    tuft = [(sx, sy - int(round(0.70 * Z_UNIT))) for (sx, sy) in tuft]
                    top3 = COL_TREE_TOP_LIGHT
                    push_prism_full(q, (d[0] + 0.24, d[1]), tuft, int(BLOCK_H * 0.60),
                                    top3, shade(top3, 0.70), shade(top3, 0.90))

        for c in lane.coins:
            if c.attached_to_log is None:
                bob = math.sin(tms * 0.006) * 5.0
                push_model(q, "coin", c.x, y, cam, bounce_px=bob)

    # 3) Moving obstacles
    for y in range(start_row, end_row + 1):
        lane = g.lanes.get(y)
        if lane is None:
            continue

        if lane.kind == "road":
            for car in lane.cars:
                # body tiles
                for i in range(car.length):
                    cx_tile = car.x + i
                    base = footprint_pts(cx_tile, y, cx_tile + 1, y + 1, cam)
                    cx, cy = cx_tile + 0.5, y + 0.5
                    d = depth_key_from_world(cx, cy)
                    top = car.color
                    push_prism_full(q, (d[0] + 0.30, d[1]), base, int(BLOCK_H * 0.55),
                                    top, shade(top, 0.68), shade(top, 0.86))
                # roof
                roof = footprint_pts(car.x + 0.25, y + 0.18, car.x + car.length - 0.25, y + 0.82, cam)
                rcx, rcy = car.x + car.length / 2, y + 0.5
                rd = depth_key_from_world(rcx, rcy)
                top = (245, 245, 250)
                push_prism_full(q, (rd[0] + 0.32, rd[1]), roof, int(BLOCK_H * 0.35),
                                top, shade(top, 0.78), shade(top, 0.92))

        elif lane.kind == "river":
            for lg in lane.logs:
                for i in range(lg.length):
                    lx = lg.x + i
                    base = footprint_pts(lx, y, lx + 1, y + 1, cam)
                    cx, cy = lx + 0.5, y + 0.5
                    d = depth_key_from_world(cx, cy)
                    top = COL_LOG
                    push_prism_full(q, (d[0] + 0.26, d[1]), base, int(BLOCK_H * 0.35),
                                    top, shade(top, 0.70), shade(top, 0.86))

            # attached coins
            for c in lane.coins:
                if c.attached_to_log is not None and 0 <= c.attached_to_log < len(lane.logs):
                    lg = lane.logs[c.attached_to_log]
                    cx = lg.x + c.offset
                    bob = math.sin(tms * 0.006) * 5.0
                    push_model(q, "coin", cx - 0.5, y, cam, bounce_px=bob)

        elif lane.kind == "track":
            if lane.track_state == "PASSING" and lane.train is not None:
                tr = lane.train
                for i in range(tr.length):
                    tx = tr.x + i
                    base = footprint_pts(tx, y, tx + 1, y + 1, cam)
                    cx, cy = tx + 0.5, y + 0.5
                    d = depth_key_from_world(cx, cy)
                    top = (210, 210, 220)
                    push_prism_full(q, (d[0] + 0.36, d[1]), base, int(BLOCK_H * 0.65),
                                    top, shade(top, 0.70), shade(top, 0.88))
                stripe = footprint_pts(tr.x + 0.10, y + 0.42, tr.x + tr.length - 0.10, y + 0.58, cam)
                dcx, dcy = tr.x + tr.length / 2, y + 0.5
                dd = depth_key_from_world(dcx, dcy)
                top = (215, 80, 80)
                push_prism_full(q, (dd[0] + 0.38, dd[1]), stripe, int(BLOCK_H * 0.10),
                                top, shade(top, 0.75), shade(top, 0.92))

    # 4) Player
    p = g.player
    hop_arc = 0.0
    if p.hopping:
        t = min(1.0, p.hop_t)
        hop_arc = math.sin(math.pi * t) * 16.0
    push_model(q, "player", p.xf - 0.5, p.yf, cam, bounce_px=hop_arc)

    # Draw sorted by depth (screen_y then screen_x)
    q.sort(key=lambda it: it[0])

    for _, kind, payload in q:
        if kind == "tile":
            pts, col = payload
            pygame.draw.polygon(surface, col, pts)
        elif kind == "poly":
            pts, col = payload
            pygame.draw.polygon(surface, col, pts)
        elif kind == "prism_full":
            base_pts, h, top, left, right = payload
            draw_prism(surface, base_pts, h, top, left, right)

    # UI
    surface.blit(font.render(f"Score: {g.score}", True, COL_UI), (18, 16))
    surface.blit(font.render(f"Coins: {g.coins}", True, COL_UI), (18, 44))
    if g.game_over:
        tmsg = bigfont.render("GAME OVER", True, COL_UI_BAD)
        treason = font.render(g.death_reason, True, COL_UI)
        tret = font.render("Press R to restart", True, COL_UI)
        surface.blit(tmsg, (SCREEN_W // 2 - tmsg.get_width() // 2, 120))
        surface.blit(treason, (SCREEN_W // 2 - treason.get_width() // 2, 170))
        surface.blit(tret, (SCREEN_W // 2 - tret.get_width() // 2, 204))

# ----------------------------
# Reset
# ----------------------------
def new_game(seed: Optional[int] = None) -> Game:
    rng = random.Random(seed if seed is not None else random.randrange(1_000_000_000))
    g = Game(rng=rng)
    g.player = Player(6, 0)
    g.score = 0
    g.coins = 0
    g.max_forward_y = 0
    g.game_over = False
    g.death_reason = ""
    g.camera_y = 0.0
    g.path_x = 6
    g.recent_kinds = []
    g.lanes = {}

    for y in range(0, INTRO_SAFE_LANES):
        g.lanes[y] = create_lane(g, y)
    g.lanes[FIRST_HAZARD_Y] = create_lane(g, FIRST_HAZARD_Y)

    ensure_lanes(g)
    return g

# ----------------------------
# Main loop
# ----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("Crossy Road (Isometric) - Pygame")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 22)
    bigfont = pygame.font.SysFont("Menlo", 52, bold=True)

    g = new_game()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 1.0 / 20.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    g = new_game()

                if not g.game_over:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        try_start_hop(g, 0, +1)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        try_start_hop(g, 0, -1)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        try_start_hop(g, -1, 0)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        try_start_hop(g, +1, 0)

        if not g.game_over:
            update_lanes(g, dt)
            update_player(g, dt)
            check_collisions(g)

        render(g, screen, font, bigfont)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

