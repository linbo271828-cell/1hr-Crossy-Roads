import pygame
import sys
import random
import math

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 50
BLOCK_HEIGHT = 20

# Palette
WHITE = (240, 240, 255)

# Terrain Colors
GRASS_TOP = (100, 200, 100); GRASS_SIDE = (80, 180, 80)
ROAD_TOP = (100, 100, 100);  ROAD_SIDE = (80, 80, 80)
WATER_TOP = (50, 150, 250);  WATER_SIDE = (40, 130, 220)

# Voxel Palette
COLOR_TRUNK = (139, 69, 19)
COLOR_LEAVES = (34, 139, 34)
COLOR_LEAVES_LIGHT = (50, 205, 50)
COLOR_STONE = (120, 120, 120)
COLOR_COIN = (255, 215, 0)
COLOR_COIN_C = (200, 0, 0)
COLOR_CHICKEN = (255, 255, 255) 
COLOR_BEAK = (255, 50, 50) 
COLOR_COMB = (255, 0, 0)
COLOR_TIRE = (20, 20, 20)
COLOR_WINDOW = (150, 220, 255)

# Settings
ROAD_WIDTH = 6 
WORLD_OFFSET_X = 0 
WORLD_OFFSET_Y = 100

# --- Voxel Models ---
def get_sedan_model(body_color):
    m = []
    # Wheels
    m.append((0.3, 0.25, 0, COLOR_TIRE, 0.25, 0.25))
    m.append((0.3, -0.25, 0, COLOR_TIRE, 0.25, 0.25))
    m.append((-0.3, 0.25, 0, COLOR_TIRE, 0.25, 0.25))
    m.append((-0.3, -0.25, 0, COLOR_TIRE, 0.25, 0.25))
    # Body
    m.append((0, 0, 0.2, body_color, 0.9, 0.3))
    # Cabin
    m.append((-0.1, 0, 0.5, COLOR_WINDOW, 0.5, 0.3)) 
    m.append((-0.1, 0, 0.8, body_color, 0.55, 0.1))
    return m

MODELS = {
    'player': [
        (0, 0, 0, COLOR_CHICKEN, 0.6, 0.6),
        (0.2, 0, 0.6, COLOR_CHICKEN, 0.4, 0.4),
        (0.5, 0, 0.8, COLOR_BEAK, 0.2, 0.1),
        (0.2, 0, 1.1, COLOR_COMB, 0.1, 0.2),
        (0.35, 0.2, 0.8, (0,0,0), 0.05, 0.05),
        (0.35, -0.2, 0.8, (0,0,0), 0.05, 0.05),
    ],
    'tree': [
        (0, 0, 0, COLOR_TRUNK, 0.3, 0.6),
        (0, 0, 0.6, COLOR_LEAVES, 0.8, 0.5),
        (0, 0, 1.1, COLOR_LEAVES_LIGHT, 0.5, 0.5)
    ],
    'rock': [
        (0, 0, 0, COLOR_STONE, 0.7, 0.4),
        (0.1, 0.1, 0.2, (100,100,100), 0.5, 0.3)
    ],
    'coin': [
        (0, 0, 0.2, COLOR_COIN, 0.3, 0.3),
        (0, 0, 0.3, COLOR_COIN_C, 0.1, 0.2)
    ]
}

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Crossy Voxel: Fixed Physics")
clock = pygame.time.Clock()
font_small = pygame.font.SysFont("Arial", 30, bold=True)
font_large = pygame.font.SysFont("Arial", 60, bold=True)

# --- Classes ---
class GameObject:
    def __init__(self, row, col, obj_type):
        self.row = row 
        self.col = col
        self.type = obj_type
        if obj_type in MODELS: self.model = MODELS[obj_type]
        else: self.model = []

class Lane:
    def __init__(self, row_index, lane_type):
        self.row_index = row_index
        self.type = lane_type
        self.obstacles = []
        self.static_objects = []
        self.speed = 0
        self.direction = 1 
        
        # Generation Logic
        if lane_type == 'grass':
            if row_index < 5: return 

            obstacles_placed = 0
            # Coins
            if random.random() < 0.20: 
                c = random.randint(-ROAD_WIDTH, ROAD_WIDTH)
                self.static_objects.append(GameObject(row_index, c, 'coin'))

            # Trees/Rocks
            available_cols = list(range(-ROAD_WIDTH, ROAD_WIDTH + 1))
            random.shuffle(available_cols)
            for c in available_cols:
                if obstacles_placed >= 4: break
                if any(o.col == c for o in self.static_objects): continue

                if random.random() < 0.30:
                    t = 'tree' if random.random() < 0.7 else 'rock'
                    self.static_objects.append(GameObject(row_index, c, t))
                    obstacles_placed += 1

        elif lane_type == 'road':
            self.speed = random.uniform(0.03, 0.08)
            self.direction = random.choice([-1, 1])
            color = random.choice([(200, 50, 50), (50, 50, 200), (220, 100, 50), (100, 100, 100)])
            for _ in range(random.randint(1, 2)):
                start_col = random.randint(-ROAD_WIDTH, ROAD_WIDTH)
                car = GameObject(row_index, start_col, 'car')
                car.width = 1
                car.model = get_sedan_model(color)
                self.obstacles.append(car)

        elif lane_type == 'water':
            self.speed = random.uniform(0.03, 0.06)
            self.direction = random.choice([-1, 1])
            for _ in range(random.randint(2, 3)):
                start_col = random.randint(-ROAD_WIDTH, ROAD_WIDTH)
                log = GameObject(row_index, start_col, 'log')
                log.width = random.randint(2, 3) 
                self.obstacles.append(log)

    def update(self):
        for obs in self.obstacles:
            obs.col += self.speed * self.direction
            if self.direction == 1 and obs.col > ROAD_WIDTH + 4: obs.col = -ROAD_WIDTH - 4
            elif self.direction == -1 and obs.col < -ROAD_WIDTH - 4: obs.col = ROAD_WIDTH + 4

# --- Projection Engine ---
def to_iso(row, col, cam_x, cam_y):
    r_row = row - cam_x
    r_col = col - cam_y
    iso_x = (r_col + r_row) * (TILE_SIZE // 2)
    iso_y = (r_row - r_col) * (TILE_SIZE // 4) * -1
    return iso_x + SCREEN_WIDTH // 2 + WORLD_OFFSET_X, iso_y + SCREEN_HEIGHT // 2 + WORLD_OFFSET_Y

def draw_face(surface, color, pts):
    pygame.draw.polygon(surface, color, pts)

def draw_voxel(surface, x, y, color, w_scale=1.0, h_scale=1.0, height_offset=0):
    hw = (TILE_SIZE // 2) * w_scale
    hh = (TILE_SIZE // 4) * w_scale
    bh = BLOCK_HEIGHT * h_scale
    y -= height_offset 

    c_top = color
    c_right = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40))
    c_left = (max(0, color[0]-20), max(0, color[1]-20), max(0, color[2]-20))

    draw_face(surface, c_top, [(x, y - hh * 2), (x + hw, y - hh), (x, y), (x - hw, y - hh)])
    draw_face(surface, c_right, [(x + hw, y - hh), (x + hw, y - hh + bh), (x, y + bh), (x, y)])
    draw_face(surface, c_left, [(x, y), (x, y + bh), (x - hw, y - hh + bh), (x - hw, y - hh)])

def draw_model(surface, model, row, col, cam_x, cam_y, bounce_offset=0, facing_dir=1):
    center_x, center_y = to_iso(row, col, cam_x, cam_y)
    center_y += bounce_offset
    
    for voxel in model:
        d_row, d_col, dz, color, ws, hs = voxel
        eff_row = row + d_row
        eff_col = col + d_col * facing_dir
        vx, vy = to_iso(eff_row, eff_col, cam_x, cam_y)
        vy += bounce_offset
        draw_voxel(surface, vx, vy, color, ws, hs, height_offset=dz * 20)

# --- Game State ---
lanes = {} 
player_row = 0; player_col = 0
camera_row = 0; camera_col = 0
score = 0; coins = 0
game_over = False

def generate_lane(row_index):
    if row_index < 5: return Lane(row_index, 'grass')
    rnd = random.random()
    if rnd < 0.45: return Lane(row_index, 'grass')
    elif rnd < 0.75: return Lane(row_index, 'road')
    else: return Lane(row_index, 'water')

def get_or_create_lane(row):
    if row not in lanes:
        lanes[row] = generate_lane(row)
    return lanes[row]

def is_blocked(row, col):
    # Ensure coordinates are integers for static objects
    check_col = int(round(col)) 
    lane = get_or_create_lane(row)
    for obj in lane.static_objects:
        if obj.col == check_col and obj.type in ['tree', 'rock']: 
            return True
    return False

def check_coin(row, col):
    global coins
    check_col = int(round(col)) # Force integer alignment
    lane = get_or_create_lane(row)
    for obj in lane.static_objects[:]: 
        if obj.col == check_col and obj.type == 'coin':
            coins += 1
            lane.static_objects.remove(obj)
            return True
    return False

# Init initial world
for i in range(50): get_or_create_lane(i)

# --- Main Loop ---
running = True
while running:
    current_time = pygame.time.get_ticks()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        
        if event.type == pygame.KEYDOWN and not game_over:
            
            # --- CRITICAL FIX: Snap to grid before moving ---
            # If we are drifting on a log (float), we snap to int for the next hop
            current_grid_c = int(round(player_col))
            new_r, new_c = player_row, current_grid_c
            moved = False
            
            if event.key == pygame.K_UP: new_r += 1; moved = True
            elif event.key == pygame.K_DOWN: new_r -= 1; moved = True
            elif event.key == pygame.K_LEFT: new_c -= 1; moved = True
            elif event.key == pygame.K_RIGHT: new_c += 1; moved = True
            
            if -ROAD_WIDTH <= new_c <= ROAD_WIDTH and new_r > camera_row - 2:
                # Now we pass integer coordinates to collision check
                if not is_blocked(new_r, new_c):
                    player_row, player_col = new_r, new_c
                    if moved:
                        score = max(score, player_row)
                        check_coin(player_row, player_col)
        
        elif event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_SPACE:
                # RESET
                lanes = {}
                player_row = 0; player_col = 0; camera_row = 0; score = 0; coins = 0; game_over = False
                for i in range(50): get_or_create_lane(i)

    if not game_over:
        camera_row += (player_row - 4 - camera_row) * 0.1
        on_log = False; on_water = False
        
        for r in list(lanes.keys()):
            if r < camera_row - 10: del lanes[r]

        active_rows = range(int(camera_row)-5, int(camera_row)+25)
        for r in active_rows:
            lane = get_or_create_lane(r)
            lane.update()
            
            if r == player_row:
                if lane.type == 'water': on_water = True
                
                for obs in lane.obstacles:
                    # Hitbox for Cars/Logs
                    if obs.col - 0.8 < player_col < obs.col + obs.width - 0.2:
                        if lane.type == 'road': game_over = True 
                        elif lane.type == 'water': 
                            on_log = True
                            player_col += lane.speed * lane.direction 

        if on_water and not on_log: game_over = True 
        if player_col < -ROAD_WIDTH - 1 or player_col > ROAD_WIDTH + 1: game_over = True

    # --- Draw ---
    screen.fill(WHITE)
    
    start_row = int(camera_row) - 2
    end_row = int(camera_row) + 22
    
    for row in range(end_row, start_row - 1, -1):
        lane = get_or_create_lane(row)
        
        # Floor
        for col in range(-ROAD_WIDTH, ROAD_WIDTH + 1):
            if lane.type == 'grass': c = GRASS_TOP
            elif lane.type == 'road': c = ROAD_TOP
            elif lane.type == 'water': c = WATER_TOP
            draw_voxel(screen, *to_iso(row, col, camera_row, camera_col), c)
        
        # Static
        for obj in lane.static_objects:
            bob = math.sin(current_time * 0.005) * 5 if obj.type == 'coin' else 0
            draw_model(screen, obj.model, row, obj.col, camera_row, camera_col, bounce_offset=bob)

        # Moving
        for obs in lane.obstacles:
            if lane.type == 'road':
                draw_model(screen, obs.model, row, obs.col, camera_row, camera_col, facing_dir=lane.direction)
            elif lane.type == 'water':
                lc = (139, 69, 19)
                for w in range(obs.width):
                    lx, ly = to_iso(row, obs.col + w, camera_row, camera_col)
                    draw_voxel(screen, lx, ly, lc, 1.0, 0.5, height_offset=-5)

        # Player
        if row == player_row:
            if not game_over:
                draw_model(screen, MODELS['player'], row, player_col, camera_row, camera_col, bounce_offset=-10)
            else:
                 lx, ly = to_iso(row, player_col, camera_row, camera_col)
                 draw_voxel(screen, lx, ly, (200,200,200), 1.0, 0.1)

    # UI
    s_txt = font_small.render(f"Score: {score}", True, (0,0,0)); screen.blit(s_txt, (12, 12))
    c_txt = font_small.render(f"Coins: {coins}", True, (0,0,0)); screen.blit(c_txt, (12, 42))
    
    s_txt = font_small.render(f"Score: {score}", True, (255, 255, 255)); screen.blit(s_txt, (10, 10))
    c_txt = font_small.render(f"Coins: {coins}", True, (255, 200, 0)); screen.blit(c_txt, (10, 40))

    if game_over:
        go_txt = font_large.render("GAME OVER", True, (0,0,0))
        screen.blit(go_txt, (SCREEN_WIDTH//2 - go_txt.get_width()//2 + 2, SCREEN_HEIGHT//2 - 20 + 2))
        go_txt = font_large.render("GAME OVER", True, (255, 50, 50))
        screen.blit(go_txt, (SCREEN_WIDTH//2 - go_txt.get_width()//2, SCREEN_HEIGHT//2 - 20))
        restart_txt = font_small.render("Press SPACE to Restart", True, (100, 100, 100))
        screen.blit(restart_txt, (SCREEN_WIDTH//2 - restart_txt.get_width()//2, SCREEN_HEIGHT//2 + 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
