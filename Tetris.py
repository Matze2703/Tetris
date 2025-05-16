"""

To do:
        olkek:
            - Shape spawn algorithmus bearbeiten
            - Shape move während gameplay funktion
            - Potentiell besseres rotate mit Q und E
            - Transition funktion mit Aimationen untersuchen
            - Punkte-System Overhaul: mehr Konditionen um Punkte zu geben und visuell verbessern

        Matze:
            - Sounds für UI (Kurzes 8-bit bop für buttons)
            - Vielleicht sounds für shape placement und line clear
            - Speziall sound für "Tetris" (4x Line clear)

"""

import pygame, sys, random, time
pygame.init()

#############
# CONSTANTS #
#############
WIDTH, HEIGHT = 1000, 800
GAME_WIDTH, GAME_HEIGHT = 300, 600
BLOCK_SIZE = 30
COLS, ROWS = 10, 20
FPS = 60

# Musik
music_tracks = ["Original_Theme.mp3","Piano_Theme.mp3","TAKEO_ENDBOSS.mp3"]
selected_track = 1
music_volume = 0.5
pygame.mixer.music.load("sound_design\\" + music_tracks[selected_track-1])
pygame.mixer.music.play(-1, 0.0)    # -1 = Loopen lassen
pygame.mixer.music.set_volume(0.5)

#SFX
sfx_volume = 0.5
def play_sound(soundfile):
    sound = pygame.mixer.Sound(f"sound_design\\{soundfile}")
    sound.set_volume(sfx_volume)
    sound.play()



WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARKGREY = (50, 50, 50)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()

# Game States
MENU, GAME, PAUSE, GAME_OVER, OPTIONS = "menu", "game", "pause", "game_over", "options"
state = MENU

font = pygame.font.Font("game_design\\Pixel_Emulator.otf", 40)
small_font = pygame.font.SysFont("Arial", 24)

# Shapes and Colors
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'J': [[1, 0, 0], [1, 1, 1]],
    'L': [[0, 0, 1], [1, 1, 1]],
}
SHAPE_COLORS = {
    'I': (0, 255, 255),
    'O': (255, 255, 0),
    'T': (128, 0, 128),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0),
}


###########
# KLASSEN #
###########

# Button class
class Button:
    def __init__(self, text, x, y, w, h, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback

    def draw(self, surface):
        draw_text_centered(
            self.text,
            y=self.rect.centery,
            x=self.rect.centerx,
            bg_img="game_design\\Border_2.png",
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()


def start_game():
    reset_game()
    global state 
    state = GAME

def return_to_menu():
    global state 
    state = MENU

def resume_game():
    global state 
    state = GAME


#Platzierung der Buttons korrigieren
def get_menu_buttons(width, height):
    return [
        Button("Start", width // 2 - 100, height // 2 , 200, 50, start_game),
        Button("Options", width // 2 - 100, height // 2 + 100, 200, 50, start_game),
    ]

def get_pause_buttons(width, height):
    return [
        Button("Continue", width // 2 - 100, height // 2 , 200, 50, resume_game),
        Button("Main Menu", width // 2 - 100, height // 2 + 100, 200, 50, return_to_menu),
    ]

def get_game_over_buttons(width, height):
    return [
        Button("Try Again", WIDTH // 2 - 100, HEIGHT // 2 + 60, 200, 50, start_game),
        Button("Main Menu", WIDTH // 2 - 100, HEIGHT // 2 + 130, 200, 50, return_to_menu),
    ]

##########################
# GAME LOGIC & MECHANICS #
##########################

score = 0
level = 1
lines_cleared = 0
score_popup = []
hold_piece = None
next_queue = []
used_hold = False

class ScorePopup:
    def __init__(self, text, x, y):
        self.text = text
        self.x = x
        self.y = y
        self.timer = 60

    def draw(self):
        txt_surface = small_font.render(self.text, True, BLACK)
        screen.blit(txt_surface, (self.x, self.y))
        self.y -= 1
        self.timer -= 1

    def is_alive(self):
        return self.timer > 0


##########################
# GAME LOGIC & MECHANICS #
##########################

score = 0
level = 1
lines_cleared = 0
score_popup = []
hold_piece = None
next_queue = []
used_hold = False

def create_piece():
    shape = random.choice(list(SHAPES.keys()))
    return {
        'shape': shape,
        'matrix': SHAPES[shape],
        'x': COLS // 2 - len(SHAPES[shape][0]) // 2,
        'y': 0,
        'color': SHAPE_COLORS[shape]
    }

def rotate(matrix):
    return [list(row)[::-1] for row in zip(*matrix)]

def wall_kick(piece, rotated):
    for dx in [0, -1, 1, -2, 2]:
        if valid_position(piece, dx=dx, rotated=rotated):
            piece['x'] += dx
            piece['matrix'] = rotated
            return

def valid_position(piece, dx=0, dy=0, rotated=None):
    shape = rotated if rotated else piece['matrix']
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                new_x = piece['x'] + x + dx
                new_y = piece['y'] + y + dy
                if new_x < 0 or new_x >= COLS or new_y >= ROWS:
                    return False
                if new_y >= 0 and board[new_y][new_x]:
                    return False
    return True

def merge_piece(piece):
    for y, row in enumerate(piece['matrix']):
        for x, cell in enumerate(row):
            if cell:
                board[piece['y'] + y][piece['x'] + x] = piece['color']

def clear_lines():
    global board, score, lines_cleared, level, fall_speed
    new_board = []
    cleared = 0
    for row in board:
        if all(row):
            cleared += 1
        else:
            new_board.append(row)
    for _ in range(cleared):
        new_board.insert(0, [0 for _ in range(COLS)])
    board = new_board
    if cleared:
        base_scores = [0, 100, 300, 500, 800]
        score_add = base_scores[cleared] * level
        score_popup.append(ScorePopup(f"{cleared}x Line clear", 30, 250))
        score_popup.append(ScorePopup(f"+{score_add} pts", 30, 280))
        score += score_add
        lines_cleared += cleared
        level = 1 + lines_cleared // 10  # Adjusted to increase level every 10 lines
        fall_speed = max(100, 500 - (level - 1) * 30)

def move_piece(dx, dy):
    if valid_position(current_piece, dx, dy):
        current_piece['x'] += dx
        current_piece['y'] += dy
        return True
    return False

def drop_piece():
    global current_piece, used_hold
    if not move_piece(0, 1):
        merge_piece(current_piece)
        clear_lines()
        current_piece = next_queue.pop(0)
        next_queue.append(create_piece())
        used_hold = False
        if not valid_position(current_piece):
            time.sleep(1)
            game_over()

def hard_drop():
    global score
    drops = 0
    while move_piece(0, 1):
        drops += 1
    score += drops * 2
    score_popup.append(ScorePopup(f"Hard Drop", 30, 320))
    score_popup.append(ScorePopup(f"+{drops * 2} pts", 30, 350))
    drop_piece()

def reset_game():
    global board, current_piece, next_queue, fall_time, fall_speed, score, level, lines_cleared, score_popup, hold_piece, used_hold
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    next_queue = [create_piece() for _ in range(3)]
    current_piece = next_queue.pop(0)
    next_queue.append(create_piece())
    fall_time = pygame.time.get_ticks()  # Initialize to current time
    fall_speed = 500
    score = 0
    level = 1
    lines_cleared = 0
    score_popup = []
    hold_piece = None
    used_hold = False

def hold_current_piece():
    global hold_piece, current_piece, next_queue, used_hold
    if used_hold:
        return
    used_hold = True
    if hold_piece is None:
        hold_piece = current_piece
        current_piece = next_queue.pop(0)
        next_queue.append(create_piece())
    else:
        hold_piece, current_piece = current_piece, hold_piece
    current_piece['x'] = COLS // 2 - len(current_piece['matrix'][0]) // 2
    current_piece['y'] = 0

def draw_piece_in_box(piece, offset_x, offset_y, scale=1.0):
    matrix = piece['matrix']
    color = piece['color']
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(offset_x + x * BLOCK_SIZE * scale, offset_y + y * BLOCK_SIZE * scale,
                                   BLOCK_SIZE * scale, BLOCK_SIZE * scale)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)


def draw_next_pieces():
    box_width, box_height = 160, 260
    NEXT_PIECE_X = WIDTH // 2 + 200
    NEXT_PIECE_Y = 100

    pygame.draw.rect(screen, BLACK, (NEXT_PIECE_X, NEXT_PIECE_Y, box_width, box_height))
    pygame.draw.rect(screen, GREY, (NEXT_PIECE_X, NEXT_PIECE_Y, box_width, box_height), 4)

    draw_text_centered("Next:", NEXT_PIECE_Y-45, NEXT_PIECE_X+75)
    
    for i, piece in enumerate(next_queue[:3]):
        matrix = piece['matrix']
        piece_width = len(matrix[0]) * BLOCK_SIZE * 0.75
        piece_height = len(matrix) * BLOCK_SIZE * 0.75
        x_offset = NEXT_PIECE_X + (box_width - piece_width) // 2 - 15
        y_offset = NEXT_PIECE_Y + i * 80 + (60 - piece_height) // 2 + 20
        draw_piece_in_box(piece, x_offset, y_offset, 0.75)


def draw_hold_piece():
    box_width, box_height = 120, 120
    HOLD_PIECE_X = WIDTH // 2 - 350
    HOLD_PIECE_Y = 50

    pygame.draw.rect(screen, BLACK, (HOLD_PIECE_X - 10, HOLD_PIECE_Y - 10, box_width, box_height), 0)
    pygame.draw.rect(screen, GREY, (HOLD_PIECE_X - 10, HOLD_PIECE_Y - 10, box_width, box_height), 5)
    
    if hold_piece:
        matrix = hold_piece['matrix']
        shape_width = len(matrix[0]) * BLOCK_SIZE
        shape_height = len(matrix) * BLOCK_SIZE
        x_offset = HOLD_PIECE_X + (box_width - shape_width) // 2 - 10
        y_offset = HOLD_PIECE_Y + (box_height - shape_height) // 2 - 10
        draw_piece_in_box(hold_piece, x_offset, y_offset, 1)

    draw_text_centered("Press F", HOLD_PIECE_Y+150, HOLD_PIECE_X + 50)


def game_over():
    global state
    state = GAME_OVER

def start_game():
    reset_game()
    global state
    state = GAME

def return_to_menu():
    global state
    state = MENU

def go_to_options():
    global state
    state = OPTIONS

def resume_game():
    global state
    state = GAME

def change_music_volume(delta):
    global music_volume
    music_volume = min(max(round((music_volume + delta) * 20) / 20, 0.0), 1.0)
    pygame.mixer.music.set_volume(music_volume)

def change_music_track(delta):
    global selected_track
    if delta == -1 and selected_track != 1:
        selected_track += delta
    if delta == +1 and selected_track != (len(music_tracks)):
        selected_track += delta
    pygame.mixer.music.load("sound_design\\" + music_tracks[selected_track-1])
    pygame.mixer.music.play(-1, 0.0)





##########
# DESIGN #
##########

def get_menu_buttons(width, height):
    return [
        Button("Start", width // 2 - 100, height // 2 , 200, 50, start_game),
        Button("Options", width // 2 - 100, height // 2 + 100, 200, 50, go_to_options),
    ]

def get_options_UI(width, height):
    return [
        Button("<", width // 2 -250, height // 2 -25, 80, 50, lambda: change_music_volume(-0.05)),
        Button(">", width // 2 +170, height // 2 -25, 80, 50, lambda: change_music_volume(+0.05)),
        Button("<", width // 2 -250, height // 2 -125, 80, 50, lambda: change_music_track(-1)),
        Button(">", width // 2 +170, height // 2 -125, 80, 50, lambda: change_music_track(+1)),
        Button("Back", width // 2 -300, height // 2 +100, 150, 80, return_to_menu),
    ]

def get_pause_buttons(width, height):
    return [
        Button("Continue", width // 2 - 100, height // 2 , 200, 50, resume_game),
        Button("Main Menu", width // 2 - 100, height // 2 + 100, 200, 50, return_to_menu),
    ]

def get_game_over_buttons(width, height):
    return [
        Button("Try Again", width // 2 - 100, height // 2 + 60, 200, 50, start_game),
        Button("Main Menu", width // 2 - 100, height // 2 + 130, 200, 50, return_to_menu),
    ]


def draw_board(offset_x, offset_y):
    for y in range(ROWS):
        for x in range(COLS):
            rect = pygame.Rect(offset_x + x * BLOCK_SIZE, offset_y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, DARKGREY, rect, 1)
            color = board[y][x]
            if color:
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)


def draw_piece(piece, offset_x, offset_y, ghost=False):
    color = piece['color']
    alpha = 100 if ghost else 255
    for y, row in enumerate(piece['matrix']):
        for x, cell in enumerate(row):
            if cell:
                px = piece['x'] + x
                py = piece['y'] + y
                if py >= 0:
                    rect = pygame.Rect(offset_x + px * BLOCK_SIZE, offset_y + py * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    s = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
                    s.fill((*color, alpha))
                    screen.blit(s, rect.topleft)
                    pygame.draw.rect(screen, BLACK, rect, 1)

def get_ghost_piece(piece):
    ghost = dict(piece)
    while valid_position(ghost, dy=1):
        ghost['y'] += 1
    return ghost


def draw_text_centered(text, y, x=None, bg_img="game_design\\Border_2.png", colour=WHITE):
    fnt = pygame.font.Font("game_design\\Pixel_Emulator.otf", 40)
    txt_surface = fnt.render(text, True, colour)

    if x is None:
        txt_rect = txt_surface.get_rect(center=(WIDTH // 2, y))
    else:
        txt_rect = txt_surface.get_rect(center=(x, y))

    padding = 20
    box_rect = txt_rect.inflate(padding * 2, padding * 2)

    pygame.draw.rect(screen, BLACK, box_rect)

    border = pygame.image.load(bg_img).convert_alpha()

    corner = 20
    bw, bh = border.get_size()

    top_left     = border.subsurface((0, 0, corner, corner))
    top_right    = border.subsurface((bw - corner, 0, corner, corner))
    bottom_left  = border.subsurface((0, bh - corner, corner, corner))
    bottom_right = border.subsurface((bw - corner, bh - corner, corner, corner))

    top    = border.subsurface((corner, 0, bw - 2 * corner, corner))
    bottom = border.subsurface((corner, bh - corner, bw - 2 * corner, corner))
    left   = border.subsurface((0, corner, corner, bh - 2 * corner))
    right  = border.subsurface((bw - corner, corner, corner, bh - 2 * corner))

    screen.blit(top_left, (box_rect.left, box_rect.top))
    screen.blit(top_right, (box_rect.right - corner, box_rect.top))
    screen.blit(bottom_left, (box_rect.left, box_rect.bottom - corner))
    screen.blit(bottom_right, (box_rect.right - corner, box_rect.bottom - corner))

    screen.blit(pygame.transform.scale(top, (box_rect.width - 2 * corner, corner)), 
                (box_rect.left + corner, box_rect.top))
    screen.blit(pygame.transform.scale(bottom, (box_rect.width - 2 * corner, corner)), 
                (box_rect.left + corner, box_rect.bottom - corner))
    screen.blit(pygame.transform.scale(left, (corner, box_rect.height - 2 * corner)), 
                (box_rect.left, box_rect.top + corner))
    screen.blit(pygame.transform.scale(right, (corner, box_rect.height - 2 * corner)), 
                (box_rect.right - corner, box_rect.top + corner))

    screen.blit(txt_surface, txt_rect)


bg_tile = pygame.image.load("game_design\\Background.png").convert()
tile_width, tile_height = bg_tile.get_size()

def draw_background():
    for x in range(0, WIDTH, tile_width):
        for y in range(0, HEIGHT, tile_height):
            screen.blit(bg_tile, (x, y))


def update_GUI():
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = screen.get_size()
    screen.fill(WHITE)
    draw_background()


#############
# MAIN LOOP #
#############
reset_game()
running = True

while running:
    update_GUI()
    clock.tick(FPS)
    now = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == MENU:
            for btn in get_menu_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)
        
        elif state == OPTIONS:
            for btn in get_options_UI(WIDTH, HEIGHT):
                btn.handle_event(event)
        
        elif state == GAME:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state = PAUSE
                elif event.key == pygame.K_a:
                    play_sound("move.mp3")
                    move_piece(-1, 0)
                elif event.key == pygame.K_d:
                    play_sound("move.mp3")
                    move_piece(1, 0)
                elif event.key == pygame.K_s:
                    drop_piece()
                elif event.key == pygame.K_w:
                    play_sound("rotate.mp3")
                    rotated = rotate(current_piece['matrix'])
                    wall_kick(current_piece, rotated)
                elif event.key == pygame.K_SPACE:
                    play_sound("drop.mp3")
                    hard_drop()
                elif event.key == pygame.K_f:
                    hold_current_piece()
        
        elif state == PAUSE:
            for btn in get_pause_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)
        
        elif state == GAME_OVER:
            for btn in get_game_over_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)

    if state == GAME and now - fall_time > fall_speed:
        drop_piece()
        fall_time = now

    offset_x = WIDTH // 2 - GAME_WIDTH // 2
    offset_y = 0

    if state == MENU:
        draw_text_centered("TETRIS", 200, None, "game_design\\Border.png", (30, 30, 150))
        for btn in get_menu_buttons(WIDTH, HEIGHT):
            btn.draw(screen)
    
    elif state == OPTIONS:
        draw_text_centered(f"TRACK: {selected_track}", HEIGHT // 2 -100)
        draw_text_centered(f"MUSIK: {round(music_volume*100)}%", HEIGHT // 2)
        for btn in get_options_UI(WIDTH, HEIGHT):
            btn.draw(screen)

    elif state == GAME:
        pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        draw_board(offset_x, offset_y)
        draw_piece(get_ghost_piece(current_piece), offset_x, offset_y, ghost=True)
        draw_piece(current_piece, offset_x, offset_y)

        draw_text_centered(f"Score: {score}", 450, WIDTH // 2 +300)
        draw_text_centered(f"Level: {level}", 550, WIDTH // 2 +300)
        draw_text_centered(f"Lines: {lines_cleared}", 650, WIDTH // 2 +300)
        draw_next_pieces()
        draw_hold_piece()
        for popup in score_popup:
            if popup.is_alive():
                popup.draw()
        score_popup[:] = [p for p in score_popup if p.is_alive()]
    
    elif state == PAUSE:
        draw_text_centered("PAUSED", 200, None, "game_design\\Border.png", (30, 30, 150))
        for btn in get_pause_buttons(WIDTH, HEIGHT):
            btn.draw(screen)
    
    elif state == GAME_OVER:
        draw_text_centered("GAME OVER", 200, None, "game_design\\Border.png", (30, 30, 150))
        draw_text_centered(f"Final Score: {score}", 300, None, "game_design\\Border.png")
        for btn in get_game_over_buttons(WIDTH, HEIGHT):
            btn.draw(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()