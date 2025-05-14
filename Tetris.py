import pygame
import sys
import random

pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GAME_WIDTH, GAME_HEIGHT = 300, 600
BLOCK_SIZE = 30
COLS, ROWS = 10, 20
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARKGREY = (50, 50, 50)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()

# Game States
MENU, GAME, PAUSE, TRANSITION = "menu", "game", "pause", "transition"
state = MENU

font = pygame.font.SysFont("Arial", 40)

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

# Button class
class Button:
    def __init__(self, text, x, y, w, h, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback

    def draw(self, surface):
        pygame.draw.rect(surface, GREY, self.rect)
        pygame.draw.rect(surface, DARKGREY, self.rect, 3)
        txt_surface = font.render(self.text, True, BLACK)
        txt_rect = txt_surface.get_rect(center=self.rect.center)
        surface.blit(txt_surface, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()

# Transition
transition_target = None
transition_y = HEIGHT

def start_transition(target_state):
    global state, transition_target, transition_y
    transition_target = target_state
    transition_y = HEIGHT
    state = TRANSITION

def start_game():
    reset_game()
    start_transition(GAME)

def return_to_menu():
    start_transition(MENU)

def resume_game():
    start_transition(GAME)

def get_menu_buttons(width, height):
    return [Button("Start Game", width // 2 - 100, height // 2 + 50, 200, 50, start_game)]

def get_pause_buttons(width, height):
    return [
        Button("Continue", width // 2 - 100, height // 2 - 60, 200, 50, resume_game),
        Button("Main Menu", width // 2 - 100, height // 2 + 10, 200, 50, return_to_menu),
    ]

# Game logic
score = 0
level = 1
lines_cleared = 0

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
        score_add = [0, 40, 100, 300, 1200][cleared] * level
        score += score_add
        lines_cleared += cleared
        level = 1 + lines_cleared // 10
        fall_speed = max(100, 500 - (level - 1) * 30)

def move_piece(dx, dy):
    if valid_position(current_piece, dx, dy):
        current_piece['x'] += dx
        current_piece['y'] += dy
        return True
    return False

def drop_piece():
    global current_piece
    if not move_piece(0, 1):
        merge_piece(current_piece)
        clear_lines()
        current_piece = create_piece()
        if not valid_position(current_piece):
            return_to_menu()

def hard_drop():
    while move_piece(0, 1):
        pass
    drop_piece()

def reset_game():
    global board, current_piece, fall_time, fall_speed, score, level, lines_cleared
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    current_piece = create_piece()
    fall_time = 0
    fall_speed = 500
    score = 0
    level = 1
    lines_cleared = 0

def get_ghost_piece(piece):
    ghost = dict(piece)
    while valid_position(ghost, dy=1):
        ghost['y'] += 1
    return ghost

# Drawing functions
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

def draw_text_centered(text, size, y, colour=BLACK):
    fnt = pygame.font.SysFont("Arial", size)
    txt_surface = fnt.render(text, True, colour)
    txt_rect = txt_surface.get_rect(center=(WIDTH // 2, y))
    screen.blit(txt_surface, txt_rect)

# Hintergrundbild laden
bg_tile = pygame.image.load("Background.png").convert()
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

# Main Loop
reset_game()
running = True
last_fall = pygame.time.get_ticks()

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
        elif state == GAME:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    start_transition(PAUSE)
                elif event.key == pygame.K_a:
                    move_piece(-1, 0)
                elif event.key == pygame.K_d:
                    move_piece(1, 0)
                elif event.key == pygame.K_s:
                    drop_piece()
                elif event.key == pygame.K_w:
                    rotated = rotate(current_piece['matrix'])
                    if valid_position(current_piece, rotated=rotated):
                        current_piece['matrix'] = rotated
                elif event.key == pygame.K_SPACE:
                    hard_drop()
        elif state == PAUSE:
            for btn in get_pause_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)

    if state == GAME and now - last_fall > fall_speed:
        drop_piece()
        last_fall = now

    offset_x = WIDTH // 2 - GAME_WIDTH // 2
    offset_y = 0

    if state == MENU:
        draw_text_centered("★ TETRIS ★", 80, HEIGHT // 2 - 120, (30, 30, 150))
        for btn in get_menu_buttons(WIDTH, HEIGHT):
            btn.draw(screen)

    elif state == GAME:
        pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        draw_board(offset_x, offset_y)
        ghost_piece = get_ghost_piece(current_piece)
        draw_piece(ghost_piece, offset_x, offset_y, ghost=True)
        draw_piece(current_piece, offset_x, offset_y)
        info_x = offset_x + GAME_WIDTH + 40
        screen.blit(font.render(f"Score: {score}", True, BLACK), (info_x, 100))
        screen.blit(font.render(f"Level: {level}", True, BLACK), (info_x, 150))
        screen.blit(font.render(f"Lines: {lines_cleared}", True, BLACK), (info_x, 200))

    elif state == PAUSE:
        draw_text_centered("PAUSED", 60, HEIGHT // 2 - 120)
        for btn in get_pause_buttons(WIDTH, HEIGHT):
            btn.draw(screen)

    elif state == TRANSITION:
        transition_y -= 30
        if transition_y <= 0:
            state = transition_target
        else:
            pygame.draw.rect(screen, WHITE, (0, HEIGHT - transition_y, WIDTH, transition_y))

    pygame.display.flip()

pygame.quit()
sys.exit()
