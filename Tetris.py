import pygame
import random

# Initialisierung
pygame.init()

# Konstanten
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
COLUMNS = SCREEN_WIDTH // BLOCK_SIZE
ROWS = SCREEN_HEIGHT // BLOCK_SIZE

# States
MENU = "menu"
PAUSE = "pause"
GAME = "game"

# Farben
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
COLORS = [
    (0, 255, 255),  # I
    (0, 0, 255),    # J
    (255, 165, 0),  # L
    (255, 255, 0),  # O
    (0, 255, 0),    # S
    (128, 0, 128),  # T
    (255, 0, 0)     # Z
]

# Formen der Tetriminos
TETROMINOS = {
    'I': [[1, 1, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'Z': [[1, 1, 0],
          [0, 1, 1]]
}

# Bildschirm
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

clock = pygame.time.Clock()
FPS = 60

# Klasse für Tetriminos
class Tetromino:
    def __init__(self):
        self.shape_name = random.choice(list(TETROMINOS))
        self.shape = TETROMINOS[self.shape_name]
        self.color = COLORS[list(TETROMINOS.keys()).index(self.shape_name)]
        self.x = COLUMNS // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

    def get_coords(self):
        coords = []
        for dy, row in enumerate(self.shape):
            for dx, val in enumerate(row):
                if val:
                    coords.append((self.x + dx, self.y + dy))
        return coords

    
    def collision(self, grid):
        for x, y in self.get_coords():
            if x < 0 or x >= COLUMNS or y >= ROWS:
                return True
            if (x, y) in grid:
                return True
        return False


# Funktionen
def create_grid(locked=None):
    if locked is None:
        locked = {}
    grid = [[(0, 0, 0) for _ in range(COLUMNS)] for _ in range(ROWS)]
    for y in range(ROWS):
        for x in range(COLUMNS):
            if (x, y) in locked:
                grid[y][x] = locked[(x, y)]
    return grid

def draw_grid(surface, grid):
    for y in range(ROWS):
        for x in range(COLUMNS):
            pygame.draw.rect(surface, grid[y][x],
                             (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    for y in range(ROWS):
        pygame.draw.line(surface, GRAY, (0, y * BLOCK_SIZE), (SCREEN_WIDTH, y * BLOCK_SIZE))
    for x in range(COLUMNS):
        pygame.draw.line(surface, GRAY, (x * BLOCK_SIZE, 0), (x * BLOCK_SIZE, SCREEN_HEIGHT))

def clear_rows(grid, locked):
    cleared = 0
    for y in range(ROWS - 1, -1, -1):
        if (0, 0, 0) not in grid[y]:
            cleared += 1
            for x in range(COLUMNS):
                try:
                    del locked[(x, y)]
                except:
                    continue
            for yy in range(y - 1, -1, -1):
                for x in range(COLUMNS):
                    if (x, yy) in locked:
                        locked[(x, yy + 1)] = locked.pop((x, yy))
    return cleared

# Spiel-Loop
def main():
    locked_positions = {0,0,0,0,0,0,0,0,0,0}
    grid = create_grid(locked_positions)

    current_piece = Tetromino()
    fall_time = 0
    fall_speed = 0.5
    change_piece = False

    run = True
    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick(FPS)

        if fall_time / 1000 >= fall_speed:
            current_piece.y += 1
            if current_piece.collision(grid):
                current_piece.y -= 1
                change_piece = True
            fall_time = 0

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if current_piece.collision(grid):
                        current_piece.x += 1
                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if current_piece.collision(grid):
                        current_piece.x -= 1
                elif event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if current_piece.collision(grid):
                        current_piece.y -= 1
                elif event.key == pygame.K_UP:
                    current_piece.rotate()
                    if current_piece.collision(grid):
                        current_piece.rotate()
                        current_piece.rotate()
                        current_piece.rotate()

        for x, y in current_piece.get_coords():
            if y >= 0:
                grid[y][x] = current_piece.color

        if change_piece:
            for x, y in current_piece.get_coords():
                locked_positions[(x, y)] = current_piece.color
            current_piece = Tetromino()
            change_piece = False
            clear_rows(grid, locked_positions)

        # Zeichnen
        screen.fill(BLACK)
        draw_grid(screen, grid)
        pygame.display.update()

main()
