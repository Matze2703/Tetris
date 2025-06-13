import numpy as np
import pygame
import random
import gymnasium as gym
from gymnasium import spaces

# Spielfeld-Größe
ROWS = 20
COLS = 10

# Tetromino-Formen
TETROMINOS = {
    'I': [[1],
          [1],
          [1],
          [1],],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'S': [[1, 0],
          [1, 1],
          [0, 1]],
    'Z': [[0, 1],
          [1, 1],
          [1, 0]],
    'J': [[0, 1],
          [0, 1],
          [1, 1]],
    'L': [[1, 0],
          [1, 0],
          [1, 1]],
}

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.max_height = 0

        self.action_space = spaces.Discrete(6)  # 0: nichts, 1: links, 2: rechts, 3: runter, 4: drehen, 5: harter Drop
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, ROWS, COLS), dtype=np.float32)
        self.ticks = 0
        self.fall_interval = 2  # Alle x ticks ein automatischer Fall


        self.current_piece = None
        self.previous_shape = None
        self.piece_x = 0
        self.piece_y = 0

        # Pygame für Rendering
        self.window_size = 400
        self.block_size = self.window_size // ROWS
        self.screen = None
        self.clock = None
        self.skip_render = True  # Rendering im Training deaktivieren


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.board.fill(0)
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self._spawn_piece()
        return self._get_obs(), {}

    def _spawn_piece(self):
        while self.current_piece == self.previous_shape:
            self.current_piece = random.choice(list(TETROMINOS.values()))
        self.previous_shape = self.current_piece
        
        self.piece_y = 0
        self.piece_x = COLS // 2 - len(self.current_piece[0]) // 2

    def _valid_position(self, piece, x, y):
        for py, row in enumerate(piece):
            for px, cell in enumerate(row):
                if cell:
                    nx, ny = x + px, y + py
                    if nx < 0 or nx >= COLS or ny >= ROWS:
                        return False
                    if ny >= 0 and self.board[ny][nx]:
                        return False
        return True


    #Bestrafung für Höhe
    def _get_max_height(self):
        for y in range(ROWS):
            if np.any(self.board[y]):
                return ROWS - y
        return 0


    def _lock_piece(self):
        reward = 0

        for py, row in enumerate(self.current_piece):
            for px, cell in enumerate(row):
                if cell:
                    x, y = self.piece_x + px, self.piece_y + py
                    if 0 <= y < ROWS and 0 <= x < COLS:
                        self.board[y][x] = 1

        # Linien löschen
        lines_before = np.count_nonzero(self.board.sum(axis=1) == COLS)
        self.board = np.array([row for row in self.board if not all(row)]).tolist()
        new_rows = ROWS - len(self.board)
        self.board = [[0] * COLS for _ in range(new_rows)] + self.board
        self.board = np.array(self.board)

        self.lines_cleared += lines_before
        line_reward = (lines_before ** 2) * 100
        reward += line_reward       # Reward für KI
        self.score += line_reward   # Score

        self.level = 1 + self.lines_cleared // 5
        self._spawn_piece()

        # Belohnung für Platzieren
        reward += 5

        # Bestrafe hohe Stapel leicht
        current_height = self._get_max_height()
        if current_height > self.max_height:
            reward -= (current_height - self.max_height) * 5    # Für jeden Schritt nach oben -x Punkte
            self.max_height = current_height

        return reward


    def _get_obs(self):
        board_obs = np.copy(self.board)
        piece_obs = np.zeros_like(self.board)

        for py, row in enumerate(self.current_piece):
            for px, cell in enumerate(row):
                if cell:
                    x, y = self.piece_x + px, self.piece_y + py
                    if 0 <= y < ROWS and 0 <= x < COLS:
                        piece_obs[y][x] = 1

        obs = np.stack([board_obs, piece_obs], axis=0)
        return obs.astype(np.float32)
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Spieleraktion
        if action == 1:  # links
            if self._valid_position(self.current_piece, self.piece_x - 1, self.piece_y):
                self.piece_x -= 1
        elif action == 2:  # rechts
            if self._valid_position(self.current_piece, self.piece_x + 1, self.piece_y):
                self.piece_x += 1
        elif action == 3:  # runter
            if self._valid_position(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            else:
                reward += self._lock_piece()
        elif action == 4:  # drehen
            rotated = list(zip(*self.current_piece[::-1]))
            rotated = [list(row) for row in rotated]
            if self._valid_position(rotated, self.piece_x, self.piece_y):
                self.current_piece = rotated
        elif action == 5:  # Hard Drop
            while self._valid_position(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            reward += self._lock_piece()

        # Automatischer Fall
        self.ticks += 1
        if self.ticks >= self.fall_interval:
            self.ticks = 0
            if self._valid_position(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            else:
                reward += self._lock_piece()

        # Game Over prüfen
        if not self._valid_position(self.current_piece, self.piece_x, self.piece_y):
            terminated = True
            reward -= 100

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}


    def _draw_text(self, text, x, y, color=(255, 255, 255)):
        if not hasattr(self, "font") or self.font is None:
            self.font = pygame.font.SysFont("Arial", 20)
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))


    def render(self, mode="human"):
        if self.skip_render:
            return
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((COLS * self.block_size + 150, ROWS * self.block_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Hintergrund
        self.screen.fill((0, 0, 0))

        # Spielbrett zeichnen
        for y in range(ROWS):
            for x in range(COLS):
                color = (255, 255, 255) if self.board[y][x] else (40, 40, 40)
                pygame.draw.rect(self.screen, color, (x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        # Aktuelles Tetromino zeichnen
        for py, row in enumerate(self.current_piece):
            for px, cell in enumerate(row):
                if cell:
                    x = self.piece_x + px
                    y = self.piece_y + py
                    if 0 <= x < COLS and 0 <= y < ROWS:
                        pygame.draw.rect(self.screen, (0, 255, 0), (x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        # Score-Anzeige rechts neben dem Spielfeld
        info_x = COLS * self.block_size + 10
        self._draw_text(f"Score: {self.score}", info_x, 20)
        self._draw_text(f"Level: {self.level}", info_x, 50)
        self._draw_text(f"Lines: {self.lines_cleared}", info_x, 80)

        pygame.display.flip()
        self.clock.tick(10 + self.level)


    def close(self):
        if self.screen:
            pygame.quit()
