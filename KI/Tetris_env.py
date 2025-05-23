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
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'Z': [[1, 1, 0],
          [0, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]],
}


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.score = 0
        self.level = 0
        self.lines_cleared = 0

        self.action_space = spaces.Discrete(6)  # 0: nichts, 1: links, 2: rechts, 3: runter, 4: drehen, 5: harter Drop
        self.observation_space = spaces.Box(low=0, high=1, shape=(ROWS, COLS), dtype=np.float32)

        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0

        # Pygame für Rendering
        self.window_size = 400
        self.block_size = self.window_size // ROWS
        self.screen = None
        self.clock = None
        self.skip_render = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.board.fill(0)
        self.score = 0
        self.level = 0
        self.lines_cleared = 0
        self._spawn_piece()
        return self._get_obs(), {}

    def _spawn_piece(self):
        self.current_piece = random.choice(list(TETROMINOS.values()))
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

    def _lock_piece(self):
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
        self.score += (lines_before ** 2) * 100
        self.level = self.lines_cleared // 10
        self._spawn_piece()

    def _get_obs(self):
        obs = np.copy(self.board)
        for py, row in enumerate(self.current_piece):
            for px, cell in enumerate(row):
                if cell:
                    x, y = self.piece_x + px, self.piece_y + py
                    if 0 <= y < ROWS and 0 <= x < COLS:
                        obs[y][x] = 1
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

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
                self._lock_piece()
        elif action == 4:  # drehen
            rotated = list(zip(*self.current_piece[::-1]))
            rotated = [list(row) for row in rotated]
            if self._valid_position(rotated, self.piece_x, self.piece_y):
                self.current_piece = rotated
        elif action == 5:  # harter Drop
            while self._valid_position(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            self._lock_piece()
            reward += 10

        if not self._valid_position(self.current_piece, self.piece_x, self.piece_y):
            terminated = True
            reward -= 10

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.skip_render:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((COLS * self.block_size, ROWS * self.block_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))
        for y in range(ROWS):
            for x in range(COLS):
                color = (255, 255, 255) if self.board[y][x] else (40, 40, 40)
                pygame.draw.rect(self.screen, color, (x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        for py, row in enumerate(self.current_piece):
            for px, cell in enumerate(row):
                if cell:
                    x = self.piece_x + px
                    y = self.piece_y + py
                    if 0 <= x < COLS and 0 <= y < ROWS:
                        pygame.draw.rect(self.screen, (0, 255, 0), (x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        pygame.display.flip()
        self.clock.tick(10 + self.level)

    def close(self):
        if self.screen:
            pygame.quit()
