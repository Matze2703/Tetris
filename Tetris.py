"""





████████╗███████╗████████╗ ██████╗  ██╗███████╗           
╚══██╔══╝██╔════╝╚══██╔══╝ ██╔══██╗ ██║██╔════╝ 
   ██║   █████╗     ██║    ██████╔╝ ██║███████╗ 
   ██║   ██╔══╝     ██║    ██╔══██╗ ██║╚════██║  
   ██║   ███████╗   ██║    ██║  ██║ ██║███████║  ...2
   ╚═╝   ╚══════╝   ╚═╝    ╚═╝  ╚═╝ ╚═╝╚══════╝ 

   Das beste Spiel seit "Tetris"!

   ©Copyright Polnische Mafia

    





"""

import importlib.util
import subprocess
import sys

def install_and_import(package_name):
    if importlib.util.find_spec(package_name) is None:
        print(f"Modul '{package_name}' nicht gefunden. Installiere...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Alle Module installieren
required_modules = ["requests", "os", "random","time","cryptography","pygame"]

for module in required_modules:
    install_and_import(module)



import pygame, random, os, time
from cryptography.fernet import Fernet
import database_access as sql_db
pygame.init()

#############
# CONSTANTS #
#############
WIDTH, HEIGHT = 1200, 800
GAME_WIDTH, GAME_HEIGHT = 300, 600
BLOCK_SIZE = 30
COLS, ROWS = 10, 20
FPS = 60
player_name = ''



#Config für Musik (und weiteres in Zukunft)
if not os.path.isfile("config.txt"):
    with open("config.txt", "w") as datei:
        print("config erstellt")
        datei.write("1\n")
        datei.write("0.5\n")
        datei.write("0.5\n")
        datei.write("False\n")
        datei.write("1\n")


# Verschlüsselung der Scores
# Schlüssel generieren und speichern
def generate_key():
    print("Neuer Key erstellt")
    key = Fernet.generate_key()
    with open("schluessel.key", "wb") as key_file:
        key_file.write(key)

# Schlüssel laden
def load_key():
    return open("schluessel.key", "rb").read()

# Datei verschlüsseln
def encrypt_file(filename):
    key = load_key()
    f = Fernet(key)

    with open(filename, "rb") as file:
        file_data = file.read()

    encrypted_data = f.encrypt(file_data)

    with open(filename + ".enc", "wb") as file:
        file.write(encrypted_data)
    os.remove(filename)

# Datei entschlüsseln
def decrypt_file(encrypted_filename, output_filename):
    key = load_key()
    f = Fernet(key)

    with open(encrypted_filename, "rb") as file:
        encrypted_data = file.read()

    decrypted_data = f.decrypt(encrypted_data)

    with open(output_filename, "wb") as file:
        file.write(decrypted_data)


# Schlüssel nur generieren wenn noch keiner vorhanden
if not os.path.isfile("schluessel.key"):
    generate_key()
    
# Verschlüsseln der neuen Scores Datei, wenn sie vorher noch nicht vorhanden war
if not os.path.isfile("Scores.txt") and not os.path.isfile("Scores.txt.enc"):
    print("Neue Score-Datei erstellt")
    with open("Scores.txt", "w") as datei:
        pass
    encrypt_file("Scores.txt")
    

# Einstellungen importieren
with open("config.txt", "r") as datei:
    selected_track = int(datei.readline().strip())
    music_volume = float(datei.readline().strip())
    sfx_volume = float(datei.readline().strip())
    fullscreen = datei.readline().strip().lower() == "true"
    bg_nr = int(datei.readline().strip())

# Musik
music_tracks = ["Original_Theme.mp3","Piano_Theme.mp3","TAKEO_ENDBOSS.mp3"]
backgrounds = ["1", "2", "3", "4","5", "goofy"]
pygame.mixer.music.load("sound_design\\" + music_tracks[selected_track-1])
pygame.mixer.music.play(-1, 0.0)    # -1 = Loopen lassen
pygame.mixer.music.set_volume(music_volume)

#SFX
def play_sound(soundfile):
    sound = pygame.mixer.Sound(f"sound_design\\{soundfile}")
    sound.set_volume(sfx_volume)
    sound.play()


def update_config():
    with open("config.txt", "w") as datei:
        datei.write(f"{selected_track}\n")
        datei.write(f"{music_volume}\n")
        datei.write(f"{sfx_volume}\n")
        datei.write(f"{fullscreen}\n")
        datei.write(f"{bg_nr}\n")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARKGREY = (50, 50, 50)
YELLOW = (242, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
if fullscreen:
    pygame.display.toggle_fullscreen()
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()

icon = pygame.image.load("game_design\\icon-p.png").convert_alpha()
pygame.display.set_icon(icon)



# Set Game state
state = "BOOTUP_SEQUENCE"
previous_state = ''
bootup_start_time = pygame.time.get_ticks()
bootup_duration = 5000  # milliseconds (5 seconds)
fade_duration = 1000    # milliseconds (1 second for fade in/out)

font = pygame.font.Font(r"game_design\Pixel_Emulator.otf", 40)
small_font = pygame.font.Font(r"game_design\Pixel_Emulator.otf", 24)
big_font = pygame.font.Font(r"game_design\Pixel_Emulator.otf", 64)

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
# BUTTONS #
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
            play_sound("button_bop.mp3")
            self.callback()


def start_game():
    global ingame_ui_anim_active, ingame_ui_anim_start_time
    reset_game()
    ingame_ui_anim_active = True
    ingame_ui_anim_start_time = pygame.time.get_ticks()
    global state 
    state = "GAME"
    

def go_to_options():
    global state
    state = "OPTIONS"

def return_to_menu():
    global state
    state = "MENU"

def resume_game():
    global state 
    state = "GAME"

def go_back():
    global state, previous_state
    state = previous_state

def game_over():
    global state
    state = "GAME_OVER"

def save_score():
    global state
    state = "ENTER_NAME"

def change_volume(variable,delta):
    if variable == "music_volume":
        global music_volume
        music_volume = min(max(round((music_volume + delta) * 20) / 20, 0.0), 1.0)
        pygame.mixer.music.set_volume(music_volume)
    elif variable == "sfx_volume":
        global sfx_volume
        sfx_volume = min(max(round((sfx_volume + delta) * 20) / 20, 0.0), 1.0)
    update_config()

def change_music_track(delta):
    global selected_track
    if delta == -1 and selected_track != 1:
        selected_track += delta
    if delta == +1 and selected_track != (len(music_tracks)):
        selected_track += delta
    pygame.mixer.music.load("sound_design\\" + music_tracks[selected_track-1])
    pygame.mixer.music.play(-1, 0.0)
    update_config()

def change_background(delta):
    global bg_nr
    if delta == -1 and bg_nr != 1:
        bg_nr += delta
    if delta == +1 and bg_nr != (len(backgrounds)):
        bg_nr += delta
    bg_tile = pygame.image.load(f"game_design\\bg{bg_nr}.png").convert()
    tile_width, tile_height = bg_tile.get_size()
    draw_background()
    update_config()

def refresh_leaderboard():
    global getting_scores
    getting_scores = True

def show_leaderboard():
    global state, getting_scores, is_online, width, height
    state = "LEADERBOARD"
    leaderboard = {}
    
    # Scores von Server laden und verschlüsselt speichern
    while getting_scores:
        draw_text_centered(f"Loading...", HEIGHT // 2)
        pygame.display.flip()
        is_online = sql_db.get_scores()
        if is_online:
            print("Connected to Server")
            # lokal gespeicherte Scores abgleichen und hochladen
            if os.path.isfile("not_uploaded.txt.enc"):
                # Logik fürs abgleichen
                print("Uploading local scores...")
                decrypt_file("not_uploaded.txt.enc", "not_uploaded.txt")
                sql_db.update_scores()

            encrypt_file("Scores.txt")
        getting_scores = False

    # Ansonsten lokal gespeicherte Scores verwenden
    if not is_online:
        draw_text_centered("OFFLINE",WIDTH // 2 + 450, HEIGHT // 2 - 300, colour=(255,0,0))
    

    decrypt_file("Scores.txt.enc", "Scores.txt")

    with open("Scores.txt", "r") as datei:
        #Dictionary aus Datei erstellen
        for line in datei:
            line = line.strip()
            name = ''
            score = ''
            
            if ':' in line:
                parts = line.split(':')
                name = parts[0].strip()
                score = ''.join(filter(str.isdigit, parts[1]))
            
            if name and score.isdigit():
                score = int(score)
                if name not in leaderboard or score > leaderboard[name]:
                    leaderboard[name] = score    
    
    encrypt_file("Scores.txt")

    # Sortieren
    sorted_leaderboard = sorted(leaderboard.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    
    # Ausgeben
    line_height = 1
    for i in range (5):
        if i < len(sorted_leaderboard):
            draw_text_centered(f"{i+1}. {sorted_leaderboard[i][0]}: {sorted_leaderboard[i][1]}", 100*line_height, None, "game_design\\Border.png", WHITE)
        line_height += 1

def toggle_fullscreen():
    global fullscreen
    pygame.display.toggle_fullscreen()
    fullscreen = not fullscreen
    update_config()

def quit_game():
    pygame.quit()
    sys.exit()



#Platzierung der Buttons korrigieren
def get_menu_buttons(width, height):
    return [
        Button("Start", width // 2 - 100, height // 2 , 200, 50, start_game),
        Button("Options", width // 2 - 100, height // 2 + 100, 200, 50, go_to_options),
        Button("Leaderboard", width // 2 - 100, height // 2 + 200, 200, 50, show_leaderboard),
        Button("QUIT", width // 2 - 100, height // 2 + 300, 200, 50, quit_game)
    ]

def get_leaderboard_UI(width, height):
    return [
        Button("Back",width // 2 - 550, height // 2 + 300, 150, 80, go_back),
        Button("Refresh", width // 2 + 450, height // 2 - 300, 200, 80, refresh_leaderboard),
    ]

def get_options_UI(width, height):
    return [
        Button("<", width // 2 -250, height // 2 -125, 80, 50, lambda: change_music_track(-1)),
        Button(">", width // 2 +170, height // 2 -125, 80, 50, lambda: change_music_track(+1)),
        Button("<", width // 2 -250, height // 2 -25, 80, 50, lambda: change_volume("music_volume", -0.05)),
        Button(">", width // 2 +170, height // 2 -25, 80, 50, lambda: change_volume("music_volume", +0.05)),
        Button("<", width // 2 -250, height // 2 +75, 80, 50, lambda: change_volume("sfx_volume", -0.05)),
        Button(">", width // 2 +170, height // 2 +75, 80, 50, lambda: change_volume("sfx_volume", +0.05)),

        Button("<", width // 2 -275, height // 2 -225, 80, 50, lambda: change_background(-1)),
        Button(">", width // 2 +195, height // 2 -225, 80, 50, lambda: change_background(+1)),
    
        Button("Back", width // 2 -300, height // 2 +300, 150, 80, go_back),
        Button("Toggle Fullscreen",  width // 2 - 70, height // 2 +170, 150, 80, toggle_fullscreen),
    ]

def get_pause_buttons(width, height):
    return [
        Button("Continue", width // 2 - 100, height // 2 , 200, 80, resume_game),
        Button("Restart", width // 2 - 100, height // 2 +100, 200, 80, start_game),
        Button("Options", width // 2 - 100, height // 2 +200, 200, 80, go_to_options),
        Button("Main Menu", width // 2 - 100, height // 2 +300, 200, 80, return_to_menu),
    ]

def get_game_over_buttons(width, height):
    return [
        Button("Try Again", WIDTH // 2 - 100, HEIGHT // 2 + 60, 200, 50, start_game),
        Button("Main Menu", WIDTH // 2 - 100, HEIGHT // 2 + 160, 200, 50, return_to_menu),
        Button("Save Score", WIDTH // 2 - 100, HEIGHT // 2 + 260, 200, 50, save_score),
    ]

def get_enter_name_buttons(width, height):
    return [
        Button("Cancel", WIDTH // 2 - 100, HEIGHT // 2 + 60, 200, 50, go_back),
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
previous_shape = None
shape_bag = []
new_highscore = False
highscore = 0

# --- Lock delay variables ---
lock_delay = 300 - level*10 # milliseconds
lock_timer = None
lock_pending = False

# --- online Leaderboard ---
getting_scores = True
is_online = True

# --- Soft Drop ---
soft_drop_active = False
s_pressed_time = None  # Zeitpunkt, an dem "S" gedrückt wurde
LONG_PRESS_THRESHOLD = 0.5  # Schwelle in Sekunden für langes Drücken


def delay(milsec):
    return pygame.time.delay(milsec)

class ScorePopup:
    def __init__(self, text, x, y, big = False):
        self.text = text
        self.x = x
        self.y = y
        self.timer = 45
        self.big = big

    def draw(self):
        # Draw black outline
        if self.big:
            txt_surface = big_font.render(self.text,True, YELLOW)
        else:
            txt_surface = small_font.render(self.text, True, WHITE)
        outline_color = BLACK
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    if self.big:
                        outline_surface = big_font.render(self.text, True, outline_color)
                    else:
                        outline_surface = small_font.render(self.text, True, outline_color)
                    screen.blit(outline_surface, (self.x + dx, self.y + dy))
        # Draw main text
        screen.blit(txt_surface, (self.x, self.y))
        self.y -= 1
        self.timer -= 1

    def is_alive(self):
        return self.timer > 0


def create_piece():
    #classic tetris bag system
    global shape_bag, previous_shape 
    if shape_bag == []:
        shape_bag = list(SHAPES.keys())
        random.shuffle(shape_bag)
    shape = shape_bag.pop()
    while shape == previous_shape:
        random.shuffle(shape_bag)
        shape = shape_bag.pop()
    previous_shape = shape

    return {
        'shape': shape,
        'matrix': SHAPES[shape],
        'x': COLS // 2 - len(SHAPES[shape][0]) // 2,
        'y': 0,
        'color': SHAPE_COLORS[shape]
    }

def rotate(matrix, input):
    if input == 113:    # Key Q
        return list(zip(*matrix))[::-1]
    else:   # Key E,W,UP
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
    global board, score, lines_cleared, level, fall_speed, combo_count
    old_level = level
    new_board = []
    cleared = 0
    base_scores = [0, 100, 300, 500, 800]
    popup_x = WIDTH // 2 + GAME_WIDTH // 2 - 540
    popup_y = HEIGHT // 2 - GAME_HEIGHT // 2 + 200

    # Anzahl cleared lines herausfinden
    for row in board:
        if all(row):
            cleared += 1
        else:
            new_board.append(row)
    lines_cleared += cleared
        
    #line clear Sounds
    if cleared in (1,2,3):
        play_sound("line_clear.mp3")

    # Tetris Sounds + Animation
    elif cleared >= 4:

        play_sound("4x_line_clear.mp3")
        score_popup.append(ScorePopup("TETRIS", popup_x+255, popup_y+100, big=True))

        # Indizes der gelöschten Zeilen erfassen
        clear_rows = [i for i, row in enumerate(board) if all(row)]
        # Animation: Von innen nach außen Spalten auf 0 setzen
        for step in range(COLS // 2):
            for row in clear_rows:
                board[row][COLS // 2 - step - 1] = 0  # nach links
                board[row][COLS // 2 + step] = 0      # nach rechts

            # Board aktualisieren
            pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
            draw_board(offset_x, offset_y)
            draw_text_centered(f"Score: {int(score)}", 450, WIDTH // 2 +300 +10*len(str(int(score))), font_size=30)
            draw_text_centered(f"Level: {level}", 550, WIDTH // 2 +300, font_size=30)
            draw_text_centered(f"Lines: {lines_cleared}", 650, WIDTH // 2 +300, font_size=30)
            draw_text_centered(f"Highscore:", 600, WIDTH // 2 -300, font_size=30)
            draw_text_centered(f"{highscore}", 670, WIDTH // 2 -300, font_size=40)
            draw_next_pieces()
            draw_hold_piece()
            pygame.display.update()

            pygame.time.delay(75)  # Pause pro Schritt

    # combo mombo + level
    if cleared:
        # Bissl kompliziertere Logik für level um Sound einzubauen
        level = 1 + lines_cleared // 10  #<-- je x zeilen wird level erhöht
        if old_level != level:
            play_sound("level_up.mp3")
        fall_speed = max(16, int(1000 * (0.8 - (level - 1) * 0.007) ** (level - 1)))
        
        combo_count += 1
        score_add = int(round(base_scores[cleared] *  level,0))
        score_popup.append(ScorePopup(f"{cleared}x Line clear", popup_x, popup_y))
        score_popup.append(ScorePopup(f"+{score_add} pts", popup_x, popup_y + 30))
        if combo_count > 1:
            combo_score = 50 * combo_count * level
            score += combo_score
            score_popup.append(ScorePopup(f"{combo_count}x COMBO", popup_x, popup_y+160))
            score_popup.append(ScorePopup(f"+{combo_score} pts", popup_x, popup_y + 190))
        else:
            score += score_add

    if not cleared:
        combo_count = 0
    
    # Neues Board rendern
    for _ in range(cleared):
        new_board.insert(0, [0 for _ in range(COLS)])
    board = new_board
    
        
def move_piece(dx, dy):
    global lock_timer, lock_pending, current_piece
    if valid_position(current_piece, dx, dy):
        current_piece['x'] += dx
        current_piece['y'] += dy
        # If piece is moved during lock delay, reset timer
        if lock_pending:
            lock_timer = pygame.time.get_ticks()
        return True
    return False

def drop_piece():
    global current_piece, used_hold, lock_timer, lock_pending
    if not move_piece(0, 1):
        if not lock_pending:
            lock_timer = pygame.time.get_ticks()
            lock_pending = True

def soft_drop():
    global current_piece, used_hold, lock_timer, lock_pending, score, soft_drop_active
    drops = 0
    while valid_position(current_piece, 0, 1) and soft_drop_active:
        # Tetromino moven + Score
        move_piece(0, 1)
        drops += 1

        # Soft Drop durch beliebige andere Aktion abbrechen
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if soft_drop_active and event.key != pygame.K_ESCAPE:
                    drops = 0
                    soft_drop_active = False
        
        # Board aktualisieren
        pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        draw_board(offset_x, offset_y)
        draw_text_centered(f"Score: {int(score)}", 450, WIDTH // 2 +300 +10*len(str(int(score))), font_size=30)
        draw_text_centered(f"Level: {level}", 550, WIDTH // 2 +300, font_size=30)
        draw_text_centered(f"Lines: {lines_cleared}", 650, WIDTH // 2 +300, font_size=30)
        draw_text_centered(f"Highscore:", 600, WIDTH // 2 -300, font_size=30)
        draw_text_centered(f"{highscore}", 670, WIDTH // 2 -300, font_size=40)
        draw_piece(get_ghost_piece(current_piece), offset_x, offset_y, ghost=True)
        draw_piece(current_piece, offset_x, offset_y)
        draw_next_pieces()
        draw_hold_piece()
        pygame.display.update()
        pygame.time.delay(50)  # Pause pro Schritt
    
    soft_drop_active = False
    popup_x = WIDTH // 2 + GAME_WIDTH // 2 - 540
    popup_y = HEIGHT // 2 - GAME_HEIGHT // 2 + 280
    if drops:
        lock_timer = pygame.time.get_ticks()
        lock_pending = True
        score += drops
        score_popup.append(ScorePopup(f"Soft Drop", popup_x, popup_y))
        score_popup.append(ScorePopup(f"+{drops} pts", popup_x, popup_y +30))

def hard_drop():
    global score, lock_pending, used_hold, current_piece
    drops = 0
    while move_piece(0, 1):
        drops += 1
    score += drops * 2
    popup_x = WIDTH // 2 + GAME_WIDTH // 2 - 540
    popup_y = HEIGHT // 2 - GAME_HEIGHT // 2 + 280
    if drops:
        score_popup.append(ScorePopup(f"Hard Drop", popup_x, popup_y))
        score_popup.append(ScorePopup(f"+{drops * 2} pts", popup_x, popup_y +30))
    # Immediately lock and merge the piece, skip lock delay
    merge_piece(current_piece)
    clear_lines()
    current_piece = next_queue.pop(0)
    dyn_icon(current_piece)
    next_queue.append(create_piece())
    used_hold = False
    lock_pending = False
    if not valid_position(current_piece):
        pygame.time.delay(500)
        game_over()

def dyn_icon(current_piece):
    # Dynamisches Game Icon
    shape = current_piece['shape']
    if shape == "I":
        dyn_icon = "I"
    elif shape == "O":
        dyn_icon = "y"
    elif shape == "T":
        dyn_icon = "p"
    elif shape == "S":
        dyn_icon = "g"
    elif shape == "Z":
        dyn_icon = "r"
    elif shape == "J":
        dyn_icon = "b"
    elif shape == "L":
        dyn_icon = "o"
    try:
        icon = pygame.image.load(f"game_design\\icon-{dyn_icon}.png").convert_alpha()
        pygame.display.set_icon(icon)
    except:
        None


def reset_game():
    global dyn_icon, icon, board, current_piece, next_queue, fall_time, fall_speed, score, level, lines_cleared, score_popup, hold_piece, used_hold
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    next_queue = [create_piece() for _ in range(3)]
    current_piece = next_queue.pop(0)
    
    dyn_icon(current_piece)

    next_queue.append(create_piece())
    fall_time = pygame.time.get_ticks()  # Initialize to current time
    fall_speed = 1000
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
        dyn_icon(current_piece)
        next_queue.append(create_piece())
    else:
        hold_piece, current_piece = current_piece, hold_piece
    current_piece['x'] = COLS // 2 - len(current_piece['matrix'][0]) // 2
    current_piece['y'] = 0



#############
# RENDERING #
#############

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




# Hold Piece 
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

def draw_board(offset_x, offset_y):
    for y in range(ROWS):
        for x in range(COLS):
            rect = pygame.Rect(offset_x + x * BLOCK_SIZE, offset_y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, DARKGREY, rect, 1)
            color = board[y][x]
            if color:
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)

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
    NEXT_PIECE_X = WIDTH // 2 + 230
    NEXT_PIECE_Y = 100

    pygame.draw.rect(screen, BLACK, (NEXT_PIECE_X, NEXT_PIECE_Y, box_width, box_height))
    pygame.draw.rect(screen, GREY, (NEXT_PIECE_X, NEXT_PIECE_Y, box_width, box_height), 4)

    draw_text_centered("Next:", NEXT_PIECE_Y-45, NEXT_PIECE_X+75)
    
    for i, piece in enumerate(next_queue[:3]):
        matrix = piece['matrix']
        piece_width = len(matrix[0]) * BLOCK_SIZE * 0.75
        piece_height = len(matrix) * BLOCK_SIZE * 0.75
        x_offset = NEXT_PIECE_X + (box_width - piece_width) // 2 - 0
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

def draw_text_centered(text, y, x=None, bg_img="game_design\\Border_2.png", colour=WHITE, font_size = 40, surface=None):
    if surface is None:
        surface = screen
    fnt = pygame.font.Font("game_design\\Pixel_Emulator.otf", font_size)
    txt_surface = fnt.render(text, True, colour)
    if x is None:
        txt_rect = txt_surface.get_rect(center=(WIDTH // 2, y))
    else:
        txt_rect = txt_surface.get_rect(center=(x, y))
    padding = 20
    box_rect = txt_rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, BLACK, box_rect)
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
    surface.blit(top_left, (box_rect.left, box_rect.top))
    surface.blit(top_right, (box_rect.right - corner, box_rect.top))
    surface.blit(bottom_left, (box_rect.left, box_rect.bottom - corner))
    surface.blit(bottom_right, (box_rect.right - corner, box_rect.bottom - corner))
    surface.blit(pygame.transform.scale(top, (box_rect.width - 2 * corner, corner)), 
                (box_rect.left + corner, box_rect.top))
    surface.blit(pygame.transform.scale(bottom, (box_rect.width - 2 * corner, corner)), 
                (box_rect.left + corner, box_rect.bottom - corner))
    surface.blit(pygame.transform.scale(left, (corner, box_rect.height - 2 * corner)), 
                (box_rect.left, box_rect.top + corner))
    surface.blit(pygame.transform.scale(right, (corner, box_rect.height - 2 * corner)), 
                (box_rect.right - corner, box_rect.top + corner))
    surface.blit(txt_surface, txt_rect)


def get_bg_tile():
    bg_tile = pygame.image.load(f"game_design\\bg{bg_nr}.png").convert()
    tile_width, tile_height = bg_tile.get_size()
    return bg_tile, tile_width, tile_height

def draw_background():
    bg_tile, tile_width, tile_height = get_bg_tile()
    for x in range(0, WIDTH, tile_width):
        for y in range(0, HEIGHT, tile_height):
            screen.blit(bg_tile, (x, y))


bg_tile = pygame.image.load(f"game_design\\bg{bg_nr}.png").convert()
tile_width, tile_height = bg_tile.get_size()

def update_GUI():
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = screen.get_size()
    screen.fill(BLACK)
    draw_background()


#############
# MAIN LOOP #
#############
reset_game()
running = True

def is_piece_fully_in_air(piece):
    for y, row in enumerate(piece['matrix']):
        for x, cell in enumerate(row):
            if cell:
                board_x = piece['x'] + x
                board_y = piece['y'] + y
                # If at the bottom row
                if board_y + 1 >= ROWS:
                    return False
                # If block below is occupied
                if board[board_y + 1][board_x]:
                    return False
    return True

# --- Game Over Blink Variables ---
game_over_blink_timer = 0
game_over_blink_visible = True

# --- Menu Fade Variables ---
menu_fade_alpha = 0
menu_fade_start_time = None
menu_fade_duration = 1000  # milliseconds (1 second)

skip_intro = 0

MENU_LOGO = " TETRIS "
MENU_LOGO_FONT_SIZE = 80
MENU_TRANSITION_BASE_SPEED = 100  # ms per char
MENU_TRANSITION_VARIANCE = 40    # ms random extra per char
menu_transition_active = False
menu_transition_start_time = 0
menu_transition_chars = []
menu_transition_done = False

# --- In-game UI Animation Variables ---
ingame_ui_anim_active = False
ingame_ui_anim_start_time = 0
ingame_ui_anim_duration = 1000  # ms for full text

def start_menu_transition(buttons):
    global menu_transition_active, menu_transition_start_time, menu_transition_chars, menu_transition_done
    menu_transition_active = True
    menu_transition_start_time = pygame.time.get_ticks()
    menu_transition_done = False
    menu_transition_chars.clear()
    # Logo
    logo_times = []
    t = 0
    for c in MENU_LOGO:
        dt = MENU_TRANSITION_BASE_SPEED + random.randint(0, MENU_TRANSITION_VARIANCE)
        t += dt
        logo_times.append((c, t))
    menu_transition_chars.append(('logo', logo_times))
    # Buttons
    for btn in buttons:
        btn_times = []
        t = 0
        for c in btn.text:
            dt = MENU_TRANSITION_BASE_SPEED + random.randint(0, MENU_TRANSITION_VARIANCE)
            t += dt
            btn_times.append((c, t))
        menu_transition_chars.append((btn, btn_times))



while running:
    update_GUI()
    clock.tick(FPS)
    now = pygame.time.get_ticks()

    # BOOTUP SEQUENCE STATE
    if state == "BOOTUP_SEQUENCE":
        elapsed = now - bootup_start_time
        # Fade in for first fade_duration ms, fade out for last fade_duration ms
        if elapsed < fade_duration:
            alpha = int(255 * (elapsed / fade_duration))
        elif elapsed > bootup_duration - fade_duration:
            alpha = int(255 * ((bootup_duration - elapsed) / fade_duration))
        else:
            alpha = 255

        # Draw black background
        screen.fill((0, 0, 0))
        # Render text surface
        bootup_font = pygame.font.Font(r"game_design\Pixel_Emulator.otf", 24)
        text_surface = bootup_font.render("content intended for mature audiences only", True, (255, 255, 255))
        text_surface.set_alpha(alpha)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(text_surface, text_rect)
        text_surface2 = bootup_font.render("Pegi 18", True, (255,50,50))
        text_surface2.set_alpha(alpha)
        text_rect2 = text_surface2.get_rect(center=(WIDTH // 2, HEIGHT //2 -50))
        screen.blit(text_surface2,text_rect2)
        
        pygame.display.flip()
        # After bootup_duration ms, switch to MENU
        if elapsed >= bootup_duration or skip_intro == 1:
            state = "MENU"
            previous_state = "MENU"
            menu_fade_alpha = 0
            menu_fade_start_time = pygame.time.get_ticks()

    # Handle GAME_OVER and PAUSE blinking
    if state in ("GAME_OVER", "PAUSE"):
        game_over_blink_timer += clock.get_time()
        if game_over_blink_timer > 500:  # Blink every 500ms
            game_over_blink_visible = not game_over_blink_visible
            game_over_blink_timer = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()

        if state == "BOOTUP_SEQUENCE":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or pygame.K_SPACE:

                        skip_intro = 1
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    skip_intro = 1

        if state == "MENU":
            getting_scores = True # Für Online-Leaderboard
            # Always get fresh button positions for current window size
            menu_buttons = get_menu_buttons(WIDTH, HEIGHT)
            if menu_transition_done:
                for btn in menu_buttons:
                    btn.handle_event(event)
            previous_state = "MENU"
            if event.type == pygame.KEYDOWN and menu_transition_done:
                if event.key == pygame.K_ESCAPE:
                    go_back()
        
        elif state == "OPTIONS":
            for btn in get_options_UI(WIDTH, HEIGHT):
                btn.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    go_back()
                            
        
        elif state == "LEADERBOARD":
            for btn in get_leaderboard_UI(WIDTH, HEIGHT):
                btn.handle_event(event)
                previous_state = "MENU"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    go_back()
            
        elif state == "GAME":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state = "PAUSE"
                    previous_state = "PAUSE"
                if event.key in (pygame.K_a, pygame.K_LEFT):
                    if move_piece(-1, 0) and lock_pending:
                        lock_timer = pygame.time.get_ticks()
                        play_sound("move.mp3")
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    if move_piece(1, 0) and lock_pending:
                        lock_timer = pygame.time.get_ticks()
                        play_sound("move.mp3")
                if event.key in (pygame.K_s, pygame.K_DOWN):
                    # Zeitpunkt des Drückens für Softdrop speichern
                    s_pressed_time = time.time()
                
                #Bissreres Rotate
                if event.key in (pygame.K_q, pygame.K_e, pygame.K_w, pygame.K_UP):
                    play_sound("rotate.mp3")
                    rotated = rotate(current_piece['matrix'],event.key)
                    wall_kick(current_piece, rotated)
                elif event.key == pygame.K_SPACE:
                    play_sound("drop.mp3")
                    hard_drop()
                    
                elif event.key == pygame.K_f:
                    hold_current_piece()
                    dyn_icon(current_piece)

            # Richtiger Soft Drop
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_s and s_pressed_time is not None:
                    duration = time.time() - s_pressed_time
                    if duration >= LONG_PRESS_THRESHOLD:
                        # Soft Drop durch gedrückt halten (0.5s) aktivieren
                        soft_drop_active = True
                    else:
                        drop_piece()
                    s_pressed_time = None
            
            # Soft Drop ausführen, wenn aktiv
            if soft_drop_active:
                soft_drop()
        
        elif state == "PAUSE":
            for btn in get_pause_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state = "GAME"

        elif state == "GAME_OVER":
            for btn in get_game_over_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state = "MENU"                
        
        elif state == "ENTER_NAME":
            for btn in get_enter_name_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)
            previous_state = "GAME_OVER"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    
                    # Lokal speichern für Leaderboard
                    decrypt_file("Scores.txt.enc", "Scores.txt")
                    with open("Scores.txt","a") as datei:
                        datei.write(f"{player_name.upper()}: {score}\n")

                    # Lokal speichern für Backup
                    if os.path.isfile("not_uploaded.txt.enc"):
                        decrypt_file("not_uploaded.txt.enc", "not_uploaded.txt")
                        with open("not_uploaded.txt","a") as datei:
                            datei.write(f"{player_name.upper()}: {score}\n")
                    else:
                        with open("not_uploaded.txt","w") as datei:
                            datei.write(f"{player_name.upper()}: {score}\n")
                    
                    #Versuchen Score auf Server hoch zu laden
                    if sql_db.get_scores():
                        sql_db.update_scores()
                    # Wenn nicht hochgeladen, Backup verschlüsseln um Cheating zu verhindern
                    if os.path.isfile("not_uploaded.txt"):
                        encrypt_file("not_uploaded.txt")

                    encrypt_file("Scores.txt")
                    player_name = ''
                    state = "GAME_OVER"
                elif event.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]  # Letztes Zeichen löschen
                elif event.key == pygame.K_ESCAPE: # Wieder raus ohne speichern
                    go_back()
                else:
                    player_name += event.unicode  # Zeichen hinzufügen

        elif state == "GAME_OVER":
            for btn in get_game_over_buttons(WIDTH, HEIGHT):
                btn.handle_event(event)

    if state == "GAME" and now - fall_time > fall_speed:
        drop_piece()
        fall_time = now
    
    if state == "GAME":
        if lock_pending:
            if pygame.time.get_ticks() - lock_timer >= lock_delay:
                # Only merge if the piece cannot move down
                if not valid_position(current_piece, dy=1):
                    merge_piece(current_piece)
                    clear_lines()
                    current_piece = next_queue.pop(0)
                    dyn_icon(current_piece)
                    next_queue.append(create_piece())
                    used_hold = False
                    lock_pending = False
                    if not valid_position(current_piece):
                        pygame.time.delay(500)
                        game_over()
                else:
                    # Reset lock timer if still able to fall
                    lock_pending = False

    offset_x = WIDTH // 2 - GAME_WIDTH // 2
    offset_y = HEIGHT // 2 - GAME_HEIGHT // 2

    if state == "MENU":
        # Fade in effect for menu background only
        if menu_fade_alpha < 255:
            if menu_fade_start_time is None:
                menu_fade_start_time = pygame.time.get_ticks()
            elapsed_fade = now - menu_fade_start_time
            menu_fade_alpha = min(255, int(255 * (elapsed_fade / menu_fade_duration)))
        else:
            menu_fade_alpha = 255
        # Draw black background
        screen.fill(BLACK)
        # Draw faded-in background image
        bg_tile, tile_width, tile_height = get_bg_tile()
        bg_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for x in range(0, WIDTH, tile_width):
            for y in range(0, HEIGHT, tile_height):
                bg_surface.blit(bg_tile, (x, y))
        bg_surface.set_alpha(menu_fade_alpha)
        screen.blit(bg_surface, (0, 0))
        # Always get fresh button positions for current window size
        menu_buttons = get_menu_buttons(WIDTH, HEIGHT)
        # Start transition if not already started
        if not menu_transition_active:
            start_menu_transition(menu_buttons)
        # Draw animated logo and buttons
        elapsed_anim = pygame.time.get_ticks() - menu_transition_start_time
        logo_chars = [c for c, t in menu_transition_chars[0][1] if elapsed_anim >= t]
        logo_str = "".join(logo_chars)
        logo_palette = [
            (0, 174, 239),   # cyan
            (255, 213, 0),   # yellow
            (255, 121, 0),   # orange
            (237, 28, 36),   # red
            (0, 166, 81),    # green
            (128,0,128),     # purple
        ]
        palette_offset = (pygame.time.get_ticks() // 100) % len(logo_palette)
        if logo_str:
            fnt = pygame.font.Font("game_design\Pixel_Emulator.otf", MENU_LOGO_FONT_SIZE)
            char_surfaces = [fnt.render(c, True, logo_palette[(i + palette_offset) % len(logo_palette)]) for i, c in enumerate(logo_str)]
            total_width = sum(s.get_width() for s in char_surfaces)
            x = WIDTH // 2 - total_width // 2
            y = 200
            txt_height = char_surfaces[0].get_height() if char_surfaces else 0
            padding = 20
            box_rect = pygame.Rect(x, y - txt_height//2, total_width, txt_height).inflate(padding*2, padding*2)
            pygame.draw.rect(screen, BLACK, box_rect)
            border = pygame.image.load("game_design\\Border.png").convert_alpha()
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
            # Draw each letter
            cx = x
            for surf in char_surfaces:
                screen.blit(surf, (cx, y - surf.get_height()//2))
                cx += surf.get_width()
        # Draw buttons only during animation, not after (no border for animated text)
        all_done = True
        if not menu_transition_done:
            # Recalculate button positions every frame for animation
            animated_buttons = get_menu_buttons(WIDTH, HEIGHT)
            for i, (btn, btn_times) in enumerate(menu_transition_chars[1:]):
                chars = [c for c, t in btn_times if elapsed_anim >= t]
                if len(chars) < len(btn.text):
                    all_done = False
                if chars:
                    # Use the current position from the recalculated button
                    anim_btn = animated_buttons[i]
                    draw_text_centered(
                        "".join(chars),
                        anim_btn.rect.centery,
                        anim_btn.rect.centerx,
                        "game_design\\Border_2.png"
                    )
        menu_transition_done = all_done and menu_fade_alpha == 255

        # Only allow button interaction and draw full buttons after animation is done
        if menu_transition_done:
            for btn in menu_buttons:
                btn.draw(screen)
    elif state == "OPTIONS":
        draw_text_centered(f"TRACK: {selected_track}", HEIGHT // 2 -100)
        draw_text_centered(f"MUSIK: {round(music_volume*100)}%", HEIGHT // 2)
        draw_text_centered(f"SFX: {round(sfx_volume*100)}%", HEIGHT // 2 +100)
        draw_text_centered(f"Background:{bg_nr}", HEIGHT // 2-200 )
        for btn in get_options_UI(WIDTH, HEIGHT):
            btn.draw(screen)

    elif state == "LEADERBOARD":
        for btn in get_leaderboard_UI(WIDTH, HEIGHT):
            btn.draw(screen)
        show_leaderboard()

    elif state == "GAME":
        pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        draw_board(offset_x, offset_y)
        draw_piece(get_ghost_piece(current_piece), offset_x, offset_y, ghost=True)
        draw_piece(current_piece, offset_x, offset_y)

        # aktuellen Highscore herausfinden
        decrypt_file("Scores.txt.enc", "Scores.txt")
        with open("Scores.txt", "r") as datei:
            for line in datei:
                line = line.strip()
                name = ''
                data_score = ''
                if ':' in line:
                    parts = line.split(':')
                    name = parts[0].strip()
                    data_score = int(''.join(filter(str.isdigit, parts[1])))
                    if data_score > highscore:
                        highscore = data_score
        encrypt_file("Scores.txt")

        # --- Animated UI Text ---
        ui_texts = [
            (f"Score: {int(score)}", 450, WIDTH // 2 +300 +10*len(str(int(score))), 30),
            (f"Level: {level}", 550, WIDTH // 2 +300, 30),
            (f"Lines: {lines_cleared}", 650, WIDTH // 2 +300, 30),
            (f"Highscore:", 600, WIDTH // 2 -300, 30),
            (f"{highscore}", 670, WIDTH // 2 -300, 40),
        ]
        palette = [
            (0, 174, 239),   # cyan
            (255, 213, 0),   # yellow
            (255, 121, 0),   # orange
            (237, 28, 36),   # red
            (0, 166, 81),    # green
            (128,0,128),     # purple
        ]
        palette_offset = (pygame.time.get_ticks() // 100) % len(palette)
        if ingame_ui_anim_active:
            elapsed_anim = pygame.time.get_ticks() - ingame_ui_anim_start_time
            for idx, (text, y, x, font_size) in enumerate(ui_texts):
                chars_to_show = int(len(text) * min(1, elapsed_anim / ingame_ui_anim_duration))
                shown = text[:chars_to_show]
                if shown:
                    fnt = pygame.font.Font("game_design\\Pixel_Emulator.otf", font_size)
                    char_surfaces = [fnt.render(c, True, WHITE) for c in shown]
                    total_width = sum(s.get_width() for s in char_surfaces)
                    tx = x - total_width // 2
                    ty = y
                    txt_height = char_surfaces[0].get_height() if char_surfaces else 0
                    padding = 20
                    box_rect = pygame.Rect(tx, ty - txt_height//2, total_width, txt_height).inflate(padding*2, padding*2)
                    pygame.draw.rect(screen, BLACK, box_rect)
                    border = pygame.image.load("game_design\\Border_2.png").convert_alpha()
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
                    cx = tx
                    for surf in char_surfaces:
                        screen.blit(surf, (cx, ty - surf.get_height()//2))
                        cx += surf.get_width()
            if elapsed_anim >= ingame_ui_anim_duration:
                ingame_ui_anim_active = False
                
        else:
            draw_text_centered(f"Score: {int(score)}", 450, WIDTH // 2 +300 +10*len(str(int(score))), font_size=30)
            draw_text_centered(f"Level: {level}", 550, WIDTH // 2 +300, font_size=30)
            draw_text_centered(f"Lines: {lines_cleared}", 650, WIDTH // 2 +300, font_size=30)
            draw_text_centered(f"Highscore:", 600, WIDTH // 2 -300, font_size=30)
            draw_text_centered(f"{highscore}", 670, WIDTH // 2 -300, font_size=40)
            

        draw_next_pieces()
        draw_hold_piece()
        for popup in score_popup:
            if popup.is_alive():
                popup.draw()
        score_popup[:] = [p for p in score_popup if p.is_alive()]
    
    elif state == "PAUSE":
        if game_over_blink_visible:
            draw_text_centered(" PAUSED ", 200, None, "game_design\\Border.png", (30, 30, 150))
        else:
            draw_text_centered(" PAUSED ", 200, None, "game_design\\Border.png", (0, 0, 0))
        for btn in get_pause_buttons(WIDTH, HEIGHT):
            btn.draw(screen)
    
    elif state == "ENTER_NAME":
        draw_text_centered(f"ENTER NAME: {player_name}", 200, None, "game_design\\Border.png", (30, 30, 150))
        for btn in get_enter_name_buttons(WIDTH, HEIGHT):
            btn.draw(screen)

    elif state == "GAME_OVER":
        # Always draw the border/background at the correct size
        if game_over_blink_visible:
            draw_text_centered(" GAME OVER ", 200, None, "game_design\\Border.png", (255,0,0))
        else:
            draw_text_centered(" GAME OVER ", 200, None, "game_design\\Border.png", (0,0,0))
        draw_text_centered(f"Final Score: {score}", 300, None, "game_design\\Border.png")
        for btn in get_game_over_buttons(WIDTH, HEIGHT):
            btn.draw(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()