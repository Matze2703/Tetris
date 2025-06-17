# Tetris 

A feature-rich Tetris clone built with Python and Pygame.

## Features

- Retro pixel-style UI with animated transitions
- Multiple music tracks and sound effects
- Hold system and multi-piece preview queue
- Online and local leaderboards (with encryption)
- Configurable volume for music and sound effects
- Ghost piece, hard/soft drop, and wall-kick mechanics
- Score saving with encryption using `cryptography.Fernet`
- Fullscreen and windowed display modes
- Game settings saved in `config.txt`

## Requirements

The required Python modules are automatically installed if missing:

- `pygame`
- `cryptography`
- `requests`
- `os` 
- `random`
- `time`
- `sys`
- `subprocess`
- `importlib`.


## Controls

| Key        | Action                       |
|------------|------------------------------|
| A / Left   | Move left                    |
| D / Right  | Move right                   |
| S / Down   | Soft drop (hold for auto)    |
| Space      | Hard drop                    |
| Q / E / W / Up | Rotate piece             |
| F          | Hold piece                   |
| ESC        | Pause or go back             |
| F11        | Toggle fullscreen            |
| Enter      | Confirm name on score screen |
| Backspace  | Delete character in name     |

## Score System

- Scores are encrypted using a generated key.
- If online, scores are uploaded to the server.
- If offline, scores are saved locally and uploaded when possible.

## Notes

- Have fun!

## File Structure
.
├── Tetris.py # Main game script
├── database_access.py # Handles online leaderboard interaction
├── sound_design/ # Music and sound files
├── game_design/ # Icons, borders, backgrounds, font
├── config.txt # Generated settings file
├── schluessel.key # Encryption key for scores
├── Scores.txt.enc # Encrypted local scores
└── not_uploaded.txt.enc # Local backup of unsynced scores

## Disclaimer

This project is intended for educational and entertainment use only.

