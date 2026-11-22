# RPS-1 Rock-Paper-Scissors

A webcam-based Rock-Paper-Scissors project using computer vision to recognize hand gestures and determine the winner.

## MediaPipe version

The primary two-player game is `main2.py`. It uses MediaPipe Hand Landmarker to recognize two hands, orders them from left to right, classifies each as rock, paper, scissors, or unknown, and displays and prints the round result. Show one clear gesture per player. Press **q** or close the video window to exit.

The `hand_landmarker.task` model file must be in the project folder beside `main2.py`.

### Setup and run (PowerShell)

Create and activate a virtual environment if you do not already have one, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\main2.py
```

If PowerShell activation is unavailable, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main2.py
```

The computer must have a working webcam. The gesture classifier uses finger landmark positions, so clear hand visibility and gestures help recognition.

## Project files

| File | Purpose |
| --- | --- |
| `main2.py` | MediaPipe two-player RPS game and webcam display |
| `game_logic.py` | Determines whether the left player, right player, or neither wins |
| `hand_landmarker.task` | MediaPipe hand landmark model used by `main2.py` |
| `main.py` | Alternative YOLO-based webcam application |
| `cv_processor.py` | YOLO hand detection used by `main.py` |
| `camera.py` | Webcam capture helper |
| `media_pipe_hands.py` | Standalone MediaPipe hand landmark demo |
| `requirements.txt` | Python package dependencies |
