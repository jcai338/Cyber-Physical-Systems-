# RPS-1 Software System

A real-time computer vision software system designed to play the **RPS-1** game using a webcam. Developed as a collaborative 4-person group project.

## Project Overview
According to project specifications, the system features:
* **Computer Vision (CV):** Recognizes hand shapes in real time via a webcam feed.
* **Two-Stage Game Strategy:** Manages gameplay in two distinct stages—first tracking two hands, followed by tracking the hand withdrawal action[cite: 1].
* **Winner Determination:** Automatically evaluates moves and determines the round winner using a CV decision module[cite: 1].

---

## Project Structure
```text
rps-1-software-system/
│
├── src/
│   ├── __init__.py
│   ├── main.py             # Main execution loop & integration
│   ├── camera.py           # Webcam stream ingestion
│   ├── cv_processor.py     # Hand shape recognition & CV logic
│   ├── game_logic.py       # Two-stage game state machine
│   └── ui_display.py       # Visual overlays & winner determination
│
├── tests/                  # Unit tests for game logic and vision modules
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
