# Trackmania AI

Can an AI beat my **34.094-second** time on a Trackmania 2020 test map? That was my best after 15 minutes of practice. The agent uses Openplanet telemetry, screen-derived road distances, a small road image and a virtual gamepad. It learns with PyTorch Soft Actor-Critic; the recorded route supplies progress rewards, not human driving inputs.

The latest 10-second fullscreen check ran at about 47 fresh image/input updates per second, but **the AI has not finished a lap yet**.

## Run it

On Windows, install Python 3.11, Trackmania 2020, Openplanet's TMRL plugin and ViGEmBus ([TMRL setup](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)). Screen capture uses DXcam. The test map is not included; set `TRACKMANIA_MAP_PATH` to its file. The code checks its hash before driving.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

With the game foreground, fully visible on one monitor, and the car at the start:

```powershell
.\.venv\Scripts\python.exe train.py --episodes 10 --seconds 60 --warmup-steps 2000
```

Runs, replay data and checkpoints are saved locally under `runs/`. Continue a run made with the current observation and reward settings using `train.py --resume-run .\runs\train-YYYYMMDDTHHMMSSZ --episodes 10`. The map, ghost, virtual environment and training files are excluded from Git; the route trace and human benchmark are included in `data/`.
