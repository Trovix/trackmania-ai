# Trackmania AI

Can an AI beat my **34.094-second** time on a Trackmania 2020 test map? That was my best after 15 minutes of practice. A PyTorch actor steers from Openplanet telemetry; a route-following driver supplied training examples without recorded human controls. Screen images and road-distance rays are collected, but the current imitated actor starts with their weights masked. SAC is implemented but has not improved this result.

The learned steering actor, with a speed governor, has finished in **46.332 seconds** officially without detected wall scraping. A 44.114-second finish included wall scraping. Neither beats the human benchmark; these are development runs, not a final 20-attempt evaluation.

## Run it

On Windows, install Python 3.11, Trackmania 2020, Openplanet's TMRL plugin and ViGEmBus ([TMRL setup](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)). Screen capture uses DXcam. The test map is not included; set `TRACKMANIA_MAP_PATH` to its file. The code checks its hash before driving.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

With the game foreground, fully visible on one monitor, and the car at the start:

```powershell
.\.venv\Scripts\python.exe bootstrap.py --guide-attempts 1 --actor-attempts 1 --seconds 90
```

`bootstrap.py` keeps only complete guided laps without detected wall scrapes, fits the actor to their observations and actions, then tests it without route-guide actions. The actor receives route progress, heading error and offset derived from telemetry. Imitation initially masks the image features so changing stadium graphics cannot steer the car. `relabel_trace.py` can fit it again on guide corrections from a failed actor trace. SAC is experimental; to start it from the guided examples:

```powershell
.\.venv\Scripts\python.exe train.py --bootstrap-run .\runs\bootstrap-YYYYMMDDTHHMMSSZ --episodes 10 --seconds 90 --warmup-steps 2000
```

Runs, replay data and checkpoints are saved locally under `runs/`. Continue a run made with the current observation and reward settings using `train.py --resume-run .\runs\train-YYYYMMDDTHHMMSSZ --episodes 10`. The map, ghost, virtual environment and training files are excluded from Git; the route trace and human benchmark are included in `data/`.
