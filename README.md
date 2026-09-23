# Trackmania AI: can it beat 34.094?

A weekend experiment on one Trackmania 2020 map. My best official time after 15 minutes of timed practice was **34.094 seconds** on keyboard. The AI gets more training time, but starts without a driving demonstration or pretrained policy.

The live game connection, route reward, replay buffer and PyTorch Soft Actor-Critic (SAC) learner are implemented. Short checks put both an exploratory driver and the SAC actor on the track. **The actor has not yet learned a useful lap or finished the map.** The learner was built with substantial agentic coding help at my request.

## Repository contents

- `game_bridge.py`, `episode.py`, `explore.py`: fresh Openplanet telemetry, TMRL's screenshot-derived 19-ray road sensor, virtual gamepad control, reset and bounded driving attempts.
- `route_progress.py`: reward for new distance along a separate post-benchmark position trace. It contains no human steering or throttle labels.
- `replay.py`, `sac.py`, `train.py`: replay storage, two-critic SAC learner, episode-boundary updates, checkpoints and resume.
- `tests/`: offline checks for route progress, replay wraparound, SAC update and checkpoint reload.
- `data/route-20260923T131307Z/` and `data/human-benchmark.md`: the recorded route and frozen human result.

The `.venv`, training runs, installers, map file and best-run ghost remain local and are ignored by Git. The map and ghost can be shared separately if needed. The installed map must have SHA-256 `1477f55a124739f26b9d7a9b16633e6ffd3eb93deebc02fa7c7ab0e2c8e1e6c3`; the code checks this before driving. The route is about 1,902 position units long. Its recorder's 61.453-second elapsed time is **not** an official lap time.

## Setup on Windows

This project was checked with Python 3.11, Trackmania 2020, Openplanet 1.29.14, TMRL 0.7.1, ViGEmBus, and an RTX 4060 Ti 16 GB. Install Openplanet's TMRL GrabData plugin and the virtual gamepad driver following [TMRL's installation guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md). Put the benchmark map at `~/Documents/Trackmania/Maps/My Maps/Aimap.Map.Gbx`, or set `TRACKMANIA_MAP_PATH` to its path. Keep Trackmania foreground in driving mode with the front camera/car-hidden view and consistent graphics and window size.

The pinned Python environment can be recreated with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
.\.venv\Scripts\python.exe probe.py gpu
.\.venv\Scripts\python.exe probe.py telemetry --seconds 5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The CUDA PyTorch wheel is pinned in `requirements-lock.txt`. The probe scripts also check the virtual gamepad and road sensor; they write local results under ignored `runs/`.

## Drive and train

Reset to Aimap's start and keep the game foreground. The scripts release inputs on exit and use the verified full-restart key between attempts.

```powershell
.\.venv\Scripts\python.exe explore.py --episodes 1 --seconds 15
.\.venv\Scripts\python.exe train.py --episodes 10 --seconds 60
```

The exploratory driver uses full throttle and slowly changing random steering. It is a data-collection heuristic, not a learned policy. The trainer begins using its randomly initialised SAC actor after a warmup buffer is ready. Each run writes configuration, episode summaries, learner diagnostics, a checkpoint and replay data to `runs/train-<UTC>/`. To add five attempts to an existing run:

```powershell
.\.venv\Scripts\python.exe train.py --resume-run .\runs\train-YYYYMMDDTHHMMSSZ --episodes 5
```

The 80-value observation contains scaled speed, route progress, four recent 19-ray readings, and the previous two actions. The actor produces continuous steering and a signed drive value. The game adapter maps drive to full gas/coast/full brake because half-throttle did not register in a control probe. The reward is 10 points spread along new route progress plus 10 for the game's valid finish signal. An off-route, stuck or time-limited attempt ends and is reset. The current plugin does not export an official finish time to Python; that must be solved before a scored comparison.

## What has actually been checked

A 7-second exploratory attempt reached 194.445 route units (10.22%). A two-attempt trainer check collected 280 transitions and made 70 SAC updates. The actor controlled the second attempt and reached 22.35 units (1.18%). A resumed third attempt increased the buffer to 420 transitions and 105 updates; the actor reached 8.029 units (0.42%). These are connection and training-loop checks, **not** evidence that performance is improving. All seven offline tests pass, the saved model and buffer reload, and the car was reset to the start with inputs released.

Before a long unattended run, the road sensor needs inspection throughout the track, the provisional route corridor/jump thresholds need observation under varied driving, and the learner needs a longer supervised check. The benchmark remains 34.094 seconds until an AI records a valid official finish faster than that.
