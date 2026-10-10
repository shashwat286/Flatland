# Flatland MARL

This project trains a group of trains to share a railway network without crashing into or blocking each other.
It uses the [Flatland](https://flatland.aicrowd.com/) environment and trains the agents with PPO. A simple
rule-based "expert" stops the agents from picking moves that obviously don't make sense, which makes
training much faster.

The main pieces:

- `Training.py` is the script you run. It starts several environments in parallel and trains one shared network.
- `config.py` holds the environment settings (map size, number of trains, malfunctions, and so on).
- `NewAgentInitObs.py` and `shortestpath.py` build what each train "sees" at every step.
- `agents/models.py` defines the neural network.

## How to run

Install the dependencies (Python 3.9 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flatland-rl==4.0.3 torch numpy
```

The code is written against the Flatland 4.x API, so other versions may break it.

Then start training:

```bash
python Training.py
```

Progress prints to the terminal. Checkpoints are saved to the `checkpoints/` folder every 50 episodes. If you stop and
run the script again, it picks up from the latest checkpoint automatically.

The number of parallel environments is based on your CPU core count, up to 11. If your machine still slows down,
lower `N_WORKERS` at the top of `Training.py`.

Flatland only gives a reward at the end of an episode, so each train also gets a small bonus for every cell it moves
closer to its target (and a penalty for moving away). You can change this with `SHAPING_COEF` in `Training.py`,
or set it to `0.0` to turn it off. The `EpReward` in the logs is always Flatland's own reward, so runs stay comparable.

## Running on Colab

Run `python build_colab.py` to create `Flatland_Colab_Ready.ipynb`. Upload it to Colab and pick a GPU runtime.
Mount your Google Drive so checkpoints are kept, then run all the cells.
