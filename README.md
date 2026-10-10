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

Install the dependencies (Python 3.10 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flatland-rl torch numpy
```

Before you start, open `config.py` and point `checkpoint_dir` to a local folder. The default is a Google Drive path for Colab:

```python
checkpoint_dir = "./checkpoints"
```

Then start training:

```bash
python Training.py
```

Progress prints to the terminal. Checkpoints are saved every 50 episodes. If you stop and run the script again,
it picks up from the latest checkpoint automatically.

If your machine slows to a crawl, lower `N_WORKERS` at the top of `Training.py`. It's set to 11, which suits a 12-core machine.

## Running on Colab

Run `python build_colab.py` to create `Flatland_Colab_Ready.ipynb`. Upload it to Colab and pick a GPU runtime.
Mount your Google Drive so checkpoints are kept, then run all the cells.
