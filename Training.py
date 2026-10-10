"""
Training.py  —  GPU-optimised PPO for Flatland MARL
=====================================================

Architecture:
  - 11 CPU worker processes  (mp.Process, spawn)  — each owns one Flatland env
  - 1  main process          (GPU)                 — batched inference + PPO update
  - Communication via mp.Pipe (worker→main: obs/rew/done, main→worker: actions)

Why spawn (not fork):
  - Main process initialises CUDA before workers start.
  - fork() copies the CUDA context into child processes → corrupts it.
  - spawn() starts a fresh Python interpreter per worker → always safe.

Why workers build their own env (not passed from main):
  - Flatland's StateMaskingObs uses copy.deepcopy on attr-slotted objects.
  - These cannot be pickled/sent across process boundaries.
  - This was the original crash: KeyboardInterrupt trace ending in
    slots_getstate inside DistToNextJunction → deepcopy → mp pipe send.
  - Fix: each worker calls make_env(seed_offset=rank) itself.

GPU optimisations:
  - 11 envs step in parallel on CPU while main does PPO backward.
  - All agents × all workers batched into one tensor → one GPU forward pass.
  - Rollout buffer pre-allocated on GPU — no repeated CPU→GPU copies.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters

from config import Config
from NewAgentInitObs import StateMaskingObs
from agents.models import ACNet
from shortestpath import clear_trans_cache   # clear shared transition cache on env reset
from flatland.envs.step_utils.states import TrainState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WORKERS    = max(1, min(11, (os.cpu_count() or 2) - 1))   # 11 on a 12-core Colab L4; fewer on smaller machines
ROLLOUT_LEN  = 128      # steps collected per worker before PPO update
PPO_EPOCHS   = 4
PPO_CLIP     = 0.2
ENTROPY_COEF = 0.01
GAMMA        = 0.99
LR           = 3e-4
MAX_EPISODES = 11000
SAVE_EVERY   = 50
A_SIZE       = 4        # internal: 0=Left, 1=Forward, 2=Right, 3=Stop

# Reward shaping: Flatland 4.x only gives reward at episode end, so the agents
# get almost no feedback during 1500-step episodes. Each step, every agent
# additionally receives SHAPING_COEF × (cells it moved closer to its target).
# Moving away is penalised by the same amount; waiting / malfunctions give 0.
# Set to 0.0 to train on Flatland's raw reward only.
SHAPING_COEF = 0.1

_SHUTDOWN = "SHUTDOWN"
_RESET    = "RESET"


# ---------------------------------------------------------------------------
# Environment factory  (called inside worker processes only)
# ---------------------------------------------------------------------------

def make_env(rank: int = 0) -> RailEnv:
    # seed=None: the generator then uses the env's RNG, which we re-seed on every
    # reset(random_seed=...). A fixed seed here would rebuild the SAME map every episode.
    rail_gen = sparse_rail_generator(
        max_num_cities=Config.n_cities,
        max_rails_between_cities=Config.max_rails_between_cities,
        max_rail_pairs_in_city=Config.max_rail_pairs_in_city,
        seed=None,
        grid_mode=False,
    )
    malfunction_gen = ParamMalfunctionGen(MalfunctionParameters(
        malfunction_rate=Config.malfunction_rate,
        min_duration=Config.min_malfunction_duration,
        max_duration=Config.max_malfunction_duration,
    ))
    env = RailEnv(
        width=Config.width,
        height=Config.height,
        rail_generator=rail_gen,
        # SparseLineGen needs an int seed (flatland 4.x computes seed + num_resets,
        # so seed=None crashes). It already changes every reset; rank keeps
        # workers distinct.
        line_generator=SparseLineGen(seed=Config.seed + 100_000 * rank),
        number_of_agents=Config.n_agents,
        obs_builder_object=StateMaskingObs(),
        malfunction_generator=malfunction_gen,
    )
    return env


# ---------------------------------------------------------------------------
# Expert heuristics  (called inside env_worker — env is local to that process)
# ---------------------------------------------------------------------------

def state_classifier(env, pos, direction, _cache=None):
    """
    Classify current cell:
      0: Normal  1: Stopping Point  2: Junction  3: Must Stop  4: Stopping+Must Stop
    Uses an optional dict cache to avoid repeated rail.get_transitions() calls.
    """
    key = (pos, direction)
    if _cache is not None and key in _cache:
        avb_moves = _cache[key]
    else:
        avb_moves = env.rail.get_transitions(*pos, direction)
        if _cache is not None:
            _cache[key] = avb_moves
    if sum(avb_moves) > 1:
        return 2
    avb_idx     = avb_moves.index(1)
    dir_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    next_pos    = (pos[0] + dir_offsets[avb_idx][0],
                   pos[1] + dir_offsets[avb_idx][1])
    is_next_junction = False
    for d in range(4):
        nkey = (next_pos, d)
        if _cache is not None and nkey in _cache:
            trans = _cache[nkey]
        else:
            trans = env.rail.get_transitions(*next_pos, d)
            if _cache is not None:
                _cache[nkey] = trans
        if sum(trans) > 2:
            is_next_junction = True
            break
    occupied = any(a.position == next_pos for a in env.agents)
    if is_next_junction:
        return 4 if occupied else 1
    return 3 if occupied else 0


def get_valid_actions(env, obs, agent_id, _cache=None):
    """
    Port of _NextValidActions from the NeurIPS reference code.
    Returns a list of valid internal actions: 0=Left, 1=Forward, 2=Right, 3=Stop.
    _cache: optional dict for caching rail.get_transitions results.
    """
    traffic_signal = obs[24]
    homo_junctions = obs[27:30]
    if traffic_signal == -1:
        return [3]
    currentobs  = np.reshape(obs[0:24], (3, -1))
    agent       = env.agents[agent_id]
    actual_pos  = agent.position  if agent.position  is not None else agent.initial_position
    actual_dir  = agent.direction if agent.direction is not None else agent.initial_direction
    state = state_classifier(env, actual_pos, actual_dir, _cache)
    if state in [3, 4]:
        return [3]
    elif state == 0:
        return [1]
    elif state == 1:
        SolExist         = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
        agentsblocking   = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
        agentsblockjunct = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
        for i in range(3):
            if SolExist[i] == 1 and agentsblocking[i] == 0:
                return [1, 3]
        if int(np.sum(homo_junctions == 1)) >= 2:
            for i in range(3):
                if SolExist[i] == 1 and agentsblockjunct[i] == 0 and homo_junctions[i] == 1:
                    return [1, 3]
        return [3]
    else:  # Junction
        SolExist         = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
        agentsblocking   = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
        agentsblockjunct = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
        validactions = []
        for i in range(3):
            if SolExist[i] == 1 and agentsblocking[i] == 0:
                validactions.append(i)
        if validactions:
            return validactions
        if int(np.sum(homo_junctions == 1)) >= 2:
            for i in range(3):
                if SolExist[i] == 1 and agentsblockjunct[i] == 0 and homo_junctions[i] == 1:
                    validactions.append(i)
                    break
            if validactions:
                return validactions
        return [3]


# ---------------------------------------------------------------------------
# Action masking helper  (used for sampling AND in the PPO update so the
# importance ratio is computed on the same distribution actions came from)
# ---------------------------------------------------------------------------

def masked_policy(policies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renormalise policies over valid actions. Falls back to uniform-over-valid
    if the network puts ~zero mass on every valid action."""
    p = policies.clamp(min=0) * mask
    s = p.sum(-1, keepdim=True)
    uniform = mask / mask.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.where(s > 1e-8, p / s.clamp(min=1e-8), uniform)


# ---------------------------------------------------------------------------
# Worker process  (pure CPU — no CUDA, no network)
# ---------------------------------------------------------------------------

def env_worker(rank: int, obs_pipe, act_pipe):
    """
    Owns one Flatland env. Protocol:
      Startup  : sends (obs [n_agents, obs_size], valid_tgts [n_agents, A_SIZE])
      Loop     : recvs action dict {agent_id: flatland_action (int)}
                 sends (obs, shaped_rewards, done_dict, valid_tgts, raw_rewards)
      _RESET   : resets env, sends (obs, valid_tgts)
      _SHUTDOWN: exits
    """
    env    = make_env(rank)
    n      = Config.n_agents
    obs_sz = Config.obs_size
    _trans_cache: dict = {}   # per-episode rail transition cache (pos,dir) -> transitions
    # Per-worker seed stream: reproducible, but a different map every episode
    # and different across workers.
    seed_rng = np.random.default_rng(Config.seed + rank)

    def next_seed() -> int:
        return int(seed_rng.integers(0, 2**31 - 1))

    def pack_obs(obs_dict):
        arr = np.zeros((n, obs_sz), dtype=np.float32)
        for i in range(n):
            if obs_dict.get(i) is not None:
                arr[i] = obs_dict[i]
        return arr

    def compute_valid_tgts(obs_dict):
        """Multi-hot [n_agents, A_SIZE] mask of expert-valid actions."""
        vtgts = np.zeros((n, A_SIZE), dtype=np.float32)
        for i in range(n):
            if obs_dict.get(i) is not None:
                try:
                    valid = get_valid_actions(env, obs_dict[i], i, _trans_cache)
                    vtgts[i, valid] = 1.0
                except Exception:
                    vtgts[i, 3] = 1.0   # fallback: stop
            else:
                vtgts[i, 3] = 1.0       # agent done/missing
        return vtgts

    max_dist = float(Config.width * Config.height)   # cap for unreachable cells (inf)

    def target_distances() -> np.ndarray:
        """Shortest-path distance (cells) from each agent to its target."""
        dmap = env.distance_map.get()
        out  = np.zeros(n, dtype=np.float32)
        for i, a in enumerate(env.agents):
            if a.state == TrainState.DONE:
                continue                                   # arrived -> 0
            pos = a.position  if a.position  is not None else a.initial_position
            dr  = a.direction if a.direction is not None else a.initial_direction
            d   = dmap[i, pos[0], pos[1], dr]
            out[i] = max_dist if not np.isfinite(d) else min(float(d), max_dist)
        return out

    obs_dict, _ = env.reset(random_seed=next_seed())
    env._max_episode_steps = Config.max_episode_steps   # override timetable value
    prev_dist = target_distances()
    obs_pipe.send((pack_obs(obs_dict), compute_valid_tgts(obs_dict)))

    while True:
        msg = act_pipe.recv()

        if msg == _SHUTDOWN:
            break

        if msg == _RESET:
            _trans_cache.clear()    # rail regenerates — old transitions invalid
            clear_trans_cache()     # also clear shortestpath.py module cache
            obs_dict, _ = env.reset(random_seed=next_seed())
            env._max_episode_steps = Config.max_episode_steps   # override timetable value
            prev_dist = target_distances()
            obs_pipe.send((pack_obs(obs_dict), compute_valid_tgts(obs_dict)))
            continue

        # Normal env step. Flatland returns (obs, rewards, dones, info); there is no
        # separate truncation flag — on timeout Flatland marks every agent done itself.
        obs_dict, rew_dict, term, _info = env.step(msg)

        obs  = pack_obs(obs_dict)
        rews = np.array([rew_dict.get(i, 0.0) for i in range(n)], dtype=np.float32)

        cur_dist  = target_distances()
        shaped    = rews + SHAPING_COEF * (prev_dist - cur_dist)
        prev_dist = cur_dist

        done = {i: bool(term.get(i, False)) for i in range(n)}
        done['__all__'] = bool(term.get('__all__', False)) or all(done.values())

        obs_pipe.send((obs, shaped.astype(np.float32), done, compute_valid_tgts(obs_dict), rews))


# ---------------------------------------------------------------------------
# Rollout buffer  (pre-allocated on GPU)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Shape: [T, W, A, ...] — T=timesteps, W=workers, A=agents."""

    def __init__(self, T, W, A, obs_sz, act_sz, device):
        self.T      = T
        self.device = device
        kw = dict(device=device)
        self.obs        = torch.zeros(T, W, A, obs_sz, **kw)
        self.actions    = torch.zeros(T, W, A,          dtype=torch.long, **kw)
        self.rewards    = torch.zeros(T, W, A,          **kw)
        self.dones      = torch.zeros(T, W,             **kw)
        self.valid_tgts = torch.zeros(T, W, A, act_sz,  **kw)
        self.log_probs  = torch.zeros(T, W, A,          **kw)
        self.values     = torch.zeros(T, W,             **kw)
        self.ptr = 0

    def store(self, obs, actions, rewards, dones, valid_tgts, log_probs, values):
        t = self.ptr
        self.obs[t]        = obs
        self.actions[t]    = actions
        self.rewards[t]    = rewards
        self.dones[t]      = dones
        self.valid_tgts[t] = valid_tgts
        self.log_probs[t]  = log_probs
        self.values[t]     = values
        self.ptr += 1

    def reset_ptr(self):
        self.ptr = 0

    def compute_returns(self, bootstrap: torch.Tensor) -> torch.Tensor:
        """Monte-Carlo returns. bootstrap: [W]. Returns [T, W]."""
        returns       = torch.zeros(self.T, bootstrap.shape[0], device=self.device)
        joint_rewards = self.rewards.sum(dim=-1)  # [T, W]
        R = bootstrap.clone()
        for t in reversed(range(self.T)):
            R          = joint_rewards[t] + GAMMA * R * (1.0 - self.dones[t])
            returns[t] = R
        return returns


# ---------------------------------------------------------------------------
# PPO update  (everything on GPU)
# ---------------------------------------------------------------------------

def ppo_update(network, optimizer, buffer, returns):
    T, W, A, obs_sz = buffer.obs.shape

    flat_obs      = buffer.obs.view(T * W, A, obs_sz)
    flat_acts     = buffer.actions.view(T * W, A)
    flat_returns  = returns.view(T * W)
    flat_tgts     = buffer.valid_tgts.view(T * W, A, -1)
    flat_lp_old   = buffer.log_probs.view(T * W, A)
    flat_vals_old = buffer.values.view(T * W)

    # Only agents that actually had a choice (>1 valid action) contribute to the
    # actor/entropy loss. Done agents and forced moves carry no learning signal.
    has_choice = (flat_tgts.sum(-1) > 1).float()           # [T*W, A]
    n_choice   = has_choice.sum().clamp(min=1.0)

    adv = flat_returns - flat_vals_old
    if adv.numel() > 1 and adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total_loss = 0.0
    for _ in range(PPO_EPOCHS):
        policies, estimations, v = network(flat_obs)
        cur_v = v.squeeze(-1)

        # Same masked distribution the actions were sampled from
        m_pol  = masked_policy(policies, flat_tgts)

        # Actor loss (PPO clip)
        act_p  = m_pol.gather(2, flat_acts.unsqueeze(-1)).squeeze(-1)
        log_p  = torch.log(act_p + 1e-8)
        ratio  = torch.exp(log_p - flat_lp_old)
        adv_e  = adv.unsqueeze(1)
        surr1  = ratio * adv_e
        surr2  = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_e
        a_loss = -(torch.min(surr1, surr2) * has_choice).sum() / n_choice

        # Critic loss
        std    = flat_returns.std().clamp(min=1e-8)
        r_norm = (flat_returns - flat_returns.mean()) / std
        v_norm = (cur_v        - flat_returns.mean()) / std
        c_loss = F.mse_loss(v_norm, r_norm)

        # Entropy over valid actions only (unmasked entropy would push mass onto invalid moves)
        ent    = (-(m_pol * torch.log(m_pol + 1e-8)).sum(-1) * has_choice).sum() / n_choice

        # Estimator (valid-action imitation)
        e_loss = F.binary_cross_entropy_with_logits(estimations, flat_tgts)

        loss = a_loss + 0.5 * c_loss - ENTROPY_COEF * ent + 0.5 * e_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

    return total_loss / PPO_EPOCHS


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Workers: {N_WORKERS}  (cores used: {N_WORKERS + 1} / {os.cpu_count()})")
    print()

    # ---- Network + optimiser ----
    network = ACNet(
        state_size=Config.obs_size,
        action_size=A_SIZE,
        n_agents=Config.n_agents,
    ).to(device)
    optimizer = optim.AdamW(network.parameters(), lr=LR,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # ---- Resume from checkpoint ----
    start_episode = 0
    ckpt_dir = os.path.abspath(Config.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = [f for f in os.listdir(ckpt_dir)
             if f.startswith("checkpoint_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda f: int(f.split("_")[1].split(".")[0]))
        ckpt   = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
        network.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = ckpt.get("episode", 0)
        print(f"Resumed: {latest}  (episode {start_episode})")

    # ---- Spawn worker processes ----
    # spawn is set in __main__ block before train() is called.
    # Workers are daemon=True so they die automatically if main crashes.
    workers, obs_pipes, act_pipes = [], [], []
    for rank in range(N_WORKERS):
        p_obs, c_obs = mp.Pipe(duplex=False)   # worker → main  (p_obs=read, c_obs=write)
        p_act, c_act = mp.Pipe(duplex=False)   # main → worker  (p_act=read, c_act=write)
        # Worker needs: c_obs (write end, to send obs) + p_act (read end, to recv actions)
        # Main  needs: p_obs (read end, to recv obs)  + c_act (write end, to send actions)
        p = mp.Process(target=env_worker,
                       args=(rank, c_obs, p_act),
                       daemon=True)
        p.start()
        workers.append(p)
        obs_pipes.append(p_obs)
        act_pipes.append(c_act)
    print(f"Spawned {N_WORKERS} env workers.")

    # Collect initial obs + valid-action masks from each worker
    init_data     = [obs_pipes[w].recv() for w in range(N_WORKERS)]
    obs_np        = np.stack([d[0] for d in init_data])   # [W, A, obs]
    valid_tgts_np = np.stack([d[1] for d in init_data])   # [W, A, A_SIZE]
    dones  = [{'__all__': False, **{i: False for i in range(Config.n_agents)}}
              for _ in range(N_WORKERS)]

    # ---- Rollout buffer (GPU) ----
    buffer = RolloutBuffer(
        T=ROLLOUT_LEN, W=N_WORKERS, A=Config.n_agents,
        obs_sz=Config.obs_size, act_sz=A_SIZE, device=device,
    )

    episode_count = start_episode
    last_saved_ep = start_episode
    ep_reward_buf = [0.0] * N_WORKERS   # running reward sum per worker, reset each episode
    t_start       = time.time()
    print(f"Training: episodes {start_episode} → {MAX_EPISODES}\n")

    # ==========================================================================
    # Main loop
    # ==========================================================================
    while episode_count < MAX_EPISODES:
        buffer.reset_ptr()
        rollout_ep_rewards: list[float] = []   # EpRewards of episodes completed this rollout
        network.eval()

        # ----------------------------------------------------------------------
        # ROLLOUT COLLECTION
        # Key timing:
        #   1. GPU forward pass  (~ms)
        #   2. Send actions to all workers simultaneously
        #   3. Workers step envs IN PARALLEL on their cores  (~seconds, the bottleneck)
        #   4. Collect results
        # Step 3 overlaps with any CPU work we do between send and recv.
        # ----------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(ROLLOUT_LEN):

                # 1. Batched GPU inference — all workers × all agents in one call
                obs_t              = torch.from_numpy(obs_np).to(device)      # [W, A, obs]
                policies, _, v     = network(obs_t)                            # [W, A, act], [W,1]
                values_t           = v.squeeze(-1)                             # [W]

                # 2. Build the current-state action mask. Done agents (and any row
                #    with no valid action) are forced to Stop, so they contribute
                #    log-prob 0 and no gradient.
                for w in range(N_WORKERS):
                    for i in range(Config.n_agents):
                        if dones[w].get(i, False) or valid_tgts_np[w, i].sum() <= 0:
                            valid_tgts_np[w, i] = 0.0
                            valid_tgts_np[w, i, 3] = 1.0
                valid_t = torch.from_numpy(valid_tgts_np).to(device)          # [W, A, A_SIZE]

                # 3. Sample from the masked policy and record ITS log-prob
                m_pol     = masked_policy(torch.nan_to_num(policies), valid_t)
                actions_t = Categorical(probs=m_pol).sample()                 # [W, A]
                log_probs = torch.log(m_pol.gather(2, actions_t.unsqueeze(-1)).squeeze(-1) + 1e-8)
                actions_np = actions_t.cpu().numpy()

                # internal 0=L,1=F,2=R,3=Stop → Flatland 1=L,2=F,3=R,4=Stop
                actions_dicts = [
                    {i: int(a) + 1 if a < 3 else 4 for i, a in enumerate(actions_np[w])}
                    for w in range(N_WORKERS)
                ]

                # 4. Send actions to ALL workers — they start stepping in parallel NOW
                for w in range(N_WORKERS):
                    act_pipes[w].send(actions_dicts[w])

                # 5. Collect results — blocks until all workers finish
                results          = [obs_pipes[w].recv() for w in range(N_WORKERS)]
                next_obs_np      = np.stack([r[0] for r in results])          # [W, A, obs]
                rewards_np       = np.stack([r[1] for r in results])          # [W, A] shaped (for learning)
                raw_rewards_np   = np.stack([r[4] for r in results])          # [W, A] Flatland reward (for logs)
                next_dones       = [r[2]            for r in results]
                valid_tgts_np    = np.stack([r[3] for r in results])          # [W, A, A_SIZE] next-state masks

                rewards_t = torch.from_numpy(rewards_np).to(device)
                dones_t   = torch.tensor(
                    [float(next_dones[w]['__all__']) for w in range(N_WORKERS)],
                    device=device,
                )

                buffer.store(obs_t, actions_t, rewards_t, dones_t,
                             valid_t, log_probs, values_t)

                # Accumulate rewards BEFORE the done check so each episode's final
                # step (which carries Flatland's end-of-episode penalty) is counted
                # in that episode, not the next one.
                for w in range(N_WORKERS):
                    ep_reward_buf[w] += float(raw_rewards_np[w].sum())

                # ---- Inline reset + per-episode logging ----
                for w in range(N_WORKERS):
                    if next_dones[w]['__all__']:
                        episode_count += 1
                        rollout_ep_rewards.append(ep_reward_buf[w])
                        elapsed = time.time() - t_start
                        print(f"  [W{w:02d}] Ep {episode_count:4d}/{MAX_EPISODES} | "
                              f"EpReward {ep_reward_buf[w]:8.2f} | "
                              f"{elapsed/60:.1f}min")
                        ep_reward_buf[w] = 0.0   # reset accumulator

                        act_pipes[w].send(_RESET)
                        reset_data        = obs_pipes[w].recv()
                        next_obs_np[w]    = reset_data[0]
                        valid_tgts_np[w]  = reset_data[1]
                        next_dones[w]     = {'__all__': False,
                                             **{i: False for i in range(Config.n_agents)}}

                obs_np = next_obs_np
                dones  = next_dones

        # ----------------------------------------------------------------------
        # PPO UPDATE  (GPU — workers idle during this, waiting for next actions)
        # ----------------------------------------------------------------------
        network.train()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device)
            _, _, v_boot = network(obs_t)
            bootstrap    = v_boot.squeeze(-1)          # [W]

        returns  = buffer.compute_returns(bootstrap)   # [T, W]
        avg_loss = ppo_update(network, optimizer, buffer, returns)

        # ----------------------------------------------------------------------
        # LOGGING
        # ----------------------------------------------------------------------
        elapsed = time.time() - t_start
        if rollout_ep_rewards:
            ep_reward_str = f"EpReward {sum(rollout_ep_rewards)/len(rollout_ep_rewards):8.2f} (n={len(rollout_ep_rewards)})"
        else:
            # No episode finished this rollout — fall back to per-step mean
            ep_reward_str = f"ShapedReward/step {buffer.rewards.sum(dim=-1).mean().item():.3f} (no ep done)"
        print(f"Ep {episode_count:4d}/{MAX_EPISODES} | "
              f"Loss {avg_loss:.4f} | "
              f"{ep_reward_str} | "
              f"{elapsed/60:.1f}min elapsed")

        # ----------------------------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------------------------
        # Episode count can jump by several per rollout (many workers), so a
        # `% SAVE_EVERY == 0` test would often be skipped. Track the gap instead.
        if episode_count - last_saved_ep >= SAVE_EVERY:
            path = os.path.join(ckpt_dir, f"checkpoint_{episode_count}.pth")
            torch.save({
                "model_state_dict":     network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode":              episode_count,
            }, path)
            last_saved_ep = episode_count
            print(f"  >> Saved {path}")

        # (Workers are reset inline during the rollout — nothing to do here.)

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    for w in range(N_WORKERS):
        act_pipes[w].send(_SHUTDOWN)
    for p in workers:
        p.join(timeout=10)

    final = os.path.join(ckpt_dir, "appo_final.pth")
    torch.save({
        "model_state_dict":     network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              episode_count,
    }, final)
    print(f"\nDone. Saved → {final}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Must be set before ANY CUDA call and before spawning.
    # force=True in case something imported above already set it.
    mp.set_start_method("spawn", force=True)
    train()
