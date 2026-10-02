import torch
import torch.multiprocessing as mp
import torch.optim as optim

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import SparseLineGen
from config import Config
from NewAgentInitObs import StateMaskingObs
from torch.distributions import Categorical
import os

# Assuming models.py contains your PyTorch Actor-Critic Network
from agents.models import ACNet

class Worker(mp.Process):
    def __init__(self, gameEnv, metaAgentID, workerID, a_size, global_network, global_optimizer, groupLock, start_episode=0, checkpoint_dir=None):
        """
        PyTorch A3C Worker Process.
        Inherits from mp.Process to run on a completely separate CPU core.
        """
        super(Worker, self).__init__()
        self.workerID = workerID
        self.env = gameEnv
        self.metaAgentID = metaAgentID
        self.name = "worker_" + str(workerID)
        self.groupLock = groupLock
        self.a_size = a_size
        self.start_episode = start_episode  # Resume from this episode
        self.checkpoint_dir = checkpoint_dir  # Absolute path resolved in __main__
        
        self.global_network = global_network
        self.global_optimizer = global_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Local Network (The agent's personal brain)
        # We initialize it with the same structure as the global network
        from config import Config
        self.local_network = ACNet(state_size=Config.obs_size, action_size=a_size, n_agents=Config.n_agents).to(self.device)

    def synchronize(self):
        """Wait for all other workers in the simulation to be ready."""
        if self.groupLock is not None:
            self.groupLock.wait()

    def train_step(self, rollout, gamma, bootstrap_value, imitation=False):
        """
        Calculates local PPO gradients and PUSHES them to the Global Network.
        rollout: list of tuples (state, action, reward, next_state, done)
        """
        if not rollout:
            return 0.0

        # PPO Hyperparameters
        ppo_epochs = 4
        ppo_clip = 0.2
        entropy_coef = 0.01

        # Unpack rollout
        # states: [batch, n_agents, state_size]
        states = torch.FloatTensor(np.array([r[0] for r in rollout])).to(self.device)
        actions = torch.LongTensor(np.array([r[1] for r in rollout])).to(self.device)
        rewards = torch.FloatTensor(np.array([r[2] for r in rollout])).to(self.device)
        dones = torch.FloatTensor(np.array([r[4] for r in rollout])).to(self.device)
        valid_targets = torch.FloatTensor(np.array([r[5] for r in rollout])).to(self.device)
        
        # Calculate Returns and Advantages using the old policy/values
        with torch.no_grad():
            old_policies, _, v = self.local_network(states)
            values = v.squeeze(-1) # [batch]
            if values.dim() == 2:
                values = values.squeeze(1) # [batch]
            
            action_probs = old_policies.gather(2, actions.unsqueeze(-1)).squeeze(-1) # [batch, n_agents]
            old_log_probs = torch.log(action_probs + 1e-8)
            
            # Centralized critic uses joint rewards
            joint_rewards = rewards.sum(dim=1)
            joint_dones = dones.max(dim=1)[0]
            
            returns = torch.zeros_like(joint_rewards)
            R = bootstrap_value
            for t in reversed(range(len(rollout))):
                R = joint_rewards[t] + gamma * R * (1.0 - joint_dones[t])
                returns[t] = R
                
            advantages = returns - values
            # Normalize advantages only when batch has > 1 element and non-zero std
            if advantages.numel() > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_estimator_loss = 0.0

        for _ in range(ppo_epochs):
            policies, estimations, v = self.local_network(states)
            
            current_v = v.squeeze(-1)
            if current_v.dim() == 2:
                current_v = current_v.squeeze(1)
                
            action_probs = policies.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            log_probs = torch.log(action_probs + 1e-8)
            
            # Ratio for PPO
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Surrogate losses
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages.unsqueeze(1)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss — normalize returns (guard against single-element batches where std=0)
            returns_std = returns.std() if returns.numel() > 1 else torch.tensor(1.0)
            if returns_std < 1e-8:
                returns_std = torch.tensor(1.0)
            returns_norm = (returns - returns.mean()) / returns_std
            current_v_norm = (current_v - returns.mean()) / returns_std
            critic_loss = torch.nn.functional.mse_loss(current_v_norm, returns_norm)
                          
            # Entropy bonus
            entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=-1).mean()
            
            # Estimator Loss (BCE with logits for numerical stability)
            estimator_loss = torch.nn.functional.binary_cross_entropy_with_logits(estimations, valid_targets)
            
            # Total PPO loss
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy + 0.5 * estimator_loss
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_estimator_loss += estimator_loss.item()

            # PUSH: Apply gradients to Global Network
            self.global_optimizer.zero_grad()
            self.local_network.zero_grad()
            loss.backward()
            
            # Manually copy the local gradients over to the global network parameters
            for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
                if local_param.grad is not None:
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.clone()
                    else:
                        global_param.grad.copy_(local_param.grad)

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), max_norm=0.5)

            # Update the global Master Brain
            self.global_optimizer.step()
            
            # PULL: Sync local weights with newly updated global weights
            self.local_network.load_state_dict(self.global_network.state_dict())
            
        if self.workerID == 0:
            print(f"    Loss breakdown — Actor: {total_actor_loss/ppo_epochs:.4f} | "
                  f"Critic: {total_critic_loss/ppo_epochs:.4f} | "
                  f"Estimator: {total_estimator_loss/ppo_epochs:.4f}")
        return total_loss / ppo_epochs

    def StateClassifier(self, pos, direction):
        """
        0: Normal, 1: Stopping Point, 2: Junction, 3: Must Stop, 4: Stopping+Must Stop
        """
        avb_moves = self.env.rail.get_transitions((pos, direction))
        if sum(avb_moves) > 1: return 2
        
        # Look ahead for junction
        avb_idx = avb_moves.index(1)
        dir_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_pos = (pos[0] + dir_offsets[avb_idx][0], pos[1] + dir_offsets[avb_idx][1])
        
        is_next_junction = False
        for d in range(4):
            if sum(self.env.rail.get_transitions((tuple(next_pos), d))) > 2:
                is_next_junction = True
                break
        
        occupied = any(a.position == next_pos for a in self.env.agents)
        if is_next_junction:
            return 4 if occupied else 1
        return 3 if occupied else 0

    def get_valid_actions(self, obs, agentID):
        """
        Implements the exact _NextValidActions from the NeurIPS code.
        Returns a list of valid actions (0: Left, 1: Forward, 2: Right, 3: Stop)
        """
        traffic_signal = obs[24]
        homo_junctions = obs[27:30]
        
        if traffic_signal == -1:
            return [3]
            
        currentobs = np.reshape(obs[0:24], (3, -1))
        
        agent = self.env.agents[agentID]
        actual_pos = agent.position if agent.position is not None else agent.initial_position
        actual_dir = agent.direction if agent.direction is not None else agent.initial_direction
        
        state = self.StateClassifier(actual_pos, actual_dir)
        
        if state in [3, 4]:  # Must Stop Point
            return [3]
        elif state == 0:  # Currently at NO decision point
            return [1]
        elif state == 1:  # Currently at stopping point
            SolExist = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
            agentsblocking = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
            agentsblockingjunction = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
            
            for i in range(0, 3):
                if (SolExist[i] == 1) and (agentsblocking[i] == 0):
                    return [1, 3]
                    
            if homo_junctions.count(1) >= 2:
                for i in range(0, 3):
                    if (SolExist[i] == 1) and (agentsblockingjunction[i] == 0) and homo_junctions[i] == 1:
                        return [1, 3]
            return [3]
        else:  # Currently at junction
            SolExist = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
            agentsblocking = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
            agentsblockingjunction = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
            
            validactions = []
            for i in range(0, 3):
                if (SolExist[i] == 1) and (agentsblocking[i] == 0):
                    validactions.append(i)
                    
            if validactions:
                return validactions
            else:
                if homo_junctions.count(1) >= 2:
                    for i in range(0, 3):
                        if (SolExist[i] == 1) and (agentsblockingjunction[i] == 0) and homo_junctions[i] == 1:
                            validactions.append(i)
                            break
                    if validactions:
                        return validactions
                return [3]

    def run(self):
        """
        Executes the main Flatland multi-agent loop for this specific worker.
        """
        print(f"Worker {self.workerID} started.")
        self.local_network.load_state_dict(self.global_network.state_dict())
        
        # 1. Initialize the Worker's Personal Environment
        # In TF 1.x, they shared the environment via threads, but in PyTorch multiprocessing,
        # it is best to give each worker its own environment instance to avoid GIL and pickling issues.

        rail_gen = sparse_rail_generator(
            max_num_cities=Config.n_cities,
            max_rails_between_cities=Config.max_rails_between_cities,
            max_rail_pairs_in_city=Config.max_rail_pairs_in_city,
            seed=Config.seed + self.workerID,
            grid_mode=False
        )
        line_gen = SparseLineGen()
        
        from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
        malfunction_gen = ParamMalfunctionGen(MalfunctionParameters(
            malfunction_rate=Config.malfunction_rate,
            min_duration=Config.min_malfunction_duration,
            max_duration=Config.max_malfunction_duration
        ))
        
        self.env = RailEnv(
            width=Config.width,
            height=Config.height,
            rail_generator=rail_gen,
            line_generator=line_gen,
            number_of_agents=Config.n_agents,
            obs_builder_object=StateMaskingObs(),
            malfunction_generator=malfunction_gen
        )
        # Note: step_penalty and global_reward are set in Config and applied manually in reward shaping.
        # Flatland 4.x does not expose these as env attributes.
        
        # 2. RL Execution Loop
        episode_count = self.start_episode  # Resume from checkpoint if applicable
        gamma = 0.99
        rollout_length = 128
        max_episodes = 1000
        
        print(f"Worker {self.workerID} | Starting from episode {episode_count}")
        while episode_count < max_episodes: # Max episodes per worker
            obs_dict, _ = self.env.reset()
            rollout = []
            done = {i: False for i in range(Config.n_agents)}
            done['__all__'] = False
            
            while not done['__all__']:
                joint_obs = np.zeros((Config.n_agents, Config.obs_size))
                valid_actions_dict = {}
                valid_actions_targets = np.zeros((Config.n_agents, self.a_size))
                
                # Build Joint Observation & Extract Masks
                for i in range(Config.n_agents):
                    if not done[i] and obs_dict[i] is not None:
                        joint_obs[i] = obs_dict[i]
                        valid_actions_dict[i] = self.get_valid_actions(obs_dict[i], i)
                    else:
                        valid_actions_dict[i] = [3] # If done or no obs, just stop
                        
                    valid_actions_targets[i, valid_actions_dict[i]] = 1.0
                        
                # Forward Pass
                states_t = torch.FloatTensor(joint_obs).unsqueeze(0).to(self.device) # [1, n_agents, obs_size]
                with torch.no_grad():
                    policies, _, _ = self.local_network(states_t)
                
                # Masking & Sampling
                actions_dict = {}
                joint_action = np.zeros(Config.n_agents, dtype=int)
                
                for i in range(Config.n_agents):
                    if not done[i]:
                        probs = policies[0, i].cpu().clone() # [action_size]
                        
                        # Apply the Hard Mask (Pruning)
                        mask = torch.zeros_like(probs)
                        mask[valid_actions_dict[i]] = 1.0
                        
                        probs = probs * mask
                        # NaN guard: if network produced NaN weights, fall back to uniform over valid actions
                        if torch.isnan(probs).any() or probs.sum() <= 0:
                            probs = mask.clone()
                            if probs.sum() <= 0:
                                probs[3] = 1.0  # Force stop
                        probs /= probs.sum()
                            
                        dist = Categorical(probs=probs)
                        act = dist.sample().item()
                        
                        # Flatland expects 1-4 for movement, ACNet output is 0-3
                        # We map our ACNet output (0:L, 1:F, 2:R, 3:S) to Flatland's action space.
                        # Assuming Flatland: 1:L, 2:F, 3:R, 4:Stop
                        flatland_action = act + 1 if act < 3 else 4
                        actions_dict[i] = flatland_action
                        joint_action[i] = act # Store internal 0-3 action for training
                        
                # Step Environment — Flatland 4.x returns 5-tuple (obs, rew, terminated, truncated, info)
                step_result = self.env.step(actions_dict)
                next_obs_dict = step_result[0]
                rewards_dict  = step_result[1]
                # Merge terminated + truncated into a single done dict
                terminated    = step_result[2]
                truncated     = step_result[3] if len(step_result) > 3 else {}
                done = {k: terminated.get(k, False) or truncated.get(k, False)
                        for k in terminated}
                if '__all__' not in done:
                    done['__all__'] = all(done.get(i, False) for i in range(Config.n_agents))
                
                joint_reward = np.zeros(Config.n_agents)
                joint_done = np.zeros(Config.n_agents)
                for i in range(Config.n_agents):
                    joint_reward[i] = rewards_dict.get(i, 0)
                    joint_done[i] = done.get(i, True)
                    
                rollout.append((joint_obs, joint_action, joint_reward, None, joint_done, valid_actions_targets))
                obs_dict = next_obs_dict
                
                # Truncated Backpropagation (Update over partial episode)
                if len(rollout) >= rollout_length and not done['__all__']:
                    next_joint_obs = np.zeros((Config.n_agents, Config.obs_size))
                    for i in range(Config.n_agents):
                        if not done[i] and obs_dict[i] is not None:
                            next_joint_obs[i] = obs_dict[i]
                            
                    states_t = torch.FloatTensor(next_joint_obs).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _, _, v = self.local_network(states_t)
                        bootstrap_val = v.item()
                        
                    loss = self.train_step(rollout, gamma, bootstrap_val)
                    rollout = []
                    
            # End of Episode Update
            loss = 0.0
            if len(rollout) > 0:
                loss = self.train_step(rollout, gamma, 0.0)
                
            episode_count += 1
            print(f"Worker {self.workerID} | Episode {episode_count} | Loss: {loss:.4f}")

            # Checkpoint every 20 episodes (worker 0 saves on behalf of all)
            if episode_count % 20 == 0 and self.workerID == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode_count}.pth")
                torch.save({
                    'model_state_dict': self.global_network.state_dict(),
                    'optimizer_state_dict': self.global_optimizer.state_dict(),
                    'episode': episode_count,
                }, save_path)
                print(f"  >> Checkpoint saved: {save_path}")


if __name__ == "__main__":
    # 1. CRITICAL: Required for PyTorch multiprocessing
    mp.set_start_method('spawn', force=True)

    # ----------------------------parameters----------------------------------------------
    max_episode_length      = 384
    gamma                   = .95  # discount rate for advantage estimation and reward discounting
    EXPERIENCE_BUFFER_SIZE  = 64
    ENVIRONMENT_SIZE        = (15, 51)
    a_size                  = 4  # removed DO_NOTHING. Otherwise should equal 5
    NUM_META_AGENTS         = 3
    NUM_THREADS             = 8  # number of parallel workers/agents
    EPISODE_SAMPLES         = EXPERIENCE_BUFFER_SIZE
    ENV_CHANGE_FREQUENCY    = 10
    SPARSE_POSSIBILITY      = 1
    FUSE_LAYERS             = 80
    LR_Q                    = 2.e-5  
    ADAPT_LR                = False
    ADAPT_COEFF             = 5.e-6
    PRUNE_ACTION            = 1
    ON_SCREEN_RENDERING     = False
    load_model              = False 
    RESET_TRAINER           = False
    version                 = 'dynamic_prune'
    model_path              = './model_' + version
    gifs_path               = './gifs_' + version
    train_path              = 'train_' + version
    
    # Imitation options
    PRIMING_LENGTH          = 0  
    DEMONSTRATION_PROB      = 0
    
    # Testing
    TRAINING                = True
    GREEDY                  = False
    NUM_EXPS                = 100
    
    # Network input/output configs
    from config import Config
    state_size = Config.obs_size # From StateMaskingObs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Master running on: {device}")

    # 2. Create the Master Brain and SHARE ITS MEMORY
    # This allows all independent processes to read/write to the same weights
    global_network = ACNet(state_size=state_size, action_size=a_size, n_agents=Config.n_agents).to(device)
    global_network.share_memory()

    # 3. Create Global Optimizer
    # Using AdamW for better weight decay regularization
    global_optimizer = optim.AdamW(global_network.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # --- Resume from latest checkpoint if available ---
    start_episode = 0
    if os.path.exists(Config.checkpoint_dir):
        checkpoints = [
            f for f in os.listdir(Config.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]
        if checkpoints:
            # Find the one with the highest episode number
            latest = max(checkpoints, key=lambda f: int(f.split("_")[1].split(".")[0]))
            checkpoint_path = os.path.join(Config.checkpoint_dir, latest)
            print(f"Resuming from checkpoint: {latest}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            global_network.load_state_dict(checkpoint['model_state_dict'])
            global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            print(f"Resuming from episode {start_episode}")
        else:
            print("No checkpoint found. Starting fresh.")
    else:
        print("No checkpoint directory found. Starting fresh.")

    # 4. Create synchronization mechanism
    # PyTorch's 'Barrier' acts identically to the custom GroupLock
    groupLock = mp.Barrier(NUM_THREADS)

    # 5. Spawn Workers
    workers = []
    dummy_env = None # Pass your RailEnv here
    
    # Resolve checkpoint dir once here in the main process (absolute path guaranteed)
    checkpoint_dir = os.path.abspath(Config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    for i in range(NUM_THREADS):
        worker = Worker(
            gameEnv=dummy_env,
            metaAgentID=0,
            workerID=i,
            a_size=a_size,
            global_network=global_network,
            global_optimizer=global_optimizer,
            groupLock=groupLock,
            start_episode=start_episode,
            checkpoint_dir=checkpoint_dir,  # Pass resolved absolute path
        )
        worker.start()
        workers.append(worker)


    # 6. Wait for all training to finish
    for worker in workers:
        worker.join()
    
    print("Training Complete.")

    # 7. Final model save
    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)
    final_path = os.path.join(Config.checkpoint_dir, "appo_final.pth")
    torch.save({
        'model_state_dict': global_network.state_dict(),
        'optimizer_state_dict': global_optimizer.state_dict(),
    }, final_path)
    print(f"Final model saved to: {final_path}")

