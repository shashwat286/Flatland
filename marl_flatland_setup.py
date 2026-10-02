import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool

# --- 1. CONFIGURATION ---
class Config:
    n_agents = 3
    x_dim = 30
    y_dim = 30
    n_cities = 2
    max_rails_between_cities = 2
    max_rail_pairs_in_city = 3
    seed = 42
    
    # RL Hyperparameters
    state_size = 5  # Depends on tree depth and features
    action_size = 5 # (0: DO NOTHING, 1: LEFT, 2: FORWARD, 3: RIGHT, 4: STOP)
    learning_rate = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    memory_size = 10000

# --- 2. THE RL AGENT (SKELETON) ---
# You can implement your own MARL algorithm here.
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.memory_size)
        
        # Simple Neural Network
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.epsilon = Config.epsilon_start

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > Config.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, Config.batch_size)
        # (Standard DQN Learning logic goes here)
        pass

# --- 3. UTILITIES ---
def preprocess_obs(obs, agent_idx):
    """
    Flatland TreeObs can be nested. This function flattens or extracts 
    key features for the neural network. 
    Modify this based on your tree depth or custom features.
    """
    if obs is None:
        return np.zeros(Config.state_size)
    
    # Placeholder: Extracting very basic features from tree node
    # In practice, you'd traverse the tree or use a flattened version.
    features = [
        obs.dist_min_to_target,
        obs.dist_own_target_encountered,
        obs.dist_other_target_encountered,
        obs.dist_other_agent_encountered,
        obs.dist_potential_conflict
    ]
    return np.array(features, dtype=np.float32)

# --- 4. MAIN SETUP AND TRAINING ---
def main():
    # Observation Builder: TreeObs is recommended for RL
    tree_obs = TreeObsForRailEnv(max_depth=2)

    # Generators
    rail_generator = sparse_rail_generator(
        max_num_cities=Config.n_cities,
        grid_mode=False,
        max_rails_between_cities=Config.max_rails_between_cities,
        max_rail_pairs_in_city=Config.max_rail_pairs_in_city,
        seed=Config.seed
    )

    # Environment
    env = RailEnv(
        width=Config.x_dim,
        height=Config.y_dim,
        rail_generator=rail_generator,
        line_generator=sparse_line_generator(),
        number_of_agents=Config.n_agents,
        obs_builder_object=tree_obs
    )

    # Agent Initialization
    agents = {i: DQNAgent(Config.state_size, Config.action_size) for i in range(Config.n_agents)}

    print(f"Flatland MARL Environment Ready with {Config.n_agents} agents.")
    
    # Training Loop
    for episode in range(1, 11): # Demo with 10 episodes
        obs, info = env.reset()
        
        # Preprocess initial observations
        agent_obs = {i: preprocess_obs(obs[i], i) for i in range(Config.n_agents)}
        
        total_reward = 0
        done = {"__all__": False}
        steps = 0

        while not done["__all__"] and steps < 100:
            actions = {}
            for i in range(Config.n_agents):
                if i in obs: # Only for agents currently in the environment
                    actions[i] = agents[i].act(agent_obs[i])
            
            next_obs, rewards, done, info = env.step(actions)
            
            for i in actions:
                next_state = preprocess_obs(next_obs[i], i)
                agents[i].step(agent_obs[i], actions[i], rewards[i], next_state, done[i])
                agent_obs[i] = next_state
                total_reward += rewards[i]
            
            steps += 1
            if steps % 20 == 0:
                print(f"Episode {episode}, Step {steps}, Total Reward: {total_reward:.2f}")

        print(f"Episode {episode} finished in {steps} steps. Total Reward: {total_reward:.2f}")

    print("\nEnvironment setup and demo run complete.")

if __name__ == "__main__":
    main()
