import os
import torch
import numpy as np
import random
from tqdm import tqdm

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool

from config import Config
from observations import StateMaskingObs
from agents.marl_agent import MARLAgent

def train():
    # 1. Environment Setup (NeurIPS 2020 Style)
    obs_builder = StateMaskingObs()
    
    rail_gen = sparse_rail_generator(
        max_num_cities=Config.n_cities,
        max_rails_between_cities=Config.max_rails_between_cities,
        max_rails_in_city=Config.max_rails_in_city,
        seed=Config.seed,
        grid_mode=False
    )
    
    line_gen = sparse_line_generator()
    
    # Custom Malfunctions (from NeurIPS winner code)
    stochastic_data = {
        'malfunction_rate': Config.malfunction_rate,
        'min_duration': Config.min_malfunction_duration,
        'max_duration': Config.max_malfunction_duration
    }
    
    env = RailEnv(
        width=Config.width,
        height=Config.height,
        rail_generator=rail_gen,
        line_generator=line_gen,
        number_of_agents=Config.n_agents,
        obs_builder_object=obs_builder,
        malfunction_generator_and_process_data=stochastic_data
    )
    
    # Custom Reward Overrides
    env.step_penalty = Config.step_penalty
    env.global_reward = Config.global_reward

    # 2. Agent Initialization
    agent_handler = MARLAgent(
        state_size=Config.obs_size,
        action_size=Config.action_size,
        n_agents=Config.n_agents
    )

    # Ensure checkpoint directory exists
    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)

    # 3. Training Loop
    n_episodes = 1000
    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        done = {i: False for i in range(Config.n_agents)}
        done['__all__'] = False
        
        while not done['__all__']:
            actions = {}
            for i in range(Config.n_agents):
                if not done[i]:
                    actions[i] = agent_handler.act(obs[i], i)
            
            next_obs, rewards, next_done, info = env.step(actions)
            
            for i in range(Config.n_agents):
                if not done[i]:
                    agent_handler.step(
                        obs[i], actions[i], rewards[i], next_obs[i], next_done[i], i
                    )
                    episode_reward += rewards[i]
            
            obs = next_obs
            done = next_done

        # Logging and Checkpointing
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes} | Total Reward: {episode_reward:.2f}")
            agent_handler.save(f"marl_agent_ep{episode}.pth")

if __name__ == "__main__":
    train()
