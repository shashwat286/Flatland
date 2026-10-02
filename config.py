
class Config:
    # Environment Dimensions
    width  = 50
    height = 50
    n_agents = 12

    # Rail Generation Parameters (Sparse)
    n_cities = 5
    max_rails_between_cities = 3
    max_rail_pairs_in_city   = 2   # 2 pairs = 4 rails per city (max_rails_in_city=4)
    grid_mode = False
    max_episode_steps = 1500
    seed = 1
    
    # Malfunction Parameters
    malfunction_rate         = 0.01
    min_malfunction_duration = 20
    max_malfunction_duration = 50
    
    # Reward Logic
    step_penalty = -0.3
    global_reward = 20.0
    
    # Observation and Action Space
    obs_size = 30   # NewAgentInitObs size
    action_size = 5 # (0: stop, 1: left, 2: straight, 3: right, 4: no-op/logic)
    
    # Training Parameters
    checkpoint_dir = "/content/drive/MyDrive/Flatland/checkpoints"
    lr = 1e-4
    gamma = 0.99
    batch_size = 64
    alpha = 0.2 # Temperature parameter for SAC (if still used elsewhere)
    memory_size = 1000000 # Standard MARL buffer capacity
    
    # PPO Parameters
    ppo_epochs = 4
    ppo_clip = 0.2
    entropy_coef = 0.01
    
    # Coordination / Expert Logic Toggle
    use_expert = True
