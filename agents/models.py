import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    """
    Standard 7-layer MLP structure consistent with NeurIPS 2020 architecture.
    """
    def __init__(self, in_features, out_features):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1024), nn.ReLU(),
            nn.Linear(1024, out_features)
        )

    def forward(self, x):
        return self.net(x)

class ACNet(nn.Module):
    def __init__(self, state_size, action_size, n_agents):
        """
        Multi-Agent Actor-Critic Network
        - Single Critic with two value functions (Centralized)
        - Multiple Actors using the DeepMLP structure (Decentralized)
        - Multiple Action Estimators using the same DeepMLP structure (Decentralized)
        """
        super(ACNet, self).__init__()
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size

        # --- CENTRALIZED CRITIC ---
        self.critic_net = nn.Sequential(
            nn.Linear(state_size * n_agents, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_head = nn.Linear(256, 1)

        # --- SHARED ACTOR ---
        # Single actor network shared across all homogeneous agents
        self.actor = DeepMLP(state_size, action_size)

        # --- SHARED ESTIMATOR ---
        self.estimator = DeepMLP(state_size, action_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Adjust head gains for stability
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        
        nn.init.orthogonal_(self.actor.net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.estimator.net[-1].weight, gain=0.01)

    def forward(self, x, centralized_state=None):
        """
        Args:
            x (torch.Tensor): Tensor of individual states [batch, n_agents, state_size]
            centralized_state (torch.Tensor, optional): Centralized joint state [batch, n_agents * state_size]
        Returns:
            policies (torch.Tensor): Agent policy outputs [batch, n_agents, action_size]
            estimations (torch.Tensor): Agent estimation outputs [batch, n_agents, action_size]
            values (torch.Tensor): Value output [batch, 1]
        """
        batch_size = x.size(0)
        # 1. Process all agent states through the shared Actor
        # x is [batch_size, n_agents, state_size], Linear layers broadcast the computation automatically!
        actor_logits = self.actor(x) 
        policies = F.softmax(actor_logits, dim=-1) # [batch_size, n_agents, action_size]

        # Valid action estimation
        estimator_logits = self.estimator(x) # [batch_size, n_agents, action_size]

        # Centralized Critic
        if centralized_state is None:
            centralized_state = x.view(batch_size, -1)
            
        v = self.value_head(self.critic_net(centralized_state))
        
        return policies, estimator_logits, v
