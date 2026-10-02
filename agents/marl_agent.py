import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Categorical
from agents.base_agent import BaseMARLAgent
from agents.models import ACNet
from agents.replay_buffer import ReplayBuffer
from config import Config
import copy

class MARLAgent(BaseMARLAgent):
    def __init__(self, state_size, action_size, n_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # Device Selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Centralized Model featuring multiple Actors and Estimators
        self.model = ACNet(state_size, action_size, n_agents).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        
        # Optimized with independent gradient descent for each twin critic
        self.critic1_optimizer = optim.Adam(
            list(self.model.critic_net1.parameters()) + list(self.model.value_head1.parameters()), 
            lr=Config.lr
        )
        self.critic2_optimizer = optim.Adam(
            list(self.model.critic_net2.parameters()) + list(self.model.value_head2.parameters()), 
            lr=Config.lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.model.actors.parameters()) + list(self.model.action_estimators.parameters()), 
            lr=Config.lr
        )

        # Replay Buffer configured for joint transitions
        self.memory = ReplayBuffer(Config.memory_size)
        self.tau = 0.005 # Target network soft update parameter
        
        # Step data buffering to form joint transitions
        self._current_step_data = {}

    def act(self, state, agent_idx):
        """
        Action selection for a single agent using Categorical sampling.
        State: Individual observation from the environment (18-element vector).
        """
        # Convert state to tensor with batch dimension [1, state_size]
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Directly call the specific actor head for this agent index to get logits
            logits = self.model.actors[agent_idx](state_t)
            # Use Categorical distribution for sampling
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            
        return action

    def estimate(self, state, agent_idx):
        """
        Sampling an estimated action using the expert estimator for a specific agent.
        """
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Directly call the specific estimator head for this agent index to get logits
            logits = self.model.action_estimators[agent_idx](state_t)
            # Use Categorical distribution for sampling
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            
        return action

    def step(self, state, action, reward, next_state, done, agent_idx):
        """
        Record individual transitions into a temporary buffer and push as joint when ready.
        Note: The critic is centralized, needing joint observations and rewards.
        """
        self._current_step_data[agent_idx] = (state, action, reward, next_state, done)
        
        # Once all agents have provided their step data, form and store the joint transition
        if len(self._current_step_data) == self.n_agents:
            joint_s = np.array([self._current_step_data[i][0] for i in range(self.n_agents)])
            joint_a = np.array([self._current_step_data[i][1] for i in range(self.n_agents)])
            joint_r = np.array([self._current_step_data[i][2] for i in range(self.n_agents)])
            joint_ns = np.array([self._current_step_data[i][3] for i in range(self.n_agents)])
            joint_d = np.array([self._current_step_data[i][4] for i in range(self.n_agents)])
            
            # Store joint transition (S, A, R, S', D)
            # sum(joint_r) is the joint reward, sum(joint_d) or any(joint_d) logic for terminality
            self.memory.add(joint_s, joint_a, joint_r, joint_ns, joint_d)
            self._current_step_data = {}
            
            if len(self.memory) > Config.batch_size:
                self.learn()

    def learn(self):
        """
        Update the centralized Q/Value function using the joint target formula:
        Q(s,a) = r_joint + gamma * (Q(s',a') - ln pi(a'|s'))
        Note: The model returns 2 global values representing twin critics.
        """
        if len(self.memory) < Config.batch_size:
            return

        # Sample joint transitions
        states, actions, rewards, next_states, dones = self.memory.sample(Config.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)       # [batch, n_agents, state_size]
        next_states_t = torch.FloatTensor(next_states).to(self.device) # [batch, n_agents, state_size]
        rewards_t = torch.FloatTensor(rewards).to(self.device).sum(dim=1, keepdim=True) # [batch, 1] (Joint Reward)
        dones_t = torch.FloatTensor(dones).to(self.device).max(dim=1, keepdim=True)[0] # [batch, 1] (Joint Terminal)
        
        # 1. Update Global Critics (the 2 values)
        with torch.no_grad():
            # Get next actions and their joint entropy from current policy
            policies_next, _, (q1_target_next, q2_target_next) = self.target_model(next_states_t)
            # q1_target_next, q2_target_next: [batch, 1]
            
            # Sum of log-probs for joint policy entropy
            dist_next = Categorical(probs=policies_next)
            actions_next = dist_next.sample() # [batch, n_agents]
            log_pi_next_total = dist_next.log_prob(actions_next).sum(dim=1, keepdim=True) # [batch, 1]
            
            # Soft target calculation using provided formula: r + gamma * (Q - alpha * ln pi)
            q_min_next = torch.min(q1_target_next, q2_target_next)
            q_targets = rewards_t + Config.gamma * (1 - dones_t) * (q_min_next - Config.alpha * log_pi_next_total)

        # Get current global predictions (the 2 values)
        _, _, (q1, q2) = self.model(states_t)
        
        # Separated Gradient Descent for Critic 1
        loss1 = F.mse_loss(q1, q_targets)
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        self.critic1_optimizer.step()
        
        # Separated Gradient Descent for Critic 2
        loss2 = F.mse_loss(q2, q_targets)
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        self.critic2_optimizer.step()
        
        # 2. Update Actors (PPO Policy Gradient)
        ppo_clip = getattr(Config, 'ppo_clip', 0.2)
        ppo_epochs = getattr(Config, 'ppo_epochs', 4)
        entropy_coef = getattr(Config, 'entropy_coef', 0.01)

        with torch.no_grad():
            _, _, (q1_det, q2_det) = self.model(states_t)
            q_min_det = torch.min(q1_det, q2_det) # [batch, 1]
            
            # Compute advantage
            advantage = (q_targets - q_min_det).detach() # [batch, 1]
            
            # Get old policy from target model for the PPO ratio
            old_policies, _, _ = self.target_model(states_t) # [batch, n_agents, action_size]
            actions_t = torch.LongTensor(actions).to(self.device) # [batch, n_agents]
            
            # Gather old log probabilities for the actions actually taken
            old_action_probs = old_policies.gather(2, actions_t.unsqueeze(-1)).squeeze(-1) # [batch, n_agents]
            old_log_probs = torch.log(old_action_probs + 1e-8) # [batch, n_agents]

        for _ in range(ppo_epochs):
            policies, _, _ = self.model(states_t) # [batch, n_agents, action_size]
            
            # Gather current log probabilities for the actions taken
            action_probs = policies.gather(2, actions_t.unsqueeze(-1)).squeeze(-1) # [batch, n_agents]
            log_probs = torch.log(action_probs + 1e-8) # [batch, n_agents]
            
            # Calculate ratio: pi_theta / pi_old
            ratios = torch.exp(log_probs - old_log_probs) # [batch, n_agents]
            
            # Surrogate objective with PPO clipping
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantage
            
            # Entropy bonus for exploration
            entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=-1).mean()
            
            # Final actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Soft Update Target Network
        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """
        Save the entire multi-agent system state.
        """
        if not os.path.exists(Config.checkpoint_dir):
            os.makedirs(Config.checkpoint_dir)
        
        save_path = os.path.join(Config.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, save_path)

    def load(self, filename):
        """
        Load the saved multi-agent system state.
        """
        path = os.path.join(Config.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
