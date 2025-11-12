import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """Actor network for policy learning"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64, use_conv=False):
        super(ActorNetwork, self).__init__()
        self.use_conv = use_conv
        
        if use_conv:
            # For image-based observations (e.g., StarCraft)
            self.conv1 = nn.Conv2d(obs_dim[0], 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            conv_out_size = 64 * obs_dim[1] * obs_dim[2]
            self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        else:
            # For vector observations (MPE)
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        if self.use_conv:
            x = F.relu(self.conv1(obs))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(obs))
        
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)
    
    def get_log_prob(self, obs, action):
        probs = self.forward(obs)
        log_probs = torch.log(probs + 1e-8)
        return log_probs.gather(1, action.long().unsqueeze(-1)).squeeze(-1)


class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    def __init__(self, obs_dim, hidden_dim=64, use_conv=False):
        super(CriticNetwork, self).__init__()
        self.use_conv = use_conv
        
        if use_conv:
            # For image-based observations
            self.conv1 = nn.Conv2d(obs_dim[0], 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            conv_out_size = 64 * obs_dim[1] * obs_dim[2]
            self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        else:
            # For vector observations
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        if self.use_conv:
            x = F.relu(self.conv1(obs))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(obs))
        
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PrimalDualAgent:
    """Single agent with primal-dual actor-critic"""
    def __init__(self, agent_id, obs_dim, action_dim, hidden_dim=64, 
                 use_conv=False, lr_actor=1e-3, lr_critic=1e-3, lr_dual=1e-3, 
                 beta=0.1, gamma=0.99, device='cpu'):
        self.agent_id = agent_id
        self.gamma = gamma
        self.beta = beta
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, use_conv).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_dim, use_conv).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Dual variable (initialized to zeros matching critic parameters)
        self.dual = {name: torch.zeros_like(param.data).to(device) 
                     for name, param in self.critic.named_parameters()}
        self.lr_dual = lr_dual
        
    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().item()
    
    def get_value(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value.cpu().item()
    
    def compute_importance_ratio(self, obs, action, behavior_probs):
        """Compute importance ratio ρ = π(a|s) / μ(a|s)"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            target_probs = self.actor(obs_tensor)
            target_prob = target_probs[0, action].item()
            behavior_prob = behavior_probs[action]
        return target_prob / (behavior_prob + 1e-8)
