import torch
import torch.nn as nn
import numpy as np
from collections import deque
import copy

class PrimalDualMARL:
    """Primal-Dual Multi-Agent Off-Policy Actor-Critic"""
    def __init__(self, n_agents, obs_dims, action_dims, hidden_dim=64,
                 lr_actor=1e-3, lr_critic=1e-3, lr_dual=1e-3,
                 beta=0.1, gamma=0.99, use_conv=False, device='cpu'):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.beta = beta
        self.device = device
        
        # Import agent class
        from offpolicy.algorithms.shri.pd_marl_networks import PrimalDualAgent
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            obs_dim = obs_dims[i] if isinstance(obs_dims, list) else obs_dims
            action_dim = action_dims[i] if isinstance(action_dims, list) else action_dims
            
            agent = PrimalDualAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_conv=use_conv,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                lr_dual=lr_dual,
                beta=beta,
                gamma=gamma,
                device=device
            )
            self.agents.append(agent)
        
        # Shared critic parameters (initialized as mean of all agents)
        self.w_shared = None
        self._initialize_shared_critic()
        
    def _initialize_shared_critic(self):
        """Initialize shared critic as average of all agent critics"""
        self.w_shared = {}
        
        # Get parameter names from first agent
        for name, param in self.agents[0].critic.named_parameters():
            # Average parameters across all agents
            avg_param = torch.stack([agent.critic.state_dict()[name] 
                                    for agent in self.agents]).mean(dim=0)
            self.w_shared[name] = avg_param.clone().to(self.device)
    
    def update_shared_critic(self):
        """Update shared critic as mean of all agent critics"""
        for name in self.w_shared.keys():
            params = torch.stack([agent.critic.state_dict()[name] 
                                 for agent in self.agents])
            self.w_shared[name] = params.mean(dim=0)
    
    def update_critics(self, states, actions, rewards, next_states, 
                      behavior_probs, dones):
        """Update critics with primal-dual penalties"""
        
        for i, agent in enumerate(self.agents):
            # Convert to tensors
            s = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            s_next = torch.FloatTensor(next_states[i]).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([rewards[i]]).to(self.device)
            done = torch.FloatTensor([dones[i]]).to(self.device)
            a = actions[i]
            
            # Compute importance ratio
            rho = agent.compute_importance_ratio(states[i], a, behavior_probs[i])
            rho = torch.tensor(float(rho), device=self.device, dtype=torch.float32)
            rho = torch.clamp(rho, 0.1, 10.0)  # Clip to avoid extreme values
            
            # Compute TD error: δ = r + γ*V(s') - V(s)
            with torch.no_grad():
                v_next = agent.critic(s_next).squeeze()
                delta = r.squeeze() + self.gamma * v_next * (1 - done.squeeze())
            
            v_current = agent.critic(s).squeeze()
            delta = delta - v_current  # Complete TD error
            
            # CORRECTED: Compute critic loss properly
            # Loss = -ρ * δ * V(s) + penalty terms
            # We want gradient: g = ρ * δ * ∇V(s)
            # So we minimize: -ρ * δ * V(s)
            
            agent.critic_optimizer.zero_grad()
            
            # Critic TD loss (detach delta to avoid backprop through it)
            td_loss = -(rho * delta.detach() * v_current)
            
            # Add primal-dual penalty
            penalty_loss = 0
            for name, param in agent.critic.named_parameters():
                dual_term = agent.dual[name]
                diff = param - self.w_shared[name]
                # Penalty: λ^T(w - w_sh) + β/2 ||w - w_sh||^2
                penalty_loss += torch.sum(dual_term * diff) + (self.beta / 2) * torch.sum(diff ** 2)
            
            # Total loss
            total_loss = td_loss + penalty_loss
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=10.0)
            
            agent.critic_optimizer.step()
    
    def update_actors(self, states, actions, rewards, next_states, 
                     behavior_probs, dones):
        """Update actors with policy gradient"""
        
        for i, agent in enumerate(self.agents):
            # Convert to tensors
            s = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            s_next = torch.FloatTensor(next_states[i]).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([rewards[i]]).to(self.device)
            done = torch.FloatTensor([dones[i]]).to(self.device)
            a = torch.LongTensor([actions[i]]).to(self.device)
            
            # Compute importance ratio
            rho = agent.compute_importance_ratio(states[i], actions[i], behavior_probs[i])
            rho = torch.FloatTensor([rho]).to(self.device)
            rho = torch.clamp(rho, 0.1, 10.0)  # Clip for stability
            
            # Compute TD error
            with torch.no_grad():
                v_current = agent.critic(s).squeeze()
                v_next = agent.critic(s_next).squeeze()
                delta = r.squeeze() + self.gamma * v_next * (1 - done.squeeze()) - v_current
            
            # Policy gradient: ρ * δ * ∇_θ log π(a|s)
            agent.actor_optimizer.zero_grad()
            log_prob = agent.actor.get_log_prob(s, a)
            
            # Loss = -ρ * δ * log π(a|s)
            actor_loss = -(rho * delta * log_prob)
            actor_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=10.0)
            
            agent.actor_optimizer.step()
    
    def update_duals(self):
        """Update dual variables: λ_i += α_λ * (w_i - w_sh)"""
        
        for agent in self.agents:
            for name, param in agent.critic.named_parameters():
                # λ_i += α_λ * (w_i - w_sh)
                with torch.no_grad():
                    diff = param.data - self.w_shared[name]
                    agent.dual[name] += agent.lr_dual * diff
    
    def train_step(self, states, actions, rewards, next_states, 
                   behavior_probs, dones):
        """Single training step"""
        
        # Update critics
        self.update_critics(states, actions, rewards, next_states, 
                          behavior_probs, dones)
        
        # Update shared critic
        self.update_shared_critic()
        
        # Update actors
        self.update_actors(states, actions, rewards, next_states, 
                          behavior_probs, dones)
        
        # Update duals
        self.update_duals()
    
    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents"""
        actions = []
        probs = []
        
        for i, agent in enumerate(self.agents):
            action = agent.select_action(observations[i], deterministic)
            actions.append(action)
            
            # Get action probabilities for importance sampling
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = agent.actor(obs_tensor).cpu().numpy()[0]
            probs.append(action_probs)
        
        return actions, probs
    
    def format_actions_for_env(self, actions, env_action_spaces):
        """
        Robust formatter: convert algorithm actions (scalars or small lists) into the
        flattened one-hot / component format expected by the MPE environment.

        This version is defensive: it never does `list(a)` on an int and will
        fall back to sensible representations when unsure.
        """
        import numpy as _np
        from gym import spaces

        formatted = []
        for i, a in enumerate(actions):
            act_space = env_action_spaces[i]

            # If already array-like, convert to list for inspection
            if isinstance(a, (list, tuple, _np.ndarray)):
                arr = _np.array(a)
                a_list = arr.tolist()
            else:
                a_list = None

            # Try to detect scalar-ness robustly
            is_scalar = False
            ai = None
            try:
                # works for python ints, numpy scalars, torch scalars
                ai = int(_np.asscalar(_np.array(a)))
                is_scalar = True
            except Exception:
                is_scalar = False

            # Case: Discrete -> return flattened one-hot
            if isinstance(act_space, spaces.Discrete):
                n = act_space.n
                if is_scalar:
                    one_hot = [0] * n
                    one_hot[ai % n] = 1
                    formatted.append(one_hot)
                else:
                    # if provided a list/ndarray of correct one-hot length, pass it through
                    if a_list is not None and len(a_list) == n:
                        formatted.append(a_list)
                    else:
                        # fallback: try to coerce to int then one-hot
                        try:
                            ai2 = int(a)
                            one_hot = [0] * n
                            one_hot[ai2 % n] = 1
                            formatted.append(one_hot)
                        except Exception:
                            # last resort: sample a default (all zeros with first 1)
                            one_hot = [0] * n
                            one_hot[0] = 1
                            formatted.append(one_hot)
                continue

            # Case: MultiDiscrete-like (has high/low arrays) -> produce flattened concatenated one-hots
            if hasattr(act_space, 'high') and hasattr(act_space, 'low'):
                highs = _np.array(act_space.high)
                lows = _np.array(act_space.low)
                sizes = (highs - lows + 1).astype(int)

                # scalar index => unravel; non-scalar => treat as component indices if possible
                if is_scalar:
                    total = int(_np.prod(sizes))
                    idx = ai % total
                    components = list(_np.unravel_index(idx, tuple(sizes)))
                else:
                    if a_list is not None:
                        # if user passed flattened one-hot of correct length, pass through
                        if len(a_list) == int(_np.sum(sizes)):
                            formatted.append(a_list)
                            continue
                        # if passed component indices like [i,j,...], use those
                        try:
                            components = list(map(int, a_list))
                        except Exception:
                            # fallback to zeros
                            components = [0] * len(sizes)
                    else:
                        # fallback: single int -> convert to zeros with first index 0
                        components = [0] * len(sizes)

                # build flattened one-hot
                flat = []
                for comp, s in zip(components, sizes):
                    one = [0] * int(s)
                    one[int(comp) % int(s)] = 1
                    flat.extend(one)
                formatted.append(flat)
                continue

            # Case: Tuple of spaces (fallback) -> try to build a list per subspace
            if isinstance(act_space, spaces.Tuple):
                # If scalar, repeat or coerce to ints for each subspace
                if is_scalar:
                    formatted.append([int(ai)] * len(act_space.spaces))
                else:
                    # if iterable, convert safely to list
                    try:
                        formatted.append(list(a))
                    except Exception:
                        # fall back to a single-int-in-list
                        try:
                            formatted.append([int(a)])
                        except Exception:
                            formatted.append([0] * len(act_space.spaces))
                continue

            # If we reach here, unknown space type (Box or other). Try to return appropriate array.
            if is_scalar:
                # coerce scalar into 1-element list (environment may handle)
                formatted.append(int(ai))
            else:
                # if iterable and not listable above, try safe coercion
                try:
                    formatted.append(list(a))
                except Exception:
                    # final fallback: pass-through original value
                    formatted.append(a)

        return formatted

    
    def save(self, path):
        """Save all agent models"""
        checkpoint = {
            'n_agents': self.n_agents,
            'agents': []
        }
        
        for agent in self.agents:
            agent_state = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'dual': agent.dual
            }
            checkpoint['agents'].append(agent_state)
        
        checkpoint['w_shared'] = self.w_shared
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load all agent models"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, agent_state in enumerate(checkpoint['agents']):
            self.agents[i].actor.load_state_dict(agent_state['actor'])
            self.agents[i].critic.load_state_dict(agent_state['critic'])
            self.agents[i].dual = agent_state['dual']
        
        self.w_shared = checkpoint['w_shared']
        print(f"Model loaded from {path}")