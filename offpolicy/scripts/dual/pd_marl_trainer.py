import numpy as np
from offpolicy.algorithms.mqmix import algorithm
import torch
import argparse
import os
import sys
from collections import deque
import json
from types import SimpleNamespace
# Add offpolicy repo to path
# Assuming the repo is cloned: git clone https://github.com/marlbenchmark/off-policy.git
sys.path.append('./off-policy')

from offpolicy.envs.mpe.MPE_Env import MPEEnv
from offpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from offpolicy.envs.starcraft2.smac_maps import get_map_params


def make_env(env_name, args):
    """Create environment based on name"""
    if env_name.startswith('mpe'):
        # MPE environments: simple_spread, simple_reference, etc.
        scenario_name = env_name.replace('mpe_', '')

        # gather defaults from args or use safe fallbacks
        env_args = {
            'scenario_name': scenario_name,
            'benchmark': False,
            'num_agents': getattr(args, 'num_agents', 3),
            'episode_length': getattr(args, 'episode_length', 25),
            # many MPE scenarios use landmarks; default to 3 (like simple_spread)
            'num_landmarks': getattr(args, 'num_landmarks', 3),
            # optional world sizing used in some scenarios
            'world_length': getattr(args, 'world_length', None),
            'world_size': getattr(args, 'world_size', None),
            # keep other knobs if provided
            'render': getattr(args, 'render', False),
        }

        # drop None entries so we don't accidentally pass them
        env_args = {k: v for k, v in env_args.items() if v is not None}

        # wrap into an object with attribute access expected by MPEEnv
        env = MPEEnv(SimpleNamespace(**env_args))
        
    elif env_name.startswith('smac'):
        # StarCraft environments: 3m, 2s3z, etc.
        map_name = env_name.replace('smac_', '')
        map_params = get_map_params(map_name)
        env_args = {
            'map_name': map_name,
            'step_mul': 8,
            'difficulty': '7'
        }
        env = StarCraft2Env(env_args)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env


def train(args):
    """Main training loop"""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print(f"Creating environment: {args.env_name}")
    env = make_env(args.env_name, args)
    
    # Get environment info
    n_agents = env.num_agents
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    
    if hasattr(env.action_space[0], 'n'):
        action_dims = [env.action_space[i].n for i in range(n_agents)]
    else:
        action_dims = [env.action_space[i].shape[0] for i in range(n_agents)]
    
    print(f"Number of agents: {n_agents}")
    print(f"Observation dimensions: {obs_dims}")
    print(f"Action dimensions: {action_dims}")
    
    # Create algorithm
    from offpolicy.algorithms.shri.pd_marl_algorithm import PrimalDualMARL
    
    algorithm = PrimalDualMARL(
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_dual=args.lr_dual,
        beta=args.beta,
        gamma=args.gamma,
        use_conv=args.use_conv,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    consensus_errors = []
    
    # Training loop
    print(f"\nStarting training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = [False] * n_agents
        
        while not all(done):
            # Select actions
            actions, behavior_probs = algorithm.select_actions(obs)
            
            # Execute actions
            # format actions to match env.action_space
            formatted_actions = algorithm.format_actions_for_env(actions, env.action_space)
            next_obs, rewards, dones, infos = env.step(formatted_actions)
            dones = [bool(d[0]) if isinstance(d, (list, tuple, np.ndarray)) else bool(d) for d in dones]

            # Update algorithm
            algorithm.train_step(
                states=obs,
                actions=actions,
                rewards=rewards,
                next_states=next_obs,
                behavior_probs=behavior_probs,
                dones=dones
            )
            
            # Update tracking
            episode_reward += np.mean(rewards)
            episode_length += 1
            obs = next_obs
            done = dones
            
            if episode_length >= args.max_episode_length:
                break
        
        # Compute consensus error
        consensus_err = 0
        for agent in algorithm.agents:
            for name, param in agent.critic.named_parameters():
                diff = param.data - algorithm.w_shared[name]
                consensus_err += torch.norm(diff).item()
        consensus_err /= n_agents
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        consensus_errors.append(consensus_err)
        
        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            avg_consensus = np.mean(consensus_errors[-args.log_interval:])
            
            print(f"Episode {episode+1}/{args.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Consensus Error: {avg_consensus:.4f}")
        
        # Save model
        if (episode + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'model_ep{episode+1}.pt')
            algorithm.save(save_path)
    
    # Save final model
    os.makedirs(args.save_dir, exist_ok=True)
    algorithm.save(os.path.join(args.save_dir, 'model_final.pt'))
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'consensus_errors': consensus_errors
    }
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print("\nTraining completed!")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--env_name', type=str, default='mpe_simple_spread',
                       help='Environment name (mpe_simple_spread, smac_3m, etc.)')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents (for MPE)')
    
    # Training
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--max_episode_length', type=int, default=100,
                       help='Maximum episode length')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # Algorithm hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--lr_actor', type=float, default=1e-3,
                       help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                       help='Critic learning rate')
    parser.add_argument('--lr_dual', type=float, default=1e-3,
                       help='Dual variable learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Penalty coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--use_conv', action='store_true',
                       help='Use convolutional layers')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log every N episodes')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save model every N episodes')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                       help='Directory to save models')
    
    # Device
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    train(args)
