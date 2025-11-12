import numpy as np
import torch
import argparse
import os
import sys
from collections import defaultdict
import json
from types import SimpleNamespace
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


def evaluate(args):
    """Evaluate trained model"""
    
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
    
    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    print(f"\nLoading model from: {args.model_path}")
    algorithm.load(args.model_path)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    
    # Evaluation loop
    print(f"\nEvaluating for {args.num_eval_episodes} episodes...")
    
    for episode in range(args.num_eval_episodes):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = [False] * n_agents
        episode_success = False
        
        while not all(done):
            # Select actions (deterministic)
            actions, _ = algorithm.select_actions(obs, deterministic=args.deterministic)
            
            # Execute actions
            formatted_actions = algorithm.format_actions_for_env(actions, env.action_space)

            # Execute actions
            next_obs, rewards, dones, infos = env.step(formatted_actions)
            
            # Update tracking
            episode_reward += np.mean(rewards)
            episode_length += 1
            obs = next_obs
            done = dones
            
            # Check for success (environment-specific)
            if 'battle_won' in infos[0]:  # StarCraft
                episode_success = infos[0]['battle_won']
            
            if args.render:
                env.render()
            
            if episode_length >= args.max_episode_length:
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_success:
            success_rate.append(1.0)
        else:
            success_rate.append(0.0)
        
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode+1}/{args.num_eval_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_success = np.mean(success_rate) if success_rate else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {args.num_eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    if success_rate:
        print(f"Success Rate: {mean_success*100:.1f}%")
    print("="*50)
    
    # Save results
    if args.save_results:
        results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths]
        }
        if success_rate:
            results['success_rate'] = float(mean_success)
            results['episode_successes'] = [int(s) for s in success_rate]
        
        save_path = os.path.join(os.path.dirname(args.model_path), 'eval_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--env_name', type=str, default='mpe_simple_spread',
                       help='Environment name')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents (for MPE)')
    
    # Evaluation
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num_eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--max_episode_length', type=int, default=100,
                       help='Maximum episode length')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save evaluation results to JSON')
    
    # Algorithm hyperparameters (must match training)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_dual', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--use_conv', action='store_true')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N episodes')
    
    # Device
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    
    evaluate(args)
