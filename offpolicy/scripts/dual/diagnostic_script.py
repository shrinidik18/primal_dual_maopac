"""
Diagnostic script to test environment and action formatting
Run this to verify everything works before training
"""

import numpy as np
import torch
import sys
from types import SimpleNamespace

sys.path.append('./off-policy')

from offpolicy.envs.mpe.MPE_Env import MPEEnv


def test_environment_baseline():
    """Test 1: Environment works with random actions"""
    print("="*60)
    print("TEST 1: Environment Baseline (Random Policy)")
    print("="*60)
    
    env = MPEEnv(SimpleNamespace(
        scenario_name='simple_spread',
        num_agents=3,
        episode_length=25,
        num_landmarks=3
    ))
    
    print(f"Number of agents: {env.num_agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs = env.reset()
    print(f"\nInitial observation shape: {[o.shape for o in obs]}")
    
    # Run 5 random steps
    total_reward = 0
    for step in range(5):
        # Test: Use env.action_space.sample() directly first
        actions = [env.action_space[i].sample() for i in range(env.num_agents)]
        
        print(f"\nStep {step + 1}:")
        print(f"  Actions (raw sample): {actions}")
        print(f"  Action types: {[type(a) for a in actions]}")
        print(f"  Action shapes: {[np.array(a).shape if hasattr(a, '__len__') else 'scalar' for a in actions]}")
        
        next_obs, rewards, dones, infos = env.step(actions)
        print(f"  Rewards: {rewards}")
        print(f"  Dones: {dones}")
        
        total_reward += np.sum(rewards)
        obs = next_obs
        
        if all(dones):
            break
    
    print(f"\nTotal reward (5 steps, random): {total_reward:.2f}")
    print("✓ Environment works!\n")
    
    return env


def test_action_formatting():
    """Test 2: Action formatting"""
    print("="*60)
    print("TEST 2: Action Formatting")
    print("="*60)
    
    env = MPEEnv(SimpleNamespace(
        scenario_name='simple_spread',
        num_agents=3,
        episode_length=25,
        num_landmarks=3
    ))
    
    from offpolicy.algorithms.shri.pd_marl_algorithm import PrimalDualMARL
    
    # Create dummy algorithm
    algorithm = PrimalDualMARL(
        n_agents=3,
        obs_dims=[18, 18, 18],
        action_dims=[5, 5, 5],
        hidden_dim=64,
        device='cpu'
    )
    
    obs = env.reset()
    
    # Test 1: Policy actions
    actions, probs = algorithm.select_actions(obs)
    print(f"Raw policy actions: {actions}")
    print(f"Action types: {[type(a) for a in actions]}")
    
    # Test 2: Formatted actions
    formatted = algorithm.format_actions_for_env(actions, env.action_space)
    print(f"Formatted actions: {formatted}")
    print(f"Formatted types: {[type(a) for a in formatted]}")
    
    # Test 3: Can environment accept them?
    try:
        next_obs, rewards, dones, infos = env.step(formatted)
        print(f"✓ Environment accepted formatted actions!")
        print(f"  Rewards: {rewards}")
    except Exception as e:
        print(f"✗ Error with formatted actions: {e}")
        
        # Try with raw integers
        print("\nTrying raw integers...")
        int_actions = [int(a) for a in actions]
        print(f"Integer actions: {int_actions}")
        try:
            next_obs, rewards, dones, infos = env.step(int_actions)
            print(f"✓ Environment accepted integer actions!")
            print(f"  Rewards: {rewards}")
        except Exception as e2:
            print(f"✗ Error with integer actions: {e2}")
    
    print()


def test_training_step():
    """Test 3: Single training step"""
    print("="*60)
    print("TEST 3: Single Training Step")
    print("="*60)
    
    env = MPEEnv(SimpleNamespace(
        scenario_name='simple_spread',
        num_agents=3,
        episode_length=25,
        num_landmarks=3
    ))
    
    from offpolicy.algorithms.shri.pd_marl_algorithm import PrimalDualMARL
    
    algorithm = PrimalDualMARL(
        n_agents=3,
        obs_dims=[18, 18, 18],
        action_dims=[5, 5, 5],
        hidden_dim=64,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_dual=1e-3,
        beta=0.1,
        device='cpu'
    )
    
    obs = env.reset()
    
    # Take one step
    actions, behavior_probs = algorithm.select_actions(obs)
    formatted = algorithm.format_actions_for_env(actions, env.action_space)
    next_obs, rewards, dones, infos = env.step(formatted)
    
    print(f"Observations: {[o.shape for o in obs]}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    print(f"Behavior probs shape: {[p.shape for p in behavior_probs]}")
    
    # Normalize dones
    dones = [bool(d[0]) if isinstance(d, (list, tuple, np.ndarray)) else bool(d) for d in dones]
    print(f"Dones (normalized): {dones}")
    
    # Test training step
    print("\nTesting training step...")
    try:
        algorithm.train_step(
            states=obs,
            actions=actions,
            rewards=rewards,
            next_states=next_obs,
            behavior_probs=behavior_probs,
            dones=dones
        )
        print("✓ Training step completed!")
        
        # Check consensus error
        consensus_err = 0
        for agent in algorithm.agents:
            for name, param in agent.critic.named_parameters():
                diff = param.data - algorithm.w_shared[name]
                consensus_err += torch.norm(diff).item()
        consensus_err /= algorithm.n_agents
        
        print(f"Consensus error: {consensus_err:.6f}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_short_episode():
    """Test 4: Short training episode"""
    print("="*60)
    print("TEST 4: Short Training Episode (10 steps)")
    print("="*60)
    
    env = MPEEnv(SimpleNamespace(
        scenario_name='simple_spread',
        num_agents=3,
        episode_length=25,
        num_landmarks=3
    ))
    
    from offpolicy.algorithms.shri.pd_marl_algorithm import PrimalDualMARL
    
    algorithm = PrimalDualMARL(
        n_agents=3,
        obs_dims=[18, 18, 18],
        action_dims=[5, 5, 5],
        hidden_dim=64,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_dual=1e-3,
        beta=0.1,
        device='cpu'
    )
    
    obs = env.reset()
    episode_reward = 0
    
    for step in range(10):
        actions, behavior_probs = algorithm.select_actions(obs)
        formatted = algorithm.format_actions_for_env(actions, env.action_space)
        next_obs, rewards, dones, infos = env.step(formatted)
        
        # Normalize dones
        dones = [bool(d[0]) if isinstance(d, (list, tuple, np.ndarray)) else bool(d) for d in dones]
        
        # Train
        algorithm.train_step(
            states=obs,
            actions=actions,
            rewards=rewards,
            next_states=next_obs,
            behavior_probs=behavior_probs,
            dones=dones
        )
        
        episode_reward += np.mean(rewards)
        obs = next_obs
        
        if step % 3 == 0:
            print(f"Step {step}: reward={np.mean(rewards):.3f}, cumulative={episode_reward:.3f}")
        
        if all(dones):
            break
    
    print(f"\nFinal episode reward: {episode_reward:.2f}")
    
    # Check if policy changed
    print("\nChecking if policy is learning...")
    obs = env.reset()
    actions1, _ = algorithm.select_actions(obs)
    
    # Take a few more training steps
    for _ in range(5):
        actions, behavior_probs = algorithm.select_actions(obs)
        formatted = algorithm.format_actions_for_env(actions, env.action_space)
        next_obs, rewards, dones, infos = env.step(formatted)
        dones = [bool(d[0]) if isinstance(d, (list, tuple, np.ndarray)) else bool(d) for d in dones]
        algorithm.train_step(obs, actions, rewards, next_obs, behavior_probs, dones)
        obs = next_obs
    
    obs = env.reset()
    actions2, _ = algorithm.select_actions(obs)
    
    print(f"Actions before: {actions1}")
    print(f"Actions after:  {actions2}")
    
    if actions1 != actions2:
        print("✓ Policy is changing (learning happening)")
    else:
        print("⚠ Policy unchanged (might need more steps or higher LR)")
    
    print()


def main():
    """Run all diagnostics"""
    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS FOR PRIMAL-DUAL MARL")
    print("="*60 + "\n")
    
    try:
        test_environment_baseline()
    except Exception as e:
        print(f"✗ Environment test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        test_action_formatting()
    except Exception as e:
        print(f"✗ Action formatting test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        test_training_step()
    except Exception as e:
        print(f"✗ Training step test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        test_short_episode()
    except Exception as e:
        print(f"✗ Episode test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)
    print("\nIf all tests passed, you can start training!")
    print("Run: python pd_marl_trainer.py --env_name mpe_simple_spread --num_episodes 100")


if __name__ == '__main__':
    main()