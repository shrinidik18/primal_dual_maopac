"""
Wrapper to fix MPE action format issues
The off-policy MPE expects specific nested action format
"""

import numpy as np
from gym import spaces


class MPEActionWrapper:
    """
    Wrapper to convert between integer actions and MPE's expected format.
    
    MPE's _set_action expects: action[0][1] and action[0][2] for movement
    This means action should be: [[comm, left, right, down, up]]
    or more generally: [[discrete_action_vector]]
    """
    
    @staticmethod
    def int_to_mpe_action(action_int, action_space):
        """
        Convert integer action to MPE format.
        
        For Discrete(5) in simple_spread:
        - 0: no action
        - 1: move left
        - 2: move right  
        - 3: move down
        - 4: move up
        
        Returns: [[no_op, left, right, down, up]]
        """
        if isinstance(action_space, spaces.Discrete):
            n = action_space.n
            
            # Create action vector
            action_vec = np.zeros(n, dtype=np.float32)
            action_vec[action_int] = 1.0
            
            # Wrap in nested list structure
            return [action_vec.tolist()]
        
        elif isinstance(action_space, spaces.MultiDiscrete):
            # For MultiDiscrete, each dimension gets its own encoding
            nvec = action_space.nvec
            action_vecs = []
            
            for i, n in enumerate(nvec):
                vec = np.zeros(n, dtype=np.float32)
                # Extract the i-th component from action_int
                component = (action_int // np.prod(nvec[:i], dtype=int)) % n
                vec[component] = 1.0
                action_vecs.append(vec.tolist())
            
            return action_vecs
        
        else:
            # Unsupported action space
            return action_int
    
    @staticmethod
    def convert_actions_for_env(actions, action_spaces):
        """
        Convert list of integer actions to MPE format for all agents.
        
        Args:
            actions: List of integers [a1, a2, ..., an]
            action_spaces: List of gym.spaces for each agent
            
        Returns:
            List of formatted actions for MPE
        """
        formatted_actions = []
        
        for action, action_space in zip(actions, action_spaces):
            # If already in correct format, keep it
            if isinstance(action, list):
                formatted_actions.append(action)
            else:
                # Convert integer to MPE format
                formatted = MPEActionWrapper.int_to_mpe_action(action, action_space)
                formatted_actions.append(formatted)
        
        return formatted_actions
    
    @staticmethod
    def test_action_format(env):
        """
        Test to find what format the environment expects.
        
        Args:
            env: MPE environment instance
            
        Returns:
            A valid action that works with env.step()
        """
        obs = env.reset()
        
        # Try the wrapped format
        test_action = MPEActionWrapper.int_to_mpe_action(2, env.action_space[0])
        test_actions = [test_action] * env.num_agents
        
        try:
            next_obs, rewards, dones, infos = env.step(test_actions)
            print(f"✓ Action format works!")
            print(f"  Format: {test_action}")
            print(f"  Rewards: {rewards}")
            return test_actions
        except Exception as e:
            print(f"✗ Action format failed: {e}")
            return None


# Quick test function
def test_wrapper():
    """Test the wrapper with MPE environment"""
    import sys
    from types import SimpleNamespace
    sys.path.append('./off-policy')
    
    from offpolicy.envs.mpe.MPE_Env import MPEEnv
    
    print("Testing MPE Action Wrapper")
    print("="*60)
    
    env = MPEEnv(SimpleNamespace(
        scenario_name='simple_spread',
        num_agents=3,
        episode_length=25,
        num_landmarks=3
    ))
    
    print(f"Action space: {env.action_space}")
    
    # Test conversion
    integer_actions = [2, 1, 4]
    print(f"\nInteger actions: {integer_actions}")
    
    formatted = MPEActionWrapper.convert_actions_for_env(integer_actions, env.action_space)
    print(f"Formatted actions: {formatted}")
    
    # Test with environment
    print(f"\nTesting with environment...")
    obs = env.reset()
    
    try:
        next_obs, rewards, dones, infos = env.step(formatted)
        print(f"✓ SUCCESS!")
        print(f"  Rewards: {rewards}")
        print(f"  Reward sum: {np.sum(rewards):.3f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == '__main__':
    test_wrapper()
