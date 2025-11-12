import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
from scipy.ndimage import uniform_filter1d


def smooth(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


def plot_training_curves(metrics_path, save_dir=None):
    """Plot training curves from metrics file"""
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episode_rewards = metrics['episode_rewards']
    episode_lengths = metrics['episode_lengths']
    consensus_errors = metrics['consensus_errors']
    
    episodes = np.arange(len(episode_rewards))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Primal-Dual Multi-Agent Off-Policy Actor-Critic Training', 
                 fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(episodes, smooth(episode_rewards, 100), color='blue', 
            linewidth=2, label='Smoothed (100 eps)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Episode Rewards over Training', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    ax.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
    ax.plot(episodes, smooth(episode_lengths, 100), color='green', 
            linewidth=2, label='Smoothed (100 eps)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('Episode Lengths over Training', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Consensus Error
    ax = axes[1, 0]
    ax.plot(episodes, consensus_errors, alpha=0.3, color='red', label='Raw')
    ax.plot(episodes, smooth(consensus_errors, 100), color='red', 
            linewidth=2, label='Smoothed (100 eps)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Consensus Error', fontsize=12)
    ax.set_title('Critic Consensus Error over Training', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Rolling Average Reward (last 1000 episodes)
    ax = axes[1, 1]
    window_sizes = [100, 500, 1000]
    colors = ['blue', 'green', 'red']
    
    for window, color in zip(window_sizes, colors):
        if len(episode_rewards) >= window:
            rolling_avg = np.convolve(episode_rewards, 
                                     np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], rolling_avg, color=color, 
                   linewidth=2, label=f'Rolling avg ({window} eps)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Rolling Average Rewards', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"\nRewards:")
    print(f"  Initial (first 100): {np.mean(episode_rewards[:100]):.2f}")
    print(f"  Final (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Best: {np.max(episode_rewards):.2f}")
    print(f"  Worst: {np.min(episode_rewards):.2f}")
    print(f"\nConsensus Error:")
    print(f"  Initial (first 100): {np.mean(consensus_errors[:100]):.4f}")
    print(f"  Final (last 100): {np.mean(consensus_errors[-100:]):.4f}")
    print(f"  Min: {np.min(consensus_errors):.4f}")
    print("="*60)


def plot_evaluation_results(eval_path, save_dir=None):
    """Plot evaluation results"""
    
    # Load evaluation results
    with open(eval_path, 'r') as f:
        results = json.load(f)
    
    episode_rewards = results['episode_rewards']
    episode_lengths = results['episode_lengths']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Reward distribution
    ax = axes[0]
    ax.hist(episode_rewards, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_reward'], color='red', linestyle='--', 
              linewidth=2, label=f"Mean: {results['mean_reward']:.2f}")
    ax.set_xlabel('Episode Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Episode Rewards', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Rewards over episodes
    ax = axes[1]
    episodes = np.arange(len(episode_rewards))
    ax.plot(episodes, episode_rewards, marker='o', markersize=3, 
           alpha=0.6, color='blue')
    ax.axhline(results['mean_reward'], color='red', linestyle='--', 
              linewidth=2, label=f"Mean: {results['mean_reward']:.2f}")
    ax.fill_between(episodes, 
                    results['mean_reward'] - results['std_reward'],
                    results['mean_reward'] + results['std_reward'],
                    alpha=0.3, color='red', label='±1 std')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Rewards Across Evaluation Episodes', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'evaluation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation results saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    print(f"Number of Episodes: {len(episode_rewards)}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    if 'success_rate' in results:
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print("="*60)


def plot_comparison(metrics_paths, labels, save_dir=None):
    """Plot comparison of multiple training runs"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Comparison of Training Runs', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_paths)))
    
    for path, label, color in zip(metrics_paths, labels, colors):
        with open(path, 'r') as f:
            metrics = json.load(f)
        
        episodes = np.arange(len(metrics['episode_rewards']))
        
        # Plot rewards
        ax = axes[0]
        smoothed_rewards = smooth(metrics['episode_rewards'], 100)
        ax.plot(episodes, smoothed_rewards, color=color, linewidth=2, label=label)
        
        # Plot lengths
        ax = axes[1]
        smoothed_lengths = smooth(metrics['episode_lengths'], 100)
        ax.plot(episodes, smoothed_lengths, color=color, linewidth=2, label=label)
        
        # Plot consensus error
        ax = axes[2]
        smoothed_consensus = smooth(metrics['consensus_errors'], 100)
        ax.plot(episodes, smoothed_consensus, color=color, linewidth=2, label=label)
    
    # Configure axes
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('Episode Rewards', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Episode Length', fontsize=12)
    axes[1].set_title('Episode Lengths', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Episode', fontsize=12)
    axes[2].set_ylabel('Consensus Error', fontsize=12)
    axes[2].set_title('Consensus Error', fontsize=13, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='training',
                       choices=['training', 'evaluation', 'comparison'],
                       help='Plotting mode')
    parser.add_argument('--metrics_path', type=str,
                       help='Path to metrics.json file')
    parser.add_argument('--eval_path', type=str,
                       help='Path to eval_results.json file')
    parser.add_argument('--comparison_paths', nargs='+',
                       help='Paths to multiple metrics.json files for comparison')
    parser.add_argument('--comparison_labels', nargs='+',
                       help='Labels for comparison plots')
    parser.add_argument('--save_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.mode == 'training':
        if not args.metrics_path:
            raise ValueError("--metrics_path required for training mode")
        plot_training_curves(args.metrics_path, args.save_dir)
    
    elif args.mode == 'evaluation':
        if not args.eval_path:
            raise ValueError("--eval_path required for evaluation mode")
        plot_evaluation_results(args.eval_path, args.save_dir)
    
    elif args.mode == 'comparison':
        if not args.comparison_paths or not args.comparison_labels:
            raise ValueError("--comparison_paths and --comparison_labels required")
        if len(args.comparison_paths) != len(args.comparison_labels):
            raise ValueError("Number of paths and labels must match")
        plot_comparison(args.comparison_paths, args.comparison_labels, args.save_dir)
