"""
Main execution loop for pursuit-evasion RL environment.
Runs episodes and collects statistics.
"""

import numpy as np
import time
import argparse
from env import PursuitEvasionEnv
from policy import RushPolicy


def run_episode(env, policy, verbose=True):
    """
    Run a single episode with the given policy.

    Args:
        env: PursuitEvasionEnv instance
        policy: Policy instance
        verbose: Whether to print step-by-step info

    Returns:
        dict: Episode statistics
    """
    obs, info = env.reset()
    policy.reset()

    episode_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Get action from policy
        action = policy.get_action(obs)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        if verbose and episode_steps % 50 == 0:
            print(f"  Step {episode_steps}: "
                  f"Winning={info['evaders_winning']}, "
                  f"Captured={info['evaders_captured']}, "
                  f"Reward={reward:.2f}")

    # Collect episode statistics
    stats = {
        'total_reward': episode_reward,
        'total_steps': episode_steps,
        'result': info.get('result', 'unknown'),
        'evaders_winning': info.get('evaders_winning', 0),
        'evaders_captured': info.get('evaders_captured', 0)
    }

    return stats


def run_evaluation(env, policy, num_episodes=10):
    """
    Evaluate policy over multiple episodes.

    Args:
        env: PursuitEvasionEnv instance
        policy: Policy instance
        num_episodes: Number of episodes to run

    Returns:
        dict: Aggregated statistics
    """
    all_stats = []

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        stats = run_episode(env, policy, verbose=True)
        all_stats.append(stats)

        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Result: {stats['result']}")
        print(f"  Total Reward: {stats['total_reward']:.2f}")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Evaders Winning: {stats['evaders_winning']}")
        print(f"  Evaders Captured: {stats['evaders_captured']}")

    # Aggregate statistics
    aggregated = {
        'mean_reward': np.mean([s['total_reward'] for s in all_stats]),
        'std_reward': np.std([s['total_reward'] for s in all_stats]),
        'mean_steps': np.mean([s['total_steps'] for s in all_stats]),
        'win_rate': np.mean([s['result'] == 'win' for s in all_stats]),
        'loss_rate': np.mean([s['result'] == 'loss' for s in all_stats]),
        'mean_winning_count': np.mean([s['evaders_winning'] for s in all_stats])
    }

    return aggregated, all_stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Pursuit-Evasion RL Environment')
    parser.add_argument('--policy', type=str, default='rush',
                        choices=['rush'], help='Policy to use')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=10000000,
                        help='Maximum steps per episode')
    parser.add_argument('--env-config', type=str, default='configs/env.json',
                        help='Path to environment config')
    parser.add_argument('--swarm-config', type=str, default='configs/swarm.json',
                        help='Path to swarm config')

    args = parser.parse_args()

    print("=" * 60)
    print("Pursuit-Evasion RL Environment")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Environment Config: {args.env_config}")
    print(f"Swarm Config: {args.swarm_config}")
    print("=" * 60)

    # Initialize environment
    print("\nInitializing environment...")
    env = PursuitEvasionEnv(
        env_config_path=args.env_config,
        swarm_config_path=args.swarm_config,
        max_steps=args.max_steps
    )
    print("Environment initialized successfully!")
    print(f"  Num evaders: {env.num_evaders}")
    print(f"  Num pursuers: {env.num_pursuers}")
    print(f"  Max evader velocity: {env.max_evader_vel}")
    print(f"  Target center: {env.target_center}")

    # Initialize policy
    print(f"\nInitializing {args.policy} policy...")
    if args.policy == 'rush':
        policy = RushPolicy(num_evaders=env.num_evaders, max_vel=env.max_evader_vel)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    print("Policy initialized successfully!")

    # Run evaluation
    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print("(Make sure main_pyqt.exe simulator is running!)")
    start_time = time.time()

    try:
        aggregated, all_stats = run_evaluation(env, policy, num_episodes=args.episodes)
        elapsed_time = time.time() - start_time

        # Print final summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {args.episodes}")
        print(f"Total Time: {elapsed_time:.2f}s")
        print(f"\nPerformance Metrics:")
        print(f"  Mean Reward: {aggregated['mean_reward']:.2f} +/- {aggregated['std_reward']:.2f}")
        print(f"  Mean Steps: {aggregated['mean_steps']:.1f}")
        print(f"  Win Rate: {aggregated['win_rate']*100:.1f}%")
        print(f"  Loss Rate: {aggregated['loss_rate']*100:.1f}%")
        print(f"  Mean Winning Evaders: {aggregated['mean_winning_count']:.2f}")
        print("=" * 60)

    except TimeoutError as e:
        print(f"\nError: {e}")
        print("Make sure the simulator (main_pyqt.exe) is running!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        # Clean up
        env.close()
        print("\nEnvironment closed. Done!")


if __name__ == "__main__":
    main()
