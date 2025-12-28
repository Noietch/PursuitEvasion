"""
Evaluation script for trained IPPO models.
"""

import numpy as np
import argparse
import time
from env import PursuitEvasionEnv
from policy.ippo_policy import IPPOPolicy


def evaluate(env, policy, num_episodes=10, verbose=True):
    """
    Evaluate trained policy.

    Returns:
        stats: Aggregated statistics
    """
    all_stats = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        policy.reset()

        episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            if verbose and episode_steps % 100 == 0:
                print(f"  Step {episode_steps}: "
                      f"Winning={info['evaders_winning']}, "
                      f"Captured={info['evaders_captured']}")

        stats = {
            'total_reward': episode_reward,
            'total_steps': episode_steps,
            'result': info.get('result', 'unknown'),
            'evaders_winning': info.get('evaders_winning', 0),
            'evaders_captured': info.get('evaders_captured', 0)
        }
        all_stats.append(stats)

        if verbose:
            print(f"\nEpisode {ep + 1}/{num_episodes}:")
            print(f"  Result: {stats['result']}")
            print(f"  Reward: {stats['total_reward']:.2f}")
            print(f"  Steps: {stats['total_steps']}")
            print(f"  Winning: {stats['evaders_winning']}")
            print(f"  Captured: {stats['evaders_captured']}")

    # Aggregate
    aggregated = {
        'mean_reward': np.mean([s['total_reward'] for s in all_stats]),
        'std_reward': np.std([s['total_reward'] for s in all_stats]),
        'mean_steps': np.mean([s['total_steps'] for s in all_stats]),
        'win_rate': np.mean([s['result'] == 'win' for s in all_stats]),
        'mean_winning': np.mean([s['evaders_winning'] for s in all_stats]),
        'mean_captured': np.mean([s['evaders_captured'] for s in all_stats])
    }

    return aggregated, all_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate IPPO on Pursuit-Evasion')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--env-config', type=str, default='configs/env.json')
    parser.add_argument('--swarm-config', type=str, default='configs/swarm.json')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'

    print("=" * 60)
    print("IPPO Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Initialize
    env = PursuitEvasionEnv(
        env_config_path=args.env_config,
        swarm_config_path=args.swarm_config,
        max_steps=args.max_steps
    )

    policy = IPPOPolicy(
        num_evaders=env.num_evaders,
        num_pursuers=env.num_pursuers,
        max_vel=env.max_evader_vel,
        device=device
    )
    policy.load(args.model)
    print("Model loaded successfully!")

    print("\n(Make sure main_pyqt.exe simulator is running!)")
    print("-" * 60)

    try:
        aggregated, _ = evaluate(env, policy, num_episodes=args.episodes)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Mean Reward: {aggregated['mean_reward']:.2f} +/- {aggregated['std_reward']:.2f}")
        print(f"Mean Steps: {aggregated['mean_steps']:.1f}")
        print(f"Win Rate: {aggregated['win_rate'] * 100:.1f}%")
        print(f"Mean Evaders Winning: {aggregated['mean_winning']:.2f}")
        print(f"Mean Evaders Captured: {aggregated['mean_captured']:.2f}")
        print("=" * 60)

    except TimeoutError as e:
        print(f"\nError: {e}")
        print("Make sure the simulator (main_pyqt.exe) is running!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
