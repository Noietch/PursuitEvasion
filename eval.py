"""
Unified evaluation script for pursuit-evasion environment.
Supports all registered policies.
"""

import numpy as np
import argparse
from env import PursuitEvasionEnv
from policy import POLICY_REGISTRY, create_policy, is_trainable


def evaluate(env, policy, num_episodes=10, verbose=True):
    """
    Evaluate policy over multiple episodes.

    Returns:
        aggregated: Aggregated statistics
        all_stats: List of per-episode statistics
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
    parser = argparse.ArgumentParser(description='Evaluate policy on Pursuit-Evasion')
    parser.add_argument('--policy', type=str, default='rush',
                        choices=list(POLICY_REGISTRY.keys()),
                        help='Policy to evaluate')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (required for trainable policies)')
    parser.add_argument('--env-config', type=str, default='configs/env.json')
    parser.add_argument('--swarm-config', type=str, default='configs/swarm.json')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'

    print("=" * 60)
    print("Pursuit-Evasion Evaluation")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    if args.model:
        print(f"Model: {args.model}")
    print("=" * 60)

    env = PursuitEvasionEnv(
        env_config_path=args.env_config,
        swarm_config_path=args.swarm_config,
        max_steps=args.max_steps
    )
    print(f"Num evaders: {env.num_evaders}, Num pursuers: {env.num_pursuers}")

    policy = create_policy(args.policy, env, device=device, hidden_dim=args.hidden_dim)

    if is_trainable(args.policy):
        if args.model is None:
            print(f"Warning: {args.policy} is trainable but no --model specified. Using untrained policy.")
        else:
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

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
