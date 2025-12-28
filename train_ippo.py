"""
Training script for IPPO on pursuit-evasion environment.
"""

import numpy as np
import torch
import argparse
import time
import os
from datetime import datetime
from tqdm import tqdm
from env import PursuitEvasionEnv
from policy.ippo_policy import IPPOPolicy


def collect_episode(env, policy, max_steps=1000):
    """
    Collect a single episode trajectory.

    Returns:
        trajectories: List of trajectory dicts, one per evader
        episode_stats: Episode statistics
    """
    obs, info = env.reset()
    policy.reset()

    num_evaders = policy.num_evaders
    trajectories = [{
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': [],
        'values': [],
        'dones': []
    } for _ in range(num_evaders)]

    episode_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and episode_steps < max_steps:
        # Get action with log probs
        action, log_probs, local_obs_list = policy.get_action_with_log_prob(obs)
        values = policy.get_values(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Calculate individual rewards
        evader_rewards = calculate_individual_rewards(obs, next_obs, reward, policy.num_evaders)

        # Store transitions for each evader
        for i in range(num_evaders):
            captured = obs['evaders_captured'][i]
            winning = obs['evaders_winning'][i]

            if not (captured or winning):
                trajectories[i]['states'].append(local_obs_list[i])
                trajectories[i]['actions'].append(action[i] / policy.max_vel)  # Normalize
                trajectories[i]['log_probs'].append(log_probs[i])
                trajectories[i]['rewards'].append(evader_rewards[i])
                trajectories[i]['values'].append(values[i])
                trajectories[i]['dones'].append(terminated or truncated)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

    # Compute advantages and returns for each evader
    final_values = policy.get_values(obs)
    for i in range(num_evaders):
        if len(trajectories[i]['rewards']) > 0:
            advantages, returns = policy.agent.compute_gae(
                trajectories[i]['rewards'],
                trajectories[i]['values'],
                final_values[i],
                trajectories[i]['dones']
            )
            trajectories[i]['advantages'] = advantages
            trajectories[i]['returns'] = returns

    stats = {
        'total_reward': episode_reward,
        'total_steps': episode_steps,
        'result': info.get('result', 'unknown'),
        'evaders_winning': info.get('evaders_winning', 0),
        'evaders_captured': info.get('evaders_captured', 0)
    }

    return trajectories, stats


def calculate_individual_rewards(obs, next_obs, global_reward, num_evaders):
    """
    Calculate individual rewards for each evader.

    Reward structure per evader:
    - Progress toward target: +0.1 * distance_reduced
    - Reaching target: +10
    - Being captured: -5
    - Small time penalty: -0.01
    """
    rewards = np.zeros(num_evaders)
    target = obs['target_center']

    for i in range(num_evaders):
        prev_captured = obs['evaders_captured'][i]
        prev_winning = obs['evaders_winning'][i]
        curr_captured = next_obs['evaders_captured'][i]
        curr_winning = next_obs['evaders_winning'][i]

        if prev_captured or prev_winning:
            continue

        # Progress reward
        prev_dist = np.linalg.norm(obs['evaders_pos'][i] - target)
        curr_dist = np.linalg.norm(next_obs['evaders_pos'][i] - target)
        progress = prev_dist - curr_dist
        rewards[i] += 0.1 * progress

        # Win bonus
        if curr_winning and not prev_winning:
            rewards[i] += 10.0

        # Capture penalty
        if curr_captured and not prev_captured:
            rewards[i] -= 5.0

        # Time penalty
        rewards[i] -= 0.01

    return rewards


def train(args):
    """Main training loop."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(f"Using device: {device}")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"ippo_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize environment
    print("Initializing environment...")
    env = PursuitEvasionEnv(
        env_config_path=args.env_config,
        swarm_config_path=args.swarm_config,
        max_steps=args.max_steps
    )
    print(f"  Num evaders: {env.num_evaders}")
    print(f"  Num pursuers: {env.num_pursuers}")

    # Initialize policy
    print("Initializing IPPO policy...")
    policy = IPPOPolicy(
        num_evaders=env.num_evaders,
        num_pursuers=env.num_pursuers,
        max_vel=env.max_evader_vel,
        hidden_dim=args.hidden_dim,
        device=device
    )
    policy.agent.gamma = args.gamma
    policy.agent.lmbda = args.lmbda
    policy.agent.eps_clip = args.eps_clip

    # Load checkpoint if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        policy.load(args.load_model)

    # Training statistics
    all_rewards = []
    win_rates = []
    best_win_rate = 0.0

    print(f"\nStarting training for {args.num_episodes} episodes...")
    print("(Make sure main_pyqt.exe simulator is running!)")

    try:
        for iteration in range(args.num_iterations):
            iteration_rewards = []
            iteration_wins = []
            all_trajectories = []

            desc = f"Iteration {iteration + 1}/{args.num_iterations}"
            pbar = tqdm(range(args.episodes_per_iter), desc=desc)

            for ep in pbar:
                trajectories, stats = collect_episode(env, policy, max_steps=args.max_steps)

                iteration_rewards.append(stats['total_reward'])
                iteration_wins.append(1 if stats['result'] == 'win' else 0)

                # Accumulate trajectories
                for i, traj in enumerate(trajectories):
                    if 'advantages' in traj:
                        all_trajectories.append(traj)

                # Update progress bar
                if len(iteration_rewards) > 0:
                    avg_reward = np.mean(iteration_rewards[-min(10, len(iteration_rewards)):])
                    avg_win = np.mean(iteration_wins[-min(10, len(iteration_wins)):])
                    pbar.set_postfix({
                        'reward': f'{avg_reward:.1f}',
                        'win_rate': f'{avg_win:.2f}'
                    })

            # Update policy after collecting episodes
            if len(all_trajectories) > 0:
                policy.update(all_trajectories, epochs=args.update_epochs, batch_size=args.batch_size)

            # Log statistics
            mean_reward = np.mean(iteration_rewards)
            win_rate = np.mean(iteration_wins)
            all_rewards.extend(iteration_rewards)
            win_rates.append(win_rate)

            print(f"\nIteration {iteration + 1} Summary:")
            print(f"  Mean Reward: {mean_reward:.2f}")
            print(f"  Win Rate: {win_rate * 100:.1f}%")
            print(f"  Total Episodes: {(iteration + 1) * args.episodes_per_iter}")

            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                policy.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  New best model saved! Win rate: {win_rate * 100:.1f}%")

            # Periodic save
            if (iteration + 1) % args.save_interval == 0:
                policy.save(os.path.join(save_dir, f"model_iter_{iteration + 1}.pt"))

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    except TimeoutError as e:
        print(f"\nError: {e}")
        print("Make sure the simulator (main_pyqt.exe) is running!")

    finally:
        # Save final model
        policy.save(os.path.join(save_dir, "final_model.pt"))
        print(f"\nFinal model saved to {save_dir}")

        # Save training log
        log_path = os.path.join(save_dir, "training_log.npz")
        np.savez(log_path, rewards=all_rewards, win_rates=win_rates)
        print(f"Training log saved to {log_path}")

        env.close()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Train IPPO on Pursuit-Evasion')

    # Environment
    parser.add_argument('--env-config', type=str, default='env.json')
    parser.add_argument('--swarm-config', type=str, default='swarm.json')
    parser.add_argument('--max-steps', type=int, default=1000)

    # Training
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--episodes-per-iter', type=int, default=10)
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Total episodes (alternative to iterations)')
    parser.add_argument('--update-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)

    # PPO hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--load-model', type=str, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("IPPO Training for Pursuit-Evasion")
    print("=" * 60)
    print(f"Iterations: {args.num_iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Gamma: {args.gamma}")
    print(f"Lambda: {args.lmbda}")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
