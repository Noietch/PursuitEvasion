"""
Unified training script for pursuit-evasion environment.
Supports all trainable policies.
"""

import os
import argparse
from datetime import datetime
from env import PursuitEvasionEnv
from policy import POLICY_REGISTRY, TRAINABLE_POLICIES, create_policy, is_trainable


def main():
    parser = argparse.ArgumentParser(description='Train policy on Pursuit-Evasion')
    parser.add_argument('--policy', type=str, default='ippo',
                        choices=list(TRAINABLE_POLICIES),
                        help='Trainable policy to use')

    # Environment
    parser.add_argument('--env-config', type=str, default='configs/env.json')
    parser.add_argument('--swarm-config', type=str, default='configs/swarm.json')
    parser.add_argument('--max-steps', type=int, default=1000)

    # Training
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--episodes-per-iter', type=int, default=10)
    parser.add_argument('--update-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)

    # PPO hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--eps-clip', type=float, default=0.2)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--load-model', type=str, default=None)

    args = parser.parse_args()

    if not is_trainable(args.policy):
        print(f"Error: {args.policy} is not a trainable policy.")
        print(f"Trainable policies: {list(TRAINABLE_POLICIES)}")
        return

    device = 'cuda' if args.cuda else 'cpu'

    print("=" * 60)
    print(f"Training {args.policy.upper()} on Pursuit-Evasion")
    print("=" * 60)
    print(f"Iterations: {args.num_iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"Device: {device}")
    print("=" * 60)

    env = PursuitEvasionEnv(
        env_config_path=args.env_config,
        swarm_config_path=args.swarm_config,
        max_steps=args.max_steps
    )
    print(f"Num evaders: {env.num_evaders}, Num pursuers: {env.num_pursuers}")

    policy = create_policy(args.policy, env, device=device, hidden_dim=args.hidden_dim)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.policy}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSave directory: {save_dir}")
    print("(Make sure main_pyqt.exe simulator is running!)")
    print("-" * 60)

    try:
        policy.train(env, args, save_dir)

    except TimeoutError as e:
        print(f"\nError: {e}")
        print("Make sure the simulator (main_pyqt.exe) is running!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    finally:
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
