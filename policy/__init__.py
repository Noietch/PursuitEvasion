"""
Policy module for pursuit-evasion game.
"""

from .base_policy import BasePolicy
from .rush_policy import RushPolicy
from .ippo_policy import IPPOPolicy, PPOAgent
from .rule_based_policy import RuleBasedPolicy

__all__ = ['BasePolicy', 'RushPolicy', 'IPPOPolicy', 'PPOAgent', 'RuleBasedPolicy',
           'POLICY_REGISTRY', 'TRAINABLE_POLICIES', 'create_policy', 'is_trainable']

POLICY_REGISTRY = {
    'rush': RushPolicy,
    'ippo': IPPOPolicy,
    'rule': RuleBasedPolicy,
}

TRAINABLE_POLICIES = {'ippo'}


def create_policy(name, env, device='cpu', **kwargs):
    """
    Create a policy instance by name.

    Args:
        name: Policy name (rush, ippo, etc.)
        env: PursuitEvasionEnv instance
        device: Device for trainable policies
        **kwargs: Additional policy-specific arguments

    Returns:
        Policy instance
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICY_REGISTRY.keys())}")

    policy_cls = POLICY_REGISTRY[name]

    if name == 'rush':
        return policy_cls(num_evaders=env.num_evaders, max_vel=env.max_evader_vel)
    elif name == 'ippo':
        return policy_cls(
            num_evaders=env.num_evaders,
            num_pursuers=env.num_pursuers,
            num_obstacles=env.num_obstacles,
            max_vel=env.max_evader_vel,
            hidden_dim=kwargs.get('hidden_dim', 128),
            device=device
        )
    elif name == 'rule':
        # Extract space bounds from env config
        space = env.env_config.get('space', [[1000, 10], [1000, 1000], [10, 1000], [10, 10]])
        x_coords = [p[0] for p in space]
        y_coords = [p[1] for p in space]
        space_bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

        # Filter kwargs to only include RuleBasedPolicy parameters
        rule_kwargs = {k: v for k, v in kwargs.items() if k in [
            'attract_gain', 'repulse_gain', 'obstacle_gain', 'boundary_gain',
            'danger_radius', 'obstacle_radius', 'boundary_margin',
            'teammate_separation', 'cooperation_radius'
        ]}

        return policy_cls(
            num_evaders=env.num_evaders,
            max_vel=env.max_evader_vel,
            num_pursuers=env.num_pursuers,
            pursuer_max_vel=env.max_pursuer_vel,
            obstacle_vertices=env.env_config.get('obs', []),
            space_bounds=space_bounds,
            **rule_kwargs
        )
    else:
        return policy_cls(num_evaders=env.num_evaders, max_vel=env.max_evader_vel)


def is_trainable(name):
    """Check if a policy is trainable."""
    return name in TRAINABLE_POLICIES
