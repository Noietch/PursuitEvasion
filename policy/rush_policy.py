"""
Full-speed rush policy for pursuit-evasion game.
Each evader moves at maximum velocity directly toward the target.
"""

import numpy as np
from .base_policy import BasePolicy


class RushPolicy(BasePolicy):
    """
    Full-speed rush policy: Each evader moves at maximum velocity
    directly toward the target area center.

    This is a simple baseline policy that ignores pursuers and obstacles.
    """

    def __init__(self, num_evaders: int = 10, max_vel: float = 4.0):
        """
        Initialize rush policy.

        Args:
            num_evaders: Number of evaders to control
            max_vel: Maximum velocity for evaders
        """
        super().__init__(num_evaders, max_vel)

    def get_action(self, observation: dict) -> np.ndarray:
        """
        Compute rush action: move at max speed toward target.

        Args:
            observation: Dictionary containing:
                - evaders_pos: (num_evaders, 2) current positions
                - target_center: (2,) target area center
                - evaders_captured: (num_evaders,) capture status
                - evaders_winning: (num_evaders,) winning status

        Returns:
            np.ndarray: Velocity commands of shape (num_evaders, 2)
        """
        evaders_pos = observation['evaders_pos']
        target_center = observation['target_center']
        captured = observation['evaders_captured'].astype(bool)
        winning = observation['evaders_winning'].astype(bool)

        # Calculate direction to target for each evader
        direction = target_center - evaders_pos  # Shape: (num_evaders, 2)

        # Calculate distance to target
        distance = np.linalg.norm(direction, axis=1, keepdims=True)  # Shape: (num_evaders, 1)

        # Normalize direction (avoid division by zero)
        distance = np.maximum(distance, 1e-6)
        normalized_direction = direction / distance

        # Set velocity to max_vel in target direction
        velocity = normalized_direction * self.max_vel

        # Zero velocity for captured or winning evaders
        inactive_mask = captured | winning
        velocity[inactive_mask] = 0

        return velocity.astype(np.float32)
