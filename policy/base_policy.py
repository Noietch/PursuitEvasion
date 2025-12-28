"""
Base policy class for pursuit-evasion game.
All policies should inherit from this class.
"""

import numpy as np
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """
    Abstract base class for all policies.

    All policies must implement the get_action method which takes
    an observation and returns an action.
    """

    def __init__(self, num_evaders: int = 10, max_vel: float = 4.0):
        """
        Initialize base policy.

        Args:
            num_evaders: Number of evaders to control
            max_vel: Maximum velocity for evaders
        """
        self.num_evaders = num_evaders
        self.max_vel = max_vel

    @abstractmethod
    def get_action(self, observation: dict) -> np.ndarray:
        """
        Get action from policy given observation.

        Args:
            observation: Dictionary observation from environment
                - evaders_pos: (num_evaders, 2) current positions
                - pursuers_pos: (num_pursuers, 2) pursuer positions
                - evaders_winning: (num_evaders,) winning status
                - evaders_captured: (num_evaders,) capture status
                - target_center: (2,) target area center

        Returns:
            np.ndarray: Action array of shape (num_evaders, 2)
        """
        pass

    def reset(self):
        """Reset policy internal state (if stateful)."""
        pass
