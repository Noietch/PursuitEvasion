"""
IPPO (Independent PPO) policy for multi-agent pursuit-evasion game.
Uses parameter sharing across all evaders with continuous action space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base_policy import BasePolicy


class PolicyNet(nn.Module):
    """
    Gaussian policy network for continuous actions.
    Outputs mean of action distribution, with learnable log_std.
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))  # Output in [-1, 1]
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std


class ValueNet(nn.Module):
    """Value network for state value estimation."""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    """
    PPO agent with clipped objective for continuous actions.
    Used for training individual evader policies with parameter sharing.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Sample action from policy.

        Args:
            state: State array
            deterministic: If True, return mean action without sampling

        Returns:
            action: Sampled action in [-1, 1]
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            if deterministic:
                action = mean.squeeze(0).cpu().numpy()
                log_prob = 0.0
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1).item()
                action = action.squeeze(0).cpu().numpy()
        return action, log_prob

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor).item()
        return value

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, trajectories: Dict[str, List], epochs: int = 10, batch_size: int = 64):
        """
        Update policy and value networks using PPO.

        Args:
            trajectories: Dict with keys 'states', 'actions', 'log_probs', 'returns', 'advantages'
            epochs: Number of update epochs
            batch_size: Mini-batch size
        """
        states = torch.FloatTensor(np.array(trajectories['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(trajectories['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(self.device)
        returns = torch.FloatTensor(trajectories['returns']).to(self.device)
        advantages = torch.FloatTensor(trajectories['advantages']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]

        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Actor loss
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)

                # Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class IPPOPolicy(BasePolicy):
    """
    IPPO policy wrapper for pursuit-evasion environment.
    Uses parameter sharing: all evaders share the same policy network.
    """
    def __init__(
        self,
        num_evaders: int = 10,
        num_pursuers: int = 10,
        max_vel: float = 4.0,
        hidden_dim: int = 128,
        device: str = 'cpu'
    ):
        super().__init__(num_evaders, max_vel)
        self.num_pursuers = num_pursuers
        self.device = device

        # State dim for each evader's local observation:
        # - relative position to target (2)
        # - distances to all pursuers (num_pursuers)
        # - directions to k nearest pursuers (k * 2)
        # - own status (captured, winning) (2)
        self.k_nearest = min(3, num_pursuers)
        self.state_dim = 2 + num_pursuers + self.k_nearest * 2 + 2
        self.action_dim = 2  # velocity (vx, vy)

        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            device=device
        )

    def _build_local_obs(self, observation: Dict, evader_idx: int) -> np.ndarray:
        """
        Build local observation for a single evader.

        Args:
            observation: Global observation dict
            evader_idx: Index of evader

        Returns:
            Local observation array
        """
        evader_pos = observation['evaders_pos'][evader_idx]
        pursuers_pos = observation['pursuers_pos']
        target_center = observation['target_center']
        captured = observation['evaders_captured'][evader_idx]
        winning = observation['evaders_winning'][evader_idx]

        # Relative position to target (normalized)
        rel_target = (target_center - evader_pos) / 500.0  # Normalize by half field size

        # Distances to all pursuers (normalized)
        distances = np.linalg.norm(pursuers_pos - evader_pos, axis=1) / 1000.0

        # Directions to k nearest pursuers
        sorted_indices = np.argsort(distances)[:self.k_nearest]
        nearest_dirs = []
        for idx in sorted_indices:
            direction = pursuers_pos[idx] - evader_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                direction = direction / dist
            nearest_dirs.extend(direction.tolist())

        # Status
        status = [float(captured), float(winning)]

        local_obs = np.concatenate([
            rel_target,
            distances,
            np.array(nearest_dirs),
            np.array(status)
        ]).astype(np.float32)

        return local_obs

    def get_action(self, observation: Dict) -> np.ndarray:
        """
        Get actions for all evaders.

        Args:
            observation: Global observation dict

        Returns:
            Velocity commands of shape (num_evaders, 2)
        """
        velocities = np.zeros((self.num_evaders, 2), dtype=np.float32)

        for i in range(self.num_evaders):
            captured = observation['evaders_captured'][i]
            winning = observation['evaders_winning'][i]

            if captured or winning:
                continue

            local_obs = self._build_local_obs(observation, i)
            action, _ = self.agent.get_action(local_obs, deterministic=True)
            velocities[i] = action * self.max_vel  # Scale to velocity range

        return velocities

    def get_action_with_log_prob(
        self,
        observation: Dict,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Get actions with log probabilities for training.

        Returns:
            velocities: Action array
            log_probs: Log probabilities for each evader
            local_obs_list: Local observations for each evader
        """
        velocities = np.zeros((self.num_evaders, 2), dtype=np.float32)
        log_probs = []
        local_obs_list = []

        for i in range(self.num_evaders):
            captured = observation['evaders_captured'][i]
            winning = observation['evaders_winning'][i]

            local_obs = self._build_local_obs(observation, i)
            local_obs_list.append(local_obs)

            if captured or winning:
                log_probs.append(0.0)
                continue

            action, log_prob = self.agent.get_action(local_obs, deterministic=deterministic)
            velocities[i] = action * self.max_vel
            log_probs.append(log_prob)

        return velocities, log_probs, local_obs_list

    def get_values(self, observation: Dict) -> List[float]:
        """Get value estimates for all evaders."""
        values = []
        for i in range(self.num_evaders):
            local_obs = self._build_local_obs(observation, i)
            value = self.agent.get_value(local_obs)
            values.append(value)
        return values

    def update(self, all_trajectories: List[Dict], epochs: int = 10, batch_size: int = 64):
        """
        Update policy using collected trajectories from all evaders.

        Args:
            all_trajectories: List of trajectory dicts, one per evader
        """
        # Merge trajectories from all evaders (parameter sharing)
        merged = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'returns': [],
            'advantages': []
        }

        for traj in all_trajectories:
            merged['states'].extend(traj['states'])
            merged['actions'].extend(traj['actions'])
            merged['log_probs'].extend(traj['log_probs'])
            merged['returns'].extend(traj['returns'])
            merged['advantages'].extend(traj['advantages'])

        if len(merged['states']) > 0:
            self.agent.update(merged, epochs=epochs, batch_size=batch_size)

    def save(self, path: str):
        """Save policy weights."""
        self.agent.save(path)

    def load(self, path: str):
        """Load policy weights."""
        self.agent.load(path)
