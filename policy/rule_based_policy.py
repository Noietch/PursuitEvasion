"""
Rule-based policy for evaders - counter-strategy against Apollonius circle pursuit.
Exploits weaknesses in pursuer's LP matching and on-site capture conditions.
"""

import numpy as np
from .base_policy import BasePolicy


class RuleBasedPolicy(BasePolicy):
    """
    Counter-strategy exploiting pursuer weaknesses from strategy_p_reconstructed.py:

    1. Apollonius circle breaks when it intersects obstacles
       -> Actively move toward obstacles when threatened

    2. On-site capture fails when circle intersects target area
       -> Sprint to target when close enough

    3. LP matching is static and locks pursuers to targets
       -> Spread out to force 1v1 matchups, sacrifice some to save others

    4. Win condition: distET * alphaV - distPE > 1
       -> Maximize distPE (distance from pursuer) while minimizing distET
    """

    def __init__(
        self,
        num_evaders: int = 10,
        max_vel: float = 4.0,
        num_pursuers: int = 10,
        pursuer_max_vel: float = 6.0,
        obstacle_vertices: list = None,
        space_bounds: tuple = (10.0, 1000.0, 10.0, 1000.0),
        obstacle_gain: float = 0.8,
        boundary_gain: float = 1.0,
        boundary_margin: float = 50.0,
        teammate_separation: float = 40.0,
    ):
        super().__init__(num_evaders, max_vel)
        self.num_pursuers = num_pursuers
        self.pursuer_max_vel = pursuer_max_vel
        self.space_bounds = space_bounds
        self.obstacle_gain = obstacle_gain
        self.boundary_gain = boundary_gain
        self.boundary_margin = boundary_margin
        self.teammate_separation = teammate_separation

        self.obstacle_vertices = obstacle_vertices or []
        self.obstacle_centers = self._compute_obstacle_centers()
        self.prev_pursuers_pos = None

        self.alpha_v = pursuer_max_vel / max_vel

    def _compute_obstacle_centers(self) -> np.ndarray:
        if not self.obstacle_vertices:
            return np.array([]).reshape(0, 2)
        centers = []
        for obs in self.obstacle_vertices:
            obs = np.array(obs)
            centers.append(obs.mean(axis=0))
        return np.array(centers)

    def reset(self):
        self.prev_pursuers_pos = None

    def _get_apollonius_circle(self, e_pos: np.ndarray, p_pos: np.ndarray) -> tuple:
        if self.alpha_v <= 1:
            return None, None
        alpha_sq = self.alpha_v ** 2
        denom = alpha_sq - 1
        center = np.array([
            (alpha_sq * e_pos[0] - p_pos[0]) / denom,
            (alpha_sq * e_pos[1] - p_pos[1]) / denom
        ])
        radius = self.alpha_v * np.linalg.norm(e_pos - p_pos) / denom
        return center, radius

    def _compute_win_margin(self, e_pos: np.ndarray, p_pos: np.ndarray, target: np.ndarray) -> float:
        """
        Pursuer's win condition: distET * alphaV - distPE > 1
        Returns margin: positive = pursuer wins, negative = evader escapes
        """
        dist_pe = np.linalg.norm(e_pos - p_pos)
        dist_et = np.linalg.norm(e_pos - target)
        return dist_et * self.alpha_v - dist_pe

    def _find_best_obstacle_path(self, pos: np.ndarray, target: np.ndarray,
                                  obs_centers: np.ndarray, p_pos: np.ndarray) -> np.ndarray:
        """
        Find obstacle that can break Apollonius circle while still progressing toward target.
        Returns direction to move.
        """
        if len(obs_centers) == 0:
            return None

        to_target = target - pos
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < 1e-6:
            return None
        target_dir = to_target / dist_to_target

        best_obs = None
        best_score = -float('inf')

        for obs_center in obs_centers:
            to_obs = obs_center - pos
            dist_to_obs = np.linalg.norm(to_obs)

            if dist_to_obs < 30 or dist_to_obs > 300:
                continue

            obs_dir = to_obs / dist_to_obs

            # Check if obstacle is between evader and pursuer
            to_pursuer = p_pos - pos
            dist_to_p = np.linalg.norm(to_pursuer)
            if dist_to_p > 1e-6:
                p_dir = to_pursuer / dist_to_p

                # Score: prefer obstacles that are both toward target and block pursuer
                target_alignment = np.dot(obs_dir, target_dir)
                pursuer_blocking = np.dot(obs_dir, p_dir)

                # Weighted score: prioritize target progress, but value blocking pursuer
                score = target_alignment * 0.6 + pursuer_blocking * 0.4 - dist_to_obs / 500.0

                if score > best_score:
                    best_score = score
                    best_obs = obs_center

        if best_obs is not None and best_score > -0.3:
            to_best = best_obs - pos
            dist = np.linalg.norm(to_best)
            if dist > 1e-6:
                obs_dir = to_best / dist
                # Move tangent to obstacle (edge hugging)
                tangent = np.array([-obs_dir[1], obs_dir[0]])
                if np.dot(tangent, target_dir) < 0:
                    tangent = -tangent
                return tangent * 0.6 + target_dir * 0.4

        return None

    def _estimate_pursuer_target(self, pursuers_pos: np.ndarray, evader_idx: int,
                                  evaders_pos: np.ndarray, active_mask: np.ndarray) -> int:
        """
        Estimate which pursuer is targeting this evader based on LP matching logic.
        Pursuers prioritize evaders they can capture (high win margin).
        """
        e_pos = evaders_pos[evader_idx]

        best_p = -1
        best_score = -float('inf')

        for p_idx, p_pos in enumerate(pursuers_pos):
            # Simulate pursuer's scoring: higher score = more likely to target
            dist_pe = np.linalg.norm(e_pos - p_pos)
            win_margin = self._compute_win_margin(e_pos, p_pos, np.zeros(2))  # Placeholder

            # Pursuer prefers closer evaders with higher capture probability
            score = -dist_pe
            if score > best_score:
                best_score = score
                best_p = p_idx

        return best_p

    def _find_escape_waypoint(self, pos: np.ndarray, p_pos: np.ndarray,
                               target: np.ndarray, obs_centers: np.ndarray) -> np.ndarray:
        """
        Find a waypoint that:
        1. Puts an obstacle between evader and pursuer (breaks Apollonius)
        2. Still allows progress toward target eventually
        """
        if len(obs_centers) == 0:
            return None

        to_target = target - pos
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < 1e-6:
            return None
        target_dir = to_target / dist_to_target

        to_pursuer = p_pos - pos
        dist_to_pursuer = np.linalg.norm(to_pursuer)
        if dist_to_pursuer < 1e-6:
            return None
        pursuer_dir = to_pursuer / dist_to_pursuer

        best_waypoint = None
        best_score = -float('inf')

        for obs_center in obs_centers:
            to_obs = obs_center - pos
            dist_to_obs = np.linalg.norm(to_obs)

            if dist_to_obs < 20 or dist_to_obs > 250:
                continue

            obs_dir = to_obs / dist_to_obs

            # Calculate if obstacle is roughly between us and pursuer
            # We want to move so that obstacle blocks pursuer's path
            pursuer_to_obs = obs_center - p_pos
            dist_p_to_obs = np.linalg.norm(pursuer_to_obs)

            # Score components:
            # 1. Obstacle should be closer to pursuer's path to us
            blocking_score = np.dot(obs_dir, pursuer_dir)

            # 2. After reaching obstacle, we should be able to reach target
            obs_to_target = target - obs_center
            dist_obs_target = np.linalg.norm(obs_to_target)

            # 3. Prefer obstacles that don't add too much distance
            detour_penalty = (dist_to_obs + dist_obs_target) / (dist_to_target + 1)

            # 4. Prefer obstacles that are on the side away from pursuer
            away_from_pursuer = -np.dot(obs_dir, pursuer_dir)

            score = blocking_score * 0.3 + away_from_pursuer * 0.4 - detour_penalty * 0.3

            if score > best_score:
                best_score = score
                # Waypoint is on the far side of obstacle from pursuer
                offset = -pursuer_dir * 50  # Go past the obstacle
                best_waypoint = obs_center + offset

        return best_waypoint if best_score > -0.5 else None

    def _should_flee_opposite(self, pos: np.ndarray, p_pos: np.ndarray,
                               target: np.ndarray) -> bool:
        """
        Check if we should flee in opposite direction first.
        Returns True if pursuer is directly between us and target.
        """
        to_target = target - pos
        to_pursuer = p_pos - pos

        dist_to_target = np.linalg.norm(to_target)
        dist_to_pursuer = np.linalg.norm(to_pursuer)

        if dist_to_target < 1e-6 or dist_to_pursuer < 1e-6:
            return False

        # Check if pursuer is between us and target
        target_dir = to_target / dist_to_target
        pursuer_dir = to_pursuer / dist_to_pursuer

        # Pursuer is blocking if: close, aligned with target direction
        alignment = np.dot(pursuer_dir, target_dir)

        # Must be close and blocking our path
        return alignment > 0.7 and dist_to_pursuer < 80 and dist_to_pursuer < dist_to_target

    def _would_apollonius_intersect_obstacle(self, e_pos: np.ndarray, p_pos: np.ndarray,
                                               obs_centers: np.ndarray, obs_radius: float = 40.0) -> tuple:
        """
        Check if Apollonius circle would intersect any obstacle.
        Returns (intersects, best_obstacle_center) - the obstacle to move toward.
        """
        center, radius = self._get_apollonius_circle(e_pos, p_pos)
        if center is None:
            return False, None

        best_obs = None
        best_dist = float('inf')

        for obs_c in obs_centers:
            # Distance from Apollonius center to obstacle center
            dist = np.linalg.norm(center - obs_c)
            # If obstacle is inside or close to Apollonius circle
            if dist < radius + obs_radius:
                # Already intersects!
                return True, obs_c
            # Find closest obstacle to the circle
            gap = dist - radius - obs_radius
            if gap < best_dist and gap < 100:  # Only consider nearby obstacles
                best_dist = gap
                best_obs = obs_c

        return False, best_obs

    def get_action(self, observation: dict) -> np.ndarray:
        evaders_pos = observation['evaders_pos']
        pursuers_pos = observation['pursuers_pos']
        winning = observation['evaders_winning'].astype(bool)
        captured = observation['evaders_captured'].astype(bool)
        target_center = observation['target_center']

        obs_centers = observation.get('obstacle_centers', self.obstacle_centers)
        if isinstance(obs_centers, list):
            obs_centers = np.array(obs_centers) if obs_centers else np.array([]).reshape(0, 2)

        velocities = np.zeros((self.num_evaders, 2), dtype=np.float32)
        active_mask = ~(captured | winning)

        num_active = np.sum(active_mask)
        if num_active == 0:
            return velocities

        for i in range(self.num_evaders):
            if not active_mask[i]:
                continue

            pos = evaders_pos[i]
            to_target = target_center - pos
            dist_to_target = np.linalg.norm(to_target)

            if dist_to_target < 1e-6:
                continue

            target_dir = to_target / dist_to_target

            dists_to_pursuers = np.linalg.norm(pursuers_pos - pos, axis=1)
            closest_p_idx = np.argmin(dists_to_pursuers)
            closest_p_dist = dists_to_pursuers[closest_p_idx]
            closest_p_pos = pursuers_pos[closest_p_idx]

            vel_dir = np.zeros(2)

            # === OPTIMIZED SMART EVASION STRATEGY ===

            # Very close to target - sprint!
            if dist_to_target < 100:
                vel_dir = target_dir

            # Pursuer very close - maximum evasion
            elif closest_p_dist < 35:
                to_p = closest_p_pos - pos
                p_norm = np.linalg.norm(to_p)
                if p_norm > 1e-6:
                    p_dir = to_p / p_norm
                    perp = np.array([-p_dir[1], p_dir[0]])
                    if np.dot(perp, target_dir) < 0:
                        perp = -perp
                    vel_dir = perp * 0.65 + target_dir * 0.35
                else:
                    vel_dir = target_dir

            # Pursuer close - moderate evasion
            elif closest_p_dist < 70:
                to_p = closest_p_pos - pos
                p_norm = np.linalg.norm(to_p)
                if p_norm > 1e-6:
                    p_dir = to_p / p_norm
                    perp = np.array([-p_dir[1], p_dir[0]])
                    if np.dot(perp, target_dir) < 0:
                        perp = -perp
                    vel_dir = perp * 0.35 + target_dir * 0.65
                else:
                    vel_dir = target_dir

            # Medium/far distance - smart path with side bias
            else:
                perp_left = np.array([-target_dir[1], target_dir[0]])

                left_threat = 0
                right_threat = 0

                for p_pos in pursuers_pos:
                    to_p = p_pos - pos
                    dist_p = np.linalg.norm(to_p)
                    if dist_p < 220:
                        side = np.dot(to_p, perp_left)
                        weight = (220 - dist_p) / 220
                        if side > 0:
                            left_threat += weight
                        else:
                            right_threat += weight

                if left_threat < right_threat:
                    bias = perp_left * min(0.45, (right_threat - left_threat) * 0.12)
                elif right_threat < left_threat:
                    bias = -perp_left * min(0.45, (left_threat - right_threat) * 0.12)
                else:
                    bias = np.zeros(2)

                vel_dir = target_dir + bias

            # Apply forces
            vel_dir += self._boundary_force(pos) * self.boundary_gain
            vel_dir += self._obstacle_repulsion(pos, obs_centers) * self.obstacle_gain

            norm = np.linalg.norm(vel_dir)
            if norm > 1e-6:
                velocities[i] = (vel_dir / norm) * self.max_vel
            else:
                velocities[i] = target_dir * self.max_vel

        self.prev_pursuers_pos = pursuers_pos.copy()
        return velocities

    def _find_best_approach_angle(self, pursuers_pos: np.ndarray, target: np.ndarray) -> float:
        """Find angle with fewest pursuers between edge and target."""
        best_angle = 0
        best_score = float('inf')

        for angle_deg in range(0, 360, 30):
            angle = np.radians(angle_deg)
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Count weighted pursuers in this direction from target
            score = 0
            for p_pos in pursuers_pos:
                to_p = p_pos - target
                dist = np.linalg.norm(to_p)
                if dist < 300:
                    # How aligned is this pursuer with the approach direction
                    alignment = np.dot(to_p / (dist + 1e-6), direction)
                    if alignment > 0.5:  # Pursuer is roughly in this direction
                        score += (300 - dist) / 300

            if score < best_score:
                best_score = score
                best_angle = angle

        return best_angle

    def _nearest_obstacle_dir(self, pos: np.ndarray, obs_centers: np.ndarray) -> tuple:
        if len(obs_centers) == 0:
            return None, float('inf')
        dists = np.linalg.norm(obs_centers - pos, axis=1)
        idx = np.argmin(dists)
        return obs_centers[idx] - pos, dists[idx]

    def _obstacle_repulsion(self, pos: np.ndarray, obs_centers: np.ndarray) -> np.ndarray:
        force = np.zeros(2)
        if len(obs_centers) == 0:
            return force
        obstacle_radius = 40.0  # Reduced to allow closer approach for Apollonius break
        for center in obs_centers:
            diff = pos - center
            dist = np.linalg.norm(diff)
            if dist < obstacle_radius and dist > 1e-6:
                strength = (obstacle_radius - dist) / obstacle_radius
                force += (diff / dist) * strength * 1.0  # Reduced multiplier
        return force

    def _boundary_force(self, pos: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.space_bounds
        force = np.zeros(2)
        if pos[0] - x_min < self.boundary_margin:
            strength = (self.boundary_margin - (pos[0] - x_min)) / self.boundary_margin
            force[0] += strength
        if x_max - pos[0] < self.boundary_margin:
            strength = (self.boundary_margin - (x_max - pos[0])) / self.boundary_margin
            force[0] -= strength
        if pos[1] - y_min < self.boundary_margin:
            strength = (self.boundary_margin - (pos[1] - y_min)) / self.boundary_margin
            force[1] += strength
        if y_max - pos[1] < self.boundary_margin:
            strength = (self.boundary_margin - (y_max - pos[1])) / self.boundary_margin
            force[1] -= strength
        return force

