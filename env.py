"""
OpenAI Gym-style environment for pursuit-evasion game.
Wraps the external simulator (main_pyqt.exe) via ZMQ communication.
"""

import numpy as np
import zmq
import json
import time
from typing import Optional, Tuple, Dict, Any


class PursuitEvasionEnv:
    """
    OpenAI Gym-style environment for pursuit-evasion game.

    The environment communicates with an external simulator (main_pyqt.exe)
    via ZMQ sockets. Evaders try to reach a target area while pursuers
    try to capture them.

    Observation Space:
        Dict with keys:
        - evaders_pos: (num_evaders, 2) positions
        - pursuers_pos: (num_pursuers, 2) positions
        - evaders_winning: (num_evaders,) boolean array
        - evaders_captured: (num_evaders,) boolean array
        - target_center: (2,) target area center
        - obstacles: (num_obstacles, 3, 2) obstacle vertices (triangles)

    Action Space:
        Shape (num_evaders, 2), range [-max_vel, max_vel]
        Represents velocity commands for each evader.
    """

    def __init__(
        self,
        env_config_path: str = 'configs/env.json',
        swarm_config_path: str = 'configs/swarm.json',
        state_port: int = 5555,
        cmd_port: int = 5556,
        max_steps: int = 1000
    ):
        """
        Initialize the environment.

        Args:
            env_config_path: Path to environment configuration
            swarm_config_path: Path to swarm configuration
            state_port: Port for receiving state from simulator
            cmd_port: Port for sending commands to simulator
            max_steps: Maximum steps per episode
        """
        # Load configurations
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config = json.load(f)
        with open(swarm_config_path, 'r', encoding='utf-8') as f:
            self.swarm_config = json.load(f)

        self.num_evaders = len(self.swarm_config['evaders'])
        self.num_pursuers = len(self.swarm_config['pursuers'])
        self.max_evader_vel = self.swarm_config['evaders'][0]['max_vel']
        self.max_pursuer_vel = self.swarm_config['pursuers'][0]['max_vel']
        self.max_steps = max_steps

        # Calculate target area center
        target_vertices = np.array(self.env_config['target'])
        self.target_center = np.mean(target_vertices, axis=0).astype(np.float32)

        # Load obstacles (static, from config)
        self.obstacles = np.array(self.env_config['obs'], dtype=np.float32)
        self.num_obstacles = len(self.obstacles)

        # Define action space info (for reference)
        self.action_shape = (self.num_evaders, 2)
        self.action_low = -self.max_evader_vel
        self.action_high = self.max_evader_vel

        # Define observation space info (for reference)
        self.observation_keys = ['evaders_pos', 'pursuers_pos', 'evaders_winning',
                                  'evaders_captured', 'target_center', 'obstacles']

        # Initialize ZMQ communication
        self.context = zmq.Context()
        self.state_socket = self._create_subscriber(state_port)
        self.cmd_socket = self._create_publisher(cmd_port)

        # Internal state
        self.current_step = 0
        self.last_state = None

    def _create_publisher(self, port: int) -> zmq.Socket:
        """Create ZMQ publisher socket for sending commands."""
        socket = self.context.socket(zmq.PUB)
        socket.bind(f"tcp://localhost:{port}")
        time.sleep(0.1)  # Allow socket to bind
        return socket

    def _create_subscriber(self, port: int) -> zmq.Socket:
        """Create ZMQ subscriber socket for receiving state."""
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        return socket

    def _receive_state(self, timeout_ms: int = 100) -> Optional[Dict]:
        """
        Receive state from simulator (non-blocking with timeout).

        Returns:
            dict: State dictionary or None if no data available
        """
        poller = zmq.Poller()
        poller.register(self.state_socket, zmq.POLLIN)

        if poller.poll(timeout_ms):
            try:
                received_data = self.state_socket.recv_string(flags=zmq.NOBLOCK)
                return json.loads(received_data)
            except zmq.Again:
                pass
        return None

    def _send_command(self, evaders_vel: np.ndarray, command: str = ''):
        """
        Send velocity commands to simulator.

        Args:
            evaders_vel: Velocity array for evaders
            command: Control command ('resume', 'start_pause', '')
        """
        cmd_data = {
            'timestamp': time.time(),
            'evaders_vel': evaders_vel.tolist() if isinstance(evaders_vel, np.ndarray) else evaders_vel,
            'command': command
        }
        self.cmd_socket.send_string(json.dumps(cmd_data))

    def _wait_for_state(self, max_wait_time: float = 5.0) -> Dict:
        """
        Wait for state update from simulator.

        Args:
            max_wait_time: Maximum time to wait in seconds

        Returns:
            dict: State data

        Raises:
            TimeoutError: If no state received within timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            state = self._receive_state(timeout_ms=100)
            if state is not None:
                return state
            time.sleep(0.01)
        raise TimeoutError("Failed to receive state from simulator")

    def _build_observation(self, state: Dict) -> Dict[str, np.ndarray]:
        """
        Convert simulator state to observation format.

        Args:
            state: Raw state dictionary from simulator

        Returns:
            dict: Observation dictionary
        """
        return {
            'evaders_pos': np.array(state['evaders_pos'], dtype=np.float32),
            'pursuers_pos': np.array(state['pursuers_pos'], dtype=np.float32),
            'evaders_winning': np.array(state['evaders_winning'], dtype=np.int8),
            'evaders_captured': np.array(state['evaders_captured'], dtype=np.int8),
            'target_center': self.target_center.copy(),
            'obstacles': self.obstacles.copy()
        }

    def _calculate_reward(self, state: Dict, prev_state: Optional[Dict]) -> float:
        """
        Calculate reward based on state transition.

        Reward structure:
        - +10 for each evader reaching target (newly winning)
        - -5 for each evader captured (newly captured)
        - +0.01 * progress toward target

        Args:
            state: Current state
            prev_state: Previous state

        Returns:
            float: Reward value
        """
        if prev_state is None:
            return 0.0

        reward = 0.0

        # Reward for newly winning evaders
        prev_winning = np.array(prev_state['evaders_winning'])
        curr_winning = np.array(state['evaders_winning'])
        new_wins = np.sum(curr_winning.astype(int) - prev_winning.astype(int) > 0)
        reward += new_wins * 10.0

        # Penalty for newly captured evaders
        prev_captured = np.array(prev_state['evaders_captured'])
        curr_captured = np.array(state['evaders_captured'])
        new_captured = np.sum(curr_captured.astype(int) - prev_captured.astype(int) > 0)
        reward -= new_captured * 5.0

        # Progress reward: distance to target
        prev_pos = np.array(prev_state['evaders_pos'])
        curr_pos = np.array(state['evaders_pos'])
        prev_dist = np.linalg.norm(prev_pos - self.target_center, axis=1)
        curr_dist = np.linalg.norm(curr_pos - self.target_center, axis=1)

        # Only count active evaders
        active_mask = ~(curr_captured.astype(bool) | curr_winning.astype(bool))
        if np.any(active_mask):
            progress = prev_dist[active_mask] - curr_dist[active_mask]
            reward += 0.01 * np.sum(progress)

        return reward

    def _check_terminated(self, state: Dict) -> Tuple[bool, bool, Dict]:
        """
        Check if episode is terminated or truncated.

        Episode ends when all evaders are done (either winning or captured).

        Args:
            state: Current state dictionary

        Returns:
            tuple: (terminated, truncated, info_dict)
        """
        winning = np.array(state['evaders_winning'])
        captured = np.array(state['evaders_captured'])
        game_state = state.get('gamestate', 'run')

        terminated = False
        truncated = False
        info = {'result': 'ongoing'}

        if game_state == 'stop':
            terminated = True
            info['result'] = 'stopped'
        else:
            # Check if all evaders are done (either winning or captured)
            all_done = np.all(winning | captured)
            if all_done:
                terminated = True
                winning_count = int(np.sum(winning))
                captured_count = int(np.sum(captured))
                info['winning_count'] = winning_count
                info['captured_count'] = captured_count
                if winning_count > 0:
                    info['result'] = 'win'
                else:
                    info['result'] = 'loss'

        # Check truncation
        if self.current_step >= self.max_steps:
            truncated = True
            info['truncated'] = True

        return terminated, truncated, info

    def _flush_state_buffer(self):
        """Flush any pending messages in the state socket buffer."""
        while True:
            try:
                self.state_socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Flush old messages from buffer
        self._flush_state_buffer()

        # Send resume command to reset simulator
        zero_vel = [[0, 0] for _ in range(self.num_evaders)]
        self._send_command(zero_vel, command='resume')
        time.sleep(0.5)  # Wait longer for simulator to reset

        # Flush again after resume
        self._flush_state_buffer()

        # Send start command
        self._send_command(zero_vel, command='start_pause')
        time.sleep(0.2)

        # Clear command
        self._send_command(zero_vel, command='')

        # Wait for valid initial state (gamestate should be 'run' and no captured evaders)
        state = self._wait_for_valid_initial_state(max_wait_time=3.0)

        self.last_state = state
        self.current_step = 0

        obs = self._build_observation(state)
        info = {
            'gamestate': state.get('gamestate', 'unknown'),
            'step': 0
        }

        return obs, info

    def _wait_for_valid_initial_state(self, max_wait_time: float = 3.0) -> Dict:
        """
        Wait for a valid initial state from simulator.

        A valid initial state has gamestate='run' and no captured/winning evaders.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            state = self._receive_state(timeout_ms=100)
            if state is not None:
                gamestate = state.get('gamestate', '')
                captured = state.get('evaders_captured', [])
                winning = state.get('evaders_winning', [])

                # Check if this is a valid initial state
                if gamestate == 'run' and not any(captured) and not any(winning):
                    return state

            time.sleep(0.01)

        # If timeout, try to get any state
        state = self._wait_for_state(max_wait_time=1.0)
        return state

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one timestep within the environment.

        Args:
            action: Velocity commands for evaders, shape (num_evaders, 2)

        Returns:
            observation: Next observation
            reward: Reward for the transition
            terminated: Whether episode ended due to game logic
            truncated: Whether episode ended due to step limit
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -self.max_evader_vel, self.max_evader_vel).astype(np.float32)

        # Send action to simulator
        self._send_command(action, command='')

        # Wait for next state
        time.sleep(0.05)
        state = self._wait_for_state(max_wait_time=1.0)

        # Build observation
        obs = self._build_observation(state)

        # Calculate reward
        reward = self._calculate_reward(state, self.last_state)

        # Update step counter
        self.current_step += 1

        # Check termination
        terminated, truncated, info = self._check_terminated(state)

        # Add statistics to info
        info['step'] = self.current_step
        info['evaders_winning'] = int(np.sum(state['evaders_winning']))
        info['evaders_captured'] = int(np.sum(state['evaders_captured']))
        info['gamestate'] = state.get('gamestate', 'unknown')

        self.last_state = state

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.
        The external simulator handles visualization.
        """
        pass

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'state_socket') and self.state_socket:
            self.state_socket.close()
        if hasattr(self, 'cmd_socket') and self.cmd_socket:
            self.cmd_socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()
