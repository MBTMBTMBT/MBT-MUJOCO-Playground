import hashlib
import math
import random
from typing import Optional, Union
import gymnasium as gym
import pygame
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon, Rectangle
from numpy import cos, pi, sin
from gymnasium import spaces
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register
from gymnasium.envs.box2d.lunar_lander import LunarLander
from custom_inverted_pendulum import CustomInvertedPendulum


WELCOME = "CUSTOM ENVS OF MBTMBTMBT!!!"
# print(WELCOME)


# Register the custom environments
register(
    id="CustomFrozenLake",
    entry_point="custom_envs:CustomFrozenLakeEnv",
    kwargs={
        "render_mode": None,
        "desc": None,
        "map_name": "4x4",
        "is_slippery": True,
        "slipperiness": 0.8,
    },
    max_episode_steps=100,
)

register(
    id="CustomMountainCar",
    entry_point="custom_envs:CustomMountainCarEnv",
    kwargs={
        "render_mode": None,
        "goal_velocity": 0,
        "custom_gravity": 0.0025,
        "custom_force": 0.001,
        "goal_position": 0.5,
        "reward_type": "default",
    },
    max_episode_steps=200,
)

register(
    id="CustomAcrobot",
    entry_point="custom_envs:CustomAcrobotEnv",
    kwargs={
        "render_mode": None,
        "termination_height": 1.0,
        "friction": 0.0,
        "torque_scaling": 1.0,
        "gravity": 9.8,
        "link_lengths": (1.0, 1.0),
        "link_masses": (1.0, 1.0),
        "max_velocities": (4 * pi, 9 * pi),
        "reward_type": "default",
    },
    max_episode_steps=250,
)

register(
    id="CarRacingFixedMap",
    entry_point="custom_envs:CarRacingFixedMap",
    kwargs={
        "render_mode": None,
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": False,
        "map_seed": 0,
        "fixed_start": True,
        "backwards_tolerance": 3,
        "grass_tolerance": 15,
        "number_of_initial_states": 32,
        "init_seed": None,
        "vector_obs": False,
    },
    max_episode_steps=1000,
)

register(
    id="CustomLunarLander",
    entry_point="custom_envs:CustomLunarLander",
    kwargs={
        "render_mode": None,
        "continuous": False,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
        "lander_density": 5.0,
        "number_of_initial_states": 256,
        "use_deterministic_initial_states": True,
        "init_seed": None,
    },
    max_episode_steps=500,
)

register(
    id="CustomInvertedPendulum",
    entry_point="custom_envs:CustomInvertedPendulum",
    kwargs={
        "render_mode": None,
        "length": 0.6,
        "pole_density": 1000.0,
        "cart_density": 1000.0,
        "xml_file": "./assets/inverted_pendulum.xml",
        "reset_noise_scale": 0.01,
    },
    max_episode_steps=500,
)


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        slipperiness: float = 1.0,
    ):
        """
        Custom FrozenLake environment with adjustable slipperiness.

        Parameters:
        - render_mode: Optional rendering mode.
        - desc: Custom map description (2D list of strings).
        - map_name: Predefined map name.
        - is_slippery: Determines if the environment is slippery (original logic).
        - slipperiness: Degree of slipperiness (0 = no slip, 1 = original slippery).
        """
        super().__init__(render_mode=render_mode, desc=desc, map_name=map_name)
        self.is_slippery = is_slippery
        self.slipperiness = slipperiness

        # Override transition probabilities
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    li.clear()

                    if self.desc[row][col] in b"GH":  # Goal or Hole
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            # Custom slippery logic controlled by slipperiness
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                if b == a:  # Forward direction
                                    prob = 1 - (self.slipperiness * (2 / 3))
                                else:  # Side directions
                                    prob = self.slipperiness / 3
                                new_row, new_col = self._inc(row, col, b)
                                new_state = self._to_s(new_row, new_col)
                                new_letter = self.desc[new_row][new_col]
                                terminated = bytes(new_letter) in b"GH"
                                reward = float(new_letter == b"G")
                                li.append((prob, new_state, reward, terminated))
                        else:
                            # Deterministic logic when not slippery
                            new_row, new_col = self._inc(row, col, a)
                            new_state = self._to_s(new_row, new_col)
                            new_letter = self.desc[new_row][new_col]
                            terminated = bytes(new_letter) in b"GH"
                            reward = float(new_letter == b"G")
                            li.append((1.0, new_state, reward, terminated))

    def _inc(self, row, col, action):
        """
        Increment row and column based on the action taken.

        Parameters:
        - row: Current row index.
        - col: Current column index.
        - action: Action to take (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP).

        Returns:
        - Tuple (new_row, new_col) after taking the action.
        """
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)
        return row, col

    def _to_s(self, row, col):
        """
        Convert (row, col) to a single state index.

        Parameters:
        - row: Row index.
        - col: Column index.

        Returns:
        - State index as an integer.
        """
        return row * self.ncol + col


class CustomMountainCarEnv(MountainCarEnv):
    """
    Custom version of the MountainCar environment with adjustable gravity, reward, and episode duration.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        goal_velocity=0,
        custom_gravity=0.0025,
        custom_force=0.001,
        # max_episode_steps=200,
        goal_position=0.5,
        reward_type="default",
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        # Override gravity and max_episode_steps with custom values
        self.goal_position = goal_position
        if custom_gravity >= self.gravity:
            self.max_speed = custom_gravity / self.gravity * self.max_speed
        self.force = custom_force
        self.gravity = custom_gravity
        # self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_type = reward_type

    def step(self, action: int):
        # Override the step function to include a step counter for custom episode duration
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        # Custom reward function based on reward_type
        if self.reward_type == "default":
            reward = -1.0 if not terminated else 0.0
        elif self.reward_type == "distance":
            # Reward based on the distance from the starting position (-0.5)
            reward = abs(position + 0.5) - (
                1.0
                if position <= self.min_position or position >= self.max_position
                else 0.0
            )
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == "progress":
            # Reward based on progress towards the goal, incentivizing movement to the right and higher speed
            reward = (position - self.min_position) if position >= 0.0 else 0.0
            reward += velocity if velocity >= self.goal_velocity else 0.0
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == "sparse":
            reward = -0.0 if not terminated else 1.0
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self.state = (position, velocity)
        self.current_step += 1

        # Terminate if the maximum number of steps is reached
        truncated = False
        # truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Override the reset function to reset the step counter
        self.current_step = 0
        return super().reset(seed=seed, options=options)


class CustomAcrobotEnv(AcrobotEnv):
    """
    Customizable Acrobot environment with user-defined physics parameters.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        termination_height: float = 1.0,  # Height to trigger termination
        friction: float = 0.0,  # Friction factor for joints
        torque_scaling: float = 1.0,  # Scaling factor for torque
        gravity: float = 9.8,  # Gravity value
        link_lengths: tuple = (1.0, 1.0),  # Lengths of the links
        link_masses: tuple = (1.0, 1.0),  # Masses of the links
        max_velocities: tuple = (4 * pi, 9 * pi),  # Max angular velocities
        reward_type: str = "default",
    ):
        super().__init__(render_mode=render_mode)

        # Customizable parameters
        self.termination_height = termination_height
        self.friction = friction
        self.torque_scaling = torque_scaling
        self.gravity = gravity
        self.LINK_LENGTH_1, self.LINK_LENGTH_2 = link_lengths
        self.LINK_MASS_1, self.LINK_MASS_2 = link_masses
        self.MAX_VEL_1, self.MAX_VEL_2 = max_velocities

        # Update observation space with new velocity limits
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Scale available torques
        self.AVAIL_TORQUE = [-1.0 * torque_scaling, 0.0, 1.0 * torque_scaling]
        self.reward_type = reward_type

    def step(self, a):
        obs, reward, terminated, truncated, info = super().step(a)
        if self.reward_type == "sparse":
            reward = 1.0 if terminated else 0.0
        elif self.reward_type == "scaled":
            reward = -1.0 / 250 if not terminated else 0.0
        return obs, reward, terminated, truncated, info

    def _terminal(self):
        """Check if the free end of the acrobot has reached the termination height."""
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > self.termination_height)

    def _dsdt(self, s_augmented):
        """Computes the dynamics of the system with the customized parameters."""
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = l1 / 2.0
        lc2 = self.LINK_LENGTH_2 / 2.0
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.gravity
        friction = self.friction
        a = s_augmented[-1]
        s = s_augmented[:-1]

        theta1, theta2, dtheta1, dtheta2 = s
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )

        # Compute angular accelerations
        if self.book_or_nips == "nips":
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        # Apply friction to angular velocities
        ddtheta1 -= friction * dtheta1
        ddtheta2 -= friction * dtheta2

        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        termination_y = offset + self.termination_height * scale
        pygame.draw.line(
            surf,
            color=(0, 0, 0),
            start_pos=(-2.2 * scale + offset, termination_y),
            end_pos=(2.2 * scale + offset, termination_y),
        )

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


def _clip_value(x, min_val, max_val):
    """Simple clamp to [min_val, max_val]."""
    return min(max(x, min_val), max_val)


def _normalize_value(x, scale):
    """
    Basic approach: x / scale
    If you want further clamp, do e.g. clamp to [-1, 1].
    """
    val = x / scale
    # Optionally clamp to [-1, 1].
    return _clip_value(val, -1.0, 1.0)


CR_STATE_W = 96  # less than Atari 160x192
CR_STATE_H = 96
CR_VIDEO_W = 600
CR_VIDEO_H = 400
CR_WINDOW_W = 1000
CR_WINDOW_H = 800

CR_SCALE = 6.0  # Track scale
CR_TRACK_RAD = 900 / CR_SCALE  # Track is heavily morphed circle with this radius
CR_PLAYFIELD = 2000 / CR_SCALE  # Game over boundary
CR_FPS = 50  # Frames per second
CR_ZOOM = 2.7  # Camera zoom
CR_ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


CR_TRACK_DETAIL_STEP = 21 / CR_SCALE
CR_TRACK_TURN_RATE = 0.31
CR_TRACK_WIDTH = 40 / CR_SCALE
CR_BORDER = 8 / CR_SCALE
CR_BORDER_MIN_COUNT = 4
CR_GRASS_DIM = CR_PLAYFIELD / 20.0
CR_MAX_SHAPE_DIM = (
    max(CR_GRASS_DIM, CR_TRACK_WIDTH, CR_TRACK_DETAIL_STEP)
    * math.sqrt(2)
    * CR_ZOOM
    * CR_SCALE
)
CR_NO_FREEZE = 16384
CR_ANCHORS = [
    1,
    3,
    6,
    10,
    15,
    21,
    28,
    36,
    45,
]


class CarRacingFixedMap(CarRacing):
    def __init__(
        self,
        render_mode=None,
        verbose=False,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        map_seed=0,
        fixed_start=True,
        backwards_tolerance=5,
        grass_tolerance=3,
        number_of_initial_states=32,
        init_seed=None,
        vector_obs=False,
    ):
        self.map_seed = map_seed
        self.fixed_start = fixed_start
        self.backwards_tolerance = backwards_tolerance
        self.grass_tolerance = grass_tolerance
        self.number_of_initial_states = max(1, number_of_initial_states)
        self.init_seed = init_seed
        self.vector_obs = vector_obs

        super().__init__(
            render_mode=render_mode,
            verbose=verbose,
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous,
        )

        # Extra trackers
        self.on_grass_counter = 0
        self._last_progress_idx = None
        self._backwards_counter = 0

        # We store a deterministic permutation if fixed_start=False and init_seed is not None
        self._initial_start_indices = None
        self._initial_idx_pointer = 0

        if not self.fixed_start and self.init_seed is not None:
            # Create a RNG with init_seed
            rng = np.random.default_rng(self.init_seed)
            # We'll shuffle the range [0..(number_of_initial_states-1)]
            self._initial_start_indices = rng.permutation(self.number_of_initial_states)
        elif not self.fixed_start:
            # If init_seed=None, each reset we randomize
            self._initial_start_indices = None

        # If vector obs, override observation_space
        if self.vector_obs:
            # Example dimension: 7 + len(CR_ANCHORS)*3
            # (heading_error, car_angular_vel, v_x, v_y, local_y, distance_left, distance_right) = 7
            # plus anchor info
            obs_dim = 7 + len(CR_ANCHORS) * 3
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    def reset(self, *, seed=None, options=None):
        """
        Overriding reset:
        1) Calls parent reset() logic to do track creation, domain randomize, etc.
        2) Then chooses start_idx based on fixed_start / init_seed.
        3) Respawns car at that start_idx.
        4) If vector_obs, returns vector. Otherwise, returns state_pixels.
        """
        # Step 1: Do parent's reset (which calls _create_track(), etc.)
        obs, info = super().reset(seed=seed, options=options)

        # Step 2: Decide car start_idx
        if self.fixed_start:
            # Always index 0
            start_idx = 0
        else:
            spacing = max(1, len(self.track) // self.number_of_initial_states)
            base_indices = [i * spacing for i in range(self.number_of_initial_states)]

            if self.init_seed is None:
                # fully random each reset
                idx = random.randint(0, self.number_of_initial_states - 1)
                start_idx = base_indices[idx]
            else:
                # deterministic shuffle, cycle
                if self._initial_start_indices is None:
                    raise ValueError("No _initial_start_indices, but init_seed is set.")
                index_in_list = self._initial_start_indices[self._initial_idx_pointer]
                start_idx = base_indices[index_in_list]
                self._initial_idx_pointer = (
                    self._initial_idx_pointer + 1
                ) % self.number_of_initial_states

        # Step 3: Respawn car at chosen index
        beta, x, y = self.track[start_idx][1:4]
        if self.car is not None:
            self.car.destroy()
        self.car = Car(self.world, beta, x, y)

        # Step 4: Reset custom trackers
        self._last_progress_idx = self._get_progress_index()
        self._backwards_counter = 0
        self.on_grass_counter = 0

        # If render_mode=human, do a render
        if self.render_mode == "human":
            self.render()

        # Overwrite obs if vector_obs
        if self.vector_obs:
            obs = self.get_vector_observation()

        return obs, info

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None, "Car not initialized."

        # === Process action ===
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"Invalid action `{action}`. Expected: {self.action_space}"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        # === Step physics ===
        self.car.step(1.0 / CR_FPS)
        self.world.Step(1.0 / CR_FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / CR_FPS

        # === Observation ===
        if self.vector_obs:
            obs = self.get_vector_observation()
        else:
            obs = self._render("state_pixels")

        # === Compute reward ===
        step_reward = 0
        terminated = False
        truncated = False

        if action is not None:  # skip on first reset() call
            self.reward -= 0.1
            self.car.fuel_spent = 0.0  # reset fuel counter (not penalizing)

            # Reward difference (delta reward)
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Episode truncation (finished lap)
            if self.tile_visited_count == len(self.track) or self.new_lap:
                truncated = True

            # Check if car is out of bounds
            x, y = self.car.hull.position
            if abs(x) > CR_PLAYFIELD or abs(y) > CR_PLAYFIELD:
                terminated = True
                step_reward = -100

        # === Custom termination: backwards and grass ===
        on_grass = self._check_on_grass()
        if on_grass:
            self.on_grass_counter += 1
        else:
            self.on_grass_counter = 0

        if on_grass and self.on_grass_counter >= self.grass_tolerance:
            terminated = True

        is_backwards = self._check_if_running_backwards()
        if is_backwards:
            self._backwards_counter += 1
        else:
            self._backwards_counter = 0

        if self._backwards_counter >= self.backwards_tolerance:
            terminated = True

        # === Render if human ===
        if self.render_mode == "human":
            self.render()

        # === Return obs ===
        return obs, step_reward / 1e2, terminated, truncated, {}

    def get_vector_observation(self):
        """
        Generate a vector-based observation that describes the car's state
        and its relation to the local track geometry.

        The observation includes:
            - Car's heading error relative to the current track tangent (normalized)
            - Car's angular velocity (yaw rate), normalized
            - Car's velocity in the track-aligned coordinate frame (forward and lateral components), normalized
            - Car's lateral displacement (local_y) relative to the current track centerline, normalized
            - Distance to the left and right borders from the car's position (optional), normalized
            - Relative positions and heading differences of multiple anchor points ahead along the track, normalized

        Key properties:
            - Local coordinate frame centered on the car, aligned to the current track tangent.
            - No global track position information (no absolute coordinates, no progress index).
            - Suitable for learning purely from local observations without requiring additional shaping rewards.

        Returns:
            obs (np.ndarray): A 1D float32 array representing the vectorized observation.
        """

        assert self.car is not None, "Car object not initialized!"

        # === Car's physical state ===
        car_pos = self.car.hull.position  # (x, y) world coordinates
        car_angle = self.car.hull.angle  # heading angle (radians)
        car_speed = self.car.hull.linearVelocity  # (vx, vy) world frame linear velocity
        car_angular_vel = (
            self.car.hull.angularVelocity
        )  # angular velocity (radians/sec)

        # === Get the current progress point on the track ===
        progress_idx = self._get_progress_index()
        track_point = self.track[progress_idx]
        track_x, track_y = track_point[2:4]  # track point position (x, y)
        track_beta = track_point[1]  # track tangent direction (heading)

        # === Calculate local coordinate frame transformation ===
        dx = car_pos[0] - track_x  # displacement in global frame (x)
        dy = car_pos[1] - track_y  # displacement in global frame (y)

        cos_b = np.cos(track_beta)  # cos of track heading
        sin_b = np.sin(track_beta)  # sin of track heading

        # === Car position in track-aligned local frame ===
        # local_x: longitudinal position (rarely useful if progress_idx is closest)
        # local_y: lateral offset from centerline (positive right, negative left)
        local_x = dx * cos_b + dy * sin_b
        local_y = -dx * sin_b + dy * cos_b

        # === Car velocity in track-aligned local frame ===
        # v_x: forward speed; v_y: lateral sliding speed
        v_x = car_speed[0] * cos_b + car_speed[1] * sin_b
        v_y = -car_speed[0] * sin_b + car_speed[1] * cos_b

        # === Heading error (car's heading relative to track tangent), normalized ===
        raw_heading_error = car_angle - track_beta
        raw_heading_error = np.arctan2(
            np.sin(raw_heading_error), np.cos(raw_heading_error)
        )  # wrap to [-pi, pi]
        heading_error = _normalize_value(raw_heading_error, np.pi)

        # === Angular velocity (yaw rate), normalized ===
        norm_angular_vel = _normalize_value(car_angular_vel, 10.0)

        # === Forward and lateral velocities, normalized ===
        norm_vx = _normalize_value(v_x, 15.0)
        norm_vy = _normalize_value(v_y, 15.0)

        # === Lateral displacement relative to centerline, normalized ===
        # Assuming track width ≈ 6.66 units, scaled by 7 for margin
        norm_local_y = _normalize_value(local_y, 7.0)

        # === Distance to left and right track borders at the car's position, normalized ===
        distance_left, distance_right = self._get_border_distances(
            progress_idx, car_pos
        )
        norm_left = _normalize_value(distance_left, 7.0)
        norm_right = _normalize_value(distance_right, 7.0)

        # === Anchor points ===
        # Each anchor provides:
        # - Relative (local_x, local_y) position in car's local track frame
        # - Heading difference (anchor tangent relative to current track tangent)
        anchors = []
        for anchor_step in CR_ANCHORS:
            idx = (progress_idx + anchor_step) % len(self.track)
            anchor_x, anchor_y = self.track[idx][2:4]
            anchor_beta = self.track[idx][1]

            # Relative vector (anchor to car), in track frame
            dx_a = anchor_x - car_pos[0]
            dy_a = anchor_y - car_pos[1]

            local_anchor_x = dx_a * cos_b + dy_a * sin_b
            local_anchor_y = -dx_a * sin_b + dy_a * cos_b

            # Heading difference between anchor and current track direction
            raw_anchor_diff = anchor_beta - track_beta
            raw_anchor_diff = np.arctan2(
                np.sin(raw_anchor_diff), np.cos(raw_anchor_diff)
            )
            anchor_heading_error = _normalize_value(raw_anchor_diff, np.pi)

            # Normalize anchor positions by some large factor (~50 units)
            norm_ax = _normalize_value(local_anchor_x, 50.0)
            norm_ay = _normalize_value(local_anchor_y, 50.0)

            anchors.extend([norm_ax, norm_ay, anchor_heading_error])

        # === Concatenate features into a single observation vector ===
        obs = np.array(
            [
                heading_error,  # 1
                norm_angular_vel,  # 2
                norm_vx,
                norm_vy,  # 3-4
                norm_local_y,  # 5
                norm_left,
                norm_right,  # 6-7
            ]
            + anchors,  # Anchor point features
            dtype=np.float32,
        )

        return obs

    def _get_border_distances(self, progress_idx, pos):
        """
        Calculate distances from current position to left and right track borders
        at a given track point.
        """
        track_point = self.track[progress_idx]
        beta = track_point[1]
        center_x, center_y = track_point[2:4]

        # Compute left and right border positions
        left_x = center_x - CR_TRACK_WIDTH * np.cos(beta)
        left_y = center_y - CR_TRACK_WIDTH * np.sin(beta)

        right_x = center_x + CR_TRACK_WIDTH * np.cos(beta)
        right_y = center_y + CR_TRACK_WIDTH * np.sin(beta)

        # Euclidean distances to each border
        dist_left = np.sqrt((pos[0] - left_x) ** 2 + (pos[1] - left_y) ** 2)
        dist_right = np.sqrt((pos[0] - right_x) ** 2 + (pos[1] - right_y) ** 2)

        return dist_left, dist_right

    # === Helper functions ===
    def _get_progress_index(self):
        """
        Returns the index of the closest track segment to the car.
        """
        car_pos = self.car.hull.position
        min_dist = float("inf")
        closest_idx = 0

        for i, (_, _, x, y) in enumerate(self.track):
            dist = np.sqrt((car_pos[0] - x) ** 2 + (car_pos[1] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def _check_if_running_backwards(self):
        """
        Returns True if the car is going backwards on the track.
        """
        current_idx = self._get_progress_index()

        # First step initialization
        if self._last_progress_idx is None:
            self._last_progress_idx = current_idx
            return False

        # Compute delta progress (forward: positive, backward: negative)
        delta = current_idx - self._last_progress_idx

        # Handle track looping
        if delta > len(self.track) / 2:
            delta -= len(self.track)
        elif delta < -len(self.track) / 2:
            delta += len(self.track)

        self._last_progress_idx = current_idx

        # Negative delta means moving backward
        return delta < -1  # You can adjust threshold if needed

    def _check_on_grass(self):
        """
        Return True if ALL wheels are outside road boundaries (on grass).
        """
        assert self.car is not None
        wheels = [wheel.position for wheel in self.car.wheels]

        if not self.road or len(self.road) == 0:
            print("[DEBUG] No road tiles available!")
            return True  # Assume on grass if road isn't there

        def point_on_road(pos):
            for tile in self.road:
                if tile.fixtures[0].TestPoint(pos):
                    return True
            return False

        # If any wheel is on the road, then not fully on grass
        for wheel_pos in wheels:
            if point_on_road(wheel_pos):
                return False

        return True  # All wheels are off-road

    def _create_track(self):
        def derive_new_seed(base_seed, retry_count):
            # Combine the original map_seed and retry_count in a deterministic way
            data = f"{base_seed}-{retry_count}".encode("utf-8")
            hashed = hashlib.sha256(data).hexdigest()
            # Convert hex digest to int32 range
            new_seed = int(hashed, 16) % (2**31)
            return new_seed

        max_retries = np.inf  # To avoid infinite loops
        retries = 0

        while retries < max_retries:
            # Deterministic init_seed transformation
            current_seed = derive_new_seed(self.map_seed, retries)
            map_random = np.random.RandomState(current_seed)

            # === START OF ORIGINAL GENERATION ===
            CHECKPOINTS = 12
            checkpoints = []
            for c in range(CHECKPOINTS):
                noise = map_random.uniform(0, 2 * np.pi * 1 / CHECKPOINTS)
                alpha = 2 * np.pi * c / CHECKPOINTS + noise
                rad = map_random.uniform(CR_TRACK_RAD / 3, CR_TRACK_RAD)

                if c == 0:
                    alpha = 0
                    rad = 1.5 * CR_TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * np.pi * c / CHECKPOINTS
                    self.start_alpha = 2 * np.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * CR_TRACK_RAD

                checkpoints.append((alpha, rad * np.cos(alpha), rad * np.sin(alpha)))

            self.road = []
            x, y, beta = 1.5 * CR_TRACK_RAD, 0, 0
            dest_i = 0
            laps = 0
            track = []
            no_freeze = CR_NO_FREEZE
            visited_other_side = False
            while True:
                alpha = np.arctan2(y, x)
                if visited_other_side and alpha > 0:
                    laps += 1
                    visited_other_side = False
                if alpha < 0:
                    visited_other_side = True
                    alpha += 2 * np.pi

                while True:
                    failed = True
                    while True:
                        dest_alpha, dest_x, dest_y = checkpoints[
                            dest_i % len(checkpoints)
                        ]
                        if alpha <= dest_alpha:
                            failed = False
                            break
                        dest_i += 1
                        if dest_i % len(checkpoints) == 0:
                            break

                    if not failed:
                        break

                    alpha -= 2 * np.pi
                    continue

                r1x = np.cos(beta)
                r1y = np.sin(beta)
                p1x = -r1y
                p1y = r1x
                dest_dx = dest_x - x
                dest_dy = dest_y - y
                proj = r1x * dest_dx + r1y * dest_dy

                while beta - alpha > 1.5 * np.pi:
                    beta -= 2 * np.pi
                while beta - alpha < -1.5 * np.pi:
                    beta += 2 * np.pi
                prev_beta = beta
                proj *= CR_SCALE
                if proj > 0.3:
                    beta -= min(CR_TRACK_TURN_RATE, abs(0.001 * proj))
                if proj < -0.3:
                    beta += min(CR_TRACK_TURN_RATE, abs(0.001 * proj))

                x += p1x * CR_TRACK_DETAIL_STEP
                y += p1y * CR_TRACK_DETAIL_STEP
                track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

                if laps > 4:
                    break
                no_freeze -= 1
                if no_freeze == 0:
                    break

            # Closed loop detection
            i1, i2 = -1, -1
            i = len(track)
            while True:
                i -= 1
                if i == 0:
                    success = False
                    break
                pass_through_start = track[i][0] > self.start_alpha >= track[i - 1][0]
                if pass_through_start and i2 == -1:
                    i2 = i
                elif pass_through_start and i1 == -1:
                    i1 = i
                    break

            if i1 == -1 or i2 == -1:
                success = False
            else:
                track = track[i1 : i2 - 1]
                first_beta = track[0][1]
                first_perp_x = np.cos(first_beta)
                first_perp_y = np.sin(first_beta)
                well_glued_together = np.sqrt(
                    (first_perp_x * (track[0][2] - track[-1][2])) ** 2
                    + (first_perp_y * (track[0][3] - track[-1][3])) ** 2
                )
                if well_glued_together > CR_TRACK_DETAIL_STEP:
                    success = False
                else:
                    success = True

            if success:
                break
            else:
                retries += 1
                if self.verbose:
                    print(
                        f"Retry {retries} for map_seed {self.map_seed} -> init_seed {current_seed}"
                    )

        if not success:
            raise RuntimeError(
                f"Failed to generate valid track after {max_retries} retries for map_seed {self.map_seed}"
            )

        # === REST OF ORIGINAL GENERATION ===
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(CR_BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > CR_TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == CR_BORDER_MIN_COUNT
            border[i] = good

        for i in range(len(track)):
            for neg in range(CR_BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - CR_TRACK_WIDTH * np.cos(beta1),
                y1 - CR_TRACK_WIDTH * np.sin(beta1),
            )
            road1_r = (
                x1 + CR_TRACK_WIDTH * np.cos(beta1),
                y1 + CR_TRACK_WIDTH * np.sin(beta1),
            )
            road2_l = (
                x2 - CR_TRACK_WIDTH * np.cos(beta2),
                y2 - CR_TRACK_WIDTH * np.sin(beta2),
            )
            road2_r = (
                x2 + CR_TRACK_WIDTH * np.cos(beta2),
                y2 + CR_TRACK_WIDTH * np.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]

            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append((vertices, t.color))
            self.road.append(t)

            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * CR_TRACK_WIDTH * np.cos(beta1),
                    y1 + side * CR_TRACK_WIDTH * np.sin(beta1),
                )
                b1_r = (
                    x1 + side * (CR_TRACK_WIDTH + CR_BORDER) * np.cos(beta1),
                    y1 + side * (CR_TRACK_WIDTH + CR_BORDER) * np.sin(beta1),
                )
                b2_l = (
                    x2 + side * CR_TRACK_WIDTH * np.cos(beta2),
                    y2 + side * CR_TRACK_WIDTH * np.sin(beta2),
                )
                b2_r = (
                    x2 + side * (CR_TRACK_WIDTH + CR_BORDER) * np.cos(beta2),
                    y2 + side * (CR_TRACK_WIDTH + CR_BORDER) * np.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )

        self.track = track
        return True

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((CR_WINDOW_W, CR_WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((CR_WINDOW_W, CR_WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * CR_SCALE * max(1 - self.t, 0) + CR_ZOOM * CR_SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (CR_WINDOW_W / 2 + trans[0], CR_WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(CR_WINDOW_W, CR_WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, CR_WINDOW_H - CR_WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (CR_VIDEO_W, CR_VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (CR_STATE_W, CR_STATE_H))
        else:
            return self.isopen

    def get_track_image(self, figsize=(10, 10), return_rgb=True):
        playfield = 2000 / CR_SCALE
        grass_dim = playfield / 20.0

        fig, ax = plt.subplots(figsize=figsize)

        # Remove padding and axis
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])

        # Background color
        bg_color = np.array(self.bg_color) / 255.0
        ax.set_facecolor(bg_color)

        # Limits
        ax.set_xlim(-playfield, playfield)
        ax.set_ylim(-playfield, playfield)

        # Grass patches
        grass_color = np.copy(bg_color)
        idx = np.random.randint(3)
        grass_color[idx] += 20 / 255.0

        step = 1
        for x in range(-20, 20, step):
            for y in range(-20, 20, step):
                rect_x = grass_dim * x
                rect_y = grass_dim * y
                rect = Rectangle(
                    (rect_x, rect_y),
                    grass_dim,
                    grass_dim,
                    facecolor=grass_color,
                    edgecolor="none",
                    antialiased=True,
                )
                ax.add_patch(rect)

        # Road tiles
        for poly, color in self.road_poly:
            poly_array = np.array(poly)
            color_norm = np.array(color) / 255.0
            patch = Polygon(
                poly_array,
                closed=True,
                facecolor=color_norm,
                edgecolor="none",
                antialiased=True,
            )
            ax.add_patch(patch)

        ax.set_aspect("equal")

        # Render as numpy
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )

        plt.close(fig)

        if return_rgb:
            return img
        else:
            return img[:, :, ::-1]


LL_INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder


class CustomLunarLander(LunarLander):
    """
    A custom LunarLander environment that allows:
    1) Custom gravity value.
    2) Fixed number of possible initial linear/angular velocities (e.g. 256),
       which can be used either in a deterministic cyclic order or picked
       randomly each reset, depending on the parameter use_deterministic_initial_states.

    Usage Example:
    -------------
    env = CustomLunarLander(
        render_mode="human",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        number_of_initial_states=256,
        use_deterministic_initial_states=True
    )

    Notes:
    ------
    - This environment retains the original step() logic from LunarLander.
    - The main difference is how we generate and assign the initial velocities
      (linear and angular) in reset().
    - The gravity is now directly set to the user-defined value via super().__init__.
    """

    def __init__(
        self,
        render_mode=None,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        lander_density=5.0,
        number_of_initial_states=256,
        use_deterministic_initial_states=True,
        init_seed=None,
    ):
        """
        Initialize the custom LunarLander environment.

        Parameters
        ----------
        render_mode : str or None
            Same meaning as in the original LunarLander (e.g., "human", "rgb_array").
        continuous : bool
            Whether to use the continuous action space version (same as the parent class).
        gravity : float
            Gravity to use in the simulation (must be between -12.0 and 0.0 in the original constraints).
        enable_wind : bool
            Whether to enable wind effects (same usage as parent class).
        wind_power : float
            Maximum magnitude of linear wind (same usage as parent class).
        turbulence_power : float
            Maximum magnitude of rotational wind (same usage as parent class).
        number_of_initial_states : int
            How many distinct initial velocity/angle-velocity combinations we store.
        use_deterministic_initial_states : bool
            - True: We cycle through the stored states in a fixed order (determined by init_seed).
            - False: We pick a random one from the stored states for each reset.
        init_seed : int or None
            Seed used to generate the set of initial states (and the deterministic order).
            If None, behavior follows default random seeding.
        """
        super().__init__(
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,  # user-defined gravity
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        self.lander_density = lander_density
        self.number_of_initial_states = number_of_initial_states
        self.use_deterministic_initial_states = use_deterministic_initial_states
        self.init_seed = init_seed

        # We will store the possible initial (vx, vy, omega) in this list.
        self._initial_states = []

        # We use the internal RNG from the parent or create a new one if needed.
        # If you need a specific random generator approach, you can do so here.
        if self.init_seed is not None:
            # Force a separate RNG if you want full control
            self._rng = np.random.default_rng(self.init_seed)
        else:
            # Fall back to parent's self.np_random
            self._rng = self.np_random

        # Generate all initial states
        self._generate_initial_states()

        # If deterministic, shuffle them once in a stable manner and cycle.
        # If not deterministic, we will pick randomly each reset.
        if self.use_deterministic_initial_states:
            # Shuffle them in a reproducible way using self._rng
            self._rng.shuffle(self._initial_states)
            self._current_idx = 0

    def _generate_initial_states(self):
        """
        Generate a fixed set of (vx, vy, omega) states
        with uniformly distributed directions and magnitudes,
        within a reasonable range to avoid lander 'flying away'.
        """
        self._initial_states = []

        max_speed = 5.0  # Reasonable maximum linear speed (units per second)
        min_speed = 0.0  # Optional: can set > 0 if you want to avoid zero speed

        max_angular_speed = (
            2.0  # Reasonable maximum angular velocity (radians per second)
        )

        for _ in range(self.number_of_initial_states):
            # Sample speed magnitude uniformly
            speed = self._rng.uniform(min_speed, max_speed)

            # Sample direction uniformly on [0, 2pi)
            theta = self._rng.uniform(0, 2 * np.pi)

            # Convert polar coordinates to vx, vy
            vx = speed * np.cos(theta)
            vy = speed * np.sin(theta)

            # Sample angular velocity uniformly
            omega = self._rng.uniform(-max_angular_speed, max_angular_speed)

            self._initial_states.append((vx, vy, omega))

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment:
        1) Calls the original parent's reset logic, which sets up terrain, lander body, etc.
           We skip the parent's random velocity assignment logic by temporarily disabling it.
        2) We then manually set the lander's linearVelocity and angularVelocity using
           one of the fixed initial states (either deterministic cycling or random).
        """
        # We want to keep parent's terrain and body creation, but not parent's random velocity.
        # So we do the following: temporarily modify LL_INITIAL_RANDOM = 0, so that the parent's
        # applyForce random velocity is effectively null. Then restore it after calling super().
        global LL_INITIAL_RANDOM
        original_val = LL_INITIAL_RANDOM
        LL_INITIAL_RANDOM = 0.0  # Force no random impulse from the parent's reset

        # Call parent's reset
        obs, info = super().reset(seed=seed, options=options)

        for fixture in self.lander.fixtures:
            fixture.density = self.lander_density
        self.lander.ResetMassData()

        # Restore original value
        LL_INITIAL_RANDOM = original_val

        # Now manually set the velocity & angular velocity from our stored states
        if self.use_deterministic_initial_states:
            vx, vy, omega = self._initial_states[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._initial_states)
        else:
            # completely random pick from the stored states
            idx = self._rng.integers(0, len(self._initial_states))
            vx, vy, omega = self._initial_states[idx]

        # Must ensure self.lander exists (it should after super().reset)
        if self.lander is not None:
            self.lander.linearVelocity = (vx, vy)
            self.lander.angularVelocity = omega

        return obs, info
