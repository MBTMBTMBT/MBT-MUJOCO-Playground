import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def modify_inverted_pendulum_xml(
    xml_file: str,
    length: float,
    pole_density: float,
    cart_density: float,
) -> str:
    """
    Modify an inverted pendulum XML by updating pendulum length and density values for pole and cart.

    Args:
        xml_file (str): Path to the original XML file.
        length (float): Pendulum length in meters.
        pole_density (float): Density of the pendulum body (kg/m³).
        cart_density (float): Density of the cart body (kg/m³).

    Returns:
        str: Path to the modified temporary XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Update pole length
    cpole = root.find(".//geom[@name='cpole']")
    if cpole is not None:
        cpole.set("fromto", f"0 0 0 0.001 0 {length}")
        cpole.set("density", str(pole_density))
    else:
        raise ValueError("Cannot find geom with name 'cpole' in the XML.")

    # Update cart density
    cart_geom = root.find(".//geom[@name='cart']")
    if cart_geom is not None:
        cart_geom.set("density", str(cart_density))
    else:
        raise ValueError("Cannot find geom with name 'cart' in the XML.")

    # Remove <inertial> if any (optional, MuJoCo will recompute from density and size)
    pole_body = root.find(".//body[@name='pole']")
    if pole_body is not None:
        existing_inertial = pole_body.find("inertial")
        if existing_inertial is not None:
            pole_body.remove(existing_inertial)
    else:
        raise ValueError("Cannot find body with name 'pole' in the XML.")

    # Save updated XML to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomInvertedPendulum(InvertedPendulumEnv):
    """
    Extended version of the MuJoCo Inverted Pendulum environment,
    which allows precise control over the initial state distribution,
    system parameters, and the way that episodes are reset.

    Key features:
    - Ability to modify the pendulum length, pole density, and cart density via XML patching at construction time.
    - Flexible initial state sampling: support for uniform or Gaussian distributions, or providing a user-defined finite set of initial states.
    - Option to set a fixed number of sampled initial states (`n_states`), not just grid discretisation per state dimension.
    - Ability to explicitly specify the value ranges for each initial state variable (`init_ranges`).
    - Multiple sampling policies for the reset order: purely random, sequential cycling, or pseudo-random deterministic cycling using a given seed.

    Args:
        length (float): Length in metres of the pendulum pole.
        pole_density (float): Density (kg/m³) for the pendulum body.
        cart_density (float): Density (kg/m³) for the cart body.
        xml_file (str): Path to the MuJoCo XML describing the inverted pendulum.
        initial_states (np.ndarray or None): If a numpy array/list is provided, use this finite set of states for environment reset.
        init_dist (str): Sampling distribution if `initial_states` is not provided, 'uniform' or 'gaussian' (default 'uniform').
        n_states (int): The total number of initial states to generate (default 100).
        init_ranges (dict or None): Value range for each element of the state.
        sample_mode (str): One of 'random', 'sequential', or 'seeded'. Determines the policy for selecting the next initial state at reset.
        seed (int or None): RNG seed for reproducibility if applicable.
        **kwargs: Any extra kwargs are passed to the superclass (InvertedPendulumEnv).

    Notes:
        - The code ensures that the underlying MuJoCo model XML is modified on-the-fly each time an instance is created, and the temporary file
          is properly cleaned up when the environment is closed.
        - For init_dist='uniform', sampling is done independently per state variable within the specified ranges.
        - For init_dist='gaussian', sampling is also independent per dimension, but clipped to the bounds set by init_ranges.
        - Explicitly providing `initial_states` has highest priority and overrides all automatic sampling.
        - State is always specified in the order: [cart position, cart velocity, pole angle, pole angular velocity].

    Typical usage:
        env = CustomInvertedPendulum(length=0.7, pole_density=1200, n_states=500,
                                    init_ranges=[(-.1, .1)]*4, sample_mode="sequential")

    """

    DEFAULT_INIT_RANGES = {
        'cart_position': (-0.05, 0.05),
        'cart_velocity': (-0.05, 0.05),
        'pole_angle': (-0.05, 0.05),
        'pole_ang_vel': (-0.05, 0.05)
    }

    STATE_KEYS = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_ang_vel']

    def __init__(
        self,
        length: float = 0.6,
        pole_density: float = 1000.0,
        cart_density: float = 1000.0,
        xml_file: str = "./assets/inverted_pendulum.xml",
        initial_states=None,
        init_dist: str = "uniform",
        n_states: int = 100,
        init_ranges: dict = None,
        sample_mode: str = "random",
        dense_reward: bool = False,
        seed: int = None,
        **kwargs,
    ):
        # Patch the XML with the specified cart/pole parameters.
        modified_xml_file = modify_inverted_pendulum_xml(
            xml_file, length, pole_density, cart_density
        )
        super().__init__(xml_file=modified_xml_file, **kwargs)
        self._temp_xml_path = modified_xml_file

        self.dense_reward = dense_reward
        self.cart_range = 1.0
        self.length = length

        # Initialise RNG for determinism if seed is provided.
        self._rng = np.random.default_rng(seed)
        self.sample_mode = sample_mode
        self._init_index = 0  # Used for sequential and seeded sampling.

        # Use supplied state set if provided, otherwise sample finite set as requested.
        if init_ranges is None:
            self.init_ranges = self.DEFAULT_INIT_RANGES.copy()
        else:
            self.init_ranges = self.DEFAULT_INIT_RANGES.copy()
            for key in init_ranges:
                if key in self.init_ranges:
                    self.init_ranges[key] = init_ranges[key]
                else:
                    raise KeyError(f"Unknown state key: {key}")

        if initial_states is not None:
            self.initial_states = np.asarray(initial_states)
        else:
            lows = np.array([self.init_ranges[key][0] for key in self.STATE_KEYS])
            highs = np.array([self.init_ranges[key][1] for key in self.STATE_KEYS])
            if init_dist == "uniform":
                self.initial_states = self._rng.uniform(
                    low=lows, high=highs, size=(n_states, 4)
                )
            elif init_dist == "gaussian":
                self.initial_states = np.clip(
                    self._rng.normal(loc=0, scale=0.02, size=(n_states, 4)),
                    lows, highs
                )
            else:
                raise ValueError(
                    "Unsupported init_dist: choose 'uniform', 'gaussian' or provide initial_states."
                )

        # Precompute the sampling order for non-random reset modes.
        if self.sample_mode == "seeded":
            self._index_order = np.arange(len(self.initial_states))
            self._rng.shuffle(self._index_order)
        elif self.sample_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None  # Random sampling.

    def reset(self, **kwargs):
        """
        Reset the environment and set the initial simulator state according to the configured sampling mode.

        Returns:
            obs (np.ndarray): The observation after environment reset.
            info (dict): Additional environment information (empty by default).
        """
        # Select an initial state from the pool, according to the requested sampling policy.
        if self.sample_mode in ("sequential", "seeded"):
            idx = self._index_order[self._init_index]
            self._init_index = (self._init_index + 1) % len(self.initial_states)
            state = self.initial_states[idx]
        else:
            idx = self._rng.integers(len(self.initial_states))
            state = self.initial_states[idx]

        # State order: [cart pos, cart vel, pole angle, pole ang vel]
        qpos = np.array([state[0], state[2]], dtype=np.float64)   # Position-based components.
        qvel = np.array([state[1], state[3]], dtype=np.float64)   # Velocity-based components.
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.dense_reward:
            # Get current cart position and pole angle (assuming obs[0] is cart position, obs[2] is pole angle)
            cart_pos = obs[0]
            pole_angle = obs[2]  # pole angle in radians

            # Calculate horizontal position of the pole tip
            pole_tip_pos = cart_pos + self.length * np.sin(pole_angle)

            # Normalize distance: absolute distance of pole tip from the center line,
            # divided by the maximum possible (cart max range + pole length)
            max_dist = self.cart_range + self.length
            distance = np.abs(pole_tip_pos)
            norm_dist = distance / max_dist

            # Exponential reward: closer to the center line, higher the reward at a much faster rate.
            # alpha > 0 adjusts the steepness, typical value in [3, 10]
            alpha = 7.0
            reward = np.exp(-alpha * norm_dist)  # Best at the center (reward approaches 1), drops off quickly

        return obs, reward, terminated, truncated, info

    def set_state(self, qpos, qvel):
        """
        Set the state of the MuJoCo simulator explicitly.

        Args:
            qpos (np.ndarray): Array representing the positional states.
            qvel (np.ndarray): Array representing the velocity states.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        """
        Clean up the temporary XML file created for this environment instance.
        Always call this method when the environment is no longer needed to avoid
        leaving unnecessary files in the filesystem.
        """
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)


if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from sbx import SAC
    from utils import EvalProgressGifCallback
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    # Set the main output directory
    main_save_dir = "./exp_compare_length"
    os.makedirs(main_save_dir, exist_ok=True)

    # Experiment pole lengths and algorithm settings
    pole_lengths = [1.0, 2.0, 3.0]
    algorithms = [("SAC", SAC, 125_000), ]
    num_runs = 5

    # Shared environment configuration
    common_kwargs = dict(
        pole_density=1000,
        cart_density=1000,
        xml_file="./assets/inverted_pendulum.xml"
    )

    env_configs = {
        "train": dict(
            n_states=20,
            init_ranges={
                'cart_position': (-0.25, 0.25),
                'cart_velocity': (-0.1, 0.1),
                'pole_angle': (-0.1, 0.1),
                'pole_ang_vel': (-0.1, 0.1),
            },
            init_dist="uniform",
            sample_mode="random",
            dense_reward=True,
            seed=None,
            render_mode=None,
        ),
        "eval": dict(
            n_states=32,
            init_ranges={
                'cart_position': (-0.25, 0.25),
                'cart_velocity': (-0.1, 0.1),
                'pole_angle': (-0.1, 0.1),
                'pole_ang_vel': (-0.1, 0.1),
            },
            init_dist="uniform",
            sample_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode=None,
        ),
        "gif": dict(
            n_states=32,
            init_ranges={
                'cart_position': (-0.25, 0.25),
                'cart_velocity': (-0.1, 0.1),
                'pole_angle': (-0.1, 0.1),
                'pole_ang_vel': (-0.1, 0.1),
            },
            init_dist="uniform",
            sample_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode="rgb_array"
        )
    }
    max_steps = 250
    n_envs = 10

    def make_env(env_type, length):
        """
        Utility to instantiate an environment for a given type and pole length.
        """
        kwargs = {**common_kwargs, "length": length, **env_configs[env_type]}
        env = CustomInvertedPendulum(**kwargs)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env


    # Store result file locations
    all_result_csvs = []

    # Run experiments for each pole length and repetition (SAC only)
    for length in pole_lengths:
        print(f"\n=== Experiment for Pole Length: {length} ===\n")
        for algo_name, algo_class, total_timesteps in algorithms:
            for run_id in range(1, num_runs + 1):
                print(f"[{algo_name}] Length={length}, Run {run_id}/{num_runs}")
                save_dir = os.path.join(main_save_dir, f"{algo_name}_Length{length}_Run{run_id}")
                os.makedirs(save_dir, exist_ok=True)
                exp_name = f"{algo_name}_Length{length}_Run{run_id}"

                train_env = SubprocVecEnv([lambda: make_env("train", length)] * n_envs)
                eval_env = DummyVecEnv([lambda: make_env("eval", length)])
                gif_env = DummyVecEnv([lambda: make_env("gif", length)])

                eval_episodes = env_configs["eval"]["n_states"]
                eval_interval = total_timesteps // 25
                optimal_score = max_steps * 0.8

                callback = EvalProgressGifCallback(
                    name=exp_name,
                    eval_env=eval_env,
                    eval_episodes=eval_episodes,
                    eval_interval=eval_interval,
                    save_dir=save_dir,
                    total_timesteps=total_timesteps,
                    optimal_score=optimal_score,
                    gif_env=gif_env,
                    gif_num_episodes=env_configs["gif"]["n_states"],
                    verbose=1,
                )

                model = algo_class(
                    "MlpPolicy",
                    train_env,
                    learning_rate=1e-4,
                    buffer_size=total_timesteps // 10,
                    batch_size=1000,
                    train_freq=n_envs,
                    gradient_steps=n_envs,
                    learning_starts=1000,
                    verbose=0,
                    tensorboard_log=os.path.join(save_dir, "tb"),
                )

                # Start training for one run
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                )
                csv_path = os.path.join(save_dir, f"{exp_name}_result.csv")
                all_result_csvs.append({
                    "algo": algo_name,
                    "length": length,
                    "run": run_id,
                    "csv": csv_path,
                    "curve_img": os.path.join(save_dir, f"{exp_name}_curve.png"),
                })

    # ======================= Merge and Plot Comparative Curves ========================
    # Aggregate all results into a single DataFrame
    all_dfs = []
    for info in all_result_csvs:
        df = pd.read_csv(info["csv"])
        df["algorithm"] = info["algo"]
        df["length"] = info["length"]
        df["run"] = info["run"]
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    # First plot: raw average reward curve
    plt.figure(figsize=(10, 6))
    for length in pole_lengths:
        sub = all_df[(all_df["length"] == length)]
        grouped = sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        plt.plot(grouped["timesteps"], grouped["mean_reward"]["mean"],
                 label=f"SAC length={length}")
        plt.fill_between(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"] - grouped["mean_reward"]["std"],
            grouped["mean_reward"]["mean"] + grouped["mean_reward"]["std"],
            alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("SAC: Comparison for Different Pole Lengths (Averaged, ±STD)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(main_save_dir, "compare_lengths_raw.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Raw comparison figure saved to: {save_path}")

    # Second plot: independently normalised (each curve divided by its own maximum)
    plt.figure(figsize=(10, 6))
    for length in pole_lengths:
        sub = all_df[(all_df["length"] == length)]
        grouped = sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        # Each curve uses its max for normalisation
        local_max = grouped["mean_reward"]["mean"].max()
        norm_mean = grouped["mean_reward"]["mean"] / local_max
        norm_std = grouped["mean_reward"]["std"] / local_max
        plt.plot(grouped["timesteps"], norm_mean,
                 label=f"SAC length={length}")
        plt.fill_between(
            grouped["timesteps"],
            norm_mean - norm_std,
            norm_mean + norm_std,
            alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Normalised Mean Reward (per curve)")
    plt.title("SAC: Curve-wise Normalised Comparison (Averaged, ±STD)")
    plt.legend()
    plt.grid(True)
    norm_path = os.path.join(main_save_dir, "compare_lengths_curvewise_normalised.png")
    plt.savefig(norm_path)
    plt.show()
    print(f"Curve-wise normalised comparison figure saved to: {norm_path}")
