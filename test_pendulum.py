import os
import pandas as pd
import matplotlib.pyplot as plt
from sbx import SAC, PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import custom_mujoco


if __name__ == "__main__":
    # Set the main output directory
    main_save_dir = "./exp_compare_length_pendulum"
    os.makedirs(main_save_dir, exist_ok=True)

    # Experiment pole lengths and algorithm settings
    pole_lengths = [1.0, 2.0, 3.0]
    algorithms = [
        # ("PPO", 450_000),
        ("SAC", 150_000),
    ]
    num_runs = 1

    # Shared environment configuration
    common_kwargs = dict(
        pole_density=1000,
        cart_density=1000,
    )

    env_configs = {
        "train": dict(
            n_states=20,
            init_ranges={
                "cart_position": (-0.25, 0.25),
                "cart_velocity": (-0.1, 0.1),
                "pole_angle": (-0.1, 0.1),
                "pole_ang_vel": (-0.1, 0.1),
            },
            init_dist="uniform",
            init_mode="random",
            dense_reward=True,
            seed=None,
            render_mode=None,
        ),
        "eval": dict(
            n_states=32,
            init_ranges={
                "cart_position": (-0.25, 0.25),
                "cart_velocity": (-0.1, 0.1),
                "pole_angle": (-0.1, 0.1),
                "pole_ang_vel": (-0.1, 0.1),
            },
            init_dist="uniform",
            init_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode=None,
        ),
        "gif": dict(
            n_states=32,
            init_ranges={
                "cart_position": (-0.25, 0.25),
                "cart_velocity": (-0.1, 0.1),
                "pole_angle": (-0.1, 0.1),
                "pole_ang_vel": (-0.1, 0.1),
            },
            init_dist="uniform",
            init_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode="rgb_array",
        ),
    }
    max_steps = 250
    n_envs = 10

    def make_env(env_type, length):
        """
        Utility to instantiate an environment for a given type and pole length.
        """
        kwargs = {
            **common_kwargs,
            "length": length,
            **env_configs[env_type],
            "max_episode_steps": max_steps,
        }
        env = gym.make("CustomInvertedPendulum", **kwargs)
        return env

    # Store result file locations
    all_result_csvs = []

    # Run experiments for each pole length and repetition (SAC only)
    for length in pole_lengths:
        print(f"\n=== Experiment for Pole Length: {length} ===\n")
        for algo_name, total_timesteps in algorithms:
            for run_id in range(1, num_runs + 1):
                print(f"[{algo_name}] Length={length}, Run {run_id}/{num_runs}")
                save_dir = os.path.join(
                    main_save_dir, f"{algo_name}_Length{length}_Run{run_id}"
                )
                os.makedirs(save_dir, exist_ok=True)
                exp_name = f"{algo_name}_Length{length}_Run{run_id}"

                train_env = SubprocVecEnv([lambda: make_env("train", length)] * n_envs)
                eval_env = DummyVecEnv([lambda: make_env("eval", length)])
                gif_env = DummyVecEnv([lambda: make_env("gif", length)])

                eval_episodes = env_configs["eval"]["n_states"]
                eval_interval = total_timesteps // 25
                optimal_score = max_steps * 0.8

                callback = custom_mujoco.EvalProgressGifCallback(
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

                if algo_name == "PPO":
                    model = PPO(
                        "MlpPolicy",
                        train_env,
                        learning_rate=1e-4,
                        verbose=0,
                        tensorboard_log=os.path.join(save_dir, "tb"),
                    )

                elif algo_name == "SAC":
                    model = SAC(
                        "MlpPolicy",
                        train_env,
                        learning_rate=1e-4,
                        buffer_size=total_timesteps // 5,
                        batch_size=1000,
                        train_freq=n_envs,
                        gradient_steps=n_envs,
                        learning_starts=1000,
                        verbose=0,
                        tensorboard_log=os.path.join(save_dir, "tb"),
                    )

                else:
                    model = None

                # Start training for one run
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                )
                csv_path = os.path.join(save_dir, f"{exp_name}_result.csv")
                all_result_csvs.append(
                    {
                        "algo": algo_name,
                        "length": length,
                        "run": run_id,
                        "csv": csv_path,
                        "curve_img": os.path.join(save_dir, f"{exp_name}_curve.png"),
                    }
                )

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
        grouped = (
            sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        )
        plt.plot(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"],
            label=f"SAC length={length}",
        )
        plt.fill_between(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"] - grouped["mean_reward"]["std"],
            grouped["mean_reward"]["mean"] + grouped["mean_reward"]["std"],
            alpha=0.2,
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("SAC: Comparison for Different Pole Lengths (Averaged, ±STD)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(main_save_dir, "compare_lengths_raw.png")
    plt.savefig(save_path)
    # plt.show()
    print(f"Raw comparison figure saved to: {save_path}")

    # Second plot: independently normalised (each curve divided by its own maximum)
    plt.figure(figsize=(10, 6))
    for length in pole_lengths:
        sub = all_df[(all_df["length"] == length)]
        grouped = (
            sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        )
        # Each curve uses its max for normalisation
        local_max = grouped["mean_reward"]["mean"].max()
        norm_mean = grouped["mean_reward"]["mean"] / local_max
        norm_std = grouped["mean_reward"]["std"] / local_max
        plt.plot(grouped["timesteps"], norm_mean, label=f"SAC length={length}")
        plt.fill_between(
            grouped["timesteps"], norm_mean - norm_std, norm_mean + norm_std, alpha=0.2
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Normalised Mean Reward (per curve)")
    plt.title("SAC: Curve-wise Normalised Comparison (Averaged, ±STD)")
    plt.legend()
    plt.grid(True)
    norm_path = os.path.join(main_save_dir, "compare_lengths_curvewise_normalised.png")
    plt.savefig(norm_path)
    # plt.show()
    print(f"Curve-wise normalised comparison figure saved to: {norm_path}")
