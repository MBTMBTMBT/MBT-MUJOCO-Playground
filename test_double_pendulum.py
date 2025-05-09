import os
import pandas as pd
import matplotlib.pyplot as plt
from sbx import SAC, PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import custom_mujoco

if __name__ == "__main__":
    # Output directory for experiment results
    main_save_dir = "./exp_compare_length_config_doublependulum"
    os.makedirs(main_save_dir, exist_ok=True)

    # Vary the lower pole (pole1) length while keeping total length constant at 1.2
    pole1_lengths = [0.9, 0.6, 0.3]
    total_length = 1.2
    algorithms = [
        ("PPO", 750_000),
        ("SAC", 250_000),
    ]
    num_runs = 1

    # Fixed environment parameters
    common_kwargs = dict(
        pole1_density=1000.0,
        pole2_density=1000.0,
        cart_density=1000.0,
    )

    # Shared evaluation configurations
    state_ranges = [
        (-0.25, 0.25),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),
    ]

    env_configs = {
        "train": dict(
            n_states=20,
            init_ranges=state_ranges,
            init_dist="uniform",
            init_mode="random",
            dense_reward=True,
            seed=None,
            render_mode=None,
        ),
        "eval": dict(
            n_states=32,
            init_ranges=state_ranges,
            init_dist="uniform",
            init_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode=None,
        ),
        "gif": dict(
            n_states=32,
            init_ranges=state_ranges,
            init_dist="uniform",
            init_mode="sequential",
            dense_reward=True,
            seed=123,
            render_mode="rgb_array",
        ),
    }

    max_steps = 250
    n_envs = 10

    def make_env(env_type, pole1_length):
        pole2_length = total_length - pole1_length
        kwargs = {
            **common_kwargs,
            "pole1_length": pole1_length,
            "pole2_length": pole2_length,
            **env_configs[env_type],
            "max_episode_steps": max_steps,
        }
        return gym.make("CustomInvertedDoublePendulum", **kwargs)

    all_result_csvs = []

    for length in pole1_lengths:
        print(
            f"\n=== Experiment: Pole1 Length {length}, Pole2 Length {total_length - length} ===\n"
        )
        for algo_name, total_timesteps in algorithms:
            for run_id in range(1, num_runs + 1):
                save_dir = os.path.join(
                    main_save_dir, f"{algo_name}_Pole1Len{length}_Run{run_id}"
                )
                os.makedirs(save_dir, exist_ok=True)
                exp_name = f"{algo_name}_Pole1Len{length}_Run{run_id}"

                train_env = SubprocVecEnv([lambda: make_env("train", length)] * n_envs)
                eval_env = DummyVecEnv([lambda: make_env("eval", length)])
                gif_env = DummyVecEnv([lambda: make_env("gif", length)])

                eval_episodes = env_configs["eval"]["n_states"]
                eval_interval = total_timesteps // 25
                optimal_score = max_steps * 0.9

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

                model.learn(total_timesteps=total_timesteps, callback=callback)
                csv_path = os.path.join(save_dir, f"{exp_name}_result.csv")
                all_result_csvs.append(
                    {
                        "algo": algo_name,
                        "pole1_length": length,
                        "run": run_id,
                        "csv": csv_path,
                        "curve_img": os.path.join(save_dir, f"{exp_name}_curve.png"),
                    }
                )

    all_dfs = []
    for info in all_result_csvs:
        df = pd.read_csv(info["csv"])
        df["algorithm"] = info["algo"]
        df["pole1_length"] = info["pole1_length"]
        df["run"] = info["run"]
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    # Plot 1: raw mean reward
    plt.figure(figsize=(10, 6))
    for length in pole1_lengths:
        sub = all_df[(all_df["pole1_length"] == length)]
        grouped = (
            sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        )
        plt.plot(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"],
            label=f"Pole1 Len={length:.2f}",
        )
        plt.fill_between(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"] - grouped["mean_reward"]["std"],
            grouped["mean_reward"]["mean"] + grouped["mean_reward"]["std"],
            alpha=0.2,
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("SAC: Varying Pole1 Length (Total Length = 1.2)")
    plt.legend()
    plt.grid(True)
    raw_path = os.path.join(main_save_dir, "compare_pole_lengths_raw.png")
    plt.savefig(raw_path)
    print(f"Raw comparison figure saved to: {raw_path}")

    # Plot 2: normalized curves
    plt.figure(figsize=(10, 6))
    for length in pole1_lengths:
        sub = all_df[(all_df["pole1_length"] == length)]
        grouped = (
            sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        )
        local_max = grouped["mean_reward"]["mean"].max()
        norm_mean = grouped["mean_reward"]["mean"] / local_max
        norm_std = grouped["mean_reward"]["std"] / local_max
        plt.plot(grouped["timesteps"], norm_mean, label=f"Pole1 Len={length:.2f}")
        plt.fill_between(
            grouped["timesteps"], norm_mean - norm_std, norm_mean + norm_std, alpha=0.2
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Mean Reward")
    plt.title("SAC: Normalized Performance vs Pole1 Length")
    plt.legend()
    plt.grid(True)
    norm_path = os.path.join(main_save_dir, "compare_pole_lengths_normalized.png")
    plt.savefig(norm_path)
    print(f"Normalized comparison figure saved to: {norm_path}")
