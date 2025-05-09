import os
import pandas as pd
import matplotlib.pyplot as plt
from sbx import SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import custom_mujoco

if __name__ == "__main__":
    # Set the main output directory
    main_save_dir = "./exp_compare_top_density_doublependulum"
    os.makedirs(main_save_dir, exist_ok=True)

    # Vary the density of the upper pole only
    pole1_densities = [1250.0, 2500.0, 5000.0]
    algorithms = [("SAC", SAC, 150_000)]
    num_runs = 1

    # Shared environment parameters
    common_kwargs = dict(
        pole1_length=0.6,
        pole2_length=0.6,
        pole2_density=1000.0,
        cart_density=1000.0,
    )

    # Shared evaluation configurations
    state_ranges = [(-0.25, 0.25), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]

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
        )
    }
    max_steps = 250
    n_envs = 10

    def make_env(env_type, pole1_density):
        kwargs = {
            **common_kwargs,
            "pole1_density": pole1_density,
            **env_configs[env_type],
            "max_episode_steps": max_steps,
        }
        return gym.make("CustomInvertedDoublePendulum", **kwargs)

    all_result_csvs = []

    for density in pole1_densities:
        print(f"\n=== Experiment for Pole1 Density: {density} ===\n")
        for algo_name, algo_class, total_timesteps in algorithms:
            for run_id in range(1, num_runs + 1):
                print(f"[{algo_name}] Density={density}, Run {run_id}/{num_runs}")
                save_dir = os.path.join(main_save_dir, f"{algo_name}_Pole1Density{density}_Run{run_id}")
                os.makedirs(save_dir, exist_ok=True)
                exp_name = f"{algo_name}_Pole1Density{density}_Run{run_id}"

                train_env = SubprocVecEnv([lambda: make_env("train", density)] * n_envs)
                eval_env = DummyVecEnv([lambda: make_env("eval", density)])
                gif_env = DummyVecEnv([lambda: make_env("gif", density)])

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

                model.learn(total_timesteps=total_timesteps, callback=callback)
                csv_path = os.path.join(save_dir, f"{exp_name}_result.csv")
                all_result_csvs.append({
                    "algo": algo_name,
                    "density": density,
                    "run": run_id,
                    "csv": csv_path,
                    "curve_img": os.path.join(save_dir, f"{exp_name}_curve.png"),
                })

    all_dfs = []
    for info in all_result_csvs:
        df = pd.read_csv(info["csv"])
        df["algorithm"] = info["algo"]
        df["density"] = info["density"]
        df["run"] = info["run"]
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    # Plot 1: raw mean reward
    plt.figure(figsize=(10, 6))
    for density in pole1_densities:
        sub = all_df[(all_df["density"] == density)]
        grouped = sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        plt.plot(grouped["timesteps"], grouped["mean_reward"]["mean"],
                 label=f"SAC pole1_density={density}")
        plt.fill_between(
            grouped["timesteps"],
            grouped["mean_reward"]["mean"] - grouped["mean_reward"]["std"],
            grouped["mean_reward"]["mean"] + grouped["mean_reward"]["std"],
            alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("SAC: Comparison for Top Pole Densities (Averaged, ±STD)")
    plt.legend()
    plt.grid(True)
    raw_path = os.path.join(main_save_dir, "compare_top_pole_densities_raw.png")
    plt.savefig(raw_path)
    print(f"Raw comparison figure saved to: {raw_path}")

    # Plot 2: normalised reward curves
    plt.figure(figsize=(10, 6))
    for density in pole1_densities:
        sub = all_df[(all_df["density"] == density)]
        grouped = sub.groupby("timesteps").agg({"mean_reward": ["mean", "std"]}).reset_index()
        local_max = grouped["mean_reward"]["mean"].max()
        norm_mean = grouped["mean_reward"]["mean"] / local_max
        norm_std = grouped["mean_reward"]["std"] / local_max
        plt.plot(grouped["timesteps"], norm_mean,
                 label=f"SAC pole1_density={density}")
        plt.fill_between(
            grouped["timesteps"],
            norm_mean - norm_std,
            norm_mean + norm_std,
            alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Normalised Mean Reward (per curve)")
    plt.title("SAC: Top Pole Density - Curve-wise Normalised Comparison")
    plt.legend()
    plt.grid(True)
    norm_path = os.path.join(main_save_dir, "compare_top_pole_densities_normalised.png")
    plt.savefig(norm_path)
    print(f"Normalised comparison figure saved to: {norm_path}")
