import os
from typing import Union

import cv2
import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def make_venv():
    pass


class EvalProgressGifCallback(BaseCallback):
    def __init__(
        self,
        name: str,
        eval_env,
        eval_episodes,
        eval_interval: int,
        save_dir: str,
        total_timesteps: int,
        optimal_score: Union[int, float],
        gif_env=None,
        gif_num_episodes=8,
        verbose=1,
    ):
        super().__init__(verbose)
        self.name = name
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score
        self.save_dir = save_dir
        self.gif_env = gif_env
        self.gif_num_episodes = gif_num_episodes
        self.eval_episodes = eval_episodes

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.rewards = []
        self.last_eval_step = 0
        self.original_model = None

        # Save path for best model
        self.best_model_path = os.path.join(
            self.save_dir,
            f"{self.name}_best.zip",
        )

        self.total_timesteps = total_timesteps
        self.pbar = None
        self._last_num_timesteps = 0

    def _on_training_start(self):
        self.step_reached_optimal = None
        # Force evaluation at step 0
        self.last_eval_step = -self.eval_interval
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            mininterval=5,
            maxinterval=25,
            smoothing=0.9,
            dynamic_ncols=True,
        )
        self._last_num_timesteps = 0
        self._on_step()

    def _on_step(self) -> bool:
        # Update progress bar
        delta_steps = self.num_timesteps - self._last_num_timesteps
        self.pbar.update(delta_steps)
        self._last_num_timesteps = self.num_timesteps

        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                env=self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Record evaluation results
            self.rewards.append((self.num_timesteps, mean_reward, std_reward))

            # Save best model if applicable
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)

            # Record first time reaching optimal score
            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

            # Update progress bar display
            opt_text = (
                f"{self.step_reached_optimal}" if self.step_reached_optimal else "--"
            )
            self.pbar.set_postfix_str(
                f"R:{mean_reward:.1f}±{std_reward:.1f} Opt:{opt_text}"
            )

        return True

    def _on_training_end(self):
        """
        Save evaluation logs to a CSV file after training ends.
        Automatically pad shorter records with NaN at the front to align with the longest sequence.
        """
        self.pbar.close()
        print("[EvalCallback] Training ended. Saving evaluation logs.")
        if self.gif_env:
            self.save_gif()

        # Prepare log file path
        log_path = os.path.join(self.save_dir, f"{self.name}_result.csv")

        # Save rewards to CSV
        df = pd.DataFrame(
            self.rewards, columns=["timesteps", "mean_reward", "std_reward"]
        )
        df.to_csv(log_path, index=False)

        print(f"[EvalCallback] Log saved to {log_path}")

        # Plot training curve
        timesteps = df["timesteps"]
        mean_rewards = df["mean_reward"]
        std_rewards = df["std_reward"]

        plt.figure()
        plt.plot(timesteps, mean_rewards, label="Mean Reward")
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.3,
            label="Std Dev",
        )
        plt.axhline(
            self.optimal_score, linestyle="--", color="red", label="Optimal Score"
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title(f"Evaluation Results: {self.name}")
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(self.save_dir, f"{self.name}_curve.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"[EvalCallback] Curve saved to {plot_path}")

        # Close evaluation environments
        self.eval_env.close()
        if self.gif_env is not None:
            self.gif_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 8

        for idx in range(initial_state_count):

            obs = self.gif_env.reset()
            episode_frames = []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.gif_env.step(action)

                frame = self.gif_env.render()
                episode_frames.append(frame)

                if dones[0]:
                    break

            frames.extend(episode_frames)

        self.gif_env.close()

        new_frames = []
        for frame in frames:
            resized = cv2.resize(
                frame,
                (frame.shape[1] // 2, frame.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            new_frames.append(resized)

        gif_path = os.path.join(
            self.save_dir,
            f"{self.name}.gif",
        )

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0)
        print(f"[GIF Saved] {gif_path}")


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self._last_num_timesteps = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            mininterval=5,
            maxinterval=25,
            smoothing=0.9,
            dynamic_ncols=True,
        )
        self._last_num_timesteps = 0

    def _on_step(self):
        delta_steps = self.num_timesteps - self._last_num_timesteps
        self.pbar.update(delta_steps)
        self._last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self):
        self.pbar.close()
