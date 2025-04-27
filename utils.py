from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def make_venv():
    pass


class EvalAndGifCallback(BaseCallback):
    def __init__(
        self,
        config: dict,
        env_param: Union[int, float, Any],
        n_eval_envs: int,
        run_idx: int,
        eval_interval: int,
        optimal_score: Union[int, float],
        verbose=1,
        temp_dir=".",
        use_default_policy=True,
    ):
        super().__init__(verbose)
        self.config = config
        self.env_param = env_param
        self.run_idx = run_idx
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.rewards = []
        self.last_eval_step = 0

        self.eval_episodes = self.config["eval_episodes"]
        self.n_eval_envs = n_eval_envs

        self.temp_dir = temp_dir
        self.original_model = None

        # check the config to find the environment type
        self.eval_env = eval_env

        # Save path for best model
        self.best_model_path = os.path.join(
            config["save_path"],
            f"sac_env_param_{self.env_param}_run_{self.run_idx}_best.zip",
        )

        self.use_default_policy = use_default_policy

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            # Evaluate with prior policy sampling
            mean_reward, std_reward = evaluate_policy(
                self.original_model,
                self.model,
                use_default_policy=self.use_default_policy,
                use_default_policy_for_prior=False,
                env=self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Unpack reward results
            self.rewards.append((self.num_timesteps, mean_reward, std_reward))

            if self.verbose:
                # Prepare table content
                table_data = [
                    ["Env Param", self.env_param],
                    ["Repeat", self.run_idx],
                    ["Steps", self.num_timesteps],
                    ["Mean Reward", f"{mean_reward:.2f} ± {std_reward:.2f}"],
                    ["-- Prior Policy Metrics --", ""],
                ]

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(
                    f"[Best Model] Saving new best model at step {self.num_timesteps} "
                    f"with mean reward {mean_reward:.2f}"
                )
                self.model.save(self.best_model_path)

                if (
                    self.config["near_optimal_score"] > 0
                    and mean_reward >= (self.config["near_optimal_score"] / 2)
                ) or self.config["near_optimal_score"] <= 0:
                    pass
                    # self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True

    def _on_training_start(self):
        self.step_reached_optimal = None
        # Force evaluation at step 0
        self.last_eval_step = -self.eval_interval
        self._on_step()

    def _on_training_end(self):
        """
        Save evaluation logs to a CSV file after training ends.
        Automatically pad shorter records with NaN at the front to align with the longest sequence.
        """
        print("[EvalCallback] Training ended. Saving evaluation logs.")
        self.save_gif()

        config = self.config
        env_type = config["env_type"]
        assert env_type in ["lunarlander", "carracing"], "Unsupported env_type."

        # Determine maximum length among all records
        max_len = max(len(v) for v in self.records.values())

        def pad_front(data, target_len):
            pad_len = target_len - len(data)
            if pad_len <= 0:
                return data
            return [np.nan] * pad_len + data

        # Initialize dataframe
        df = pd.DataFrame()

        # Handle Timesteps (assume reward always has timesteps)
        timesteps = [x[0] for x in self.rewards]
        df["Timesteps"] = pad_front(timesteps, max_len)

        # Add reward columns
        df["reward_mean"] = pad_front([x[1] for x in self.rewards], max_len)
        df["reward_std"] = pad_front([x[2] for x in self.rewards], max_len)

        # Add other metrics
        for key in self.records.keys():
            if key == "reward":
                continue
            df[f"{key}_mean"] = pad_front([x[1] for x in self.rewards], max_len)
            df[f"{key}_std"] = pad_front([x[2] for x in self.rewards], max_len)

        # Generate save filename
        log_name = f"{self.env_name}_eval_log.csv"

        # Generate full save path
        log_path = os.path.join(config["save_path"], log_name)

        # Save to CSV
        df.to_csv(log_path, index=False)

        print(f"[EvalCallback] Log saved to {log_path}")

        # Close evaluation environment
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 8

        single_env = None

        for idx in range(initial_state_count):

            obs = single_env.reset()
            episode_frames = []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = single_env.step(action)

                frame = single_env.render(mode="rgb_array")
                episode_frames.append(frame)

                if dones[0]:
                    break

            frames.extend(episode_frames)

        single_env.close()

        new_frames = []
        for frame in frames:
            resized = cv2.resize(
                frame,
                (frame.shape[1] // 2, frame.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            new_frames.append(resized)

        gif_path = os.path.join(
            self.config["save_path"],
            f"sac_env_param_{self.env_param}_repeat_{self.run_idx}_all_initial_states.gif",
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
            dynamic_ncols=True
        )
        self._last_num_timesteps = 0

    def _on_step(self):
        delta_steps = self.num_timesteps - self._last_num_timesteps
        self.pbar.update(delta_steps)
        self._last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self):
        self.pbar.close()


