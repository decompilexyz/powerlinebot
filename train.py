import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

class PowerlineCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class DummyPowerlineEnv(gym.Env):
    """Temporary dummy environment so we can test training on the 3060.
    We will replace this later with the real Powerline environment."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # left, straight, right, boost
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        reward = 1.0 if action == 1 else 0.1
        terminated = self.steps >= 500
        truncated = False
        return obs, reward, terminated, truncated, {}

if __name__ == "__main__":
    print("Creating environments...")
    env = make_vec_env(DummyPowerlineEnv, n_envs=4)

    policy_kwargs = dict(
        features_extractor_class=PowerlineCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1,
        device="cuda",
        tensorboard_log="./tb_logs/"
    )

    print("Starting training on", torch.cuda.get_device_name(0))
    model.learn(total_timesteps=200_000)
    model.save("powerline_ppo_dummy")
    print("Training finished and model saved!")
