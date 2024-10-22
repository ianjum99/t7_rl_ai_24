import gym
from gym import spaces
import numpy as np
from scripts.capture_game_state import capture_game_state
from scripts.perform_actions import perform_action

class TekkenEnvironment(gym.Env):
    def __init__(self):
        super(TekkenEnvironment, self).__init__()
        self.action_space = spaces.Discrete(5)  # Define the action space (e.g., punch, kick)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # State: [player_health, enemy_health, distance]

    def reset(self):
        return capture_game_state()

    def step(self, action):
        perform_action(action)
        next_state = capture_game_state()
        reward = self.calculate_reward(next_state)
        done = self.is_game_over(next_state)
        return next_state, reward, done, {}

    def calculate_reward(self, state):
        player_health, enemy_health, distance = state
        return (enemy_health - player_health) * 100  # Simple reward: difference in health

    def is_game_over(self, state):
        player_health, enemy_health, _ = state
        return player_health <= 0 or enemy_health <= 0  # Game over when health <= 0

