"""
Tabular Q-Learning with a Custom Gymnasium Environment and Taxi-v3 Benchmark.
Author: Tchaleté Joël Arnaud Shep
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class CustomGridWorldEnv(gym.Env):
    """
    Custom 2D Grid-World Environment adhering to the Gymnasium API.
    Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size: int = 5, start=(0, 0), goal=(4, 4), traps=None):
        super().__init__()
        self.grid_size = grid_size
        self.start_pos = np.array(start, dtype=int)
        self.goal_pos = np.array(goal, dtype=int)
        self.traps = [np.array(t, dtype=int) for t in (traps or [(1, 2), (2, 2), (3, 1)])]

        self.observation_space = spaces.Discrete(grid_size * grid_size)
        self.action_space = spaces.Discrete(4)
        self.state = self.start_pos.copy()

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return int(pos[0] * self.grid_size + pos[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        return self._pos_to_state(self.state), {}

    def step(self, action: int):
        # 0: Up, 1: Right, 2: Down, 3: Left
        moves = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
        }
        next_pos = np.clip(self.state + moves[action], 0, self.grid_size - 1)
        self.state = next_pos

        if np.array_equal(self.state, self.goal_pos):
            return self._pos_to_state(self.state), 10.0, True, False, {}

        if any(np.array_equal(self.state, trap) for trap in self.traps):
            return self._pos_to_state(self.state), -10.0, True, False, {}

        return self._pos_to_state(self.state), -0.1, False, False, {}


class QLearningAgent:
    """Tabular Q-Learning Agent with exponential epsilon-greedy decay."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env: gym.Env, agent: QLearningAgent, n_episodes: int = 10000, max_steps: int = 100):
    """Trains the Q-Learning agent and records episode rewards."""
    rewards_history = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

    return rewards_history


def plot_training_curve(rewards: list, title: str, window: int = 200):
    """Plots raw and moving-average episode rewards."""
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.25, label="Episode Reward")
    plt.plot(range(window - 1, len(rewards)), moving_avg, color="crimson", linewidth=2, label=f"{window}-Ep Moving Avg")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_trajectory(env: CustomGridWorldEnv, agent: QLearningAgent, max_steps: int = 50):
    """Visualizes the learned greedy policy trajectory on the 2D GridWorld."""
    state, _ = env.reset()
    path = [env.state.copy()]

    for _ in range(max_steps):
        action = agent.select_action(state, greedy=True)
        state, _, terminated, truncated, _ = env.step(action)
        path.append(env.state.copy())
        if terminated or truncated:
            break

    path = np.array(path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True)
    ax.invert_yaxis()

    # Plot traps, start, goal, and trajectory
    for trap in env.traps:
        ax.scatter(trap[1], trap[0], color="red", s=200, marker="X", label="Trap")
    ax.scatter(env.start_pos[1], env.start_pos[0], color="blue", s=150, marker="o", label="Start")
    ax.scatter(env.goal_pos[1], env.goal_pos[0], color="green", s=200, marker="*", label="Goal")
    ax.plot(path[:, 1], path[:, 0], color="black", linewidth=2, marker=".", label="Learned Path")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.set_title("Learned Optimal Trajectory (GridWorld)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Custom GridWorld Experiment
    grid_env = CustomGridWorldEnv(grid_size=5)
    grid_agent = QLearningAgent(
        n_states=grid_env.observation_space.n,
        n_actions=grid_env.action_space.n,
        epsilon_decay=0.999,
    )
    grid_rewards = train_agent(grid_env, grid_agent, n_episodes=5000, max_steps=100)
    print(f"GridWorld Final 100-Ep Mean Reward: {np.mean(grid_rewards[-100:]):.2f}")
    plot_training_curve(grid_rewards, "Custom GridWorld Q-Learning Convergence")
    plot_trajectory(grid_env, grid_agent)

    # 2. Gymnasium Taxi-v3 Benchmark
    taxi_env = gym.make("Taxi-v3")
    taxi_agent = QLearningAgent(
        n_states=taxi_env.observation_space.n,
        n_actions=taxi_env.action_space.n,
        lr=0.1,
        gamma=0.99,
        epsilon_decay=0.9995,
    )
    taxi_rewards = train_agent(taxi_env, taxi_agent, n_episodes=10000, max_steps=200)
    print(f"Taxi-v3 Final 100-Ep Mean Reward: {np.mean(taxi_rewards[-100:]):.2f}")
    plot_training_curve(taxi_rewards, "Taxi-v3 Benchmark Q-Learning Convergence")
    taxi_env.close()