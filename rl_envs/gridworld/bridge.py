"""
The simplest environment of MDP. The bridge is a 1D grid world with 10 states and 2 actions.
S={s0,s1,s2,s3,s4,s5,s6,s7,s8,s9}. The agent starts from s3, and s0 and s9 are absorbing states.
A={a0,a1}. a0 represents left and a1 represents right. Those actions move the agent left or right
until it reaches the absorbing states.

The reward function is defined as follows:
r(s1,a0)=0.1, r(s3,a1)=-0.2, r(s4,a1)=0.1, r(s8,a1)=1.0, r(s,a)=0.0 otherwise.
"""
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('TkAgg')  # Or another interactive backend like 'Qt5Agg', 'GTK3Agg', etc.
# matplotlib.use('agg')


class BridgeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }  # Define render modes

    def __init__(self, render_mode="human"):
        super(BridgeEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # Two actions: a0 (left) and a1 (right)
        self.observation_space = spaces.Discrete(10)  # Ten states: s0 to s9

        self.render_mode = self.metadata["render_modes"][0] if render_mode is None else render_mode
        self.fig = None
        self.axs = None

        self.state = 2  # Start state is s3
        self.v_values = np.zeros((1, 10))
        self.q_values = np.zeros((1, 10, 2))
        self.heuristic_values = np.zeros((1, 10))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Define rewards
        reward = 0.0
        if (self.state == 1 and action == 0) or (self.state == 4 and action == 1):
            reward = 0.1
        elif self.state == 3 and action == 1:
            reward = -0.2
        elif self.state == 8 and action == 1:
            reward = 1.0

        if action == 0:  # a0 (left)
            self.state -= 1
        elif action == 1:  # a1 (right)
            self.state += 1

        self.state = max(0, min(self.state, 9))  # Ensure state is within [0, 9]

        done = self.state in [0, 9]  # Episode is done if state is s0 or s9

        if self.render_mode == 'human':
            self.render()
        # print(self.state, action, reward, done, False, {})
        return self.state, reward, done, False, {}  # obs, reward, terminate, truncate, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = 2  # Reset to start state
        # self.v_values = np.zeros((1, 10))
        # self.heuristic_values = np.zeros((1, 10))

        if self.render_mode == 'human':
            self.render()

        # print(f"reset: {self.state}")

        return self.state, {}  # obs, info

    def render(self):
        if self.fig is None:
            plt.ion()
            # Create figure and axes
            self.fig, self.axs = plt.subplots(4, 1)

        if self.render_mode in ['human', 'rgb_array']:
            # Create a 2D array for the heatmap
            heatmap_data = np.zeros((1, 10))

            # Define rewards for states
            rewards = {1: 0.1, 3: -0.2, 4: 0.1, 8: 1.0}

            # Fill in the reward values
            for state, reward in rewards.items():
                heatmap_data[0, state] = reward

            # Plot Reward Heatmap
            self.plot_heatmap(self.axs[0], heatmap_data, "Reward Heatmap", show_agent=True)
            # Plot Value Function Heatmap
            self.plot_heatmap(self.axs[1], self.v_values, "Value Function Heatmap")
            # Plot Policy
            self.plot_policy(self.axs[2], self.q_values, "Policy")
            # Plot Heuristic Heatmap
            self.plot_heatmap(self.axs[3], self.heuristic_values, "Heuristic Function Heatmap")

            if self.render_mode == 'human':
                # Show the plot
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                for ax in self.axs:
                    ax.cla()
                # self.fig.canvas.
            elif self.render_mode == 'rgb_array':
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                for ax in self.axs:
                    ax.cla()
                return img

    def set_v_values(self, v_values):
        self.v_values = v_values

    def set_q_values(self, q_values):
        self.q_values = q_values

    def set_heuristic_values(self, heuristic_values):
        self.heuristic_values = heuristic_values

    def plot_heatmap(self, ax, data, title, show_agent=False):
        # Define the color map and normalization
        cmap = plt.cm.PiYG
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

        # Create the heatmap
        cax = ax.pcolor(data, cmap=cmap, norm=norm)

        # # Add color bar
        # plt.colorbar(cax, orientation='vertical', ax=ax)

        # Annotate each cell with the value
        for state in range(10):
            value = data[0, state]
            text_color = 'black'  # Adjust text color for visibility
            ax.text(state + 0.5, 0.5, f'{value:.2f}', ha='center', va='center', color=text_color)

        # Highlight the agent's position
        if show_agent:
            ax.plot(self.state + 0.5, 0.5, 'ro')  # Red circle for the agent

        # Set the ticks and labels
        ax.set_xticks(np.arange(10) + 0.5, minor=False)
        ax.set_xticklabels(np.arange(10))
        ax.set_yticks([0.5])
        ax.set_yticklabels(["State"])
        ax.set_title(title, fontweight='bold')

        # Remove y-ticks and set aspect
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    def plot_policy(self, ax, data, title, show_agent=False):
        # Map from basic unicode to prettier arrows
        arrow_map = {
            0: '\u2194',  # Left or Right
            -2: '\u2190',  # Left
            2: '\u2192',  # Right
        }

        # Add non-action column to data and reshape to (1, 10, 3)
        data = np.concatenate((np.zeros((1, 10, 1)), data), axis=2)
        print(data)

        data = data.argmax(axis=2).reshape((1, 10))

        # Replace the action index with color-coded index
        data[data == 0] = 0
        data[data == 1] = -2
        data[data == 2] = 2

        # Create the heatmap
        # Define the color map and normalization
        cmap = plt.cm.coolwarm
        norm = mcolors.Normalize(vmin=-4.0, vmax=4.0)

        # Create the heatmap
        cax = ax.pcolor(data, cmap=cmap, norm=norm)

        # Annotate each cell with the value
        text_color = 'white'
        # ax.text(0 + 0.5, 0.5, arrow_map[2], ha='center', va='center', color=text_color)
        # ax.text(9 + 0.5, 0.5, arrow_map[2], ha='center', va='center', color=text_color)
        for state in range(10):
            action = data[0, state]
            ax.text(state + 0.5, 0.5, arrow_map[action], ha='center', va='center', color=text_color)

        # Set the ticks and labels
        ax.set_xticks(np.arange(10) + 0.5, minor=False)
        ax.set_xticklabels(np.arange(10))
        ax.set_yticks([0.5])
        ax.set_yticklabels(["State"])
        ax.set_title(title, fontweight='bold')

        # Remove y-ticks and set aspect
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    env = BridgeEnv()
    obs, info = env.reset()
    env.render()
    print('Initial observation:', obs)
    for t in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        env.v_values = np.random.rand(1, 10)
        print('t=%d: action=%d, observation=%d, reward=%.1f, done=%d' % (t, action, obs, reward, done))
        if done:
            break
