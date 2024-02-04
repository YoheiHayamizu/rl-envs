from collections import defaultdict
from typing import Optional
import warnings
from enum import Enum

from rl_envs.gridworld.mdp import *
from rl_envs.gridworld.rendering_utils import *


class GridWorld(MDPEnv):
    LEFT = 0  # '\u25C4'
    UP = 1  # '\u25B2'
    RIGHT = 2  # '\u25BA'
    DOWN = 3  # '\u25BC'
    TERMINATE = 4

    metadata = {"render_modes": ["human", "rgb_array", "ansi"],
                "render_fps": 4, }

    def __init__(
        self,
        noise=0.1,
        width=4,
        height=3,
        discount_factor=0.9,
        blocked_states=[(1, 1)],
        action_cost=0.0,
        initial_state=(0, 0),
        goals=None,
        render_mode: Optional[str] = None,
    ):
        super(GridWorld, self).__init__()

        self.noise = noise
        self.width = width
        self.height = height
        self.blocked_states = blocked_states
        self.discount_factor = discount_factor
        self.action_cost = action_cost
        self.initial_state = initial_state
        if goals is None:
            self.goal_states = dict(
                [((width - 1, height - 1), 1), ((width - 1, height - 2), -1)]
            )
        else:
            self.goal_states = dict(goals)

        self.TERMINAL = (self.width, self.height)

        # A list of lists thatrecords all rewards given at each step
        # for each episode of a simulated gridworld
        self.rewards = []
        # The rewards for the current episode
        self.episode_rewards = []

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete(
            [self.width + 1, self.height + 1])

        self.state = self.get_initial_state()

        self.render_mode = self.set_render_mode(render_mode)
        self.fig = None
        self.ax = None
        self.img = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = self.get_initial_state()
        self.episode_rewards = []

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def step(self, action):
        transitions = self.get_transitions(self.state, action)
        next_state = self.sample_transition(transitions)
        reward = self.get_reward(self.state, action, next_state)
        terminated = self.is_terminal(next_state)
        truncated = False
        self.state = next_state

        if self.render_mode == "human":
            self.render()

        return next_state, reward, terminated, truncated, {}

    def sample_transition(self, transitions):
        import random
        r = random.random()
        # Calculate the cumulative probability list of each transition
        # and sample from it
        cumulative_probability = 0.0
        for (state, probability) in transitions:
            cumulative_probability += probability
            if r <= cumulative_probability:
                return state
        # print("Error: No transition found", cumulative_probability, r, transitions)
        return self.TERMINAL

    def get_states(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blocked_states:
                    states.append((x, y))
        return states

    def get_actions(self, state=None):

        actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.TERMINATE]
        if state is None:
            return actions

        valid_actions = []
        for action in actions:
            for (new_state, probability) in self.get_transitions(state, action):
                if probability > 0:
                    valid_actions.append(action)
                    break
        return valid_actions

    def get_initial_state(self):
        self.episode_rewards = []
        return self.initial_state

    def get_goal_states(self):
        return self.goal_states

    def valid_add(self, state, new_state, probability):
        # If the next state is blocked, stay in the same state
        if probability == 0.0:
            return []

        if new_state in self.blocked_states:
            return [(state, probability)]

        # Move to the next space if it is not off the grid
        (x, y) = new_state
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            return [((x, y), probability)]

        # If off the grid, state in the same state
        return [(state, probability)]

    def get_transitions(self, state, action):
        transitions = []

        if state in self.get_goal_states().keys():
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return [(self.TERMINAL, 1.0)]

        # Probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if state in self.get_goal_states().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == self.UP:
            transitions += self.valid_add(state, (x, y + 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.DOWN:
            transitions += self.valid_add(state, (x, y - 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.RIGHT:
            transitions += self.valid_add(state, (x + 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        elif action == self.LEFT:
            transitions += self.valid_add(state, (x - 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)
        # else:  # TERMINATE
        #     transitions += [(state, 1.0)]

        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        # print(f"transitions: {transitions}")
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

        return transitions

    def get_reward(self, state, action, new_state):
        reward = 0.0
        if state in self.get_goal_states().keys() and new_state == self.TERMINAL:
            reward = self.get_goal_states().get(state)
        else:
            reward = self.action_cost
        step = len(self.episode_rewards)
        self.episode_rewards += [reward * (self.discount_factor ** step)]
        return reward

    def get_discount_factor(self):
        return self.discount_factor

    def is_terminal(self, state):
        if state == self.TERMINAL:
            # self.rewards += [self.episode_rewards]
            return True
        return False

    def get_rewards(self):
        """
        Returns a list of lists, which records all rewards given at each step
        for each episodeof a simulated gridworld
        """
        return self.rewards

    def close(self):
        pass

    @staticmethod
    def create(string):
        """
        Create a gridworld from an array of strings: one for each line
        - First line is rewards as a dictionary from cell to value: {'A': 1, ...}
        - space is an empty cell
        - # is a blocked cell
        - @ is the agent (initial state)
        - new 'line' is a new row
        - a letter is a cell with a reward for transitioning
        into that cell. The reward defined by the first line.
        """
        # Parse the reward on the first line
        import ast

        rewards = ast.literal_eval(string[0])

        width = 0
        height = len(string) - 1

        blocked_cells = []
        initial_state = (0, 0)
        goals = []
        row = 0
        for next_row in string[1:]:
            column = 0
            for cell in next_row:
                if cell == "#":
                    blocked_cells += [(column, row)]
                elif cell == "@":
                    initial_state = (column, row)
                elif cell.isalpha():
                    goals += [((column, row), rewards[cell])]
                column += 1
            width = max(width, column)
            row += 1
        return GridWorld(
            width=width,
            height=height,
            blocked_states=blocked_cells,
            initial_state=initial_state,
            goals=goals,
        )

    @staticmethod
    def open(file):
        file = open(file, "r")
        string = file.read().splitlines()
        file.close()
        return GridWorld.create(string)

    @staticmethod
    def matplotlib_installed():
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            return True
        except ModuleNotFoundError:
            warnings.warn(
                "Matplotlib not installed. Cannot visualise gridworld.")
            return False

    def set_render_mode(self, render_mode):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode is None:
            render_mode = self.metadata["render_modes"][0]

        # Check if matplotlib is installed and warn if not and render mode is not ansi
        if not self.matplotlib_installed() and render_mode != 'ansi':
            self.render_mode = 'ansi'
            warnings.warn(
                "Matplotlib not installed. Cannot visualise gridworld.")

        return render_mode

    def render(self, mode='human'):
        """ Visualise a Grid World problem """
        if mode == 'ansi':
            return self.visualise_text(title="Grid World")
        else:
            if self.fig is None:
                plt.ion()
                self.fig, self.ax, self.img = self.initialise_grid()
            return self.visualise_frame(title="Grid World")

    def visualise_frame(self, title="", grid_size=1.5):
        self.visualise_as_image(title=title, grid_size=grid_size)
        if self.render_mode == 'human':
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.cla()
        elif self.render_mode == 'rgb_array':
            rgba_buf = self.fig.canvas.buffer_rgba()
            (w, h) = self.fig.canvas.get_width_height()
            rgba_arr = np.frombuffer(rgba_buf, np.uint8).reshape((h, w, 4))
            return rgba_arr

    def visualise_text(self, title=""):
        return self.to_string(title=title)

    def visualise_value_function(self, value_function, title="", grid_size=1.5):
        """ Visualise a Grid World value function """
        if self.matplotlib_installed():
            return self.visualise_value_function_as_image(value_function, title=title, grid_size=grid_size)
        else:
            print(self.value_function_to_string(value_function, title=title))

    def visualise_q_function(self, qfunction, title="", grid_size=2.0):
        """ Visualise a Grid World Q value function """
        if self.matplotlib_installed():
            return self.visualise_q_function_as_image(qfunction, title=title, grid_size=grid_size)
        else:
            print(self.q_function_to_string(qfunction, title=title))

    def visualise_policy(self, policy, title="", grid_size=1.5):
        """ Visualise a Grid World policy """
        if self.matplotlib_installed():
            return self.visualise_policy_as_image(policy, title=title, grid_size=grid_size)
        else:
            print(self.policy_to_string(policy, title=title))

    def visualise_stochastic_policy(self, policy, title="", grid_size=1.5):
        """ Visualise a Grid World policy """
        if self.matplotlib_installed():
            return self.visualise_stochastic_policy_as_image(policy, title=title, grid_size=grid_size)
        else:
            # TODO make a stochastic policy to string
            pass

    def to_string(self, title=""):
        """ Visualise a grid world problem as a formatted string """
        left_arrow = "\u25C4"
        up_arrow = "\u25B2"
        right_arrow = "\u25BA"
        down_arrow = "\u25BC"

        space = " |              "
        block = " | #############"

        line = "  "
        for x in range(self.width):
            line += "--------------- "
        line += "\n"

        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.get_goal_states().keys():
                    result += space
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += " |       {}      ".format(up_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |     _____    "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |    ||o  o|   "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " | {}  ||  * |  {}".format(
                        left_arrow, right_arrow)
                elif (x, y) in self.blocked_states:
                    result += block
                elif (x, y) in self.get_goal_states().keys():
                    result += " |     {:+0.2f}    ".format(
                        self.get_goal_states()[(x, y)]
                    )
                else:
                    result += " | {}           {}".format(
                        left_arrow, right_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |    ||====|   ".format(
                        left_arrow, right_arrow)
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |     -----    "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.get_goal_states().keys():
                    result += space
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += " |       {}      ".format(down_arrow)
            result += " |\n"
            result += line
        return result

    def value_function_to_string(self, values, title=""):
        """ Convert a grid world value function to a formatted string """
        line = " {:-^{n}}\n".format("", n=len(" | +0.00") * self.width + 1)
        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " | #####"
                else:
                    result += " | {:+0.2f}".format(values.get_value((x, y)))
            result += " |\n"
            result += line

        return result

    def q_function_to_string(self, qfunction, title=""):
        """ Convert a grid world Q function to a formatted string """
        left_arrow = "\u25C4"
        up_arrow = "\u25B2"
        right_arrow = "\u25BA"
        down_arrow = "\u25BC"

        space = " |               "

        line = "  "
        for x in range(self.width):
            line += "---------------- "
        line += "\n"

        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |       {}       ".format(up_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(
                        qfunction.get_q_value((x, y), self.UP)
                    )
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " |     #####     "
                elif (x, y) in self.get_goal_states().keys():
                    result += " |     {:+0.2f}     ".format(
                        self.get_goal_states()[(x, y)]
                    )
                else:
                    result += " | {}{:+0.2f}  {:+0.2f}{}".format(
                        left_arrow,
                        qfunction.get_q_value((x, y), self.LEFT),
                        qfunction.get_q_value((x, y), self.RIGHT),
                        right_arrow,
                    )
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(
                        qfunction.get_q_value((x, y), self.DOWN)
                    )
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |       {}       ".format(down_arrow)
            result += " |\n"
            result += line
        return result

    def policy_to_string(self, policy, title=""):
        """ Convert a grid world policy to a formatted string """
        line = " {:-^{n}}\n".format("", n=len(" |  N ") * self.width + 1)
        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " | ###"
                elif policy[(x, y)] == self.TERMINATE:
                    result += " | {:+0d} ".format(self.goal_states[(x, y)])
                else:
                    result += " |  " + policy[(x, y)] + " "
            result += " |\n"
            result += line

        return result

    def initialise_grid(self, grid_size=1.5):
        """ Initialise a gridworld grid """
        fig = plt.figure(
            figsize=(self.width * grid_size, self.height * grid_size))
        plt.subplots_adjust(top=0.92, bottom=0.01, right=1,
                            left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1)

        # Initialise the map to all white
        img = [[COLOURS['white']
                for _ in range(self.width)] for _ in range(self.height)]

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                if (x, y) in self.goal_states:
                    img[y][x] = COLOURS['red'] if self.goal_states[(
                        x, y)] < 0 else COLOURS['green']
                elif (x, y) in self.blocked_states:
                    img[y][x] = COLOURS['grey']

        ax.xaxis.set_ticklabels([])  # clear x tick labels
        ax.axes.yaxis.set_ticklabels([])  # clear y tick labels
        ax.tick_params(which='both', top=False, left=False,
                       right=False, bottom=False)
        ax.set_xticks([w - 0.5 for w in range(0, self.width, 1)])
        ax.set_yticks([h - 0.5 for h in range(0, self.height, 1)])
        ax.grid(color='lightgrey')
        return fig, ax, img

    def visualise_as_image(self, title="", grid_size=1.5):
        """ visualise the gridworld problem as a matplotlib image """
        current_position = self.state

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                if (x, y) == current_position:
                    self.ax.scatter(x, y, s=2000, marker='o',
                                    edgecolors='none')
                elif (x, y) in self.goal_states:
                    plt.text(
                        x,
                        y,
                        f"{self.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
        plt.imshow(self.img, origin="lower")
        plt.title(title)

    def render_tile(self, x, y, tile_size, img, tile_type=None):
        """Render each tile individually depending on the current state of the cell"""
        ymin = y * tile_size
        ymax = (y + 1) * tile_size
        xmin = x * tile_size
        xmax = (x + 1) * tile_size

        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if i == ymin or i == ymax - 1 or j == xmin or j == xmax + 1:
                    draw_grid_lines(i, j, img)
                else:
                    if tile_type == "goal":
                        render_goal(
                            i,
                            j,
                            img,
                            reward=self.goal_states[(x, y)],
                            reward_max=max(self.get_goal_states().values()),
                            reward_min=min(self.get_goal_states().values()),
                        )
                    elif tile_type == "blocked":
                        render_blocked_tile(i, j, img)
                    elif tile_type == "agent":
                        render_agent(
                            i,
                            j,
                            img,
                            center_x=xmin + tile_size / 2,
                            center_y=ymin + tile_size / 2,
                            radius=tile_size / 4,
                        )
                    elif tile_type == "empty":
                        img[i][j] = [255, 255, 255]
                    else:
                        raise ValueError("Invalid tile type")

    def visualise_value_function_as_image(self, value_function, title="", grid_size=1.5):
        """ Visualise the value function """
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                value = value_function.get_v_value((x, y))
                if (x, y) not in self.blocked_states:
                    text = plt.text(
                        x,
                        y,
                        f"{float(value):+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color='lightgrey' if value == 0.0 else 'black',
                    )
                    texts.append(text)
        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()
        return fig, ax, img

    def visualise_value_function_as_heatmap(self, value_function, title=""):
        """ Visualise the value function using a heat-map where green is high value and
        red is low value
        """
        values = [[0 for _ in range(self.width)] for _ in range(self.height)]
        fig, ax = self.initialise_grid()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    plt.text(
                        x,
                        y,
                        "#",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                else:
                    values[y][x] = value_function[(x, y)]
                    plt.text(
                        x,
                        y,
                        f"{values[y][x]:.2f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
        plt.imshow(values, origin="lower", cmap=make_red_white_green_cmap())
        plt.title(title)
        plt.show()

    def visualise_q_function_as_image(self, qfunction, title="", grid_size=2.0):
        """ Visualise the Q-function with matplotlib """
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.goal_states:
                    texts.append(plt.text(
                        x, y, f"{self.get_goal_states()[(x,y)]:+0.2f}",
                        fontsize="large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    ))

                elif (x, y) not in self.blocked_states:
                    up_value = qfunction.get_q_value((x, y), self.UP)
                    down_value = qfunction.get_q_value((x, y), self.DOWN)
                    left_value = qfunction. get_q_value((x, y), self.LEFT)
                    right_value = qfunction.get_q_value((x, y), self.RIGHT)
                    texts.append(plt.text(
                        x, y + 0.35, f"{up_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="center",
                        verticalalignment="top",
                        color='lightgrey' if up_value == 0.0 else 'black',
                    ))
                    texts.append(plt.text(
                        x, y - 0.35, f"{down_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color='lightgrey' if down_value == 0.0 else 'black',
                    ))
                    texts.append(plt.text(
                        x - 0.45, y, f"{left_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="left",
                        verticalalignment="center",
                        color='lightgrey' if left_value == 0.0 else 'black'
                    ))
                    texts.append(plt.text(
                        x + 0.45, y, f"{right_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="right",
                        verticalalignment="center",
                        color='lightgrey' if right_value == 0.0 else 'black'
                    ))
                    plt.plot([x-0.5, x+0.5], [y-0.5, y+0.5],
                             ls='-', lw=1, color='lightgrey')
                    plt.plot([x + 0.5, x - 0.5], [y - 0.5, y + 0.5],
                             ls='-', lw=1, color='lightgrey')
        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()
        return fig, ax, img

    def visualise_q_function_rendered(self, q_values, title="", tile_size=32, show_text=False, grid_size=2.0):
        """ Visualise the Q-function with a matplotlib visual"""
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = [[[0, 0, 0] for _ in range(width_px)] for _ in range(height_px)]

        # provide these to scale the colours between the highest and lowest value
        reward_max = max(self.get_goal_states().values())
        reward_min = min(self.get_goal_states().values())
        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                # Draw in the blocked states as a black and white mesh
                if (x, y) in self.blocked_states:
                    render_full_blocked_tile(
                        x * tile_size, y * tile_size, tile_size, img)
                    continue
                # Draw goal states
                if (x, y) in self.goal_states:
                    render_full_goal_tile(
                        x * tile_size,
                        y * tile_size,
                        tile_size,
                        img,
                        reward=self.goal_states[(x, y)],
                        rewardMax=reward_max,
                        rewardMin=reward_min,
                    )
                    continue

                # Draw the action value for action available in each cell
                # Break the grid up into 4 sections, using triangles that meet
                # in the middle. The base of the triangle points toward the
                # direction of the action
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.UP,
                    q_values,
                    img,
                    show_text,
                    v_text_offset=8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.DOWN,
                    q_values,
                    img,
                    show_text,
                    v_text_offset=-8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.LEFT,
                    q_values,
                    img,
                    show_text,
                    h_text_offset=-8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.RIGHT,
                    q_values,
                    img,
                    show_text,
                    h_text_offset=8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )

        ax.imshow(img, origin="lower", interpolation="bilinear")
        plt.title(title)
        plt.axis("off")
        plt.show()

    def visualise_policy_as_image(self, policy, title="", grid_size=1.5):
        """ Visualise the policy of the agent with a matplotlib visual """
        # Map from basic unicode to prettier arrows
        arrow_map = {self.UP: '\u2191',
                     self.DOWN: '\u2193',
                     self.LEFT: '\u2190',
                     self.RIGHT: '\u2192',
                     }
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.blocked_states and (x, y) not in self.goal_states:
                    if np.argmax(policy[(x, y)]) != self.TERMINATE:
                        action = arrow_map[np.argmax(policy[(x, y)])]
                        fontsize = "xx-large"
                    texts.append(plt.text(
                        x,
                        y,
                        action,
                        fontsize=fontsize,
                        horizontalalignment="center",
                        verticalalignment="center",
                    ))
                elif (x, y) in self.goal_states:
                    plt.text(
                        x,
                        y,
                        f"{self.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()

    def visualise_stochastic_policy_as_image(self, policy, title="", grid_size=1.5):
        """ Visualise the policy of the agent with a matplotlib visual """
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                prob_left = policy.get_probability((x, y), self.LEFT)
                prob_right = policy.get_probability((x, y), self.RIGHT)
                if self.height > 1:
                    prob_up = policy.get_probability((x, y), self.UP)
                    prob_down = policy.get_probability((x, y), self.DOWN)

                if (x, y) in self.goal_states:
                    plt.text(
                        x,
                        y,
                        f"{self.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                elif (x, y) not in self.blocked_states:
                    if self.height > 1:
                        texts.append(plt.text(
                            x,
                            y,
                            f"{prob_up:0.2f}\n{self.UP}\n{prob_left:0.2f}{self.LEFT} {self.RIGHT}{prob_right:0.2f}\n{self.DOWN}\n{prob_down:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        ))
                    else:
                        texts.append(plt.text(
                            x,
                            y,
                            f"{prob_left:0.2f}{self.LEFT} {self.RIGHT}{prob_right:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        ))

        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()
        return fig


class CliffWorld(GridWorld):
    def __init__(
        self,
        noise=0.0,
        discount_factor=1.0,
        width=6,
        height=4,
        blocked_states=[],
        action_cost=-0.05,
        goals=[((1, 0), -5), ((2, 0), -5), ((3, 0), -5),
               ((4, 0), -5), ((5, 0), 0)],
    ):
        super().__init__(
            noise=noise,
            discount_factor=discount_factor,
            width=width,
            height=height,
            blocked_states=blocked_states,
            action_cost=action_cost,
            goals=goals,
        )


class OneDimensionalGridWorld(GridWorld):
    """ A one dimensional GridWorld class to use with the
    Logistic regression policy gradient.
    This allows actions [left, right] and terminates when the agent reaches the
    goal state without having to use a terminate action.
    """

    def __init__(
        self,
        noise=0.1,
        width=4,
        discount_factor=0.9,
        action_cost=0.0,
        initial_state=(0, 0),
        goals=[((0, 0), 0), ((10, 0), 0)],
        rewards=[((1, 0), 0.1), ((3, 0), -0.2), ((4, 0), 0.1), ((9, 0), 1.0)],
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            noise=noise,
            width=width,
            height=1,
            blocked_states=[],
            discount_factor=discount_factor,
            action_cost=action_cost,
            initial_state=initial_state,
            goals=goals,
            render_mode=render_mode,
        )
        self.rewards = dict(goals)

    def get_reward(self, state, action, new_state):
        return self.rewards.get(state, 0.0)


if __name__ == "__main__":
    small_gridworld = GridWorld(width=4, height=3, render_mode='human')
    small_gridworld.reset()

    for _ in range(10):
        small_gridworld.step(small_gridworld.UP)
        # small_gridworld.step(small_gridworld.RIGHT)
        # small_gridworld.step(small_gridworld.DOWN)
        # small_gridworld.step(small_gridworld.LEFT)

    bridge_world = OneDimensionalGridWorld(width=10, noise=0.0, action_cost=0.0, initial_state=(5, 0),
                                           goals=[((0, 0), -1), ((9, 0), 1)], render_mode='human')
    bridge_world.reset()
    for _ in range(10):
        bridge_world.step(bridge_world.UP)
    # medium = gridworld = GridWorld(width=16, height=12)
    # medium.visualise_as_image(title="Medium")
