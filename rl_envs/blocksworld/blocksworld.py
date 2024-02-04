import numpy as np
from itertools import product
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import copy

try:
    import pygame
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[classic-control]`"
    ) from e

from rl_envs.blocksworld.utils import generate_random_state, validate_state


class BlocksworldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(
        self,
        num_blocks: int,
        init_state: Optional[list] = None,
        goal_state: Optional[list] = None,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ):
        self.num_blocks = num_blocks
        self.arg_init_state = init_state
        self.arg_goal_state = goal_state

        self.colormap = [(22, 114, 136), (140, 218, 236), (180, 82, 72), (212, 140, 132), (168, 154, 73),
                         (214, 207, 162), (60, 180, 100), (155, 221, 177), (100, 60, 106), (131, 99, 148)]

        assert 0 < self.num_blocks < 10

        # [_, A, B, C, D, ...]
        self.num2char = {}
        self.num2char[0] = "_"
        self.num2char.update({x + 1: chr(x + 65) for x in range(num_blocks)})
        self.char2num = {}
        self.char2num["_"] = 0
        self.char2num.update({chr(x + 65): x + 1 for x in range(num_blocks)})

        self.observation_space = spaces.MultiDiscrete(
            nvec=np.array([num_blocks + 1] * num_blocks * 2))
        # An action is move(X,Y). X can be a block, and Y can be a block or table.
        possible_combo = [combo for combo in product(list(self.char2num.keys())[1:], list(self.char2num.keys()))
                          if len(set(combo)) == len(combo)]  # remove (A,A), (B,B), ...
        action_space = len(possible_combo)
        self.action_space = spaces.Discrete(action_space)
        self.a2s = {x: f"move({combo[0]},{combo[1]})" for x,
                    combo in enumerate(possible_combo)}
        self.s2a = {v: k for k, v in self.a2s.items()}
        self.a2st = {x: (self.char2num[combo[0]], self.char2num[combo[1]])
                     for x, combo in enumerate(possible_combo)}
        self.s2st = {k: self.a2st[v] for k, v in self.s2a.items()}

        # All valid states
        self.valid_states = self.get_valid_states()
        self.valid_state_str = set(
            [str(self.state_decoder(x)[0]) for x in self.valid_states])
        self.valid_state_str_i2k = {i: k for i,
                                    k in enumerate(self.valid_state_str)}
        self.valid_state_str_k2i = {k: i for i,
                                    k in enumerate(self.valid_state_str)}

        # Count the number of steps
        self.number_of_steps = 0
        self.max_steps = max_steps

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Set render mode
        self.window = None
        self.clock = None

        self.window_size = (640, 480)
        self.Font = None
        self.isopen = True
        self.state = None

        gym.logger.info(
            f"observation space of BlocksworldEnv: {self.observation_space}")
        gym.logger.info(f"action space of BlocksworldEnv: {self.action_space}")

    def get_info(self):
        cur, goal = self.state_decoder(self.state)
        return {
            "cs": cur,
            "gs": goal,
        }

    def get_valid_states(self):
        all_obs = np.array(np.meshgrid(
            *[np.arange(self.observation_space[i].n) for i in range(len(self.observation_space))]
        )).T.reshape(-1, len(self.observation_space))
        valid_obs = []
        for obs in all_obs:
            if self.is_valid_state(obs):
                valid_obs.append(obs)
        return np.array(valid_obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.number_of_steps = 0
        self.init_state, self.goal_state = self.set_task()
        self.state = np.concatenate([self.init_state, self.goal_state])
        cur, goal = self.state_decoder(self.state)
        # print(self.state, cur, goal)

        if self.render_mode == "human":
            self.render()

        return self.state, {
            "num_blocks": self.num_blocks,
            "table": self.num2char[0],
            "blocks": [self.num2char[x + 1] for x in range(self.num_blocks)],
            "num_actions": self.action_space.n,
            "actions": [self.a2s[x] for x in range(self.action_space.n)],
            "cs": cur,
            "gs": goal,
        }

    def get_next_state(self, obs, action):
        state = obs.copy()

        if not self.is_valid_action(state, action):
            gym.logger.debug(f"Not a valid action: {self.a2s[action]}")
            return state

        # Separate current state and goal state
        current_state, goal_state = state[:self.num_blocks], state[self.num_blocks:]

        src, tar = self.a2st[action]
        new_state = current_state.copy()
        new_state[src - 1] = tar

        return np.concatenate((new_state, goal_state))

    def step(self, action):
        self.number_of_steps += 1

        # Check if the environment is reset
        assert self.state is not None, "Call reset before using step method."
        # Check if the action is valid
        assert self.action_space.contains(
            action), f"{action} ({self.a2s[action]}) invalid"

        state = self.state.copy()
        # print("aaa", self.state, action, self.a2s[action], current_state)

        # Execute the action
        self.state = self.get_next_state(state, action)

        # print(self.state)
        # print(self.state, action, self.a2s[action])

        # Check if the current state is the goal state
        terminated = self.is_terminated(self.state)
        truncated = self.is_truncated()
        reward = 1.0 - self.number_of_steps / self.max_steps if terminated else 0.0
        info = self.get_info()

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def is_terminated(self, state):
        # print(state[:self.num_blocks], self.goal_state)
        return np.all(state[:self.num_blocks] == self.goal_state)

    def is_truncated(self):
        return False if self.number_of_steps < self.max_steps else True

    def render(self):
        if self.render_mode is None:
            gym.logger.debug(
                "No render mode specified. Use render_mode='human' or render_mode='rgb_array'.")
            return
        return self._render_frame()

    def _render_frame(self):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window = pygame.Surface(self.window_size)

        assert self.window is not None, "Something went wrong with pygame initialization. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()

        w, h = self.window_size
        scale = 1 / (self.num_blocks + 1)
        margin = 50.0 * scale
        box_width = box_height = 250.0 * scale
        ground_y = 50

        if self.Font is None:
            self.Font = pygame.font.SysFont("Verdana", int(125 * scale))

        char = [chr(x + 65) for x in range(self.num_blocks)]

        self.surf = pygame.Surface((w, h))
        self.surf.fill((255, 255, 255))

        def get_root(state, block, parent):
            if block == 0:
                return parent, 0
            else:
                parent, depth = get_root(state, state[block - 1], block)
                return parent, depth + 1

        cur_state = self.state[:self.num_blocks]
        goal_state = self.goal_state

        # Draw current state
        for idx, block in enumerate(cur_state):
            parent, depth = get_root(cur_state, block, idx + 1)
            box_x = (box_width + margin) * parent
            box_y = ground_y + box_height / 2 + box_height * depth
            l, r, t, b = -box_width / 2, box_width / 2, box_height / 2, -box_height / 2
            box_coords = [(l, b), (l, t), (r, t), (r, b)]
            box_coords = [(c[0] + box_x, c[1] + box_y) for c in box_coords]
            pygame.draw.polygon(self.surf, self.colormap[idx], box_coords, 0)
            letter = self.Font.render(char[idx], False, (0, 0, 0))
            letter = pygame.transform.flip(letter, False, True)
            self.surf.blit(letter, (box_x - 7.5, box_y - 10))

        # Draw goal state
        for idx, block in enumerate(goal_state):
            parent, depth = get_root(goal_state, block, idx + 1)
            box_x = (box_width + margin) * parent + w // 2
            box_y = ground_y + box_height / 2 + box_height * depth
            l, r, t, b = -box_width / 2, box_width / 2, box_height / 2, -box_height / 2
            box_coords = [(l, b), (l, t), (r, t), (r, b)]
            box_coords = [(c[0] + box_x, c[1] + box_y) for c in box_coords]
            pygame.draw.polygon(self.surf, self.colormap[idx], box_coords, 0)
            letter = self.Font.render(char[idx], False, (0, 0, 0))
            letter = pygame.transform.flip(letter, False, True)
            self.surf.blit(letter, (box_x - 7.5, box_y - 10))

        pygame.draw.line(self.surf, (0, 0, 0), (0, ground_y), (w, ground_y))
        pygame.draw.line(self.surf, (0, 0, 0), (w // 2, 0), (w // 2, h))

        self.surf = pygame.transform.flip(self.surf, False, True)

        self.surf.blit(self.Font.render(
            "Current State", False, (0, 0, 0)), (10, 10))
        self.surf.blit(self.Font.render(
            "Goal State", False, (0, 0, 0)), (w // 2 + 10, 10))

        self.window.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def state_decoder(self, state: np.ndarray):
        state_str = []
        for src, tar in enumerate(state):
            _src = self.num2char[src % self.num_blocks + 1]
            _tar = self.num2char[tar]
            state_str.append(f"on({_src},{_tar})")
        return state_str[:self.num_blocks], state_str[self.num_blocks:]

    def action_decoder(self, action: int):
        return self.a2s[action]

    def action_encoder(self, action: str):
        src = self.char2num[action[5]]
        tar = self.char2num[action[7]]
        return (src - 1) * (self.num_blocks + 1) + tar

    def sample_valid_state(self):
        state = generate_random_state(self.num_blocks)
        while not validate_state(state):
            state = generate_random_state(self.num_blocks)
        return state

    def is_valid_state(self, state):
        return validate_state(state[:self.num_blocks]) and validate_state(state[self.num_blocks:])

    def is_valid_action(self, obs, action):
        # Separate current state and goal state
        state = obs.copy()
        state, _ = state[:self.num_blocks], state[self.num_blocks:]

        # Check valid action
        # action_str = "move(X,Y)"
        # If block X has a block on it, then it is not a valid action.
        # If block Y has a block on it, then it is not a valid action.
        # X and Y cannot be the same block.
        src, tar = self.a2st[action]
        # print("in is_valid_action", src, tar, state)

        new_state = state.copy()
        new_state[src - 1] = tar

        if not validate_state(new_state):
            gym.logger.debug(
                f"Action taken: move({self.num2char[src]},{self.num2char[tar]})")
            gym.logger.debug(f"Next state {self.num2char[tar]} is not valid")
            return False

        if np.all(state == new_state):
            gym.logger.debug(
                f"Action taken: move({self.num2char[src]},{self.num2char[tar]})")
            gym.logger.debug(
                f"{self.num2char[tar]} already has {self.num2char[src]}")
            return False

        if src in state:
            # print((f"Action taken: move({self.num2char[src]},{self.num2char[tar]})"),
            #       src, tar, state, new_state,  np.where(state == src))
            gym.logger.debug(
                f"Action taken: move({self.num2char[src]},{self.num2char[tar]})")
            gym.logger.debug(f"{self.num2char[src]} cannot move:" +
                             f"{self.num2char[src]} has " +
                             f"{self.num2char[np.where(state == src)[0][0] + 1]}")
            return False

        if tar != 0 and tar in state:
            # print((f"Action taken: move({self.num2char[src]},{self.num2char[tar]})"),
            #       src, tar, state, np.where(state == tar))
            gym.logger.debug(
                f"Action taken: move({self.num2char[src]},{self.num2char[tar]})")
            gym.logger.debug(f"{self.num2char[tar]} cannot have a block on it:" +
                             f"{self.num2char[tar]} has " +
                             f"{self.num2char[np.where(state == tar)[0][0] + 1]}")
            return False
        return True

    def set_task(self) -> tuple[np.ndarray, np.ndarray]:
        # Set a start state and a goal state
        # The example of a state representation is as follows:
        # stack = [3, 1, 0], goal = [0, 0, 0]
        # This means that the block A is on the block C, the block B is
        # on the block A, the block C is on the table.
        # The goal is to put all the blocks on the table.
        if self.arg_init_state is not None:
            init_state = np.array(self.arg_init_state[:])
        else:
            init_state = np.array(generate_random_state(self.num_blocks))

        if self.arg_goal_state is not None:
            goal_state = np.array(self.arg_goal_state[:])
        else:
            goal_state = np.array(generate_random_state(self.num_blocks))

        while (init_state == goal_state).all():
            if self.arg_init_state is not None:
                goal_state = np.array(generate_random_state(self.num_blocks))
            elif self.arg_goal_state is not None:
                init_state = np.array(generate_random_state(self.num_blocks))
            else:
                init_state = np.array(generate_random_state(self.num_blocks))
                goal_state = np.array(generate_random_state(self.num_blocks))
        init_state = np.array(init_state)
        goal_state = np.array(goal_state)
        return init_state, goal_state


if __name__ == "__main__":
    NUM_BLOCKS = 3
    NUM_ACTIONS = NUM_BLOCKS * NUM_BLOCKS

    env = BlocksworldEnv(NUM_BLOCKS, render_mode="human")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation space nvec: {env.observation_space.nvec}")
    print(f"Action space: {env.action_space}")
    print("-" * 50)
    state, info = env.reset()
    print(f"The number of blocks: {info['num_blocks']}")
    print(f"The blocks: {info['blocks']}")
    print(f"The number of actions: {info['num_actions']}")
    print(f"The actions: {info['actions']}")
    print("-" * 50)
    print(f"State: {state}")
    print(f"Cur: {info['cs']}, Goal: {info['gs']}")

    for _ in range(100):
        action = np.random.randint(NUM_ACTIONS)
        # while not env.get_wrapper_attr('is_valid_action')(action):
        #     action = np.random.randint(NUM_ACTIONS)
        #     # print(action)
        n_state, reward, terminated, truncated, n_info = env.step(action)
        print(f"action: {env.a2s[action]}")
        print(f"Cur: {n_info['cs']}, Goal: {n_info['gs']}")
        print(
            f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        if terminated:
            break
        state = n_state
        info = n_info
        print("-" * 50)

    env.close()
