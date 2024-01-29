from gymnasium.envs.registration import register

register(
    id='BlockWorld-n2-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 2,
    }
)
register(
    id='BlockWorld-n3-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 3,
    }
)
register(
    id='BlockWorld-n3-t1-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 3,
        "goal_state": [0, 0, 1],
    }
)
register(
    id='BlockWorld-n4-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 4,
    }
)
register(
    id='BlockWorld-n4-t1-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 4,
        "init_state": [3, 0, 0, 0],
        "goal_state": [0, 1, 0, 3],
    }
)

register(
    id='BlockWorld-n5-v0',
    entry_point='cp_envs.envs.blocksworld.blocksworld:BlocksworldEnv',
    kwargs={
        "num_blocks": 5,
    }
)

register(
    id='Bridge-v1',
    entry_point='cp_envs.envs.maze.bridge:BridgeEnv',
    kwargs={
        "render_mode": "rgb_array",
    },
)

register(
    id='Bridge-v0',
    entry_point='cp_envs.envs.maze.gridworld:OneDimensionalGridWorld',
    kwargs={
        "width": 10,
        "noise": 0.0,
        "action_cost": 0.0,
        "initial_state": (2, 0),
        "goals": [((0, 0), 0), ((9, 0), 0)],
        "render_mode": "human",
    }
)

register(
    id='GridWorld-v0',
    entry_point='cp_envs.envs.maze.gridworld:GridWorld',
    kwargs={
        "width": 4,
        "height": 3,
        "render_mode": "human",
    }
)
