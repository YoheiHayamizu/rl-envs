import gymnasium as gym
import rl_envs
import time

TIMESTEP = 1000

if __name__ == '__main__':

    env = gym.make('BlockWorld-n4-t1-v0', render_mode="human")

    # Reset the env
    state, info = env.reset()
    for t in range(TIMESTEP):
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncate, info = env.step(action)
        print(f"{state}\t{action}\t{reward}\t{n_state}")
        if terminated:
            print("Goal reached!")
            env.render()
            time.sleep(5)
            # assert reward == 1
            break
        if truncate:
            # assert reward == 0
            print("Truncate!")
            break
        state = n_state
