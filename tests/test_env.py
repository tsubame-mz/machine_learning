from context import envs
import gym


def main():
    env = gym.make("SimpleMaze-v0")
    # env.seed(123456)
    print("observation space: ", env.observation_space)
    print("action space: ", env.action_space)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(step + 1, obs, action, next_obs, reward, done, info)
        obs = next_obs
        step += 1
    env.render()
    env.close()


if __name__ == "__main__":
    main()
