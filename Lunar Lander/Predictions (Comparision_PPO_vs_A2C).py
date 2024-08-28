
import gym
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

num_episodes=2
# env = DummyVecEnv([lambda: gym.make("LunarLander-v2", render_mode=None)])
env = DummyVecEnv([lambda: gym.make("LunarLander-v2", render_mode="human")])
models_dir = "models/A2C"
model_path = f"{models_dir}/36800.zip"
model = A2C.load(model_path, env=env)
episode_reward_lst_A2C=[]

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    while not (done):
        action,_ = model.predict(observation)
        observation_, reward, terminated, info = env.step(action)
        done = (terminated or terminated)
        episode_reward += reward
        steps += 1
        if done:
            episode_reward_lst_A2C.append(float(episode_reward))
            print("Episode", episode, "total A2C Episode Reward", episode_reward)

        while steps > 400 and not done:
            action_ = [0]  # switch the engine off after 400 time-steps to make it end the episode forcibly.
            _, reward, terminated, _ = env.step(action_)

            done = (terminated or terminated)
            episode_reward += reward

            steps += 1
            if done:
                episode_reward_lst_A2C.append(float(episode_reward))
                print("Episode", episode, "total A2C Episode Reward", episode_reward)

        observation = observation_

models_dir = "models/PPO"
model_path = f"{models_dir}/335600.zip"
model = PPO.load(model_path, env=env)
episode_reward_lst_PPO=[]

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    while not (done):
        action,_ = model.predict(observation)
        observation_, reward, terminated, info = env.step(action)
        done = (terminated or terminated)
        episode_reward += reward
        steps += 1
        if done:
            episode_reward_lst_PPO.append(float(episode_reward))
            print("Episode", episode, "total PPO Episode Reward", episode_reward)

        while steps > 400 and not done:
            action_ = [0]  # switch the engine off after 400 time-steps to make it end the episode forcibly.
            _, reward, terminated, _ = env.step(action_)

            done = (terminated or terminated)
            episode_reward += reward
            steps += 1
            if done:
                episode_reward_lst_PPO.append(float(episode_reward))
                print("Episode", episode, "total PPO Episode Reward", episode_reward)

        observation = observation_




#
fig=plt.figure("A2C, PPO, Reinforce Comparison")
ax=fig.add_subplot(111)
ax.axis([0,episode+1,-600,600])
plt.ylabel('Reward')
plt.xlabel('Episodes')
ax.plot(episode_reward_lst_A2C, 'r-', label='A2C')
ax.plot(episode_reward_lst_PPO, 'g-', label='PPO')
ax.set_title("A2C, PPO, Reinforce Comparison")

ax.legend()
plt.draw()
plt.show()

fig = plt.figure("Box plot for Rewards of trained Models (PPO,A2C and Reinforce")
ax = fig.add_subplot(111)
ax.boxplot([episode_reward_lst_A2C, episode_reward_lst_PPO], labels=['A2C', 'PPO'])
ax.set_title("Box plot for Rewards of trained Models (PPO,A2C and Reinforce")
plt.show()