import numpy as np
import random
import config
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from env import make_env

import argparse

DIR_NAME = './data/rollout_'

WH = 64

def crop(image, scale):
  size = len(image)
  newsize = int(np.round(size * scale))
  border = int(round((size-newsize) / 2))
  left = border
  right = left + newsize
  top = int(np.round(left / 2 * 3))
  bottom = top + newsize
  return image[top:bottom, left:right]

def main(args):

    env_name = args.env_name
    total_episodes = int(args.total_episodes)
    time_steps = int(args.time_steps)
    render = args.render
    run_all_envs = args.run_all_envs
    action_refresh_rate = args.action_refresh_rate
    alpha = float(args.alpha)
    model_name = str(args.model_name)
    
    if run_all_envs:
        envs_to_generate = config.train_envs
    else:
        envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))
        if not os.path.isdir(DIR_NAME + model_name):
            os.mkdir(DIR_NAME + model_name)
            
        env = make_env(current_env_name)  # <1>
        s = 0

        while s < total_episodes:

            episode_id = random.randint(0, 2**31 - 1)
            filename = DIR_NAME + model_name + '/' + str(episode_id) + ".npz"

            observation = env.reset()

            env.render()

            t = 0

            obs_sequenceS = []
            obs_sequenceB = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []

            reward = -0.1
            done = False
            beta = alpha + np.random.rand() * (1-alpha)
            
            while t < time_steps:  # and not done:
                if t % action_refresh_rate == 0:
                    action = config.generate_data_action(t, env)  # <2>

                observation = config.adjust_obs(observation)  # <3>

                obs_sequenceS.append(cv2.resize(crop(observation, alpha*beta), dsize=(WH, WH), interpolation=cv2.INTER_CUBIC))
                obs_sequenceB.append(cv2.resize(crop(observation, beta), dsize=(WH, WH), interpolation=cv2.INTER_CUBIC))
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

                observation, reward, done, info = env.step(action)  # <4>

                t = t + 1

                if render:
                    env.render()

            print("Episode {} finished after {} timesteps".format(s, t))
            
            np.savez_compressed(filename, obsS=np.asarray(obs_sequenceS), obsB=np.asarray(obs_sequenceB), action=np.asarray(action_sequence),
                                reward=np.asarray(reward_sequence), done=np.asarray(done_sequence))  # <4>

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, help='name of environment')
    parser.add_argument('--total_episodes', type=int, default=200,
                        help='total number of episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=300,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--render', default=0, type=int,
                        help='render the env as data is generated')
    parser.add_argument('--alpha',default = 0.8, type=float, help='zoom level')
    parser.add_argument('--action_refresh_rate', default=20, type=int,
                        help='how often to change the random action, in frames')
    parser.add_argument('--run_all_envs', action='store_true',
                        help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')
    parser.add_argument('--model_name', type=str, default="default", help="name of the model")
    args = parser.parse_args()
    main(args)
