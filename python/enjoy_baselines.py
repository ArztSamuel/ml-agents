# # Unity ML Agents
# ## ML-Agent Learning

import logging

import os
from docopt import docopt

from unityagents import UnityEnvironment, UnityEnvironmentException

from baselines import deepq
from baselines.common import set_global_seeds

import cv2

from baselines_wrapper import make_dqn

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --seed=<n>                 Random seed used for training [default: -1].
    '''

    options = docopt(_USAGE)
    logger.info(options)

    # General parameters
    seed = int(options['--seed'])
    env_path = options['<env>']

    # set_global_seeds(seed) # TODO set seed to something different than -1
    env = UnityEnvironment(file_name=env_path, seed=seed)
    env = make_dqn(env, trainMode=False)
    act = deepq.load("3DBallModel.pkl")

    exit = False

    while not exit:
        obs, done = env.reset(), False
        episode_rew = 0
        next_episode = False
        while not done and not exit:
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)

    print("Execution stopped")
    env.close()
