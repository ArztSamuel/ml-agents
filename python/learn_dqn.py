# # Unity ML Agents
# ## ML-Agent Learning

import logging

import os
from docopt import docopt

from unityagents import UnityEnvironment, UnityEnvironmentException

from baselines import deepq
from baselines.common import set_global_seeds

import cv2
import numpy as np

from baselines_wrapper import make_dqn

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --load                     Whether to load the model or randomly initialize [default: False].
      --seed=<n>                 Random seed used for training [default: -1].
      --rewardLowerBounds=<n>    The lower bounds of the rewards of the environment [default: -inf].
      --rewardUpperBounds=<n>    The upper bounds of the rewards of the environment [default: inf].
    '''

    options = docopt(_USAGE)
    logger.info(options)

    # General parameters
    seed = int(options['--seed'])
    load_model = options['--load']
    env_path = options['<env>']
    reward_range = (-np.inf, np.inf)
    if options['--rewardLowerBounds'] != '-inf':
        reward_range = (float(options['--rewardLowerBounds']), reward_range[1])
    if options['--rewardUpperBounds'] != 'inf':
        reward_range = (reward_range[0], float(options['--rewardUpperBounds']))

    if seed == -1:
        seed = np.random.randint(0, 999999)
    set_global_seeds(seed)
    env = UnityEnvironment(file_name=env_path, seed=seed)
    env = make_dqn(env, trainMode=True, reward_range=reward_range)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(True),
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=1000000, # TODO: adjust this via parameters
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=500,
        gamma=0.99,
        print_freq=1,
        prioritized_replay=True,
        param_noise=True,
        double_q=True,
    )

    model_file_name = "DeliveryDuel.pkl"
    print("Saving model to " + model_file_name + ".")
    act.save(model_file_name)
    env.close()
