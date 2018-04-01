# # Unity ML Agents
# ## ML-Agent Learning

import logging

import os
from docopt import docopt

from baselines.common import set_global_seeds

import numpy as np

from run_dqn import learn as learn_dqn
from run_a2c import learn as learn_a2c

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --method=<n>               The method to be used for training [default: dqn][choices: dqn, a2c, acktr]
      --load                     Whether to load the model or randomly initialize [default: False].
      --seed=<n>                 Random seed used for training [default: -1].
      --rewardLowerBounds=<n>    The lower bounds of the rewards of the environment [default: -inf].
      --rewardUpperBounds=<n>    The upper bounds of the rewards of the environment [default: inf].
    '''

    options = docopt(_USAGE)
    logger.info(options)

    # General parameters
    method = options['--method']
    env_path = options['<env>']
    seed = int(options['--seed'])
    load_model = options['--load']
    reward_range = (-np.inf, np.inf)
    if options['--rewardLowerBounds'] != '-inf':
        reward_range = (float(options['--rewardLowerBounds']), reward_range[1])
    if options['--rewardUpperBounds'] != 'inf':
        reward_range = (reward_range[0], float(options['--rewardUpperBounds']))

    if seed == -1:
        seed = np.random.randint(0, 999999)
    set_global_seeds(seed)

    act = None
    if method == 'dqn':
        print("Training using DQN...")
        act = learn_dqn(env_path=env_path, seed=seed, reward_range=reward_range)
    elif method == 'a2c':
        print("Training using A2C...")
        act = learn_a2c(env_path=env_path, seed=seed, reward_range=reward_range)
    elif method == 'acktr':
        print("Training using ACKTR...")
    else:
        print("Unknown method: \"" + method + "\".")

    model_file_name = "DeliveryDuel.pkl"
    print("Saving model to " + model_file_name + ".")
    act.save(model_file_name)
    env.close()