
import logging

import os
from docopt import docopt

from baselines.common import set_global_seeds

import numpy as np
import tensorflow as tf

from time import strftime
from run_dqn import learn as learn_dqn
from run_a2c import learn as learn_a2c

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --method=<n>                  The method to be used for training [choices: dqn, a2c, acktr][default: dqn]
      --load                        Whether to load the model or randomly initialize [default: False].
      --seed=<n>                    Random seed used for training [default: -1].
      --rewardLowerBounds=<n>       The lower bounds of the rewards of the environment [default: -inf].
      --rewardUpperBounds=<n>       The upper bounds of the rewards of the environment [default: inf].
      --max-steps=<n>               The amount of timesteps before the learning process is stopped [default: 4000000].
      --base-port=<n>               The base port to be used for communication between python and the environment [default: 5005].
      --output-folder=<n>           The folder to save the trained model and summary data to [default: outputs\\]
      --custom-tag=<n>              A custom tag to distinguish this run in tensorboard [default: None]
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
    max_steps = int(options['--max-steps'])
    base_port = int(options['--base-port'])
    output_folder = options['--output-folder']
    custom_tag = options['--custom-tag']
    time_string = strftime("%Y-%m-%d.%H-%M-%S")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    custom_tag_ext = ''
    if custom_tag is not None:
        custom_tag_ext += '\\'

    summary_writer = tf.summary.FileWriter(output_folder + "summaries\\" + method + "\\" + custom_tag_ext + time_string + "\\")

    set_global_seeds(seed)

    act = None
    if method == 'dqn':
        print("Training using DQN...")
        act = learn_dqn(env_path=env_path, seed=seed, max_steps=max_steps, reward_range=reward_range, base_port=base_port, summary_writer=summary_writer)
    elif method == 'a2c':
        print("Training using A2C...")
        act = learn_a2c(env_path=env_path, seed=seed, max_steps=max_steps, reward_range=reward_range, base_port=base_port, summary_writer=summary_writer)
    elif method == 'acktr':
        print("Training using ACKTR...")
    else:
        print("Unknown method: \"" + method + "\".")

    model_file_name = os.path.basename(env_path).split('.')[0] + ".pkl"
    model_path = output_folder + "models\\" + time_string + custom_tag + "\\" + model_file_name
    print("Saving model to " + model_path + ".")
    cur_path = os.path.abspath(os.path.dirname(__file__))
    act.save(cur_path + "\\" + model_path)