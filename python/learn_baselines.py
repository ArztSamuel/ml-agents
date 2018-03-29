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
      --curriculum=<file>        Curriculum json file for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The sub-directory name for model and summary statistics [default: ppo]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0].
      --docker-target-name=<dt>       Docker Volume to store curriculum, executable and model files [default: Empty].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    # Docker Parameters
    if options['--docker-target-name'] == 'Empty':
        docker_target_name = ''
    else:
        docker_target_name = options['--docker-target-name']

    # General parameters
    run_id = options['--run-id']
    seed = int(options['--seed'])
    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    env_path = options['<env>']
    keep_checkpoints = int(options['--keep-checkpoints'])
    worker_id = int(options['--worker-id'])
    curriculum_file = str(options['--curriculum'])
    if curriculum_file == "None":
        curriculum_file = None
    lesson = int(options['--lesson'])
    fast_simulation = not bool(options['--slow'])

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    # set_global_seeds(seed) # TODO set seed to something different than -1
    env = UnityEnvironment(file_name=env_path, worker_id=worker_id, curriculum=curriculum_file, seed=seed)
    env = make_dqn(env)

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
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=1000,
        gamma=0.99,
        print_freq=50,
        prioritized_replay=True,
    )

    model_file_name = "3DBallModel.pkl"
    print("Saving model to " + model_file_name + ".")
    act.save(model_file_name)
    env.close()
