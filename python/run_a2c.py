# # Unity ML Agents
# ## ML-Agent Learning

from baselines import deepq
from baselines_wrapper import FloatToUInt8Frame
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2.policies import CnnPolicy
from baselines.a2c.a2c import learn as learn_a2c

from unityagents import UnityEnvironment, UnityEnvironmentException
from baselines_wrapper import MLToGymEnv

import tensorflow as tf
import numpy as np

def _make_a2c(env_path, num_env, seed, reward_range, base_port):
    """
    Create wrapped SubprocVecEnv for using A2C on a Unity-Environment
    """
    def make_env(rank):
        def _thunk():
            env = UnityEnvironment(file_name=env_path, seed=seed, worker_id=rank, base_port=base_port)
            env = MLToGymEnv(env, train_mode=True, reward_range=reward_range)
            env = FloatToUInt8Frame(env)
            env = WarpFrame(env)
            env = FrameStack(env, k=4)
            return env
        return _thunk
    return SubprocVecEnv([make_env(i) for i in range(num_env)])

def _create_summary_callback(summary_writer):
    def _summary_callback(local_vars, global_vars):
        batch_count = local_vars['nbatch']
        steps = local_vars['update'] * batch_count
        episode_rewards = local_vars['runner'].episode_rewards
        summary = tf.Summary()
        summary.value.add(tag='Info/Steps per Second', simple_value=local_vars['sps'])
        summary.value.add(tag='Info/Policy Entropy', simple_value=local_vars['policy_entropy'])
        summary.value.add(tag='Info/Value Loss', simple_value=local_vars['value_loss'])
        summary.value.add(tag='Info/Explained Variance', simple_value=local_vars['ev'])
        summary.value.add(tag='Info/Episode Count', simple_value=len(episode_rewards))
        summary.value.add(tag='Info/Mean 100 Reward', simple_value=round(np.mean(episode_rewards[-101:-1]), 1))
        rewards = local_vars['true_rewards']
        if rewards is not None:
            summary.value.add(tag='Info/Mean Update Reward (' + str(batch_count) + ')', simple_value=(sum(rewards) / batch_count))
        
        summary_writer.add_summary(summary, steps)
        summary_writer.flush()
        return False # we only want to log here, no need to stop algorithm

    return _summary_callback



def learn(env_path, seed, max_steps, reward_range, base_port, summary_writer):
    env = VecFrameStack(_make_a2c(env_path, num_env=8, seed=seed, reward_range=reward_range, base_port=base_port), nstack=4)

    model = learn_a2c(policy=CnnPolicy, env=env, seed=seed, ent_coef=0.00001, total_timesteps=max_steps, callback=_create_summary_callback(summary_writer=summary_writer))

    try:
        env.close()
    except Exception as e:
        print("Failed to close environment: " + str(e))

    return model