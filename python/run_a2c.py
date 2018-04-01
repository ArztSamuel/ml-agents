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

def _make_a2c(env_path, num_env, seed, reward_range):
    """
    Create wrapped SubprocVecEnv for using A2C on a Unity-Environment
    """
    def make_env(rank):
        def _thunk():
            env = UnityEnvironment(file_name=env_path, seed=(seed + rank), worker_id=rank)
            env = MLToGymEnv(env, train_mode=True, reward_range=reward_range)
            env = FloatToUInt8Frame(env)
            env = WarpFrame(env)
            env = FrameStack(env, k=4)
            return env
        return _thunk
    return SubprocVecEnv([make_env(i) for i in range(num_env)])

def learn(env_path, seed, reward_range):
    env = VecFrameStack(_make_a2c(env_path, num_env=4, seed=seed, reward_range=reward_range), nstack=4)

    learn_a2c(policy=CnnPolicy, env=env, seed=seed, total_timesteps=1000000)
    env.close()