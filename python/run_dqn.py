# # Unity ML Agents
# ## ML-Agent Learning

from baselines import deepq
from baselines_wrapper import FloatToUInt8Frame
from baselines.common.atari_wrappers import WarpFrame, FrameStack

from unityagents import UnityEnvironment, UnityEnvironmentException
from baselines_wrapper import MLToGymEnv

def _make_dqn(unity_env, train_mode, reward_range):
    env = MLToGymEnv(unity_env, train_mode=train_mode, reward_range=reward_range)
    env = FloatToUInt8Frame(env)
    env = WarpFrame(env) # Makes sure we have 84 x 84 b&w
    env = FrameStack(env, 4) # Stack last 4 frames
    return env

def learn(env_path, seed, max_steps, reward_range, base_port):
    unity_env = UnityEnvironment(file_name=env_path, seed=seed, base_port=base_port)
    env = _make_dqn(unity_env, train_mode=True, reward_range=reward_range)   

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False,
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=0.0000625,
        max_timesteps=max_steps,
        buffer_size=200000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=80000,
        target_network_update_freq=32000,
        gamma=0.99,
        print_freq=1,
        prioritized_replay=True,
        prioritized_replay_beta0=0.4,
        param_noise=False,
        double_q=True,
    )

    env.close()
    return act