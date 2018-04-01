import gym
import numpy as np
import cv2
from gym import spaces

from baselines.common.atari_wrappers import WarpFrame, FrameStack

class MLToGymEnv(gym.Env):
    def __init__(self, env, trainMode, reward_range=(-np.inf, np.inf)):
        """Wraps UnityEnvironment of ML-Agents to be used by baselines algorithms
        """
        gym.Env.__init__(self)

        self.unityEnv = env
        self.trainMode = trainMode
        self.reward_range = reward_range

        assert self.unityEnv.number_external_brains > 0, "No external brains defined in unityEnv"
        self.__externalBrainName = self.unityEnv.external_brain_names[0]
        externalBrain = self.unityEnv.brains[self.__externalBrainName]
        actionSpaceSize = externalBrain.vector_action_space_size
        assert actionSpaceSize > 0
        self.action_space = spaces.Discrete(actionSpaceSize)

        # TODO set observation space according to brain


    def step(self, action):
        action_vector = {}
        action_vector[self.__externalBrainName] = [action] # needs to be list in case of multiple agents, TODO: support more than one agent
        brain_infos = self.unityEnv.step(action_vector)
        brain_info = brain_infos[self.__externalBrainName]
        obs = brain_info.visual_observations[0][0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]
        info = None
        return obs, reward, done, info

    def reset(self):
        obs_dict = self.unityEnv.reset(train_mode=self.trainMode)
        # observations of used external brain -> visual observation -> of camera 0 of agent 0
        return obs_dict[self.__externalBrainName].visual_observations[0][0] 

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return self.unityEnv.close()

    def seed(self, seed=None):
        raise NotImplementedError

class FloatToUInt8Frame(gym.ObservationWrapper):
    def __init__(self, env):
        """Convert observation image from float64 to uint8"""
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        # convert from float64, range 0 - 1 to uint8, range 0 - 255
        frame = 255 * frame
        frame = frame.astype(np.uint8)
        frame = frame[...,::-1] #convert to bgr for opencv imshow
        return frame

def make_dqn(env, trainMode=True, reward_range=(-np.inf, np.inf)):
    env = MLToGymEnv(env, trainMode, reward_range)
    env = FloatToUInt8Frame(env)
    env = WarpFrame(env) # Makes sure we have 84 x 84 b&w
    env = FrameStack(env, 4) # Stack last 4 frames
    return env