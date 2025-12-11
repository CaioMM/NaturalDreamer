import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    # assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    if isinstance(env.action_space, gym.spaces.Box):
        observationShape = env.observation_space.shape
        actionSize = env.action_space.shape[0]
        actionLow = env.action_space.low.tolist()
        actionHigh = env.action_space.high.tolist()
        actionType="continuous"
        return observationShape, actionSize, actionLow, actionHigh, actionType
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        if isinstance(env.observation_space, gym.spaces.Dict):
            sample_obs = env.observation_space.sample()
            flat_obs = np.concatenate([sample_obs[key].flatten() for key in sample_obs.keys()])
            observationShape = flat_obs.shape
        else:
            observationShape = env.observation_space.shape
        actionSize = env.action_space.nvec.shape[0]
        actionLow = [0]*actionSize
        actionHigh = (env.action_space.nvec).tolist()
        actionType='multidiscrete'
        return observationShape, actionSize, actionLow, actionHigh, actionType
    
def flattenObservation(observation):
    """
    Wraps the raw observation in a flattened format
    suitable for RL algorithms.
    
    Args:
        observation (dict): Raw observation from the environment.

    Returns:
        np.ndarray: Flattened observation array.
    """
    flat_obs = np.concatenate([observation[key].flatten() for key in observation.keys()])

    return flat_obs
        

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs