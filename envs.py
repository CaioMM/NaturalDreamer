import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh

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

    return flat_obs, flat_obs.size

def actionsToOneHot(actions, actionDims):
    """
    Wraps raw action in one-hot encoded format to use with
    Gumbel-Softmax policy.
    Args:
        actions (np.ndarray) or (torch.Tensor): Raw actions from the environment with multi-discrete actions.
        action_dims (list): List of action dimensions for each discrete action.
    Returns:
        np.ndarray: One-hot encoded action.
    """
    actionsOneHot = []
    if isinstance(actions, np.ndarray):
        for i, action_dim in enumerate(actionDims):
            oneHot = np.zeros(action_dim, dtype=np.float32)
            oneHot[actions[i]] = 1.0
            actionsOneHot.append(oneHot)
        
        return np.concatenate(actionsOneHot, axis=-1)
    elif isinstance(actions, torch.Tensor):
        for i, action_dim in enumerate(actionDims):
            oneHot = F.one_hot(actions[:, i], num_classes=action_dim).float()
            actionsOneHot.append(oneHot)
        
        return torch.cat(actionsOneHot, dim=-1)
    else:
        raise TypeError("Actions must be either a numpy array or a torch tensor.")
    
def oneHotToActions(oneHotActions, actionDims):
    """
    Unwraps one-hot encoded action tensor to raw action format.
    Args:
        action_one_hot (torch.Tensor): One-hot encoded action tensor.
        action_dims (list): List of action dimensions for each discrete action.

    Returns:
        np.ndarray: Unwrapped action array.
    """
    # debug action_one_hot
    actions = []
    start = 0
    for action_dim in actionDims:
        action = oneHotActions[start:start + action_dim].argmax(dim=-1).item()
        actions.append(action)
        start += action_dim
    return np.array(actions)


def getMultiDiscreteEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    observation = env.observation_space.sample()
    _, observationShape = flattenObservation(observation)
    actionDims = env.action_space.nvec.tolist()
    actionSize = sum(actionDims)

    return observationShape, actionSize, actionDims

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