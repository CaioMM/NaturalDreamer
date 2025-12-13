import gymnasium as gym
import torch
import argparse
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
from UavUfpaEnv.envs.UavUfpaEnv import UavUfpaEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def main(configFile, episodes=100, checkpointFile=None, resume=True):
    assert checkpointFile is not None, "Please provide checkpointFile as argument to main function."
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    # checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    checkpointToLoad = checkpointFile
    env             = UavUfpaEnv(num_uavs=2, num_endnodes=7, grid_size=10, lambda_max=5, debug=False, evaluate_mode=False, seed=42)
    envEvaluation   = UavUfpaEnv(num_uavs=2, num_endnodes=7, grid_size=10, lambda_max=5, debug=False, evaluate_mode=True, seed=42)
    
    observationShape, actionSize, actionLow, actionHigh, actionType = getEnvProperties(env)
    
    print(f"Device: {device} | envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}, actionType {actionType}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, actionType, device, config.dreamer)
    
    if resume:
        print(f"Loading checkpoint from: {checkpointToLoad}")
        dreamer.loadCheckpoint(checkpointToLoad)
        model_identifier = checkpointToLoad.split('/')[-1]

    

    evaluationScore = dreamer.environmentInteractionEvaluation(envEvaluation, episodes, model_identifier, seed=config.seed, evaluation=True)
    print(f"Recent Score: {evaluationScore:>8.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    print(parser.parse_args())
    main(parser.parse_args().config, parser.parse_args().episodes, parser.parse_args().checkpoint)