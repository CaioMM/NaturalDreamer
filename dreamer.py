import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import numpy as np
import os
from scipy import stats

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, Encoder, Decoder, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
from envs import flattenObservation
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
from collections import Counter


class Dreamer:
    def __init__(self, observationShape, actionSize, actionLow, actionHigh, actionType, device, config):
        self.observationShape   = observationShape
        self.actionSize         = actionSize
        self.config             = config
        self.device             = device
        self.actionType         = actionType
        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.fullStateSize  = config.recurrentSize + self.latentSize

        self.actor           = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, actionType, device,                                  config.actor          ).to(self.device)
        self.critic          = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        self.encoder         = Encoder(observationShape, self.config.encodedObsSize,                                                 config.encoder        ).to(self.device)
        self.decoder         = Decoder(self.fullStateSize, observationShape,                                                     config.decoder        ).to(self.device)
        self.recurrentModel  = RecurrentModel(config.recurrentSize, self.latentSize, actionSize, config.recurrentModel, actionType, actionHigh ).to(self.device)
        self.priorNet        = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        self.posteriorNet    = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet   ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)
        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                                              config.continuation   ).to(self.device)

        self.buffer         = ReplayBuffer(observationShape, actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)

        self.totalEpisodes      = 0
        self.totalEnvSteps      = 0
        self.totalGradientSteps = 0


    def worldModelTraining(self, data):
        encodedObservations = self.encoder(data.observations.view(-1, *self.observationShape)).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState  = torch.zeros(self.config.batchSize, self.recurrentSize,    device=self.device)
        previousLatentState     = torch.zeros(self.config.batchSize, self.latentSize,       device=self.device)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState    = posterior

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        reconstructionMeans        =  self.decoder(fullStates.view(-1, self.fullStateSize)).view(self.config.batchSize, self.config.batchLength-1, *self.observationShape)
        reconstructionDistribution =  Independent(Normal(reconstructionMeans, 1), len(self.observationShape))
        reconstructionLoss         = -reconstructionDistribution.log_prob(data.observations[:, 1:]).mean()

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()
        # # After computing reward loss:
        # print(f"Reward prediction:")
        # print(f"  True rewards: min={data.rewards[:, 1:].min():.4f}, max={data.rewards[:, 1:].max():.4f}, mean={data.rewards[:, 1:].mean():.4f}")
        # print(f"  Predicted rewards: min={rewardDistribution.mean.min():.4f}, max={rewardDistribution.mean.max():.4f}, mean={rewardDistribution.mean.mean():.4f}")
        # # print(f"  Prediction error: {(rewardDistribution.mean - data.rewards[:, 1:]).abs().mean():.4f}")

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss # I think that the reconstruction loss is relatively a bit too high (11k) 

        print(f"Gradient Steps: {self.totalGradientSteps} | World model loss: {worldModelLoss.item():.4f} | reconstruction loss: {reconstructionLoss.item():.4f} | reward loss: {rewardLoss.item():.4f} | KL loss: {klLoss.item():.4f} ")
        
        
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.dones[:, 1:])
            worldModelLoss      += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}
        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            # In training loop, after actor samples action:
            # print(f"Action shape: {action.shape}, dtype: {action.dtype}")
            # print(f"Action sample: {action[0]}")  # Should be integers like [2, 4, 1, 0]
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)
        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)
        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, resetSeed, seed=None, evaluation=False, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation, _ = env.reset(seed=resetSeed)
            # print(f"uav positions at reset: {[uav.grid_id for uav in env.uavs]}")
            if isinstance(observation, dict):
                observation = flattenObservation(observation)
            
            encodedObservation = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))
            # print(f"Initial observation shape: {observation.shape}, encoded shape: {encodedObservation.shape}")
            
            currentScore, stepCount, done, frames = 0, 0, False, []
            while not done:
                recurrentState      = self.recurrentModel(recurrentState, latentState, action)
                # print(f"Recurrent state shape: {recurrentState.shape}")
                
                latentState, _      = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))
                # print(f"Latent state shape: {latentState.shape}")
                action          = self.actor(torch.cat((recurrentState, latentState), -1))
                # print(f"Action shape: {action.shape} | Action: {action}")
                actionNumpy     = action.cpu().numpy().reshape(-1)
                
                nextObservation, reward, done, truncated, _ = env.step(actionNumpy)
                done = done or truncated
                # print(f"Step {stepCount}: reward {reward}, done {done}")
                if isinstance(nextObservation, dict):
                    nextObservation = flattenObservation(nextObservation)
                if not evaluation:
                    self.buffer.add(observation, actionNumpy, reward, nextObservation, done)
                # frame = env.render()
                # print(frame.shape, frame.dtype)
                # return -1
                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio warning
                    targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                    frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
                # print(f"Next observation shape: {nextObservation.shape}, encoded shape: {encodedObservation.shape}")
                observation = nextObservation
                
                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        return sum(scores)/numEpisodes if numEpisodes else None
    
    @torch.no_grad()
    def environmentInteractionEvaluation(self, env, numEpisodes, model_identifier, seed=None, evaluation=True, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        max_steps = env.max_episode_steps
        # variables to track performance
        coverage = np.zeros((numEpisodes, max_steps))
        energy_efficiency = np.zeros((numEpisodes, max_steps))
        reward_per_step = np.zeros((numEpisodes, max_steps))

        transmission_power = np.zeros((env.num_uavs, numEpisodes, max_steps))
        spreading_factors = np.zeros((env.num_uavs, numEpisodes, max_steps))
        bandwidths = np.zeros((env.num_uavs, numEpisodes, max_steps))

        grid_positions = np.zeros((env.num_uavs, numEpisodes, max_steps))
        positions = np.zeros((env.num_uavs, numEpisodes, 2)) # Initial grid and final grid for each UAV
        duration_per_episode = np.zeros(numEpisodes)

        power_consumption = np.zeros((numEpisodes, max_steps))
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation, info = env.reset(seed=i)
            positions[:, i, 0] = [uav.grid_id for uav in env.uavs]  # initial grid positions
            if isinstance(observation, dict):
                observation = flattenObservation(observation)
            
            encodedObservation = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))
            
            currentScore, stepCount, done, frames = 0, 0, False, []
            prediction_errors = []
            while not done:

                # Logging performance metrics
                coverage[i, stepCount] = info['total_coverage']
                energy_efficiency[i, stepCount] = info['global_ee']
                if energy_efficiency[i, stepCount] < 0:
                    energy_efficiency[i, stepCount] = 0
                power_consumption[i, stepCount] = np.array([(10 ** (uav.tp / 10) / 1000) * len(uav.associated_endnodes) for uav in env.uavs]).sum()
                transmission_power[:, i, stepCount] = [uav.tp for uav in env.uavs]
                spreading_factors[:, i, stepCount] = [uav.sf for uav in env.uavs]
                bandwidths[:, i, stepCount] = [uav.bw for uav in env.uavs]
                grid_positions[:, i, stepCount] = [uav.grid_id for uav in env.uavs]

                # world model prediction before stepping the environment
                with torch.no_grad():
                    recurrentState = self.recurrentModel(recurrentState, latentState, action)
                    predicted_latent, _ = self.priorNet(recurrentState)
                    fullStatePredicted = torch.cat((recurrentState, predicted_latent), dim=-1)
                    predicted_next_obs = self.decoder(fullStatePredicted.view(-1, self.fullStateSize))
                    predicted_next_obs = predicted_next_obs.view(1, *self.observationShape)

                # actual environment step
                # recurrentState      = self.recurrentModel(recurrentState, latentState, action)                
                latentState, _      = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))
                action          = self.actor(torch.cat((recurrentState, latentState), -1))
                actionNumpy     = action.cpu().numpy().reshape(-1)
                
                nextObservation, reward, done, truncated, info = env.step(actionNumpy)
                done = done or truncated
                if isinstance(nextObservation, dict):
                    nextObservation = flattenObservation(nextObservation)
                if not evaluation:
                    self.buffer.add(observation, actionNumpy, reward, nextObservation, done)

                # compute prediction error
                actual_next_obs_tensor = torch.from_numpy(nextObservation).float().to(self.device)
                prediction_error = (predicted_next_obs.squeeze(0).cpu() - nextObservation).pow(2).mean()
                prediction_errors.append(prediction_error.item())

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
                # print(f"Step count {stepCount} prediction error: {prediction_error:.4f}") 
                observation = nextObservation
                
                reward_per_step[i, stepCount] = reward
                currentScore += reward
                stepCount += 1
                if done:
                    avg_prediction_error = np.mean(prediction_errors)
                    # print(f"Episode {i} finished. Average Prediction Error: {avg_prediction_error:.4f} | Standard Deviation: {np.std(prediction_errors):.4f}")
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount
                    break
            positions[:, i, 1] = [uav.grid_id for uav in env.uavs]  # final grid positions
            duration_per_episode[i] = stepCount

        print(f'Average Duration over {numEpisodes} episodes: {np.mean(duration_per_episode):.2f} steps')
        print(f'Average Reward per Step: {np.mean(reward_per_step[:, :]):.2f}')
        print(f'Average Coverage: {np.mean(coverage[:, :] * 100):.2f} %')
        print(f'Average Energy Efficiency: {np.mean(energy_efficiency[:, :]):.2f} bits/Joule')
        print(f'Average Power Consumption per Step: {np.mean(power_consumption[:, :]):.2f} W')

        print(f'UAV 1 Position Usage Statistics:')
        print(f'  Initial Positions: {stats.mode(positions[0, :, 0])}')
        print(f'  Final Positions: {stats.mode(positions[0, :, 1])}')
        print(f'UAV 2 Position Usage Statistics:')
        print(f'  Initial Positions: {stats.mode(positions[1, :, 0])}')
        print(f'  Final Positions: {stats.mode(positions[1, :, 1])}')
        # return -1
        # Plot histogram of grid position usage for each UAV in the same figure using subplots
        fig, axs = plt.subplots(env.num_uavs, 1, sharex=True)
        for uav_idx in range(env.num_uavs):
            initial_positions = positions[uav_idx, :, 0]
            final_positions = positions[uav_idx, :, 1]
            initial_counts = Counter(initial_positions)
            final_counts = Counter(final_positions)
            
            initial_list = list(initial_counts.keys())
            final_list = list(final_counts.keys())

            # plot initial positions
            axs[uav_idx].bar([pos - 0.2 for pos in initial_list], [initial_counts[pos] for pos in initial_list], width=0.4, label='Initial Positions', alpha=0.7)
            # plot final positions
            axs[uav_idx].bar([pos + 0.2 for pos in final_list], [final_counts[pos] for pos in final_list], width=0.4, label='Final Positions', alpha=0.7)
            axs[uav_idx].set_ylabel(f'UAV {uav_idx + 1} Counts')
            axs[uav_idx].legend()
            axs[uav_idx].grid()

        plt.suptitle(f'Grid Position Usage per UAV - Model: {model_identifier}')
        plt.xticks(np.arange(0, 101, 5))
        plt.xlabel('Grid Position ID')
        plt.tight_layout()
        plt.savefig(f'./plots/dreamer_{model_identifier}_grid_position_usage.png')
        plt.close()

        # Create first subplot for power consumption
        plt.figure()
        mean_power = np.mean(power_consumption, axis=0)
        std_power = np.std(power_consumption, axis=0)
        plt.fill_between(range(max_steps), mean_power - std_power, mean_power + std_power, alpha=0.2)
        plt.plot(range(max_steps), mean_power, label='Power Consumption', color='red')
        plt.xlabel('Steps')
        plt.ylabel('Power Consumption (W)')
        plt.title(f'Power Consumption over Steps - Model: {model_identifier}')
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/dreamer_{model_identifier}_power_consumption.png')

        # plot coverage mean and std over episodes per step considering only episodes that reached max steps
        mean_coverage = np.mean(coverage, axis=0)
        std_coverage = np.std(coverage, axis=0)
        plt.figure()
        plt.fill_between(range(max_steps), mean_coverage - std_coverage, mean_coverage + std_coverage, alpha=0.2)
        plt.plot(range(max_steps), mean_coverage, label='Coverage')
        plt.xlabel('Steps')
        plt.ylabel('Coverage')
        plt.title(f'Coverage over Steps - Model: {model_identifier}')
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/dreamer_{model_identifier}_coverage.png')

        # plot energy efficiency mean and std over episodes per step
        mean_energy_efficiency = np.mean(energy_efficiency, axis=0)
        std_energy_efficiency = np.std(energy_efficiency, axis=0)
        plt.figure()
        plt.fill_between(range(max_steps), mean_energy_efficiency - std_energy_efficiency, mean_energy_efficiency + std_energy_efficiency, alpha=0.2)
        plt.plot(range(max_steps), mean_energy_efficiency, label='Energy Efficiency', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Energy Efficiency (bits/Joule)')
        plt.title(f'Energy Efficiency over Steps - Model: {model_identifier}')
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/dreamer_{model_identifier}_energy_efficiency.png')
        return sum(scores)/numEpisodes if numEpisodes else None
    

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])
