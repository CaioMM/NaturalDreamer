import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config,  actionType='continuous', actionNvec=None):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()
        self.actionType = actionType

        # For MultiDiscrete create embedding for each action dimension
        if actionType == 'multidiscrete':
            print(f"Creating RecurrentModel with actionNvec: {actionNvec}")
            print(f"actionType: {actionType}")
            assert actionNvec is not None, "actionNvec must be provided for multidiscrete actions"
            self.actionNvec = actionNvec
            embedding = 16 # embedding size for each action dimension

            # Create separate embedding for each action component
            # e.g [5, 6, 5, 3] -> 4 embeddings of size [5, 6, 5, 3] each with embedding dim 16
            self.actionEmbeddings = nn.ModuleList([
                nn.Embedding(n, embedding_dim=embedding) for n in actionNvec
            ])
            totalActionSize = embedding * len(actionNvec)
        else:
            totalActionSize = actionSize

        self.linear = nn.Linear(latentSize + totalActionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        # print(f"RecurrentModel - action input shape: {action.shape}, dtype: {action}")
        if self.actionType == 'multidiscrete':
            # Embed each discrete action dimension
            embeddedActions = []
            for i, embedding in enumerate(self.actionEmbeddings):
                embeddedActions.append(embedding(action[:, i].long()))
            actionFeatures = torch.cat(embeddedActions, dim=-1)
        else:
            actionFeatures = action
        # print(f"RecurrentModel - actionFeatures shape: {actionFeatures.shape}, dtype: {actionFeatures}")
        combined = torch.cat((latentState, actionFeatures), -1)
        return self.recurrent(self.activation(self.linear(combined)), recurrentState)


class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)
    
class EncoderMLP(nn.Module):
    def __init__(self, inputShape, outputSize, config, useSymlog=True):
        super().__init__()
        self.config = config
        self.inputShape = inputShape
        self.outputSize = outputSize
        self.useSymlog = useSymlog

        self.network = sequentialModel1D(inputShape, [self.config.hiddenSize]*self.config.numLayers, outputSize, self.config.activation)

    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def forward(self, x):
        if self.useSymlog:
            x = self.symlog(x)
        return self.network(x)

def Encoder(inputShape, outputSize, config):
    if type(inputShape) == int:
        inputShape = (inputShape,)
    if len(inputShape) == 3:
        return EncoderConv(inputShape, outputSize, config.Conv)
    else:
        print(f"Creating MLP Encoder with input shape {inputShape} and output size {outputSize}")
        return EncoderMLP(inputShape[0], outputSize, config.MLP)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride))

    def forward(self, x):
        return self.network(x)
    
class DecoderMLP(nn.Module):
    def __init__(self, inputSize, outputShape, config, useSymlog=True):
        super().__init__()
        self.config = config
        self.outputShape = outputShape
        self.useSymlog = useSymlog
        outputSize = outputShape

        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, outputSize, self.config.activation)
    
    def symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, x):
        x = self.network(x)
        if self.useSymlog:
            x = self.symexp(x)
        return x

def Decoder(inputSize, outputShape, config):
    if type(outputShape) == int:
        outputShape = (outputShape,)
    if len(outputShape) == 3:
        return DecoderConv(inputSize, outputShape, config.Conv)
    else:
        return DecoderMLP(inputSize, outputShape[0], config.MLP)

class ActorContinuous(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x, training=False):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1) # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action
        
class ActorMultidiscrete(nn.Module):
    def __init__(self, inputSize, actionNvec, device, config):
        super().__init__()
        self.config = config
        self.actionNvec = actionNvec
        self.device = device

        self.actionHeads = nn.ModuleList([
            sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, n, self.config.activation) for n in actionNvec
        ])

    def forward(self, x, training=False):
        logitsList = [head(x) for head in self.actionHeads]

        if training:
            actions = []
            logprobs = []
            entropies = []

            for logits in logitsList:
                probs   = F.softmax(logits, dim=-1)
                uniform = torch.ones_like(probs) / probs.shape[-1]
                finalProbs = (1 - self.config.uniformMix) * probs + self.config.uniformMix * uniform

                dist = torch.distributions.Categorical(probs=finalProbs)
                action = dist.sample()
                actions.append(action)
                logprobs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

            action = torch.stack(actions, dim=-1)
            totalLogprob = torch.stack(logprobs, dim=-1).sum(-1)
            totalEntropy = torch.stack(entropies, dim=-1).sum(-1)

            return action, totalLogprob, totalEntropy
        else:
            actions = []
            for logits in logitsList:
                action = torch.argmax(logits, dim=-1)
                actions.append(action)
            action = torch.stack(actions, dim=-1)
            return action
        

def Actor(inputSize, actionSize, actionLow, actionHigh, actionType, device, config):
    if actionType == 'continuous':
        return ActorContinuous(inputSize, actionSize, actionLow, actionHigh, device, config.continuous)
    elif actionType == 'multidiscrete':
        print(f"Creating Multidiscrete Actor with actionNvec: {actionHigh}")
        return ActorMultidiscrete(inputSize, actionHigh, device, config.multidiscrete)
    else:
        raise ValueError(f"Unsupported action type: {actionType}")

class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))