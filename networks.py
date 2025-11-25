import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough, Categorical
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()
        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        return self.recurrent(self.activation(self.linear(torch.cat((latentState, action), -1))), recurrentState)


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
        self.outputSize = outputSize

        input_dim = inputShape[0] if isinstance(inputShape, (tuple, list)) else inputShape
        hidden_dim = self.config.depth * 32
        self.linearNet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            # Final projection to latent space
            nn.Linear(hidden_dim, outputSize),
            activation # Preserving the activation at the end as per your original code
        )

    def forward(self, x):
        return self.linearNet(x).view(-1, self.outputSize)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()

        # Handle outputShape being an int (52) or tuple (52,)
        self.output_dim = outputShape[0] if isinstance(outputShape, (tuple, list)) else outputShape
        
        hidden_dim = self.config.depth * 32
        self.network = nn.Sequential(
            nn.Linear(inputSize, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            # Final projection back to original input size (52)
            nn.Linear(hidden_dim, self.output_dim) 
            # No activation here to allow full range of values (matching original decoder logic)
            )

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionDims, device, config):
        super().__init__()
        self.actionSize = actionSize
        self.actionDims = actionDims
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionDimensions", torch.tensor(actionDims, device=device))

    def forward(self, x, training=False):
        # Get raw logits from the network
        all_logits = self.network(x)
        
        # Split the logits based on the size of each discrete action dimension
        # Example: splits a vector of size 8 into tensors of size [batch, 3] and [batch, 5]
        logit_splits = torch.split(all_logits, self.actionDims, dim=-1)
        
        actions = []
        log_probs = []
        entropies = []

        for logits in logit_splits:
            dist = Categorical(logits=logits)
            
            if training:
                # Sample based on probability during training
                a = dist.sample()
                actions.append(a)
                log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())
            else:
                # Greedy selection (argmax) during inference/evaluation
                a = torch.argmax(logits, dim=-1)
                actions.append(a)

        # Stack actions to shape (Batch, Num_Dimensions)
        action = torch.stack(actions, dim=-1)

        if training:
            # Stack logprobs and entropies
            log_probs = torch.stack(log_probs, dim=-1)
            entropies = torch.stack(entropies, dim=-1)
            
            # Sum logprobs across dimensions (Independence assumption: P(a,b) = P(a)*P(b) -> log P = log a + log b)
            return action, log_probs.sum(-1), entropies.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
