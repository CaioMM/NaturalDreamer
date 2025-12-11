import torch
import argparse
from networks import Encoder, Decoder, RecurrentModel, PosteriorNet, RewardModel, Actor, Critic
from dreamer import Dreamer
from utils import loadConfig, seedEverything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="car-racing-v3.yml")

config = loadConfig(parser.parse_args().config)
seedEverything(config.seed)

# Create dummy batch
batch_size = config.dreamer.batchSize
observationShape = 52
action_nvec = [5, 6, 5, 3, 5, 6, 5, 3]  # Your UAV action space

# Dummy data
dummy_obs = torch.randn(batch_size, observationShape)
dummy_actions = torch.stack([
    torch.randint(0, n, (batch_size,)) for n in action_nvec
], dim=-1)
dummy_rewards = torch.randn(batch_size)

print("=== Testing Forward Pass ===")
print(f"Observation shape: {dummy_obs.shape}")
print(f"Action shape: {dummy_actions.shape}, dtype: {dummy_actions.dtype}")
print(f"Action sample: {dummy_actions[0]}")

# Test encoder
encoder = Encoder(observationShape, config.dreamer.encodedObsSize, config.dreamer.encoder).to(device)
encoded = encoder(dummy_obs)
print(f"Encoder output: {encoded.shape}")

# Test posterior (encoder + recurrent state)
h = torch.zeros(batch_size, config.dreamer.recurrentSize)
posterior_input = torch.cat([encoded, h], dim=-1)
posteriorNet = PosteriorNet(config.dreamer.recurrentSize + config.dreamer.encodedObsSize, config.dreamer.latentLength, config.dreamer.latentClasses, config.dreamer.posteriorNet   ).to(device)
latent_posterior, posterior_logits = posteriorNet(posterior_input)
print(f"Posterior latent: {latent_posterior.shape}")

# Test recurrent model with embedded 
recurrentModel = RecurrentModel(config.dreamer.recurrentSize, config.dreamer.latentLength*config.dreamer.latentClasses, 
                                len(action_nvec), config.dreamer.recurrentModel, actionType='multidiscrete', actionNvec=action_nvec).to(device)
h_next = recurrentModel(h, latent_posterior, dummy_actions)
print(f"Recurrent output: {h_next.shape}")
assert h_next.shape == (batch_size, config.dreamer.recurrentSize), "Recurrent output wrong shape!"

# Test reward predictor
model_state = torch.cat([h_next, latent_posterior], dim=-1)
rewardModel = RewardModel(config.dreamer.recurrentSize + (config.dreamer.latentLength*config.dreamer.latentClasses), config.dreamer.reward).to(device)
reward_dist = rewardModel(model_state)
print(f"Reward prediction: mean={reward_dist.mean.shape}")

# Test decoder
decoder = Decoder(config.dreamer.recurrentSize + (config.dreamer.latentLength*config.dreamer.latentClasses), observationShape, config.dreamer.decoder).to(device)
reconstructed = decoder(model_state)
print(f"Decoder output: {reconstructed.shape}")
assert reconstructed.shape == dummy_obs.shape, "Decoder output wrong shape!"

# Test actor
actor = Actor(config.dreamer.recurrentSize + (config.dreamer.latentLength*config.dreamer.latentClasses), len(action_nvec), [0] * len(action_nvec), action_nvec, actionType='multidiscrete', device=device, config=config.dreamer.actor).to(device)

action_out, logprob, entropy = actor(model_state, training=True)
print(f"Actor output: {action_out.shape}, logprob: {logprob.shape}, entropy: {entropy.shape}")
assert action_out.shape == (batch_size, len(action_nvec)), "Actor output wrong shape!"

# Test critic
critic = Critic(config.dreamer.recurrentSize + (config.dreamer.latentLength*config.dreamer.latentClasses), config.dreamer.critic).to(device)
value_dist = critic(model_state)
print(f"Critic output: mean={value_dist.mean.shape}")

print("\n✅ ALL COMPONENTS WORKING!")

# Test actor outputs are in correct ranges
for _ in range(10):
    dummy_state = torch.randn(32, config.dreamer.recurrentSize + config.dreamer.latentLength * config.dreamer.latentClasses)
    actions, _, _ = actor(dummy_state, training=True)
    
    for i, n in enumerate(action_nvec):
        assert actions[:, i].min() >= 0, f"Action dim {i} has negative values!"
        assert actions[:, i].max() < n, f"Action dim {i} exceeds max value {n-1}!"
        assert actions[:, i].dtype in [torch.int64, torch.long], f"Action dim {i} not integer!"

print("✅ Actor outputs valid discrete actions!")

# Quick gradient check
dummy_obs = torch.randn(16, 52, requires_grad=True)
dummy_actions = torch.stack([
    torch.randint(0, n, (16,)) for n in action_nvec
], dim=-1)

# Forward pass
encoded = encoder(dummy_obs)
h = torch.zeros(16, config.dreamer.recurrentSize)
latent, _ = posteriorNet(torch.cat([encoded, h], -1))
h_next = recurrentModel(h, latent, dummy_actions)
model_state = torch.cat([h_next, latent], -1)

# Actor-critic
actions, logprobs, entropy = actor(model_state, training=True)
values = critic(model_state)

# Dummy loss
loss = -logprobs.mean() - entropy.mean() + values.mean.pow(2).mean()
loss.backward()

# Check gradients
print("=== Gradient Check ===")
for name, module in [
    ("Encoder", encoder),
    ("RecurrentModel", recurrentModel),
    ("Actor", actor),
    ("Critic", critic)
]:
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    print(f"{name}: {'HAS gradients' if has_grad else '❌ NO gradients'}")

print("✅ Gradients flowing!")