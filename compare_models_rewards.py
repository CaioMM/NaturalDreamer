import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dreamer-reward-file", type=str, default="CarRacing-v3_run_01_reduced_network_dimensions_env_reset_fixed_reward_modified_ee_log_scale_tanh.csv")
parser.add_argument("--ppo-reward-file", type=str, default="PPO-MARL-2U-1766071802_PPO_0.csv")
parser.add_argument("--dreamer-evaluation-file", type=str, default="eval_dreamer_CarRacing-v3_run_01_reduced_network_dimensions_env_reset_fixed_reward_modified_ee_log_scale_tanh_60k.pth_performance_metrics.csv")
parser.add_argument("--ppo-evaluation-file", type=str, default="eval_1766071802_ppo_MARL1080000_metrics.csv")
parser.add_argument("--random-policy-evaluation-file", type=str, default="eval_Random_Policy_metrics.csv")

args = parser.parse_args()

save_path = "comparisonPlots"

base_path = os.path.dirname(os.path.abspath(__file__))
metrics_dreamer_path = os.path.join(base_path, "metrics")
metrics_ppo_path = os.path.join(base_path, "metrics")
dreamer_eval_metrics_path = os.path.join(base_path, "metrics")
ppo_eval_metrics_path = os.path.join(base_path, "metrics")
random_policy_eval_metrics_path = os.path.join(base_path, "metrics")

dreamer_data = pd.read_csv(os.path.join(metrics_dreamer_path, args.dreamer_reward_file))
ppo_data = pd.read_csv(os.path.join(metrics_ppo_path, args.ppo_reward_file))

dreamer_eval_data = pd.read_csv(os.path.join(dreamer_eval_metrics_path, args.dreamer_evaluation_file))
ppo_eval_data = pd.read_csv(os.path.join(ppo_eval_metrics_path, args.ppo_evaluation_file))
random_policy_eval_data = pd.read_csv(os.path.join(random_policy_eval_metrics_path, args.random_policy_evaluation_file))

# print("Dreamer eval Data Head:")
# print(dreamer_eval_data.head())
# print("\nPPO eval Data Head:")
# print(ppo_eval_data.head())
# exit()
# filter PPO data to only include steps less than or equal to max dreamer envSteps
ppo_data = ppo_data[ppo_data['Step'] <= dreamer_data['envSteps'].max()]

# Plotting PPO Step x Reward
plt.figure(figsize=(10, 6))
plt.plot(dreamer_data['envSteps'], dreamer_data['totalReward'], label='Dreamer Reward', color='orange')
plt.plot(ppo_data['Step'], ppo_data['Value'], label='PPO Reward', color='blue')
plt.xlabel('Environment Steps')
plt.ylabel('Total Reward')
plt.title('Comparison of PPO and Dreamer Rewards')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "ppo_vs_dreamer_rewards.png"))
plt.show()

# Plotting PPO vs Dreamer Evaluation Metrics
plt.figure(figsize=(10, 6))
# Plot Coverage mean and std
plt.subplot(3, 1, 1)
plt.plot(dreamer_eval_data['coverage_mean'], label='Dreamer Coverage', color='orange')
plt.plot(dreamer_eval_data['coverage_mean'] + dreamer_eval_data['coverage_std'], linestyle='--', color='orange', alpha=0.5)
plt.plot(dreamer_eval_data['coverage_mean'] - dreamer_eval_data['coverage_std'], linestyle='--', color='orange', alpha=0.5)
plt.fill_between(range(len(dreamer_eval_data)),
                 dreamer_eval_data['coverage_mean'] - dreamer_eval_data['coverage_std'],
                 dreamer_eval_data['coverage_mean'] + dreamer_eval_data['coverage_std'],
                 color='orange', alpha=0.2)
plt.plot(ppo_eval_data['Coverage_Mean'], label='PPO Coverage', color='blue')
plt.plot(ppo_eval_data['Coverage_Mean'] + ppo_eval_data['Coverage_Std'], linestyle='--', color='blue', alpha=0.5)
plt.plot(ppo_eval_data['Coverage_Mean'] - ppo_eval_data['Coverage_Std'], linestyle='--', color='blue', alpha=0.5)
plt.fill_between(range(len(ppo_eval_data)),
                 ppo_eval_data['Coverage_Mean'] - ppo_eval_data['Coverage_Std'],
                 ppo_eval_data['Coverage_Mean'] + ppo_eval_data['Coverage_Std'],
                 color='blue', alpha=0.2)
plt.plot(random_policy_eval_data['Coverage_Mean'], label='Random Policy Coverage', color='green')
plt.plot(random_policy_eval_data['Coverage_Mean'] + random_policy_eval_data['Coverage_Std'], linestyle='--', color='green', alpha=0.5)
plt.plot(random_policy_eval_data['Coverage_Mean'] - random_policy_eval_data['Coverage_Std'], linestyle='--', color='green', alpha=0.5)
plt.fill_between(range(len(random_policy_eval_data)),
                 random_policy_eval_data['Coverage_Mean'] - random_policy_eval_data['Coverage_Std'],
                 random_policy_eval_data['Coverage_Mean'] + random_policy_eval_data['Coverage_Std'],
                 color='green', alpha=0.2)
plt.ylabel('Coverage')
plt.title('Comparison of PPO and Dreamer Evaluation Metrics')
plt.legend()
plt.grid()
# Plot Energy Efficiency mean and std
plt.subplot(3, 1, 2)
plt.plot(dreamer_eval_data['energy_efficiency_mean'], label='Dreamer Energy Efficiency', color='orange')
plt.plot(dreamer_eval_data['energy_efficiency_mean'] + dreamer_eval_data['energy_efficiency_std'], linestyle='--', color='orange', alpha=0.5)
plt.plot(dreamer_eval_data['energy_efficiency_mean'] - dreamer_eval_data['energy_efficiency_std'], linestyle='--', color='orange', alpha=0.5)
plt.fill_between(range(len(dreamer_eval_data)),
                 dreamer_eval_data['energy_efficiency_mean'] - dreamer_eval_data['energy_efficiency_std'],
                 dreamer_eval_data['energy_efficiency_mean'] + dreamer_eval_data['energy_efficiency_std'],
                 color='orange', alpha=0.2)
plt.plot(ppo_eval_data['Energy_Efficiency_Mean'], label='PPO Energy Efficiency', color='blue')
plt.plot(ppo_eval_data['Energy_Efficiency_Mean'] + ppo_eval_data['Energy_Efficiency_Std'], linestyle='--', color='blue', alpha=0.5)
plt.plot(ppo_eval_data['Energy_Efficiency_Mean'] - ppo_eval_data['Energy_Efficiency_Std'], linestyle='--', color='blue', alpha=0.5)
plt.fill_between(range(len(ppo_eval_data)),
                 ppo_eval_data['Energy_Efficiency_Mean'] - ppo_eval_data['Energy_Efficiency_Std'],
                 ppo_eval_data['Energy_Efficiency_Mean'] + ppo_eval_data['Energy_Efficiency_Std'],
                 color='blue', alpha=0.2)
plt.plot(random_policy_eval_data['Energy_Efficiency_Mean'], label='Random Policy Energy Efficiency', color='green')
plt.plot(random_policy_eval_data['Energy_Efficiency_Mean'] + random_policy_eval_data['Energy_Efficiency_Std'], linestyle='--', color='green', alpha=0.5)
plt.plot(random_policy_eval_data['Energy_Efficiency_Mean'] - random_policy_eval_data['Energy_Efficiency_Std'], linestyle='--', color='green', alpha=0.5)
plt.fill_between(range(len(random_policy_eval_data)),
                 random_policy_eval_data['Energy_Efficiency_Mean'] - random_policy_eval_data['Energy_Efficiency_Std'],
                 random_policy_eval_data['Energy_Efficiency_Mean'] + random_policy_eval_data['Energy_Efficiency_Std'],
                 color='green', alpha=0.2)
plt.ylabel('Energy Efficiency (bits/Joule)')
plt.legend()
plt.grid()
# Plot Power Consumption
plt.subplot(3, 1, 3)
plt.plot(dreamer_eval_data['power_consumption_mean'], label='Dreamer Power Consumption', color='orange')
plt.plot(dreamer_eval_data['power_consumption_mean'] + dreamer_eval_data['power_consumption_std'], linestyle='--', color='orange', alpha=0.5)
plt.plot(dreamer_eval_data['power_consumption_mean'] - dreamer_eval_data['power_consumption_std'], linestyle='--', color='orange', alpha=0.5)
plt.fill_between(range(len(dreamer_eval_data)),
                 dreamer_eval_data['power_consumption_mean'] - dreamer_eval_data['power_consumption_std'],
                 dreamer_eval_data['power_consumption_mean'] + dreamer_eval_data['power_consumption_std'],
                 color='orange', alpha=0.2)
plt.plot(ppo_eval_data['Power_Consumption_Mean'], label='PPO Power Consumption', color='blue')
plt.plot(ppo_eval_data['Power_Consumption_Mean'] + ppo_eval_data['Power_Consumption_Std'], linestyle='--', color='blue', alpha=0.5)
plt.plot(ppo_eval_data['Power_Consumption_Mean'] - ppo_eval_data['Power_Consumption_Std'], linestyle='--', color='blue', alpha=0.5)
plt.fill_between(range(len(ppo_eval_data)),
                 ppo_eval_data['Power_Consumption_Mean'] - ppo_eval_data['Power_Consumption_Std'],
                 ppo_eval_data['Power_Consumption_Mean'] + ppo_eval_data['Power_Consumption_Std'],
                 color='blue', alpha=0.2)
plt.plot(random_policy_eval_data['Power_Consumption_Mean'], label='Random Policy Power Consumption', color='green')
plt.plot(random_policy_eval_data['Power_Consumption_Mean'] + random_policy_eval_data['Power_Consumption_Std'], linestyle='--', color='green', alpha=0.5)
plt.plot(random_policy_eval_data['Power_Consumption_Mean'] - random_policy_eval_data['Power_Consumption_Std'], linestyle='--', color='green', alpha=0.5)
plt.fill_between(range(len(random_policy_eval_data)),
                 random_policy_eval_data['Power_Consumption_Mean'] - random_policy_eval_data['Power_Consumption_Std'],
                 random_policy_eval_data['Power_Consumption_Mean'] + random_policy_eval_data['Power_Consumption_Std'],
                 color='green', alpha=0.2)
plt.xlabel('Steps')
plt.ylabel('Power Consumption (W)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "ppo_vs_dreamer_evaluation_metrics.png"))
plt.show()