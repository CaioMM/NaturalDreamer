"""
Diagnostic tests to identify why random actions cause immediate termination
"""

import numpy as np
from collections import Counter
from UavUfpaEnv.envs.UavUfpaEnv import UavUfpaEnv as Env

def diagnoseEnvironment():
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTIC TEST")
    print("=" * 60)
    
    env = Env(num_uavs=2, num_endnodes=7, max_episode_steps=100, 
              grid_size=10, lambda_max=5, debug=False, verbose=False, seed=42)
    
    # Test 1: Check action space
    print("\n1. ACTION SPACE ANALYSIS")
    print("-" * 60)
    action_space = env.action_space
    print(f"Action space type: {type(action_space)}")
    print(f"Action space: {action_space}")
    
    if hasattr(action_space, 'nvec'):
        print(f"Action dimensions: {action_space.nvec}")
        total_combinations = np.prod(action_space.nvec)
        print(f"Total possible action combinations: {total_combinations}")
    
    # Test 2: Random action sampling
    print("\n2. RANDOM ACTION SAMPLING (100 episodes)")
    print("-" * 60)
    
    episode_lengths = []
    termination_reasons = []
    
    for episode in range(100):
        obs, info = env.reset(seed=42 + episode)
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if done:
                # Try to capture why it terminated
                termination_reasons.append(info.get('termination_reason', 'unknown'))
                break
        
        episode_lengths.append(steps)
    
    # Analyze episode lengths
    length_counts = Counter(episode_lengths)
    avg_length = np.mean(episode_lengths)
    median_length = np.median(episode_lengths)
    min_length = np.min(episode_lengths)
    max_length = np.max(episode_lengths)
    
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Median episode length: {median_length:.0f}")
    print(f"Min/Max episode length: {min_length}/{max_length}")
    print(f"\nEpisode length distribution:")
    for length in sorted(length_counts.keys())[:10]:
        count = length_counts[length]
        percentage = (count / 100) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {length:3d} steps: {bar} {count} episodes ({percentage:.0f}%)")
    
    # Check if problem is severe
    short_episodes = sum(1 for l in episode_lengths if l <= 2)
    print(f"\n⚠️ Episodes ending in ≤2 steps: {short_episodes}/100 ({short_episodes}%)")
    
    if short_episodes > 50:
        print("❌ CRITICAL: >50% episodes terminate immediately!")
        print("   This environment is NOT suitable for RL without modifications.")
    elif short_episodes > 20:
        print("⚠️ WARNING: >20% episodes terminate immediately!")
        print("   Environment may need easier initial conditions or relaxed constraints.")
    else:
        print("✅ OK: Most episodes survive random actions.")
    
    # Test 3: Action validity check
    print("\n3. ACTION VALIDITY TEST")
    print("-" * 60)
    
    obs, info = env.reset(seed=42)
    valid_actions = 0
    invalid_actions = 0
    action_lengths = []
    
    # Test each action from the same initial state
    for _ in range(50):
        obs, info = env.reset(seed=42)  # Same initial state
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            invalid_actions += 1
            action_lengths.append(1)
        else:
            valid_actions += 1
            # Take a few more random steps to see how long it survives
            steps = 1
            for _ in range(10):
                if done:
                    break
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            action_lengths.append(steps)
    
    print(f"From same initial state:")
    print(f"  Valid first actions: {valid_actions}/50 ({valid_actions*2}%)")
    print(f"  Invalid first actions: {invalid_actions}/50 ({invalid_actions*2}%)")
    
    if invalid_actions > 25:
        print("❌ CRITICAL: >50% of random actions immediately terminate!")
        print("   Problem: Action space contains too many 'invalid' actions")
    elif invalid_actions > 10:
        print("⚠️ WARNING: >20% of random actions immediately terminate!")
        print("   Consider: Action masking or constraint relaxation")
    
    # Test 4: Check if specific actions always fail
    print("\n4. DETERMINISTIC ACTION TEST")
    print("-" * 60)
    
    # Test if always taking action [0,0,...,0] works
    obs, info = env.reset(seed=42)
    zero_action = [0] * len(env.action_space.nvec) if hasattr(env.action_space, 'nvec') else 0
    steps = 0
    done = False
    
    while not done and steps < 20:
        obs, reward, done, truncated, info = env.step(zero_action)
        steps += 1
    
    print(f"Taking constant action {zero_action}:")
    print(f"  Survived for {steps} steps")
    
    if steps <= 2:
        print("  ❌ Even 'no-op' action fails immediately!")
    elif steps >= 10:
        print("  ✅ Constant action can survive - environment has valid strategies")
    
    # Test 5: Reward structure analysis
    print("\n5. REWARD STRUCTURE ANALYSIS")
    print("-" * 60)
    
    rewards_collected = []
    for episode in range(20):
        obs, info = env.reset(seed=42 + episode)
        done = False
        episode_rewards = []
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
            steps += 1
        
        rewards_collected.append(episode_rewards)
    
    # Analyze reward patterns
    all_rewards = [r for ep in rewards_collected for r in ep]
    unique_rewards = set(all_rewards)
    
    print(f"Unique reward values seen: {sorted(unique_rewards)}")
    print(f"Reward range: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")
    print(f"Average reward per step: {np.mean(all_rewards):.2f}")
    
    # Check for sparse rewards
    non_zero_rewards = sum(1 for r in all_rewards if abs(r) > 0.01)
    reward_density = non_zero_rewards / len(all_rewards) if all_rewards else 0
    
    print(f"Non-zero rewards: {non_zero_rewards}/{len(all_rewards)} ({reward_density*100:.1f}%)")
    
    if reward_density < 0.1:
        print("⚠️ WARNING: Sparse rewards (<10% non-zero)")
        print("   Consider: Reward shaping or auxiliary rewards")
    
    # Test 6: State space analysis
    print("\n6. STATE/OBSERVATION SPACE")
    print("-" * 60)
    
    obs, info = env.reset(seed=42)
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, range=[{value.min():.2f}, {value.max():.2f}]")
    elif isinstance(obs, np.ndarray):
        print(f"Observation shape: {obs.shape}")
        print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Generate recommendations based on findings
    if short_episodes > 30:
        print("\n🔧 IMMEDIATE ACTIONS NEEDED:")
        print("1. Relax environment constraints:")
        print("   - Increase grid_size (10 → 20)")
        print("   - Reduce num_endnodes (7 → 3)")
        print("   - Increase lambda_max (5 → 10)")
        print("   - Add boundary padding/wraparound")
        print("\n2. Implement action masking:")
        print("   - Mask invalid actions at each state")
        print("   - Only allow valid action combinations")
        print("\n3. Change terminal conditions:")
        print("   - Make failure conditions less strict")
        print("   - Add 'soft' penalties instead of immediate termination")
        print("\n4. Curriculum learning:")
        print("   - Start with easier version of task")
        print("   - Gradually increase difficulty")
    
    return episode_lengths, rewards_collected


# Run the diagnostic
if __name__ == "__main__":
    episode_lengths, rewards = diagnoseEnvironment()