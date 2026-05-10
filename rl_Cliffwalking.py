import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Hyperparameters ───────────────────────────────────────────────────────────

EPSILON   = 0.1
GAMMA     = 0.99
ALPHA     = 0.1
EPISODES  = 1000
MAX_STEPS = 200
RUNS      = 10
SEED_BASE = 42


#  ε-greedy policy helper

def epsilon_greedy(Q, state, rng, epsilon=EPSILON):
    """Epsilon-greedy action selection"""
    if rng.random() < epsilon:
        return rng.integers(Q.shape[1])
    return int(np.argmax(Q[state]))


# Q-Learning (Off-policy) 

def run_qlearning(seed):
    """
    Run Q-learning for one full experiment using gymnasium environment
    
    Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    Off-policy: Uses max over next actions regardless of behavior policy
    """
    rng = np.random.default_rng(seed)
    env = gym.make('CliffWalking-v1')
    
    # Initialize Q-table: 48 states (4x12 grid) × 4 actions
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    
    for episode in range(EPISODES):
        state, _ = env.reset(seed=seed + episode)  # Different seed per episode
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            action = epsilon_greedy(Q, state, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Handling of terminal states
            if terminated or truncated:
                target = reward  # No future reward for terminal states
            else:
                target = reward + GAMMA * np.max(Q[next_state])
            
            # Q-learning update
            Q[state, action] += ALPHA * (target - Q[state, action])
            
            total_reward += reward
            state = next_state
            steps += 1
        
        returns.append(total_reward)
        
        # Optional: Anneal epsilon for better convergence
        # if episode % 200 == 0 and episode > 0:
        #     globals()['EPSILON'] *= 0.95
    
    env.close()
    return np.array(returns)


# SARSA (On-policy) 

def run_sarsa(seed):
    """
    Run SARSA for one full experiment using gymnasium environment
    
    SARSA update: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
    On-policy: Uses actual next action from behavior policy
    """
    rng = np.random.default_rng(seed)
    env = gym.make('CliffWalking-v1')
    
    # Initialize Q-table: 48 states (4x12 grid) × 4 actions
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    
    for episode in range(EPISODES):
        state, _ = env.reset(seed=seed + episode)
        action = epsilon_greedy(Q, state, rng)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Handling of terminal states
            if terminated or truncated:
                target = reward  # No future reward for terminal states
                next_action = None  # Not used
            else:
                next_action = epsilon_greedy(Q, next_state, rng)
                target = reward + GAMMA * Q[next_state, next_action]
            
            # SARSA update
            Q[state, action] += ALPHA * (target - Q[state, action])
            
            total_reward += reward
            state = next_state
            if not (terminated or truncated):
                action = next_action
            steps += 1
        
        returns.append(total_reward)
    
    env.close()
    return np.array(returns)


# Run experiments with progress tracking 

print("=" * 70)
print("Q-Learning vs SARSA on CliffWalking (Gymnasium)")
print(f"Environment: CliffWalking-v1 (4×12 grid, 48 states, 4 actions)")
print(f"Parameters: ε={EPSILON}, γ={GAMMA}, α={ALPHA}")
print(f"Episodes: {EPISODES}, Runs: {RUNS}, Max Steps: {MAX_STEPS}")
print("=" * 70)

print("\n[1/2] Running Q-Learning")
ql_runs = []
for i in range(RUNS):
    seed = SEED_BASE + i
    returns = run_qlearning(seed)
    ql_runs.append(returns)
    print(f"  ✓ Q-Learning run {i+1}/{RUNS} complete (seed={seed}, "
          f"final return={returns[-100:].mean():.1f})")

ql_runs = np.array(ql_runs)
ql_avg = ql_runs.mean(axis=0)
ql_std = ql_runs.std(axis=0)
ql_sterr = ql_std / np.sqrt(RUNS)  # Standard error

print("\n[2/2] Running SARSA...")
sarsa_runs = []
for i in range(RUNS):
    seed = SEED_BASE + i + 100  # Different seeds to avoid correlation
    returns = run_sarsa(seed)
    sarsa_runs.append(returns)
    print(f"  ✓ SARSA run {i+1}/{RUNS} complete (seed={seed}, "
          f"final return={returns[-100:].mean():.1f})")

sarsa_runs = np.array(sarsa_runs)
sarsa_avg = sarsa_runs.mean(axis=0)
sarsa_std = sarsa_runs.std(axis=0)
sarsa_sterr = sarsa_std / np.sqrt(RUNS)

print("\n All experiments complete\n")


#  Statistical analysis 

# Calculate confidence intervals (95%)
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)

ql_ci_lower = ql_avg - z_score * ql_sterr
ql_ci_upper = ql_avg + z_score * ql_sterr
sarsa_ci_lower = sarsa_avg - z_score * sarsa_sterr
sarsa_ci_upper = sarsa_avg + z_score * sarsa_sterr

# Final performance comparison (last 100 episodes)
ql_final = ql_avg[-100:].mean()
sarsa_final = sarsa_avg[-100:].mean()
ql_final_std = ql_std[-100:].mean()
sarsa_final_std = sarsa_std[-100:].mean()

# Statistical significance test (paired t-test on last 100 episodes)
from scipy import stats as scipy_stats
t_stat, p_value = scipy_stats.ttest_rel(ql_runs[:, -100:].mean(axis=1), 
                                         sarsa_runs[:, -100:].mean(axis=1))

# Convergence speed (episodes to reach -20 return)
ql_convergence = np.argmax(ql_avg < -20) if np.any(ql_avg < -20) else EPISODES
sarsa_convergence = np.argmax(sarsa_avg < -20) if np.any(sarsa_avg < -20) else EPISODES

# Best returns
ql_best = ql_runs.max()
sarsa_best = sarsa_runs.max()


# Enhanced Plotting 

def smooth_curve(data, window=20):
    """Smooth curve using moving average"""
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed

window = 20
episodes_smoothed = np.arange(window - 1, EPISODES)

# Create publication-quality plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Learning curves with confidence intervals
ax1 = axes[0]
ax1.plot(episodes_smoothed, smooth_curve(ql_avg, window), 
         color='#2E86AB', linewidth=2.5, label='Q-Learning', alpha=0.9)
ax1.fill_between(episodes_smoothed, 
                 smooth_curve(ql_ci_lower, window), 
                 smooth_curve(ql_ci_upper, window),
                 alpha=0.3, color='#2E86AB', label='95% CI')

ax1.plot(episodes_smoothed, smooth_curve(sarsa_avg, window), 
         color='#A23B72', linewidth=2.5, label='SARSA', alpha=0.9)
ax1.fill_between(episodes_smoothed, 
                 smooth_curve(sarsa_ci_lower, window), 
                 smooth_curve(sarsa_ci_upper, window),
                 alpha=0.3, color='#A23B72', label='95% CI')

ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Return', fontsize=12, fontweight='bold')
ax1.set_title('Learning Curves with 95% Confidence Intervals', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, EPISODES)

# Subplot 2: Running average performance
ax2 = axes[1]
running_window = 50
running_avg_ql = np.convolve(ql_avg, np.ones(running_window)/running_window, mode='valid')
running_avg_sarsa = np.convolve(sarsa_avg, np.ones(running_window)/running_window, mode='valid')
episodes_running = np.arange(running_window - 1, EPISODES)

ax2.plot(episodes_running, running_avg_ql, color='#2E86AB', linewidth=2.5, 
         label=f'Q-Learning (window={running_window})')
ax2.plot(episodes_running, running_avg_sarsa, color='#A23B72', linewidth=2.5, 
         label=f'SARSA (window={running_window})')

# Add horizontal line for optimal return
optimal_return = -13
ax2.axhline(y=optimal_return, color='green', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Optimal return = {optimal_return}')

ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'{running_window}-Episode Running Average', fontsize=12, fontweight='bold')
ax2.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f'Q-Learning vs SARSA on CliffWalking-v0 (Gymnasium)\n'
             f'ε={EPSILON}, γ={GAMMA}, α={ALPHA} | {RUNS} runs × {EPISODES} episodes',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('qlearning_vs_sarsa_gymnasium.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Plot saved as 'qlearning_vs_sarsa_gymnasium.png'")


#Print performance statistics 

print("\n" + "=" * 70)
print("PERFORMANCE STATISTICS")
print("=" * 70)
print(f"\nFinal Performance (last 100 episodes):")
print(f"  Q-Learning: {ql_final:.2f} ± {ql_final_std:.2f}")
print(f"  SARSA:      {sarsa_final:.2f} ± {sarsa_final_std:.2f}")
print(f"  Difference: {ql_final - sarsa_final:.2f}")

print(f"\nBest Episode Return (across all runs):")
print(f"  Q-Learning: {ql_best:.2f}")
print(f"  SARSA:      {sarsa_best:.2f}")

print(f"\nConvergence Speed (episodes to reach -20 return):")
print(f"  Q-Learning: {ql_convergence} episodes")
print(f"  SARSA:      {sarsa_convergence} episodes")

print(f"\nStatistical Significance (paired t-test, last 100 eps):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✓ Difference is statistically significant (p < 0.05)")
else:
    print(f"  ✗ Difference is NOT statistically significant (p ≥ 0.05)")
