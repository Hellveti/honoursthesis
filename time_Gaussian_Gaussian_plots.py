"""
Multi-seed mode probability plotter for the time-only Gaussian–Gaussian agent.
Imports env_1d and sarsa_agent from time_Gaussian_Gaussian.py,
trains N seeds, records mode probability evolution per episode,
and plots the mean mode probability across all seeds.

Place this file in the same directory as time_Gaussian_Gaussian.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
from scipy.optimize import curve_fit


_mod = importlib.import_module("time_Gaussian_Gaussian")
env_1d = _mod.env_1d
sarsa_agent = _mod.sarsa_agent

# ============================================================
# CONFIGURATION
# ============================================================
N_SEEDS = 30         # Number of random seeds to train

# Environment / agent hyper-parameters
ENV_SIZE = 100
N_CENTERS = 200
SIGMA = 0.80
LR = 0.05
GAMMA = 0.95
EPISODES = 1000
STEPS_PER_EPISODE = 10000
AGENT_STD = [1, ]

# Whether to include the closeness reward curve
HAVE_CLOSENESS_REWARD = True

# If None, seeds are drawn randomly; otherwise provide a list of ints
SEED_LIST = None  # e.g. [42, 123, 999, ...]
# ============================================================


def train_single_seed(seed):
    #### Train one agent and return mode probability history + eval steps
    np.random.seed(seed)
    env = env_1d(start_range=(-ENV_SIZE, ENV_SIZE), target=0,
                 have_closeness_reward=HAVE_CLOSENESS_REWARD)
    agent = sarsa_agent(n_centers=N_CENTERS, sigma=SIGMA,agent_std=AGENT_STD,lr=LR, gamma=GAMMA,start_range=(-ENV_SIZE, ENV_SIZE))

    # Record mode probs at the end of each episode: shape (EPISODES, 2)
    pi_history = np.zeros((EPISODES, 2))

    for ep in range(EPISODES):
        state = env.reset()
        action, direction_id, mode_id = agent.select_action(state)
        terminated = False
        steps = 0

        while not terminated and steps < STEPS_PER_EPISODE:
            next_state, reward, terminated = env.step(action)
            if not terminated:
                next_action, next_direction_id, next_mode_id = agent.select_action(next_state)
            else:
                next_action, next_direction_id, next_mode_id = None, None, None

            agent.policy_update(state, direction_id, mode_id, reward,next_state, next_direction_id, next_mode_id, terminated)

            state = next_state
            action = next_action
            direction_id = next_direction_id
            mode_id = next_mode_id
            steps += 1

        pi_history[ep] = agent.get_mode_probs()

    # --- EVALUATION ---
    K_EVAL = 500
    eval_steps = []
    for _ in range(K_EVAL):
        state = env.reset()
        action, direction_id, mode_id = agent.select_action(state)
        terminated = False
        steps = 0
        while not terminated and steps < STEPS_PER_EPISODE:
            next_state, reward, terminated = env.step(action)
            if not terminated:
                action, direction_id, mode_id = agent.select_action(next_state)
            state = next_state
            steps += 1
        eval_steps.append(steps)
    # --- END EVALUATION ---

    return {
        "pi_history": pi_history,
        "eval_steps": eval_steps,
        "seed": seed,
    }


def main():
    # Generate or use provided seeds
    if SEED_LIST is not None:
        seeds = SEED_LIST[:N_SEEDS]
    else:
        rng = np.random.RandomState(111111)
        seeds = rng.randint(0, 100000000, size=N_SEEDS).tolist()

    print(f"Training {N_SEEDS} seeds")
    print(f"Seeds: {seeds}\n")

    

    results = []
    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{N_SEEDS}] Training seed {seed} ...")
        results.append(train_single_seed(seed))

    # Stack pi histories: shape (N_SEEDS, EPISODES, 2)
    all_pi = np.stack([r["pi_history"] for r in results])

    # Mean across seeds: shape (EPISODES, 2)
    avg_pi = all_pi.mean(axis=0)

    # pi1 is now rise, pi2 is now decay
    def pi2(x, A, gamma):
        return 1 - A * np.exp(-gamma * x)

    def pi1(x, A, gamma):
        return A * np.exp(-gamma * x)
    
    

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(12, 5))

    #Plot the data
    episodes_axis = np.arange(EPISODES)
    ax.plot(episodes_axis, avg_pi[:, 0], color='red', linewidth=2,label=f'$\sigma_1$ = {AGENT_STD[0]}',alpha=0.6)
    ax.plot(episodes_axis, avg_pi[:, 1], color='blue', linewidth=2,label=f'$\sigma_2$ = {AGENT_STD[1]}',alpha=0.6)

    colors = ['red', 'blue']

    # Detect rising vs decaying from actual data (last 10% vs first 10% of episodes)
    n = max(EPISODES // 10, 1)
    trend_0 = avg_pi[-n:, 0].mean() - avg_pi[:n, 0].mean()
    rising_idx = 0 if trend_0 > 0 else 1
    decaying_idx = 1 - rising_idx

    param_rise, _ = curve_fit(pi2, episodes_axis, avg_pi[:, rising_idx], p0=[1.0, 0.01], maxfev=5000)
    param_decay, _ = curve_fit(pi1, episodes_axis, avg_pi[:, decaying_idx], p0=[1.0, 0.01], maxfev=5000)

    fit_rise = pi2(episodes_axis, *param_rise)
    fit_decay = pi1(episodes_axis, *param_decay)

    ax.plot(episodes_axis, fit_rise, color=colors[rising_idx], linestyle='--',linewidth=2,
            label=f'$1-{param_rise[0]:.2f} e^{{-{param_rise[1]:.5f} t}}$', zorder=10)
    ax.plot(episodes_axis, fit_decay, color=colors[decaying_idx], linestyle='--',linewidth=2,
            label=f'${param_decay[0]:.2f} e^{{-{param_decay[1]:.5f} t}}$', zorder=10)


    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_title(f"Mixture Weight Evolution for $\sigma_1$={AGENT_STD[0]} vs $\sigma_2$={AGENT_STD[1]}", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)



    plt.tight_layout()
    plt.savefig(f"t_gg_mode_probs_gaussian{AGENT_STD[0]}_gaussian{AGENT_STD[1]}.png", dpi=150)
    plt.show()
    print(f"\nSaved to t_gg_mode_probs_gaussian{AGENT_STD[0]}_gaussian{AGENT_STD[1]}.png")

    #Plot 5: Lin reg of each mode probability vs number of episodes
    # filename = f"linreg_gaussian{AGENT_STD[0]}_gaussian{AGENT_STD[1]}"
    # os.makedirs(filename, exist_ok=True)

    # log_pi1 = np.log(avg_pi[:, 0])
    # log_pi2 = np.log(avg_pi[:, 1])

    # m1, b1 = np.polyfit(episodes_axis, log_pi1, 1)
    # linreg1 = m1 * episodes_axis + b1

    # m2, b2 = np.polyfit(episodes_axis, log_pi2, 1)
    # linreg2 = m2 * episodes_axis + b2
    # print(f"Linear regression for $\ln(\pi_1)$: slope={m1}, intercept={b1}")
    # print(f"Linear regression for $\ln(\pi_2)$: slope={m2}, intercept={b2}")

    
    

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # pi1fit = pi1(episodes_axis, *param1)
    # ax1.plot(episodes_axis, pi1fit, linewidth=1, color="red", label=f"$pi_2 = 1 - {param1[0]:.2f}e^{{-{param1[1]:.5f}x}}$")
    # ax1.scatter(episodes_axis, , color="green", s=5)
    # ax1.set_title("Semilog plot of $\pi_1$ over $t$")
    # ax1.set_xlabel("Episodes")
    # ax1.set_ylabel("$\ln(\pi_1)$")
    # ax1.legend()
    # ax1.grid(True)
    # ax1.autoscale(enable=True, axis='x', tight=True)
    # ax1.autoscale(enable=True, axis='y', tight=True)

    # ax2.scatter(episodes_axis, log_pi2, color="orange", s=5)
    # ax2.plot(episodes_axis, linreg2, linewidth=1, color="blue", label=f"$\ln(\pi_2)={m2:.5f}t+{b2:.5f}$")
    # ax2.set_title("Semilog plot of $\pi_2$ over $t$")
    # ax2.set_xlabel("Episodes")
    # ax2.set_ylabel("$\ln(\pi_2)$")
    # ax2.legend()
    # ax2.grid(True)
    # ax2.autoscale(enable=True, axis='x', tight=True)
    # ax2.autoscale(enable=True, axis='y', tight=True)

    # plt.tight_layout()
    # plt.savefig(f"{filename}/t_gg_pi_gaussian{AGENT_STD[0]}_gaussian{AGENT_STD[1]}.png")
    # plt.show()

    

    # --- AGGREGATE EVALUATION ACROSS SEEDS ---
    all_means = [np.mean(r["eval_steps"]) for r in results]
    all_medians = [np.median(r["eval_steps"]) for r in results]
    print(f"Across {N_SEEDS} seeds:")
    print(f"  Mean of mean steps:     {np.mean(all_means):.2f}")
    print(f"  Mean of median steps:   {np.mean(all_medians):.2f}")
    print(f"  Median of median steps: {np.median(all_medians):.2f}")
    # --- END AGGREGATE ---





if __name__ == "__main__":
    main()