"""
Multi-seed value function plotter.
Imports env_1d and sarsa_agent from time_space_Gaussian_Cauchy.py,
trains N seeds, and then make a plot of all seeds superimposed
each seed is at high transparency, then we compute an average curve across seeds and plot that in bold to show the general trend.

Place this file in the same directory as time_space_Gaussian_Cauchy.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib

# The module name starts with a digit so we use importlib
_mod = importlib.import_module("time_space_Gaussian_Cauchy")
env_1d = _mod.env_1d
sarsa_agent = _mod.sarsa_agent

# ============================================================
# CONFIGURATION
# ============================================================
N_SEEDS = 200            # Number of random seeds to train

# Environment / agent hyper-parameters
ENV_SIZE = 100
N_CENTERS = 200
SIGMA = 0.80
LR = 0.05
GAMMA = 0.95
EPISODES =  5000
STEPS_PER_EPISODE = 10000

# Plot resolution (number of x-points per subplot)
PLOT_RESOLUTION = 200

# Whether to include the closeness reward curve in the plots
HAVE_CLOSENESS_REWARD = True  

# Whether to normalise each seed's curves to [0, 1] before plotting/averaging
NORMALISE_CURVES = True

# Scale parameters
AGENT_STD = [1, 1]

# If None, seeds are drawn randomly; otherwise provide a list of ints
SEED_LIST = None  # e.g. [42, 123, 999, ...]
# ============================================================

def find_nearest_intersection(xs, curve_a, curve_b, side='right'):
    """
    Find the intersection of two curves closest to the origin.
 
    Arguments
    xs :  The x-axis positions.
    curve_a, curve_b :  The two y-value vectors to intersect.
    side : 'right' | 'left': 'right' restricts the search to x > 0, and 'left'  restricts the search to x < 0.
 
    Returns float or None
    """
    xs = np.asarray(xs)
    mask = xs > 0 if side == 'right' else xs < 0
    xs_region = xs[mask]
    diff = (np.asarray(curve_a) - np.asarray(curve_b))[mask]
 
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
 
    crossings = []
    for idx in sign_changes:
        x0, x1 = xs_region[idx], xs_region[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
        crossings.append(x_cross)
 
    crossings = np.array(crossings)
    return crossings[np.argmin(np.abs(crossings))]


def train_single_seed(seed):
    """Train one agent with the given seed and return the learned value curves."""
    np.random.seed(seed)
    env = env_1d(start_range=(-ENV_SIZE, ENV_SIZE), target=0,
                 have_closeness_reward=HAVE_CLOSENESS_REWARD)
    agent = sarsa_agent(n_centers=N_CENTERS, sigma=SIGMA, lr=LR,
                        gamma=GAMMA, start_range=(-ENV_SIZE, ENV_SIZE), agent_std = AGENT_STD)
 
    for ep in range(EPISODES):
        state = env.reset()
        action, direction_id = agent.select_action(state)
        terminated = False
        steps = 0
 
        while not terminated and steps < STEPS_PER_EPISODE:
            next_state, reward, terminated = env.step(action)
            if not terminated:
                next_action, next_direction_id = agent.select_action(next_state)
            else:
                next_action, next_direction_id = None, None
 
            agent.policy_update(state, direction_id, reward,
                                next_state, next_direction_id, terminated)
 
            state = next_state
            action = next_action
            direction_id = next_direction_id
            steps += 1
 
    # Evaluate learned value functions across the state space
    xs = np.linspace(-ENV_SIZE, ENV_SIZE, PLOT_RESOLUTION)
    L_low_arr, L_high_arr, R_low_arr, R_high_arr = [], [], [], []
    for x in xs:
        L_low, L_high, R_low, R_high, _ = agent.get_values(x)
        L_low_arr.append(L_low)
        L_high_arr.append(L_high)
        R_low_arr.append(R_low)
        R_high_arr.append(R_high)
 
    closeness = env.closeness_reward(env.get_dist(xs))

    # --- EVALUATION ---
    K_EVAL = 500
    eval_steps = []
    for _ in range(K_EVAL):
        state = env.reset()
        action, direction_id = agent.select_action(state)
        terminated = False
        steps = 0
        while not terminated and steps < STEPS_PER_EPISODE:
            next_state, reward, terminated = env.step(action)
            if not terminated:
                action, direction_id = agent.select_action(next_state)
            state = next_state
            steps += 1
        eval_steps.append(steps)
    # --- END EVALUATION ---

 
    return {
        "xs": xs,
        "L_low": np.array(L_low_arr),
        "L_high": np.array(L_high_arr),
        "R_low": np.array(R_low_arr),
        "R_high": np.array(R_high_arr),
        "closeness": closeness,
        "seed": seed,
        "eval_steps": eval_steps,
    }


def main():
    # Generate or use provided seeds
    if SEED_LIST is not None:
        seeds = SEED_LIST[:N_SEEDS]
    else:
        rng = np.random.RandomState(123456)
        seeds = rng.randint(0, 100000000, size=N_SEEDS).tolist()
 
    print(f"Training {N_SEEDS} seeds")
    print(f"Seeds: {seeds}\n")
 
    results = []
    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{N_SEEDS}] Training seed {seed} ...")
        results.append(train_single_seed(seed))
 
    xs = results[0]["xs"]
 
    # Stack all curves into (N_SEEDS, PLOT_RESOLUTION) arrays for averaging
    all_L_low  = np.stack([r["L_low"]  for r in results])
    all_L_high = np.stack([r["L_high"] for r in results])
    all_R_low  = np.stack([r["R_low"]  for r in results])
    all_R_high = np.stack([r["R_high"] for r in results])
 
    # --- START NORMALISATION (comment out this block to disable) --------
    # Normalise each seed's 4 curves jointly to [0, 1] so that every seed
    # lives on the same scale before superimposing / averaging.
    if NORMALISE_CURVES:
        for i in range(N_SEEDS):
            seed_min = min(all_L_low[i].min(), all_L_high[i].min(),
                          all_R_low[i].min(), all_R_high[i].min())
            seed_max = max(all_L_low[i].max(), all_L_high[i].max(),
                          all_R_low[i].max(), all_R_high[i].max())
            denom = seed_max - seed_min
            if denom < 1e-12:          # flat curves → avoid division by zero
                denom = 1.0
            all_L_low[i]  = (all_L_low[i]  - seed_min) / denom
            all_L_high[i] = (all_L_high[i] - seed_min) / denom
            all_R_low[i]  = (all_R_low[i]  - seed_min) / denom
            all_R_high[i] = (all_R_high[i] - seed_min) / denom
 
        # Also update the per-seed dicts so individual traces use the same scale
        for i, r in enumerate(results):
            r["L_low"]  = all_L_low[i]
            r["L_high"] = all_L_high[i]
            r["R_low"]  = all_R_low[i]
            r["R_high"] = all_R_high[i]
    # --- END NORMALISATION ----------------------------------------------
 
    avg_L_low  = all_L_low.mean(axis=0)
    avg_L_high = all_L_high.mean(axis=0)
    avg_R_low  = all_R_low.mean(axis=0)
    avg_R_high = all_R_high.mean(axis=0)
 
    # ---- Plotting 
    fig, ax = plt.subplots(figsize=(12, 7))
 
    # Individual seed curves 
    #transpaency decreases with more seeds, starts at alpha = 0.4 for 1 seed, and approaches 0.0 as N_SEEDS grows large
    for r_ind,r in enumerate(results):
        if r_ind < 400:
            alpha_ind = 0.1 * np.exp(-20 * r_ind / N_SEEDS) + 1e-8 # avoids underflow error
            ax.plot(xs, r["L_low"],  color='red',  linewidth=0.7, alpha=alpha_ind)
            ax.plot(xs, r["L_high"], color='red',  linewidth=0.7, alpha=alpha_ind, linestyle='--')
            ax.plot(xs, r["R_low"],  color='blue', linewidth=0.7, alpha=alpha_ind)
            ax.plot(xs, r["R_high"], color='blue', linewidth=0.7, alpha=alpha_ind, linestyle='--')
    
    # Bold average curves
    ax.plot(xs, avg_L_low,  color='red',  linewidth=2.5, label='Gauss Left')
    ax.plot(xs, avg_L_high, color='red',  linewidth=2.5, linestyle='--', label='Cauchy Left')
    ax.plot(xs, avg_R_low,  color='blue', linewidth=2.5, label='Gauss Right')
    ax.plot(xs, avg_R_high, color='blue', linewidth=2.5, linestyle='--', label='Cauchy Right')

    
    # Find where low/high curves cross closest to the origin
    left_intersect_x = find_nearest_intersection(xs, avg_L_low, avg_L_high, side='right')
    right_intersect_x = find_nearest_intersection(xs, avg_R_low, avg_R_high, side='left')
    print(f"Left curves (L_low vs L_high) intersect at x = {left_intersect_x}")
    print(f"Right curves (R_low vs R_high) intersect at x = {right_intersect_x}")

    # Mark intersection points on the position axis
    if left_intersect_x is not None:
        ax.plot(left_intersect_x, 0, marker='x', color='red', markersize=8,
                markeredgewidth=2.5, zorder=5,
                label=f'Left intersect: {left_intersect_x:.2f}')
    if right_intersect_x is not None:
        ax.plot(right_intersect_x, 0, marker='x', color='blue', markersize=8,
                markeredgewidth=2.5, zorder=5,
                label=f'Right intersect: {right_intersect_x:.2f}')

    ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
    ax.set_title(f"Learned Value Functions : $\sigma = {AGENT_STD[0]:.2f}$ vs $\gamma = {AGENT_STD[1]:.2f}$",
                 fontsize=13)
    ax.set_xlabel("Position")
    ax.set_ylabel("Normalised Q-Value" if NORMALISE_CURVES else "Q-Value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

 
    plt.tight_layout()
    plt.savefig(f"ts_gc_gaussian{AGENT_STD[0]}_cauchy{AGENT_STD[1]}.png", dpi=150)
    plt.show()
    print(f"\nSaved to ts_gc_gaussian{AGENT_STD[0]}_cauchy{AGENT_STD[1]}.png")


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
 
 