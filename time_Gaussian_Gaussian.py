import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation


#Set a seed to control randomness
# np.random.seed(259278)

class env_1d:

    def __init__(self, start_range = (-50, 50), target = 0, have_closeness_reward=True):
        super().__init__()
        self.target = target
        self.start_range = start_range
        self.agent_pos = 0
        self.size = abs(start_range[1] - start_range[0]) 
        self.have_closeness_reward = have_closeness_reward

    #reset by starting at a random position within the start range
    def reset(self, fixed_start=None, target=None):
        if fixed_start is not None:
            self.agent_pos = fixed_start
        else:
            self.agent_pos = np.random.uniform(self.start_range[0], self.start_range[1])
        if target is not None:
            self.target = target
        return self.agent_pos

    #Produce some reward as a function to the target
    def closeness_reward(self,dist):
        # adding 1e-8 to prevent an overflow error
        peak = 15
        reward = peak * np.exp(-dist)
        reward = np.where(reward > peak, 0, reward)
        return reward
    
        #Get the distance to the target
    def get_dist(self,loc):
        return abs(loc - self.target)
    
    def step(self,action):   
        #Now action is a vector which is a float 
        self.agent_pos += action

        low, high = -self.size/2, self.size/2
        # wrap position into [low, high)
        self.agent_pos = ((self.agent_pos - low) % self.size) + low

        # Reward: Positive reward based on closeness to target
        reward = 0
        terminated = False
        dist = self.get_dist(self.agent_pos)
        
        if dist < 0.25:
            reward += 100
            terminated = True
        # Using an exponential decaying reward
        if self.have_closeness_reward:
            reward += self.closeness_reward(dist)

        return self.agent_pos, reward, terminated


class sarsa_agent:
    def __init__(self, n_centers=200, sigma=1, agent_std=[1,5],lr=0.05, gamma=0.9, start_range=(-25,25)):
        super().__init__()
        self.n_centers = n_centers
        self.centers = np.linspace(start_range[0], start_range[1], n_centers)
        self.sigma = sigma
        
        #learning rate
        self.lr = lr
        self.gamma = gamma

        #Weights for the left and right action value function
        self.L_weights = np.zeros(n_centers)
        self.R_weights = np.zeros(n_centers)
        self.pi_values = np.array([0.5, 0.5])
        
        # variances for Gaussians one and two
        self.sigmas = agent_std
        

    #Get Radial basis function values
    def rbf(self, x, centers, sigma):
        return np.exp(-1*(x-centers)**2 /  (2 * sigma**2))
    
    #Calculate L(x) and R(x)
    def get_values(self, agent_pos):
        
        phi = self.rbf(agent_pos, self.centers, self.sigma)
        L = np.dot(self.L_weights, phi)
        R = np.dot(self.R_weights, phi)
        return L, R, phi

    def get_mode_probs(self):
        #Calculate the probabilities of selecting each mode using a softmax 
        max_pi = np.max(self.pi_values)
        exp_pi = np.exp(self.pi_values - max_pi) 
        probs = exp_pi / np.sum(exp_pi)
        return probs
    
    def select_action(self, agent_pos):

        #Select direction
        L, R, _ = self.get_values(agent_pos)

        # Calculate the left and right probabilities using a numerically stable softmax
        max_q = max(L, R)
        L -= max_q
        R -= max_q
        prob_left = np.exp(L) / (np.exp(L) + np.exp(R))
        prob_right = np.exp(R) / (np.exp(L) + np.exp(R))

        #If left then the direction -1, if right then the direction +1
        direction = np.random.choice([-1, 1], p=[prob_left, prob_right])
        if direction == -1:
            direction_id = 0
        else:
            direction_id = 1
        #Direction id is 0 for low variance, 1 for high
        probs = self.get_mode_probs()
        mode_id = np.random.choice([0, 1], p=probs)
        
        #Select Step length, high means higher variance step size distribution
        if mode_id == 0: #Low variance
            step_size = np.random.normal(0, self.sigmas[0])
        else: #High variance
            step_size = np.random.normal(0, self.sigmas[1])

        #Change to magnitude of the step size, 
        step_size = abs(step_size)

        #combine them to get the final action, direction * step size
        action = direction * step_size




        return action, direction_id, mode_id

    def policy_update(self, agent_pos, direction_id, mode_id, reward, next_agent_pos, next_direction_id, next_mode_id, terminated=False):
        """SARSA Update for direction weights and mode probabilities."""

        # --- Direction update ---
        L_curr, R_curr, phi = self.get_values(agent_pos)
        if direction_id == 0:
            current_q = L_curr 
        else:
            current_q = R_curr

        if terminated:
            target = reward
        else:
            L_next, R_next, _ = self.get_values(next_agent_pos)
            if next_direction_id == 0:
                next_q = L_next
            else:
                next_q = R_next
            target = reward + self.gamma * next_q

        td_error = target - current_q

        if direction_id == 0:
            self.L_weights += self.lr * td_error * phi
        else:
            self.R_weights += self.lr * td_error * phi

        # --- Mode update (state-independent) ---
        curr_q_mode = self.pi_values[mode_id]

        if terminated:
            target_mode = reward
        else:
            next_q_mode = self.pi_values[next_mode_id]
            target_mode = reward + self.gamma * next_q_mode

        td_error_mode = target_mode - curr_q_mode
        self.pi_values[mode_id] += self.lr * td_error_mode

if __name__ == '__main__':
    ##Training Loop
    SIZE = 100
    HAVE_CLOSENESS_REWARD = True
    # AGENT_STD = [1.482602218505602,1.482602218505602]
    AGENT_STD = [1,5]
    env = env_1d(start_range=(-SIZE, SIZE), target=0,have_closeness_reward=HAVE_CLOSENESS_REWARD)
    agent = sarsa_agent(n_centers=200, sigma=0.80,agent_std=AGENT_STD, lr=0.05, gamma=0.95, start_range=(-SIZE,SIZE))

    episodes = 1000
    step_per_episode = 1000
    reward_history = []
    eval_rewards = []

    #Keep track of the trajectory of the agent for each episode to plot later
    N_TRAJECTORIES = 300
    tracjectory = []
    
    # Shape: (episodes, 2, RESOLUTION) where 2 = [L, R]
    RESOLUTION = 200
    surface_xs = np.linspace(-SIZE, SIZE, RESOLUTION)
    q_history = np.zeros((episodes, 2, RESOLUTION))


    #Keep track of the evolution of the mode probabilities 
    pi_history = []


    for ep in tqdm(range(episodes)):
        state = env.reset()
        action, direction_id, mode_id = agent.select_action(state)
        total_reward = 0

        terminated = False
        steps = 0

        if ep >= (episodes - N_TRAJECTORIES):
            #Trajectory for this specific episode
            new_tracjectory = [state]

    
        while not terminated and steps < step_per_episode:
            next_state, reward, terminated = env.step(action)
            
            if ep >= (episodes - N_TRAJECTORIES):
                new_tracjectory.append(env.agent_pos)
            #Choose the next action and mode before the next update
            if not terminated:
                next_action, next_direction_id, next_mode_id = agent.select_action(next_state)
            else:
                next_action, next_direction_id, next_mode_id = None, None, None
            
            #Peform policy update
            agent.policy_update(state, direction_id, mode_id, reward, next_state, next_direction_id, next_mode_id, terminated)


            state = next_state
            action = next_action
            direction_id = next_direction_id
            mode_id = next_mode_id
            total_reward += reward

            steps += 1
            # global_step += 1

            
    
        if ep >= (episodes - N_TRAJECTORIES):
            tracjectory.append(new_tracjectory)
        reward_history.append(total_reward)
        pi_history.append(agent.get_mode_probs().copy())

          # --- 3D SURFACE: record Q-values for this episode ---
        L_snap, R_snap = [], []
        for x in surface_xs:
            Ll, Rl, _ = agent.get_values(x)
            L_snap.append(Ll)
            R_snap.append(Rl)
        q_history[ep, 0] = L_snap
        q_history[ep, 1] = R_snap
        # --- END 3D SURFACE RECORD ---


    # --- 4. Plotting with Moving Average ---
    plt.figure(figsize=(12, 5))

    # Plot 1: Reward History with Moving Average
    plt.subplot(1, 2, 1)

    # Calculate Moving Average (Window size = 100 episodes)
    window_size = 100
    moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')

    plt.plot(moving_avg, color='green', linewidth=2, label=f'Moving Avg (n={window_size})')
    plt.title("Training Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Learned Value Functions
    plt.subplot(1, 2, 2)
    xs = np.linspace(-SIZE, SIZE, 200)

    L_curve = []
    R_curve = []

    for x in xs:
        L, R, _ = agent.get_values(x)
        L_curve.append(L)
        R_curve.append(R)

    #Closeness Reward function
    if HAVE_CLOSENESS_REWARD:
        closeness_reward = env.closeness_reward(env.get_dist(xs))
        plt.plot(xs, closeness_reward, label='Closeness Reward Function', color='green', linewidth=2,zorder=3) 

    plt.plot(xs, L_curve, label='L(x) - Value of Left', color='red', linewidth=2)
    plt.plot(xs, R_curve, label='R(x) - Value of Right', color='blue', linewidth=2)
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Learned Value Functions")
    plt.xlabel("Position")
    plt.ylabel("Q-Values")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    #### Plot 3: Fully trained Trajectory, 
    plt.figure(figsize=(8, 5))

    last_n_trajectories = tracjectory
    linewidth_val = 1.0 
    for i, t in enumerate(last_n_trajectories):
        plt.plot(t, color='blue', linewidth=linewidth_val, alpha=0.75)

    fully_trained_trajectory = tracjectory[-1]  

    # Formatting the axes
    plt.title(f"Overlay of Last {N_TRAJECTORIES} Trajectories")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.ylim(-SIZE, SIZE)

    plt.axhline(y=env.target, color='black', linestyle='--', alpha=0.5)
    plt.plot([], [], color='purple', alpha=0.6, label=f'Target: {env.target}') 
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 4: Mode Probability Evolution
    plt.figure(figsize=(12, 5))
    p0_hist = [p[0] for p in pi_history]
    p1_hist = [p[1] for p in pi_history]
    plt.plot(p0_hist, color='red', linewidth=1, label=f"$\sigma_1$={agent.sigmas[0]}")
    plt.plot(p1_hist, color='blue', linewidth=1, label=f"$\sigma_2$={agent.sigmas[1]}")
    plt.title("Weight Probabilities over Training")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



    #  # --- 3D SURFACE PLOT FUNCTION AND CALL ---
    # def plot_q_surface(q_history, xs, episode_count, 
    #                    rstride=50, cstride=2, alpha=0.6):
    #     episodes_axis = np.arange(episode_count)
    #     X, Y = np.meshgrid(xs, episodes_axis)
    #
    #     labels = ['L', 'R']
    #     colors = ['red', 'blue']
    #
    #     fig = plt.figure(figsize=(14, 6))
    #
    #     for i in range(2):
    #         ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    #         Z = q_history[:, i, :]
    #         ax.plot_surface(X, Y, Z,
    #                         rstride=rstride, cstride=cstride,
    #                         color=colors[i], alpha=alpha,
    #                         edgecolor='none')
    #         ax.set_xlabel('Position')
    #         ax.set_ylabel('Episode')
    #         ax.set_zlabel('Q-Value')
    #         ax.set_title(labels[i])
    #
    #     fig.suptitle(
    #         f"Q-Value Evolution: σ={AGENT_STD[0]:.2f} vs σ={AGENT_STD[1]:.2f}",
    #         fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig("q_surface_evolution.png", dpi=150)
    #     plt.show()
    #
    # plot_q_surface(q_history, surface_xs, episodes, rstride=50, cstride=2)
    # # --- END 3D SURFACE PLOT ---


    print(f"The mean length of the last {N_TRAJECTORIES} trajectories is: {np.mean([len(t) for t in last_n_trajectories])}")
    print(f"The median length of the last {N_TRAJECTORIES} trajectories is: {np.median([len(t) for t in last_n_trajectories])}")