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

        #Weights for the left and right action value function, left with low/high variance and right with low/high variance
        self.L_low_weights = np.zeros(n_centers)
        self.R_low_weights = np.zeros(n_centers)
        self.L_high_weights = np.zeros(n_centers)
        self.R_high_weights = np.zeros(n_centers)
        
        # variances for Gaussians one and two
        self.sigmas = agent_std
        

    #Get Radial basis function values
    def rbf(self, x, centers, sigma):
        return np.exp(-1*(x-centers)**2 /  (2 * sigma**2))
    
    #Calculate L_low(x) and L_high(x)
    #Calculate R_low(x) and R_high(x)
    def get_values(self, agent_pos):
        
        phi = self.rbf(agent_pos, self.centers, self.sigma)

        L_low = np.dot(self.L_low_weights, phi)
        L_high = np.dot(self.L_high_weights, phi)
        R_low = np.dot(self.R_low_weights, phi)
        R_high = np.dot(self.R_high_weights, phi)

        return L_low, L_high, R_low, R_high, phi
    
    def select_action(self, agent_pos):

        #Select direction
        L_low, L_high, R_low, R_high, _ = self.get_values(agent_pos)

        #A trick to make this more numerically stable to avoid inf/inf 
        # Shift the values down so the maximum value is exactly 0
        max_q = np.max([L_low, L_high, R_low, R_high])
        L_low -= max_q
        L_high -= max_q
        R_low -= max_q
        R_high -= max_q

        #Use softmax to convert to probabilities
        sum_all = np.exp(L_low) + np.exp(L_high) + np.exp(R_low) + np.exp(R_high)
        prob_left_low = np.exp(L_low) / sum_all
        prob_left_high = np.exp(L_high) / sum_all
        prob_right_high = np.exp(R_high) / sum_all
        prob_right_low = np.exp(R_low) / sum_all

        #If left then the direction is towards the negative, if right then the direction is towards the positive
        direction_id = np.random.choice([0,1,2,3], p=[prob_left_low, prob_left_high, prob_right_low, prob_right_high])
        if direction_id in [0,1]: #Left
            direction = -1
        else: #Right
            direction = 1
        
        #Select Step length, high means higher variance step size distribution
        if direction_id in [0,2]: #Low variance
            step_size = np.random.normal(0, self.sigmas[0])
        else: #High variance
            step_size = np.random.normal(0, self.sigmas[1])

        #Change to magnitude of the step size, 
        step_size = abs(step_size)

        #combine them to get the final action, direction * step size
        action = direction * step_size

        return action, direction_id

    def policy_update(self, agent_pos, direction_id, reward, next_agent_pos, next_direction_id, terminated = False):
        """SARSA Update: Q(s,a) <-- Q(s,a) + lr * [r + gamma*Q(s',a') - Q(s,a)]"""

        #UPDATE THE DIRECTION WEIGHTS
        #Current Q-value, L or R
        L_low, L_high, R_low, R_high, phi = self.get_values(agent_pos)
        if direction_id == 0: #Left low variance
            current_q = L_low
        elif direction_id == 1: #Left high variance
            current_q = L_high
        elif direction_id == 2: #Right low variance
            current_q = R_low
        else: #Right high variance
            current_q = R_high
        
        if terminated:
            target = reward
            target_mode = reward
        else:
            #Next Q-value value, target
            L_low_next, L_high_next, R_low_next, R_high_next, _ = self.get_values(next_agent_pos)
            if next_direction_id == 0: #Left low variance
                next_q = L_low_next
            elif next_direction_id == 1: #Left high variance
                next_q = L_high_next
            elif next_direction_id == 2: #Right low variance
                next_q = R_low_next
            else: #Right high variance
                next_q = R_high_next

            #TD target
            target = reward + self.gamma * next_q

            
        #TD error
        td_error = target - current_q

        #Muptiply by e phi_curr to vectorise and scales each error by basis function value
        if direction_id == 0:
            self.L_low_weights += self.lr * td_error*phi
        elif direction_id == 1:
            self.L_high_weights += self.lr * td_error*phi
        elif direction_id == 2:
            self.R_low_weights += self.lr * td_error*phi
        else: #Right high variance
            self.R_high_weights += self.lr * td_error*phi
    


if __name__ == '__main__':
    ##Training Loop
    SIZE = 100
    HAVE_CLOSENESS_REWARD = True
    # AGENT_STD = [1.482602218505602,1.482602218505602]
    AGENT_STD = [1,5]
    env = env_1d(start_range=(-SIZE, SIZE), target=0,have_closeness_reward=HAVE_CLOSENESS_REWARD)
    agent = sarsa_agent(n_centers=200, sigma=0.80,agent_std=AGENT_STD, lr=0.05, gamma=0.95, start_range=(-SIZE,SIZE))

    episodes = 10000
    step_per_episode = 1000
    reward_history = []
    eval_rewards = []

    #Keep track of the trajectory of the agent for each episode to plot later
    N_TRAJECTORIES = 300
    tracjectory = []
    
    # Shape: (episodes, 4, RESOLUTION) where 4 = [L_low, L_high, R_low, R_high]
    RESOLUTION = 200
    surface_xs = np.linspace(-SIZE, SIZE, RESOLUTION)
    q_history = np.zeros((episodes, 4, RESOLUTION))


    #Keep track of the evolution of the mode probabilities 
    pi_history = []


    for ep in tqdm(range(episodes)):
        state = env.reset()
        action, direction_id = agent.select_action(state)
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
            #Choose the next actio and mode before the next update
            if not terminated:
                next_action, next_direction_id  = agent.select_action(next_state)
            else:
                next_action, next_direction_id = None, None
            
            #Peform policy update
            agent.policy_update(state, direction_id, reward, next_state, next_direction_id, terminated)


            state = next_state
            action = next_action
            direction_id = next_direction_id
            total_reward += reward

            steps += 1
            # global_step += 1

            
    
        if ep >= (episodes - N_TRAJECTORIES):
            tracjectory.append(new_tracjectory)
        # pi_history.append(current_pi_history)
        reward_history.append(total_reward)

          # --- 3D SURFACE: record Q-values for this episode ---
        L_snap, L_h_snap, R_snap, R_h_snap = [], [], [], []
        for x in surface_xs:
            Ll, Lh, Rl, Rh, _ = agent.get_values(x)
            L_snap.append(Ll)
            L_h_snap.append(Lh)
            R_snap.append(Rl)
            R_h_snap.append(Rh)
        q_history[ep, 0] = L_snap
        q_history[ep, 1] = L_h_snap
        q_history[ep, 2] = R_snap
        q_history[ep, 3] = R_h_snap
        # --- END 3D SURFACE RECORD ---


    # --- 4. Plotting with Moving Average ---
    plt.figure(figsize=(12, 5))

    # Plot 1: Reward History with Moving Average
    plt.subplot(1, 2, 1)

    # Calculate Moving Average (Window size = 100 episodes)
    window_size = 100
    moving_avg = np.convolve(reward_history[:1000], np.ones(window_size)/window_size, mode='valid')

    plt.plot(moving_avg, color='green', linewidth=2, label=f'Moving Avg (n={window_size})')
    plt.title("Training Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Learned Gaussians
    plt.subplot(1, 2, 2)
    xs = np.linspace(-SIZE, SIZE, 200)

    L_low_curve = []
    L_high_curve = []
    R_low_curve = []
    R_high_curve = []


    for x in xs:
        L_low, L_high, R_low, R_high, _ = agent.get_values(x)

        L_low_curve.append(L_low)
        L_high_curve.append(L_high)
        R_low_curve.append(R_low)
        R_high_curve.append(R_high)


    #Plot 3: Closeness Reward function
    if HAVE_CLOSENESS_REWARD:
        closeness_reward = env.closeness_reward(env.get_dist(xs))
        plt.plot(xs, closeness_reward, label='Closeness Reward Function', color='green', linewidth=2,zorder=3) 

    plt.plot(xs, L_low_curve, label='L_low(x) - Value of Left', color='red', linewidth=1)
    plt.plot(xs, L_high_curve, label='L_high(x) - Value of Left', color='red', linewidth=1, linestyle='--')
    plt.plot(xs, R_low_curve, label='R_low(x) - Value of Right', color='blue', linewidth=1)
    plt.plot(xs, R_high_curve, label='R_high(x) - Value of Right', color='blue', linewidth=1, linestyle='--')   
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Learned Value Functions: $\sigma_1$ = {agent.sigmas[0]:.2f} vs $\sigma_2$ = {agent.sigmas[1]:.2f}")
    plt.xlabel("Position")
    plt.ylabel("Q-Values")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    #### Plot 3: Fully trained Trajectory, we choose the last 30 trajectories to plot to demonstrate a fully trained agent
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

    #  # --- 3D SURFACE PLOT FUNCTION AND CALL ---
    # def plot_q_surface(q_history, xs, episode_count, 
    #                    rstride=50, cstride=2, alpha=0.6):
    #     """
    #     Plot a 3D surface of Q-values over (position, episode).

    #     Parameters
    #     ----------
    #     q_history  : ndarray, shape (episodes, 4, resolution)
    #     xs         : 1-D array of position values
    #     episode_count : total number of episodes
    #     rstride/cstride : step sizes for surface rendering (higher = faster)
    #     alpha      : surface transparency
    #     """
    #     episodes_axis = np.arange(episode_count)
    #     X, Y = np.meshgrid(xs, episodes_axis)  # X=position, Y=episode

    #     labels = ['L_low', 'L_high', 'R_low', 'R_high']
    #     colors = ['red', 'red', 'blue', 'blue']
    #     styles = ['solid', 'dashed', 'solid', 'dashed']

    #     fig = plt.figure(figsize=(16, 10))

    #     for i in range(4):
    #         ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    #         Z = q_history[:, i, :]  # shape (episodes, resolution)

    #         ax.plot_surface(X, Y, Z,
    #                         rstride=rstride, cstride=cstride,
    #                         color=colors[i], alpha=alpha,
    #                         edgecolor='none')

    #         ax.set_xlabel('Position')
    #         ax.set_ylabel('Episode')
    #         ax.set_zlabel('Q-Value')
    #         ax.set_title(labels[i])

    #     fig.suptitle(
    #         f"Q-Value Evolution: $\sigma_1$={AGENT_STD[0]:.2f} vs $\sigma_2$={AGENT_STD[1]:.2f}",
    #         fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig("q_surface_evolution.png", dpi=150)
    #     plt.show()
    #     print("Saved to q_surface_evolution.png")

    # # Call with rstride/cstride to keep rendering fast
    # # rstride=50 → show every 50th episode row, cstride=2 → every 2nd position
    # plot_q_surface(q_history, surface_xs, episodes, rstride=50, cstride=2)
    # # --- END 3D SURFACE PLOT ---


    print(f"The mean length of the last {N_TRAJECTORIES} trajectories is: {np.mean([len(t) for t in last_n_trajectories])}")
    print(f"The median length of the last {N_TRAJECTORIES} trajectories is: {np.median([len(t) for t in last_n_trajectories])}")

