# Imports
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Define the number of bins for each continuous feature (increased for finer granularity)
n_bins = 20

# Define the state space discretization
state_bins = [
    np.linspace(-2.4, 2.4, n_bins),  # Cart position
    np.linspace(-2.0, 2.0, n_bins),  # Cart velocity
    np.linspace(-0.5, 0.5, n_bins),  # Pole angle
    np.linspace(-2.0, 2.0, n_bins)   # Pole angular velocity
]

# Initialize Q-table (discretized state space, actions: left or right)
q_table = np.random.uniform(low=-1, high=1, size=[n_bins] * 4 + [env.action_space.n])

# Hyperparameters
learning_rate = 0.2  # Increased learning rate for faster learning
discount_factor = 0.99  # High discount factor for long-term planning
episodes = 5000  # Increased number of episodes
max_steps = 500  # Allow more steps per episode for longer training
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.9995  # Slower epsilon decay
epsilon_min = 0.01  # Minimum epsilon for exploration

# Discretize continuous state into bins
def discretize_state(state):
    """ Convert continuous state values into discrete state indices. """
    state_discretized = []
    for i, value in enumerate(state):
        state_discretized.append(np.digitize(value, state_bins[i]) - 1)
    return tuple(state_discretized)

# Reward shaping function
def reward_shaping(state, reward):
    """
    Apply reward shaping based on the current state to guide the agent
    towards better behavior.
    """
    pole_angle = state[2]
    pole_velocity = state[3]
    
    # Penalize if the pole is too far from vertical (large angle)
    angle_penalty = -0.5 * abs(pole_angle) if abs(pole_angle) > 0.1 else 0
    
    # Penalize if the pole angular velocity is too high
    velocity_penalty = -0.1 * abs(pole_velocity) if abs(pole_velocity) > 1.0 else 0

    # Combine the original reward with the penalties
    shaped_reward = reward + angle_penalty + velocity_penalty
    return shaped_reward

# Lists to store the total rewards for each episode for plotting
episode_rewards = []

# Q-learning algorithm with reward shaping
for episode in range(episodes):
    state, _ = env.reset()  # Reset environment to the initial state
    state_discretized = discretize_state(state)  # Discretize the state
    done = False
    total_reward = 0

    for step in range(max_steps):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore (random action)
        else:
            action = np.argmax(q_table[state_discretized])  # Exploit (best action)

        # Take action and observe new state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Apply reward shaping
        shaped_reward = reward_shaping(state, reward)

        # Discretize next state
        next_state_discretized = discretize_state(next_state)

        # Update Q-value using Q-learning formula
        q_table[state_discretized][action] += learning_rate * (
            shaped_reward + discount_factor * np.max(q_table[next_state_discretized]) - q_table[state_discretized][action]
        )

        state_discretized = next_state_discretized  # Transition to next state
        total_reward += shaped_reward

        if terminated or truncated:
            done = True
            break

    # Decay epsilon for more exploitation over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Store total reward for this episode
    episode_rewards.append(total_reward)

    # Print progress every 500 episodes
    if episode % 1 == 0:
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Plot progress every 100 episodes
    if episode % 100 == 0:
        plt.plot(range(episode + 1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.pause(0.1)  # Pause to allow the plot to update

# Show the final plot
plt.show()

# Test the trained agent
state, _ = env.reset()
state_discretized = discretize_state(state)
done = False
while not done:
    action = np.argmax(q_table[state_discretized])  # Choose action based on Q-table
    state, _, terminated, truncated, _ = env.step(action)
    state_discretized = discretize_state(state)
    env.render()

print("Training complete.")