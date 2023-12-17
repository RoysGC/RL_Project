#BlackJack
import gymnasium as gym
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


def basic_policy(observation):
    score = observation[0]# Extract the first element from the observation
    return 0 if score >= 20 else 1


def update_Q(env, episode, Q, alpha, gamma):
    # Unpack episode into states, actions, and rewards
    states, actions, rewards = zip(*episode)

    # Calculate discount factors for each time step in the episode
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    
    # Iterate over the episode steps to update Q-values
    for i, state in enumerate(states):
        # Retrieve the current Q-value for the given state-action pair
        old_Q = Q[state][actions[i]]

        # Calculate the target Q-value using discounted rewards
        target_Q = sum(rewards[i:] * discounts[:-(1+i)])

        # Update the Q-value
        Q[state][actions[i]] = old_Q + alpha * (target_Q - old_Q)

    # Return the updated Q-values
    return Q

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    epsilon = eps_start
    
    returns = []
    # Loop over episodes
    for i_episode in range(1, num_episodes+1):
        # Print progress every 1000 episodes
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Decay epsilon
        epsilon = max(epsilon * eps_decay, eps_min)

        # Generate an episode by following epsilon-greedy policy
        episode = []
        state = env.reset()[0]
        while True:
            # Epsilon-greedy action selection
            if state in Q and np.random.rand() > epsilon:
                action = np.argmax(Q[state])
            else: 
                action = env.action_space.sample()
            
            # Take the selected action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            
            # Append the state, action, and reward to the episode
            episode.append((state, action, reward))
            state = next_state

            # Check if the episode is done
            if done:
                break

        # Update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)

        total_return = sum([step[2] for step in episode])
        returns.append(total_return)

        # Decay epsilon
        epsilons = [max(eps_start * eps_decay**i, eps_min) for i in range(num_episodes)]

    
    # Extract the final policy from the learned Q-values
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q, returns , epsilons

def run_agentmc(basic_policy, env):
    # Set the total number of episodes for the agent to play
    total_episodes = 100000
    
    # Initialize counters for various statistics
    total_reward = 0
    total_wins = 0
    total_natural_wins = 0
    total_losses = 0
    total_draws = 0

    # Loop over episodes
    for episode in range(total_episodes):
        # Reset the environment and obtain the initial observation
        observation = env.reset()[0]
        episode_reward = 0

        # Loop over a maximum of 100 time steps in the episode
        for t in range(100):
            # Print the current observation
            print(f"Observation: {observation}")

            # Check if the current observation is a natural win (e.g., blackjack score of 21)
            if observation[0] == 21:
                total_natural_wins += 1

            # Get the action from the basic policy for the current observation
            action = basic_policy[observation]
            print(f"Taking action: {action}")

            # Take the selected action and observe the next state, reward, and other information
            observation, reward, done, term, info = env.step(action)
            episode_reward += reward

            # Check if the episode is done
            if done:
                # Print information about the end of the game and the episode's total reward
                print(f"Game ended! Reward: {episode_reward}")

                # Accumulate total reward and update win/loss/draw statistics
                total_reward += episode_reward
                if episode_reward > 0:
                    total_wins += 1
                elif episode_reward < 0:
                    total_losses += 1
                else:
                    total_draws += 1

                # Print a message indicating whether the player won, lost, or drew
                print('You won :)\n') if episode_reward > 0 else print('You lost :(\n')
                break

    # Print overall statistics after all episodes are completed
    print(f"Total Accumulated Reward: {total_reward:.3f}")
    print(f"Percentage of Wins: {(total_wins / total_episodes * 100):.3f}%")
    print(f"Percentage of Natural Wins: {(total_natural_wins / total_episodes * 100):.3f}%")
    print(f"Percentage of Losses: {(total_losses / total_episodes * 100):.3f}%")
    print(f"Percentage of Draws: {(total_draws / total_episodes * 100):.3f}%")

#Function to plot the policy in a graphical way
def plot_policy(policy):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in policy:
            return policy[x,y,usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y,usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0,1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()