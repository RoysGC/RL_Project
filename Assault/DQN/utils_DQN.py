import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple, deque
import time
from ale_py import ALEInterface
import imageio
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Configuration class for DQN parameters
class Config:
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_RATE = 0.99  
    EPISODES = 1500 
    TARGET_UPDATE = 100
    BATCH_SIZE = 128
    GAMMA = 0.999

# Replay Memory class for storing and sampling experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Model class defining the neural network architecture
class DQN(keras.Model):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.layer1 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.layer2 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.layer3 = layers.Conv2D(32, 5, strides=2, activation="relu")
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.layer4 = layers.Dense(512, activation="relu")
        self.action = layers.Dense(n_actions, activation="linear")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.layer4(x)
        return self.action(x)

# Function to select an action based on epsilon-greedy strategy
def take_action(state, epsilon, env, model):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(state[np.newaxis, ...])
        return np.argmax(q_values[0])

# Function to optimize the DQN model using experience replay
def optimize_model(memory, config, model, model_target, n_actions, loss_function, optimizer):
    # Ensure the memory has enough samples to create a batch
    if memory.__len__() < config.BATCH_SIZE:
        return
    
    # Sample a batch of transitions from the replay memory
    transitions = memory.sample(config.BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))

    # Extract components of the batch
    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    next_state_batch = np.array(batch.next_state)
    reward_batch = np.array(batch.reward)
    done_batch = np.array(batch.done, dtype=np.int8)

    # Calculate the expected future rewards using the target DQN model
    future_rewards = model_target(next_state_batch)
    target = reward_batch + config.GAMMA * tf.reduce_max(future_rewards, axis=-1) * (1 - done_batch)

    # Create a one-hot encoding for the selected actions
    action_mask = tf.one_hot(action_batch, n_actions)

    # Use gradient tape to calculate the loss and update the model
    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(tf.multiply(q_values, action_mask), axis=-1)
        loss = loss_function(target, q_action)

    # Calculate gradients and apply them using the optimizer
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Function to plot rewards per episode during training
def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # Save the plot as an image
    plot_path = f"./rewards_plot.png"
    plt.savefig(plot_path)

    # Log the plot to WandB or any other visualization tool
    # wandb.log({"Training process of DQN": wandb.Image(plot_path)})


# Function to select an action using the trained DQN for testing
def take_action_test(state, model):
    q_values = model.predict(state[np.newaxis, ...])
    return np.argmax(q_values[0])



