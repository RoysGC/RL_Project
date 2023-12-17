import os
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple, deque
from ale_py import ALEInterface
import wandb
import imageio
import matplotlib.pyplot as plt
import warnings
import logging


class Config:
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_RATE = 0.99
    EPISODES = 1500  
    BATCH_SIZE = 128
    GAMMA = 0.999
    MAX_STEPS_PER_EPISODE = 1000
    LEARNING_RATE = 1e-4 
    MEMORY_SIZE = 10000

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



class Actor(keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        #self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu", kernel_initializer='he_normal')
        #self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation=None)  # Remove activation here
        self.batch_norm1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))
        self.d2 = layers.Dense(n_actions, activation="softmax")  # Output layer for action probabilities

    def call(self, inputs):
        x = self.conv1(inputs)
        #x = self.conv2(x)
        x = self.conv2(x)
        x = tf.nn.relu(self.batch_norm1(x))
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(512, activation="relu")
        self.d2 = layers.Dense(1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
def take_action(state, epsilon, env, actor_model, n_actions):

    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        action_probabilities = actor_model.predict(state[np.newaxis, ...])
        print("Action probabilities:", action_probabilities)

        if np.isnan(action_probabilities).any():
            print("NaN detected in action probabilities")
            return env.action_space.sample() 
        return np.random.choice(n_actions, p=np.squeeze(action_probabilities))

def optimize_model(memory, config, critic_model , critic_optimizer, actor_model, actor_optimizer, n_actions):
    if len(memory) < config.BATCH_SIZE:
        return  # Exit the function if not enough samples

    # Sample a batch of transitions from the replay memory
    transitions = memory.sample(config.BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))

    # Convert the batches into numpy arrays for processing
    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    reward_batch = np.array(batch.reward)
    next_state_batch = np.array(batch.next_state)
    done_batch = np.array(batch.done)

    # Critic Update
    with tf.GradientTape() as tape:
        # Get the values from the Critic model
        values = critic_model(state_batch)
        # Create a dummy target for simplicity
        dummy_target = tf.random.uniform(shape=values.shape)
        # Compute a simple mean squared error

        values_squeezed = tf.squeeze(values)
        if len(values_squeezed.shape) > 1:
            raise ValueError("Critic model's output is not a 1D array")

        critic_loss = tf.math.reduce_mean(tf.math.square(dummy_target - values))

    critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_grads, _ = tf.clip_by_global_norm(critic_grads, 1.0)  # Gradient clipping
    critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

    # Debugging: Print gradients and corresponding variables
    for grad, var in zip(critic_grads, critic_model.trainable_variables):
        if grad is None:
            print(f"Gradient is None for variable {var.name}")

    # Filter out None gradients
    critic_grads_and_vars = [(grad, var) for grad, var in zip(critic_grads, critic_model.trainable_variables) if grad is not None]

    # Apply gradients if there are valid ones
    if critic_grads_and_vars:
        critic_optimizer.apply_gradients(critic_grads_and_vars)
    else:
        print("No valid gradients to apply.")

    # Actor Update
    with tf.GradientTape() as tape:
        # Predict the action probabilities for the current state
        #print("Input shape to the model:", state_batch.shape)
        action_probs = actor_model(state_batch)
        # Create a one-hot encoded mask for the taken actions
        action_mask = tf.one_hot(action_batch, n_actions)
        # Select the probabilities for the actions that were actually taken
        selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)

        # Adjust dummy target values shape to match the values
        dummy_target_values = np.zeros_like(values_squeezed.numpy())
        advantage = dummy_target_values - values_squeezed

        epsilon = 1e-8
        actor_loss = -tf.math.reduce_mean(tf.math.log(selected_action_probs + epsilon) * advantage)
        #actor_loss = -tf.math.reduce_mean(tf.math.log(selected_action_probs) * advantage)

    actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)  # Gradient clipping
    actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))
    for grad in actor_grads:
        if tf.reduce_any(tf.math.is_inf(grad)).numpy() or tf.reduce_any(tf.math.is_nan(grad)).numpy():
            print("Inf or NaN detected in actor gradients")

def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plot_path = "./rewards_plot.png"
    plt.savefig(plot_path)

    # Log the plot to WandB
    #wandb.log({"Training process of Actor-Critic": wandb.Image(plot_path)})

def take_action_test(state, actor_model, n_actions):
    action_probabilities = actor_model.predict(state[np.newaxis, ...])
    return np.random.choice(n_actions, p=np.squeeze(action_probabilities))