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
from utils_DDQN import *
import os

class Config:
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_RATE = 0.99  
    EPISODES = 10 
    TARGET_UPDATE = 2
    BATCH_SIZE = 128
    GAMMA = 0.999

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

def take_action(state, epsilon, env , model):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(state[np.newaxis, ...])
        return np.argmax(q_values[0])

def optimize_modelDDQN(memory, config, model, model_target , n_actions, loss_function, optimizer):
    if memory.__len__() < config.BATCH_SIZE:
        return
    transitions = memory.sample(config.BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))

    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    next_state_batch = np.array(batch.next_state)
    rewad_batch = np.array(batch.reward)
    done_batch = np.array(batch.done, dtype=np.int8)

    # Calculate Q-values for the next state using the online model
    q_values_next_state_online = model(next_state_batch)

    # Use the online model to select actions for the next state
    next_actions_online = np.argmax(q_values_next_state_online, axis=-1)

    # Use the target model to calculate Q-values for the next state and selected actions
    q_values_next_state_target = model_target(next_state_batch)
    q_values_next_state_target_selected = tf.reduce_sum(
        tf.one_hot(next_actions_online, n_actions) * q_values_next_state_target,
        axis=-1
    )

    target = rewad_batch + config.GAMMA * q_values_next_state_target_selected * (1 - done_batch)

    action_mask = tf.one_hot(action_batch, n_actions)

    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(tf.multiply(q_values, action_mask), axis=-1)
        loss = loss_function(target, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #wandb.log({"loss": loss.numpy()})
    return loss

def plot_rewards (episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    #CAMBIAR PATH a carpeta doubleDQN
    plot_path = f"./rewards_plot.png"
    plt.savefig(plot_path)

    # Log the plot to WandB
    #wandb.log({"Training process of DQN": wandb.Image(plot_path)})

def take_action_test(state, model):
    q_values = model.predict(state[np.newaxis, ...])
    return np.argmax(q_values[0])