import os
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
import warnings
import logging

class Config:
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_RATE = 0.99
    EPISODES = 1500
    TARGET_UPDATE = 100
    BATCH_SIZE = 128
    GAMMA = 0.999
    MAX_STEPS_PER_EPISODE = 1000
    LEARNING_RATE = 2.5e-4
    MEMORY_SIZE = 10000

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

class DuelingDQN(keras.Model):
    def __init__(self, n_actions):
        super(DuelingDQN, self).__init__()
        self.layer1 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.layer2 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.layer3 = layers.Conv2D(32, 5, strides=2, activation="relu")
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.layer4 = layers.Dense(512, activation="relu")

        # Dueling DQN specific layers
        self.state_value = layers.Dense(1)
        self.action_advantage = layers.Dense(n_actions)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.layer4(x)

        state_value = self.state_value(x)
        action_advantage = self.action_advantage(x)

        # Combine state and advantage values
        q_values = state_value + (action_advantage - tf.reduce_mean(action_advantage, axis=1, keepdims=True))
        return q_values

def take_action(state, epsilon, env, model):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(state[np.newaxis, ...])
        return np.argmax(q_values[0])

def optimize_model(memory, config, model, model_target, n_actions, loss_function , optimizer):
    if memory.__len__() < config.BATCH_SIZE:
        return
    transitions = memory.sample(config.BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))

    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    next_state_batch = np.array(batch.next_state)
    rewad_batch = np.array(batch.reward)
    done_batch = np.array(batch.done, dtype=np.int8)

    future_rewards = model_target(next_state_batch)
    target = rewad_batch + config.GAMMA * tf.reduce_max(future_rewards, axis=-1) * (1 - done_batch)

    action_mask = tf.one_hot(action_batch, n_actions)

    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(tf.multiply(q_values, action_mask), axis=-1)
        loss = loss_function(target, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #wandb.log({"loss": loss.numpy()})

def plot_rewards(episode_rewards, episode):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evolution of the rewards per episode")
    plt.legend()

    plt.savefig(f"./reward_plot_episode_{episode+1}.png")
    #wandb.log({"training process Dueling DQN": wandb.Image(plt)})
    plt.close()