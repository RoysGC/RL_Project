
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import cv2


def plot_loses(all_rewards, all_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")
    plt.grid()

    # Log the plot to wandb
    #wandb.log({"Rewards per Episode": wandb.Image(plt)})
    plt.show()

    # Graficar la evolución de las recompensas promedio cada 100 episodios
    average_rewards = [pd.Series(all_rewards).iloc[i:i+100].mean() for i in range(0, len(all_rewards), 100)]
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, len(all_rewards), 100), average_rewards)
    plt.title("Mean Reward for 100 episodes")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.grid()

    # Log the plot to wandb
    #wandb.log({"Mean Reward for 100 Episodes": wandb.Image(plt)})
    plt.show()

    # Graficar la evolución de la pérdida a lo largo del entrenamiento
    plt.figure(figsize=(12, 6))
    plt.plot(all_losses)
    plt.title("Training Loss Evolution")
    plt.xlabel("Training")
    plt.ylabel("Loss")
    plt.grid()

    # Log the plot to wandb
    #wandb.log({"Training Loss Evolution": wandb.Image(plt)})
    plt.show()

    #wandb.finish()

def plot_test(total_rewards_all_episodes, fig, ims):
    # Graficar la suma de recompensas por episodio
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards_all_episodes)
    plt.title("Reward return for episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward return")
    plt.grid()

    #wandb.log({"Reward return for episode": wandb.Image(plt)})

    plt.show()

    # Guardar la animación
    Writer = animation.writers['pillow']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
    im_ani.save('gif_git.gif', writer=writer)



    # Log the GIF to wandb
    wandb.log({"animation": wandb.Video("gif_git.gif", fps=4, format="gif")})

    # Finish the wandb run
    wandb.finish()
