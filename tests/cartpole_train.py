import gym
import numpy as np
import torch 
from structmechmod.trainer import HParams, train
import collections
import os
import numpy as np
import tqdm
import torch
torch.set_default_dtype(torch.float64)

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from structmechmod import utils, rigidbody, nested, models
from structmechmod.metric_tracker import MetricTracker
from structmechmod.odesolver import odestep

#set up the cartpole environment 
def generate_cartpole_data(env, num_samples):
    obs_list=[]
    next_obs_list=[]
    #initialize control input u 
    control_input = np.random.rand(32, 1)

    obs=env.reset()
    for n in range(num_samples):
        action=env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        obs_list.append(obs)
        next_obs_list.append(next_obs)
  
    return(
        np.array(obs_list, dtype=np.float64),
        control_input, 
        np.array(next_obs_list, dtype=np.float64),
    )


if __name__== "__main__":


    env=gym.make("CartPole-v1")

    train_data=generate_cartpole_data(env, num_samples=32)
    valid_data=generate_cartpole_data(env, num_samples=32)

    

    thetamask= torch.tensor([0, 1, 0, 0], dtype=torch.float64)
    mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=[32, 64, 32])

    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, hidden_sizes=[32, 64, 32])

    hparams= HParams(logdir="cartpole_logs", nepochs=5, lr=0.001, batch_size=32, dt=0.02, scheduler_step_size=10, patience=500, gradnorm=50.0)

    trained_parames = train(smm, train_data, valid_data, hparams)
    for e in range(3):
        obs=env.reset()
        for t in range(100):
            env.render()
            action=env.action_space.sample()
            print(obs)

            next_obs, reward, done, info = env.step(action)
            if done:
                break
    env.close()