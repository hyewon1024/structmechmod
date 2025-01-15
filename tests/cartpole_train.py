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

    #train_data_set 

    train_data=generate_cartpole_data(env, num_samples=32)
    valid_data=generate_cartpole_data(env, num_samples=32)

    #model set 
    thetamask= torch.tensor([0, 1, 0, 0], dtype=torch.float64)
    mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=[32, 64, 32])
    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, hidden_sizes=[32, 64, 32])

    hparams= HParams(logdir="cartpole_logs", nepochs=5, lr=0.001, batch_size=32, dt=0.02, scheduler_step_size=10, patience=500, gradnorm=50.0)

    trained_parames = train(smm, train_data, valid_data, hparams)


    obs=env.reset()
    test_datasets= generate_cartpole_data(env, num_samples=32)

    env.render()
    action=env.action_space.sample()
    x_tests=test_datasets[0]
    u_tests= test_datasets[1]
    #Lagrangian Dynamics
    q= torch.from_numpy(x_tests[:, :2]).requires_grad_()
    v= torch.from_numpy(x_tests[:, 2:]).requires_grad_()
    u=torch.from_numpy(u_tests)

    mass_matrix= smm.mass_matrix(q)
    corrioli_term = smm.corriolisforce(q, v)
    generalized_force= smm.generalized_force(q, v, u)

    obs=(torch.cat((q, v), dim=1))[0]
    print(f"mass_matrix: {mass_matrix[0]}, corrioli_term: {corrioli_term[0]},generalized_force: {generalized_force[0]}")

    #반대 Lagrangian Dynamics
    q_= -torch.from_numpy(x_tests[:, :2]).requires_grad_()
    v_= -torch.from_numpy(x_tests[:, 2:]).requires_grad_()
    u_=torch.from_numpy(u_tests)

    mass_matrix_= smm.mass_matrix(q_)
    corrioli_term_ = smm.corriolisforce(q_, v_)
    generalized_force_= smm.generalized_force(q_, v_, u)
    obs_=(torch.cat((q_, v_), dim=1))[0]
    print(obs_)
    #print(f"obs랑 {obs}, obs_:{obs_}")

    '''
    for e in range(3):
        obs=env.reset()
        test_datasets= generate_cartpole_data(env, num_samples=32)
        for t in range(10): #관찰용
            env.render()
            action=env.action_space.sample()
            x_tests=test_datasets[0]
            u_tests= test_datasets[1]
            #Lagrangian Dynamics
            q= torch.from_numpy(x_tests[:, :2]).requires_grad_()
            v= torch.from_numpy(x_tests[:, 2:]).requires_grad_()
            u=torch.from_numpy(u_tests)

            mass_matrix= smm.mass_matrix(q)
            corrioli_term = smm.corriolisforce(q, v)
            generalized_force= smm.generalized_force(q, v, u)

            obs=(torch.cat((q, v), dim=1))[0]
            print(f"mass_matrix: {mass_matrix[0]}, corrioli_term: {corrioli_term[0]},generalized_force: {generalized_force[0]}")

            #반대 Lagrangian Dynamics
            q_= -torch.from_numpy(x_tests[:, :2]).requires_grad_()
            v_= -torch.from_numpy(x_tests[:, 2:]).requires_grad_()
            u_=torch.from_numpy(u_tests)

            mass_matrix_= smm.mass_matrix(q_)
            corrioli_term_ = smm.corriolisforce(q_, v_)
            generalized_force_= smm.generalized_force(q_, v_, u)
            obs_=(torch.cat((q_, v_), dim=1))[0]
            print(f"obs랑 {obs}, obs_:{obs_}")
            break
            #print(f"mass_matrix: {mass_matrix[0]}, corrioli_term: {corrioli_term[0]},generalized_force: {generalized_force[0]}")

            #dataset 모아서 obs 하나 고름 

            #하기 추가한 것 
            
            next_obs, reward, done, info = env.step(action)
            if done:
                print("training finished!")
                break
    '''
    env.close()