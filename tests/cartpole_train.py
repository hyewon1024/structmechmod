import gym
import numpy as np
from structmechmod.trainer import HParams, train
import tqdm
import torch
import os 
import math
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from structmechmod import utils, rigidbody, nested, models
from structmechmod.metric_tracker import MetricTracker
from structmechmod.odesolver import odestep
from IPython import display as ipythondisplay
from PIL import Image
torch.set_default_dtype(torch.float64)

# Function to get Lagrangian matrix
def get_lagrangian_metrix(model, q, v, u):
    mass_matrix = model.mass_matrix(q)  # M
    corrioli_term = model.corriolisforce(q, v)  # Cv
    gravitational_term = model.gradpotential(q)  # G
    generalized_force = model.generalized_force(q, v, u)  # F
    return mass_matrix, corrioli_term, gravitational_term, generalized_force

'''
def get_real_lagrangian_metrix_cartpole(obs):

    
    #Assume no fiction 
    mass_matrix= np.array([[env.total_mass, env.length*env.masscart*]])
'''






# Set up the CartPole environment
def generate_cartpole_data(env, num_samples):
    obs_list = []
    next_obs_list = []
    control_input = np.random.rand(num_samples, 1)
    obs = env.reset()
    for n in range(num_samples):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        obs_list.append(next_obs)
        next_obs_list.append(next_obs)
    return np.array(obs_list, dtype=np.float64), control_input, np.array(next_obs_list, dtype=np.float64)


if __name__== "__main__":

    utils.set_rng_seed(42)
    env=gym.make("CartPole-v1")
    

    #train_data_set 
    train_data=generate_cartpole_data(env, num_samples=32)
    valid_data=generate_cartpole_data(env, num_samples=32) #32

    #model set 
    thetamask= torch.tensor([0, 0, 1, 1], dtype=torch.float64)
    mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=[32, 64, 32])
    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, hidden_sizes=[32, 64, 32])

    #Hyperparameter 
    hparams= HParams(logdir="cartpole_logs", nepochs=5, lr=0.001, batch_size=32, dt=0.02, scheduler_step_size=10, patience=500, gradnorm=50.0)

    #Train the model 
    trained_parames = train(smm, train_data, valid_data, hparams)

    obs=env.reset()
    test_datasets= generate_cartpole_data(env, num_samples=32)
    score=0
    i=10
    # Create a folder to save images
    image_folder = "cartpole_images"
    metric_folder = "cartpole_metrics"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(metric_folder):
        os.makedirs(metric_folder)

    for a in range(100): #환경을 reset하면서 여러 샘플 모으기 
        test_datasets= generate_cartpole_data(env, num_samples=32)
        action=env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        print(env.state)
        screen = env.render(mode='rgb_array')
        img = Image.fromarray(screen)

        x_tests=test_datasets[0]
        u_tests= test_datasets[1]
        #Lagrangian Dynamics
        q= torch.from_numpy(x_tests[:, :2]).requires_grad_()
        v= torch.from_numpy(x_tests[:, 2:]).requires_grad_()
        u= torch.from_numpy(u_tests)

        mass_matrix, corrioli_term, gravitational_term, generalized_force = get_lagrangian_metrix(smm, q, v, u)
        obs=(torch.cat((q, v), dim=1))[i]

        mask = torch.tensor([1, -1], dtype=torch.float64) #symmetry:  theta -> -theta로 변경
        #반대 Lagrangian Dynamics
        q_= torch.from_numpy(x_tests[:, :2]).requires_grad_()
        v_= -torch.from_numpy(x_tests[:, 2:]).requires_grad_()
        u_= torch.from_numpy(u_tests)

        mass_matrix_, corrioli_term_, gravitational_term_, generalized_force_ = get_lagrangian_metrix(smm, q_, v_, u_)
        obs_=(torch.cat((q_, v_), dim=1))[i]
        
        #data save 
        mass_matrix, corrioli_term, gravitational_term, generalized_force = map(lambda x: x.detach().numpy(), [mass_matrix, corrioli_term, gravitational_term, generalized_force])
        mass_matrix_, corrioli_term_, gravitational_term_, generalized_force_ = map(lambda x: x.detach().numpy(), [mass_matrix_, corrioli_term_, gravitational_term_, generalized_force_])
        # 파일에 데이터를 한 줄씩 저장
        with open(f'{metric_folder}/my_data_{a+1}.txt', 'w') as file:
            file.write('Mass Matrix, Corrioli Term, Gravitational Term, Generalized Force\n')  # 헤더
            file.write(f'{mass_matrix[i]}\n,  {corrioli_term[i]}\n, {gravitational_term[i]}\n, {generalized_force[i]}\n')

            file.write('Symmetry Metrics\n')
            file.write(f'{mass_matrix_[i]}\n,  {corrioli_term_[i]}\n, {gravitational_term_[i]}\n, {generalized_force_[i]}\n')
            file.write(f'obs: {obs} and symmetry obs: {obs_}')

        # Save the rendered image of the environment
        screen = env.render(mode='rgb_array')
        img = Image.fromarray(screen)
        img.save(f"{image_folder}/image_{a+1}.png")

        score+=reward 
        print(f"reward: {reward}")
env.close()