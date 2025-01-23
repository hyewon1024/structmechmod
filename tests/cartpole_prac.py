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

def rad_to_deg(data):
    radian_value = 0.418  

    # data[0]에 대해서 변환
    angle_array_0 = data[0][:, [2, 3]]  # angle, angle_velocity 추출
    angle_array_0 = np.degrees(angle_array_0 / radian_value)  # 라디안을 각도로 변환

    updated_first_element = data[0].copy()
    updated_first_element[:, [2, 3]] = angle_array_0  # 변환된 각도 값으로 갱신

    # data[2]에 대해서 변환
    angle_array_1 = data[2][:, [2, 3]]  # angle, angle_velocity 추출
    angle_array_1 = np.degrees(angle_array_1 / radian_value)  # 라디안을 각도로 변환

    updated_third_element = data[2].copy()
    updated_third_element[:, [2, 3]] = angle_array_1  # 변환된 각도 값으로 갱신

    updated_train_data = (updated_first_element, data[1], updated_third_element)
    return updated_train_data

def generate_cartpole_data(env, num_samples):
    obs_list = []
    next_obs_list = []
    img_list=[]
    control_input = np.random.rand(num_samples, 1)
    obs = env.reset()
    for n in range(num_samples):
        #episode img로 저장
        screen = env.render(mode='rgb_array')
        img = Image.fromarray(screen)
        img_list.append(img)
        #
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        obs_list.append(next_obs)
        next_obs_list.append(next_obs)
    return np.array(obs_list, dtype=np.float64), control_input, np.array(next_obs_list, dtype=np.float64), img_list

#Real Lagrangian dynamics 
def get_real_lagrangian_metrix_cartpole(env, state):

    #Assume no fiction 
    x, x_dot, theta, theta_dot = state
    mass_matrix= np.array([[env.total_mass, env.length*env.masspole*math.cos(theta)],
                            [env.length*env.masspole*math.cos(theta), env.masspole*(env.length**2)+env.polemass_length]]) #M
    corrioli_force=np.array([[0, -env.length*env.masspole*theta_dot*math.sin(theta)],[0, 0]]) #C
    gravitational_term= [[0],[-env.masspole*env.gravity*env.length*math.sin(theta)]] #G
    generalized_force= [[10],[0]] #action이 1이면 env.force_mag=+10
    #print(f'mass_matrix: {mass_matrix}, corrioli_force: {corrioli_force}, gravitational_term: {gravitational_term}, generalized_force: {generalized_force}')
    return mass_matrix, corrioli_force, gravitational_term, generalized_force

# Function to get Lagrangian matrix
def get_lagrangian_metrix(model, q, v, u):
    mass_matrix = model.mass_matrix(q)  # M
    corrioli_term = model.corriolisforce(q, v)  # Cv
    gravitational_term = model.gradpotential(q)  # G
    generalized_force = model.generalized_force(q, v, u)  # F
    return mass_matrix, corrioli_term, gravitational_term, generalized_force

if __name__== "__main__":

    utils.set_rng_seed(42)
    env=gym.make("CartPole-v1")
    obs = env.reset()

    train_data= generate_cartpole_data(env, num_samples=200)[:-1]
    train_data = rad_to_deg(train_data) #red to deg, theta, theta_dot이 index 2, 3에 존재
    obs_array, next_obs_array= train_data[0], train_data[2]
    obs_array[:, [1, 2]]=obs_array[:, [2, 1]]
    next_obs_array[:, [1, 2]]=next_obs_array[:, [2, 1]]
    train_data = (obs_array,) + train_data[1:2] +(next_obs_array,) + train_data[3:] #tuple update
    
    valid_data= generate_cartpole_data(env, num_samples=100)[:-1]
    valid_data = rad_to_deg(valid_data)
    obs_array_, next_obs_array_= valid_data[0], valid_data[2]
    obs_array_[:, [1, 2]]=obs_array_[:, [2, 1]]
    next_obs_array_[:, [1, 2]]=next_obs_array_[:, [2, 1]]
    valid_data = (obs_array_,) + valid_data[1:2] +(next_obs_array_,) + valid_data[3:] #tuple update

    #model set 
    thetamask= torch.tensor([0, 1, 0, 0], dtype=torch.float64) #angle : 1, x: 0
    #mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=[32, 64, 32])
    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=None, hidden_sizes=[32, 32, 32])

    #Hyperparameter 
    hparams= HParams(logdir="cartpole_logs", nepochs=5, lr=0.001, batch_size=20, dt=0.01, scheduler_step_size=2, patience=50, gradnorm=50.0)
    test_num_samples=20

    #Train the model 
    trained_parames = train(smm, train_data, valid_data, hparams)
    test_datasets= generate_cartpole_data(env, num_samples=test_num_samples)[:-1]
    test_datasets = rad_to_deg(test_datasets)
    lagrangian_metrix_list=[]

    for n in range(test_num_samples):
        states = tuple(torch.tensor(test_datasets[0][n, _]) for _ in range(4)) #state tuple, num_samples
        mass_matrix_pred, corrioli_force_pred, gravitational_term_pred, generalized_force_pred= get_real_lagrangian_metrix_cartpole(env, states)
        q =torch.tensor([[states[0], states[2]]], requires_grad=True)
        v =torch.tensor([[states[1], states[3]]], requires_grad=True)
        u= torch.tensor([test_datasets[1][n]])

        mass_matrix, corrioli_force, gravitational_term, generalized_force=get_lagrangian_metrix(smm, q, v, u)
        print(f'mass_matrix: {mass_matrix}, corrioli_force: {corrioli_force}, gravitational_term: {gravitational_term}, generalized_force: {generalized_force},      mass_matrix_pred: {mass_matrix_pred}, corrioli_force_pred: {corrioli_force_pred}, gravitational_term_pred: {gravitational_term_pred}, generalized_force_pred: {generalized_force_pred}')


    '''
    #Lagrangian Dynamics
    q= torch.from_numpy(x_tests[:, [0,2]]).requires_grad_() #x, theta 
    v= torch.from_numpy(x_tests[:, [1,3]]).requires_grad_() #x_dot, theta_dot
    u= torch.from_numpy(u_tests)

    
    #get_real_lagrangian_metrix_cartpole()
    '''