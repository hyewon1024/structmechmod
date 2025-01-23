import os
import math
import torch
import numpy as np
import gym
from PIL import Image
from structmechmod.trainer import HParams, train
from structmechmod import utils, rigidbody, models
from structmechmod.models import DelanCholeskyMMNet, ControlAffineForceNet
from structmechmod.metric_tracker import MetricTracker
import scipy.io as sio
#연습용
from generate_rand_data import generate_rand_data

def generate_cartpole_data(env, num_samples, save=False, image_folder="cartpole_images"):
    """Generate dataset from the CartPole environment."""
    x_threshold = 2.4
    obs_list, next_obs_list, action_list, img_list = [], [], [], []
def generate_cartpole_data(env, num_samples, save):
    x_threshold = 2.4
    obs_list, next_obs_list, action_list, img_list = [], [], [], []
    control_input = np.random.rand(num_samples, 1)
    obs = env.reset()

    if save: 
        image_folder = "cartpole_images"
        os.makedirs(image_folder, exist_ok=True)

    for n in range(num_samples):
        screen = env.render(mode='rgb_array')
        img = Image.fromarray(screen)

        if save:
            img.save(f"{image_folder}/image_{n + 1}.png")

        img_list.append(img)
        obs_list.append(env.state)
        action = env.action_space.sample()
        action_list.append(action)
        control_input[n] *= -1 if action == 1 else 1
        next_obs, reward, done, info = env.step(action)
        next_obs_list.append(next_obs)

        if env.state[0] < -x_threshold or env.state[0] > x_threshold:
            env.reset()

    return np.array(obs_list, dtype=np.float64), control_input, np.array(next_obs_list, dtype=np.float64), np.array(action_list), img_list

def get_real_lagrangian_metrix_cartpole(state):

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    batch_size, _ = state.shape 
    mass_matrices, corrioli_matrices, gravitational_terms, generalized_forces = [], [], [], []

    for t in range(batch_size):
        # 각 샘플에 대해 상태 값 추출
        x, x_dot, theta, theta_dot = state[t, 0], state[t, 1], state[t, 2], state[t, 3]
        force = -10 if theta > 0 else 10

        # Mass Matrix 계산
        mass_matrix = torch.tensor([
            [total_mass, length * masspole * torch.cos(theta)],
            [length * masspole * torch.cos(theta), masspole * (length ** 2) + polemass_length]
        ])

        # Coriolis Matrix 계산
        corrioli_matrix = torch.tensor([
            [0, -length * masspole * theta_dot * torch.sin(theta)],
            [0, 0]
        ])

        # Gravitational term 계산
        gravitational_term = torch.tensor([0, -masspole * gravity * length * torch.sin(theta)])

        # Generalized Force 계산
        generalized_force = torch.tensor([force, 0])

        # 결과를 리스트에 저장
        mass_matrices.append(mass_matrix)
        corrioli_matrices.append(corrioli_matrix)
        gravitational_terms.append(gravitational_term)
        generalized_forces.append(generalized_force)

    # 2차원 텐서로 반환
    return torch.stack(mass_matrices), torch.stack(corrioli_matrices), torch.stack(gravitational_terms), torch.stack(generalized_forces)



# Function to get Lagrangian matrix
def get_lagrangian_metrix(model, q, v, u):
    mass_matrix = model.mass_matrix(q)
    corrioli_term = model.corriolis(q, v, mass_matrix)
    gravitational_term = model.gradpotential(q)
    generalized_force = model.generalized_force(q, v, u)
    return mass_matrix, corrioli_term, gravitational_term, generalized_force

def solve_real_euler_lagrange(M, Cv, G, F):
    qddot = torch.linalg.solve(M, F - Cv - G.unsqueeze(2)).squeeze(2)
    return qddot

if __name__ == "__main__":
    utils.set_rng_seed(42)
    env = gym.make("CartPole-v1")
    ntrajs = 50
    traj_len = 2
    dt = 0.05
    maxu = 100.0 #general force 
    stddev = 30.0
    
    metric_folder = "cartpole_metrics"
    os.makedirs(metric_folder, exist_ok=True)
    
    train_data = generate_rand_data(env, ntrajs, traj_len, dt, maxu=maxu, stddev=stddev)
    obs_array, next_obs_array = train_data[0], train_data[2]
    obs_array[:, [1, 2]], next_obs_array[:, [1, 2]] = obs_array[:, [2, 1]], next_obs_array[:, [2, 1]]
    train_data = (obs_array,) + train_data[1:2] + (next_obs_array,)

    valid_data = generate_rand_data(env, ntrajs, traj_len, dt, maxu=maxu, stddev=stddev)
    obs_array_, next_obs_array_ = valid_data[0], valid_data[2]
    obs_array_[:, [1, 2]], next_obs_array_[:, [1, 2]] = obs_array_[:, [2, 1]], next_obs_array_[:, [2, 1]]
    valid_data = (obs_array_,) + valid_data[1:2] + (next_obs_array_,) 

    hidden_sizes = [32, 32, 32]
    thetamask = torch.tensor([0, 1, 0, 0], dtype=torch.float64)

    mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=hidden_sizes, bias=1.0)
    generalized_force_network = ControlAffineForceNet(2, 1, hidden_sizes=hidden_sizes)
    delan = rigidbody.DeLan(2, 32, 3, thetamask, activation='Tanh', udim=1, bias=1.0)
    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, hidden_sizes=hidden_sizes, generalized_force=generalized_force_network)

    for src, tgt in zip(delan._mass_matrix_network.parameters(), smm._mass_matrix.parameters()):
        tgt.data.copy_(src.data)

    for src, tgt in zip(delan._potential_network.parameters(), smm._potential.parameters()):
        tgt.data.copy_(src.data)

    hparams = HParams(None, nepochs=10, lr=1e-3, batch_size=64, dt=0.05, scheduler_step_size=100, patience=500, gradnorm=100.0)
    trained_params = train(smm, train_data, valid_data, hparams, loss_log_path="train_loss_log.mat")
    
    #test data 검증
    test_datasets = generate_rand_data(env, ntrajs=50, traj_len=3, dt=0.05, maxu=maxu, stddev=stddev)
    test_data, action_data = test_datasets[:-1], test_datasets[-1]

    x= test_data[0]
    x[:, [1, 2]] = x[:, [2, 1]]
    x= torch.tensor(x, requires_grad=True)
    q= x[:, :2]
    v= x[:, 2:]
    u = torch.tensor(test_data[1], requires_grad=True)
    mass_matrix_pred, corrioli_matrix_pred, gravitational_term_pred, generalized_force_pred= get_lagrangian_metrix(smm, q, v, u)
    qddot=solve_real_euler_lagrange(mass_matrix_pred, corrioli_matrix_pred, gravitational_term_pred, generalized_force_pred)
    mass_matrix_true, corrioli_true, gravitational_term_true, generalized_force_true = get_real_lagrangian_metrix_cartpole(x)
    v= v.unsqueeze(2)
    corrioli_force_true = corrioli_true @ v
    generalized_force_true= generalized_force_true.unsqueeze(2)
    qddot_true = solve_real_euler_lagrange(mass_matrix_true, corrioli_force_true, gravitational_term_true, generalized_force_true)

    all_data = {key: [] for key in [
        'mass_matrix', 'corrioli_matrix', 'gravitational_term', 'generalized_force',
        'mass_matrix_pred', 'corrioli_matrix_pred', 'gravitational_term_pred', 'generalized_force_pred',
        'qddot', 'qddot_true'
    ]}

    for key, value in zip([
        'mass_matrix', 'corrioli_matrix', 'gravitational_term', 'generalized_force',
        'mass_matrix_pred', 'corrioli_matrix_pred', 'gravitational_term_pred', 'generalized_force_pred', 'qddot', 'qddot_true'
    ], [
        mass_matrix_pred, corrioli_matrix_pred, gravitational_term_pred, generalized_force_pred,
        mass_matrix_true, corrioli_true, gravitational_term_true, generalized_force_true, qddot, qddot_true
    ]):
        all_data[key].append(value.detach().numpy())
    output_path = os.path.join(metric_folder, 'metrics_data.mat')
    sio.savemat(output_path, all_data)
    print(f"Metrics saved to {output_path}")
    env.close()
