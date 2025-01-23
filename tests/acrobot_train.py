import os
import math
import torch
import numpy as np
import gym
from PIL import Image
from structmechmod.trainer import HParams, train
from structmechmod import utils, rigidbody, models
from structmechmod.models import DelanCholeskyMMNet, ControlAffineForceNet, PotentialNet
from structmechmod.metric_tracker import MetricTracker
import scipy.io as sio

torch.set_default_dtype(torch.float64)

def generate_acrobot_data(env, num_samples, save=False, image_folder="acrobot_images"):
    """Generate dataset from the Acrobot environment."""
    obs_list, next_obs_list, action_list, img_list = [], [], [], []
    control_input = np.random.rand(num_samples, 1)

    if save:
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
        control_input[n] = action
        env.render()
        next_obs, reward, done, info = env.step(action)
        next_obs_list.append(env.state)
        if done:
            env.reset()
    print(next_obs_list[0].shape)
    return np.array(obs_list, dtype=np.float64), control_input, np.array(next_obs_list, dtype=np.float64), np.array(action_list), img_list

def get_real_lagrangian_metrix_acrobot(env, state, action):
    """Compute real Lagrangian matrices for the Acrobot environment."""
    theta1, theta2, theta1_dot, theta2_dot = state

    # Physical parameters
    m1, m2 = env.LINK_MASS_1, env.LINK_MASS_2
    l1, l2 = env.LINK_LENGTH_1, env.LINK_LENGTH_2
    lc1, lc2 = env.LINK_COM_POS_1, env.LINK_COM_POS_2
    I1 = env.LINK_MOI 
    I2 = env.LINK_MOI
    g = 9.8

    # Mass matrix
    d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + l1 * lc2 * math.cos(theta2)) + I1 + I2
    d12 = m2 * (lc1**2 + l1 * lc2 * math.cos(theta2)) + I2
    d22 = m2 * lc2**2 + I2
    mass_matrix = torch.tensor([
        [d11, d12],
        [d12, d22]
    ])

    # Coriolis matrix
    h = -m2 * l1 * lc2 * math.sin(theta2)
    corrioli_matrix = torch.tensor([
        [h * theta2_dot, h * (theta2_dot)],
        [-(0.5)*h * theta1_dot, 0]
    ])

    # Gravitational term
    g1 = (m1 * lc1 + m2 * l1) * g * math.cos(theta1) + m2 * lc2 * g * math.cos(theta1 + theta2)
    g2 = m2 * lc2 * g * math.cos(theta1 + theta2)
    gravitational_term = torch.tensor([[g1], [g2]])

    # Generalized force
    generalized_force = torch.tensor([[[0], [action]]])

    return mass_matrix, corrioli_matrix, gravitational_term, generalized_force

def get_lagrangian_metrix(model, q, v, u):
    """Compute Lagrangian matrices from the learned model."""
    mass_matrix = model.mass_matrix(q)
    corrioli_term = model.corriolis(q, v, mass_matrix)
    gravitational_term = model.gradpotential(q)
    generalized_force = model.generalized_force(q, v, u)
    return mass_matrix, corrioli_term, gravitational_term, generalized_force

def solve_real_euler_lagrange(M, Cv, G, F):
    """Solve the Euler-Lagrange equation."""
    M_reshape=M.unsqueeze(0)
    qddot = torch.linalg.solve(M_reshape, F - Cv - G.unsqueeze(0)).squeeze(2)
    #print(f'M: {M_reshape.shape}, F: {F.shape}, Cv: {Cv.shape}, g: {G.unsqueeze(0).shape}, qddot: {qddot.shape}')
    return qddot

# Training and Evaluation 
def setup_and_train_model(train_data, valid_data, hidden_sizes, thetamask, hparams):
    """Setup and train the model."""
    mass_matrix = DelanCholeskyMMNet(2, hidden_sizes=hidden_sizes, bias=10.0)
    generalized_force_network = ControlAffineForceNet(2, 1, hidden_sizes=hidden_sizes)
    potential_network=PotentialNet(2, hidden_sizes=hidden_sizes)
    delan = rigidbody.DeLan(2, 32, 3, thetamask, activation='Tanh', udim=1, bias=10.0)
    smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, potential=potential_network, hidden_sizes=hidden_sizes, generalized_force=generalized_force_network)

    for src, tgt in zip(delan._mass_matrix_network.parameters(), smm._mass_matrix.parameters()):
        tgt.data.copy_(src.data)

    for src, tgt in zip(delan._potential_network.parameters(), smm._potential.parameters()):
        tgt.data.copy_(src.data)

    trained_params = train(smm, train_data, valid_data, hparams)
    return smm, trained_params

def evaluate_model(env, smm, test_data, action_data, num_samples, output_path):
    """Evaluate the trained model and save metrics."""
    all_data = {key: [] for key in [
        'mass_matrix', 'corrioli_matrix', 'gravitational_term', 'generalized_force',
        'mass_matrix_pred', 'corrioli_matrix_pred', 'gravitational_term_pred', 'generalized_force_pred',
        'qdoot', 'qddot_pred'
    ]}

    for epi in range(num_samples):
        states = tuple(torch.tensor(test_data[0][epi, _]) for _ in range(4))
        q = torch.tensor([[states[0], states[1]]], requires_grad=True)
        v = torch.tensor([[states[2], states[3]]], requires_grad=True)
        u = torch.tensor([test_data[1][epi]])

        mass_matrix_pred, corrioli_matrix_pred, gravitational_term_pred, generalized_force_pred = get_real_lagrangian_metrix_acrobot(env, states, action_data[epi])
        mass_matrix, corrioli_matrix, gravitational_term, generalized_force = get_lagrangian_metrix(smm, q, v, u)

        qddot = smm.solve_euler_lagrange(q, v, u)

        v = v.unsqueeze(2)
        corrioli_force_pred = corrioli_matrix_pred @ v
        qddot_pred = solve_real_euler_lagrange(mass_matrix_pred, corrioli_force_pred, gravitational_term_pred, generalized_force_pred)
        print(
            f"Episode {epi}: \n"
            f"mass_matrix={mass_matrix}, mass_matrix_pred={mass_matrix_pred}\n"
            f"corrioli_matrix={corrioli_matrix}, corrioli_matrix_pred={corrioli_matrix_pred}\n"
            f"gravitational_term={gravitational_term}, gravitational_term_pred={gravitational_term_pred}\n"
            f"generalized_force={generalized_force}, generalized_force_pred={generalized_force_pred}\n"
            f"qddot={qddot}, qddot_pred={qddot_pred}"
)
        for key, value in zip([
            'mass_matrix', 'corrioli_matrix', 'gravitational_term', 'generalized_force',
            'mass_matrix_pred', 'corrioli_matrix_pred', 'gravitational_term_pred', 'generalized_force_pred',
            'qdoot', 'qddot_pred'
        ], [
            mass_matrix, corrioli_matrix, gravitational_term, generalized_force,
            mass_matrix_pred, corrioli_matrix_pred, gravitational_term_pred, generalized_force_pred,
            qddot, qddot_pred
        ]):
            all_data[key].append(value.detach().numpy())

    sio.savemat(output_path, all_data)
    print(f"Metrics saved to {output_path}")

# main
if __name__ == "__main__":
    utils.set_rng_seed(1339)
    env = gym.make("Acrobot-v1")
    env.reset()

    # Directories
    image_folder = "acrobot_images"
    metric_folder = "acrobot_metrics"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(metric_folder, exist_ok=True)

    # Data generation
    train_data = generate_acrobot_data(env, num_samples=200, save=False)[:-2]
    valid_data = generate_acrobot_data(env, num_samples=30, save=False)[:-2]

    # Model setup and training
    hidden_sizes = [32, 32, 32]
    thetamask = torch.tensor([1, 1, 0, 0], dtype=torch.float64)
    #n_epoch 나중 수정 
    hparams = HParams(None, nepochs=100, lr=1e-3, batch_size=128, dt=0.05, scheduler_step_size=40, patience=500, gradnorm=100.0)
    smm, _ = setup_and_train_model(train_data, valid_data, hidden_sizes, thetamask, hparams)

    # Evaluation
    test_num_samples = 50
    test_datasets = generate_acrobot_data(env, num_samples=test_num_samples, save=True)
    test_data, action_data = test_datasets[:-2], test_datasets[-2]
    output_path = os.path.join(metric_folder, 'metrics_data.mat')

    evaluate_model(env, smm, test_data, action_data, test_num_samples, output_path)
    env.close()