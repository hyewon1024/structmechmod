import collections
import os
import numpy as np
import tqdm
import torch
torch.set_default_dtype(torch.float64)
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from structmechmod import utils, rigidbody, nested, models
from structmechmod.metric_tracker import MetricTracker
from structmechmod.odesolver import odestep
import scipy.io as sio
#내가 추가한 코드 
import math 


def get_real_lagrangian_metrix_cartpole(env, state):
    batch_size, _ = state.shape  # state의 크기를 가져옵니다 (batch_size, 4)
    mass_matrices, corrioli_matrices, gravitational_terms, generalized_forces = [], [], [], []

    for t in range(batch_size):
        # 각 샘플에 대해 상태 값 추출
        x, x_dot, theta, theta_dot = state[t, 0], state[t, 1], state[t, 2], state[t, 3]
        force = -10 if theta > 0 else 10

        # Mass Matrix 계산
        mass_matrix = torch.tensor([
            [env.total_mass, env.length * env.masspole * torch.cos(theta)],
            [env.length * env.masspole * torch.cos(theta), env.masspole * (env.length ** 2) + env.polemass_length]
        ])

        # Coriolis Matrix 계산
        corrioli_matrix = torch.tensor([
            [0, -env.length * env.masspole * theta_dot * torch.sin(theta)],
            [0, 0]
        ])

        # Gravitational term 계산
        gravitational_term = torch.tensor([0, -env.masspole * env.gravity * env.length * torch.sin(theta)])

        # Generalized Force 계산
        generalized_force = torch.tensor([force, 0])

        # 결과를 리스트에 저장
        mass_matrices.append(mass_matrix)
        corrioli_matrices.append(corrioli_matrix)
        gravitational_terms.append(gravitational_term)
        generalized_forces.append(generalized_force)

    # 2차원 텐서로 반환
    return torch.stack(mass_matrices), torch.stack(corrioli_matrices), torch.stack(gravitational_terms), torch.stack(generalized_forces)

def compute_qddot_loss(model, x, u, dt, env):

    qdim = model._qdim
    q = x[:, :qdim]   # Positions
    v = x[:, qdim:]   # Velocities
    u = u.unsqueeze(1) if len(u.shape) == 1 else u

    # Compute predicted dynamics using the model
    mass_matrix_pred = model.mass_matrix(q)
    corrioli_pred = model.corriolis(q, v, mass_matrix_pred)
    gravitational_term_pred = model.gradpotential(q)
    generalized_force_pred = model.generalized_force(q, v, u)

    # Compute actual (true) dynamics (e.g., real cartpole dynamics)
    mass_matrix_true, corrioli_true, gravitational_term_true, generalized_force_true = get_real_lagrangian_metrix_cartpole(env, x)

    # Compute MSE loss components
    mse_mass_matrix = ((mass_matrix_pred - mass_matrix_true) ** 2).mean()
    mse_gravitational_term = ((gravitational_term_pred - gravitational_term_true) ** 2).mean()
    generalized_force_true= generalized_force_true.unsqueeze(1)
    mse_generalized_force = ((generalized_force_pred - generalized_force_true) ** 2).mean()

    # Combine all losses
    total_loss = mse_mass_matrix + mse_gravitational_term + mse_generalized_force

    # Debug information
    info = {
        'mse_mass_matrix': mse_mass_matrix.detach().item(),
        'mse_gravitational_term': mse_gravitational_term.detach().item(),
        'mse_generalized_force': mse_generalized_force.detach().item()
    }

    return total_loss, info


env = gym.make("CartPole-v1")  # 실제 환경 객체
x = torch.randn(10, 4, requires_grad=True)  # 임의의 상태 값
u = torch.randn(10, requires_grad=True)     # 임의의 제어 입력
dt = 0.05

# 모델 정의 (예시)
hidden_sizes = [32, 32, 32]
thetamask = torch.tensor([0, 1, 0, 0], dtype=torch.float64)

mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=hidden_sizes, bias=1.0)
delan = rigidbody.DeLan(2, 32, 3, thetamask, activation='Tanh', udim=1, bias=1.0)
smm = rigidbody.LearnedRigidBody(2, 1, thetamask=thetamask, mass_matrix=mass_matrix, hidden_sizes=hidden_sizes)

loss, info = compute_qddot_loss(smm, x, u, dt, env)
print(f"Total Loss: {loss.item()}")
print(f"Info: {info}")