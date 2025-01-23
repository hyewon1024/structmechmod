#
# File: trainer.py
#
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
# Real Lagrangian dynamics
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
        force = 10 if theta > math.pi/ 2 else (-10 if theta > 0 and theta < math.pi / 2 else (10 if theta < 0 and theta > -math.pi / 2 else -10))


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


def get_lagrangian_metrix(model, q, v, u):
    with torch.enable_grad():
        with utils.temp_require_grad((q, v)):
            mass_matrix = model.mass_matrix(q)
            corrioli_term = model.corriolis(q, v, mass_matrix)
            gravitational_term = model.gradpotential(q)
            generalized_force = model.generalized_force(q, v, u)
    return mass_matrix, corrioli_term, gravitational_term, generalized_force

def solve_real_euler_lagrange(M, Cv, G, F):
    qddot = torch.linalg.solve(M, F - Cv - G.unsqueeze(2)).squeeze(2)
    return qddot

def compute_qddot_loss(model, x, u, xp, dt):

    qdim = model._qdim
    q = x[:, :qdim].requires_grad_(True)  # Positions
    v = x[:, qdim:].requires_grad_(True)  # Velocities
    u = u.unsqueeze(1).requires_grad_(True) if len(u.shape) == 1 else u.requires_grad_(True)

    # Compute predicted dynamics using the model
    with torch.enable_grad():
        with utils.temp_require_grad((q, v)):
            qddot_pred = model.solve_euler_lagrange(q, v, u)

    # Compute actual (true) dynamics (e.g., real cartpole dynamics)
    mass_matrix_true, corrioli_true, gravitational_term_true, generalized_force_true = get_real_lagrangian_metrix_cartpole(x)
    v= v.unsqueeze(2)
    corrioli_force_true = corrioli_true @ v
    generalized_force_true= generalized_force_true.unsqueeze(2)

    qddot_true = solve_real_euler_lagrange(mass_matrix_true, corrioli_force_true, gravitational_term_true, generalized_force_true)
    # Compute MSE loss components
    mse_qddot = ((qddot_pred - qddot_true) ** 2).mean()

    # Debug information
    info = {
        'mse_qddot': mse_qddot.detach().item()  
    }

    return mse_qddot, info
##
def compute_loss(model, x, u, xp, dt):
    qdim = model._qdim
    masks = [model.thetamask[i*qdim:(i+1)*qdim] for i in range(2)]
    q = x[:, :qdim]
    v = x[:, qdim:]
    qp_hat, vp_hat = odestep(model, torch.tensor(0.), dt, (q, v), u=u,
                             method='rk4',
                             transforms=[lambda x: utils.wrap_to_pi(x, mask=masks[i]) for i in range(2)])
    xp_hat = torch.cat((qp_hat, vp_hat), dim=1)
    diff  = (utils.diffangles2(xp, xp_hat, mask=model.thetamask)**2)
    l2_loss = diff.sum(-1).mean()
    if torch.isnan(l2_loss):
        import ipdb; ipdb.set_trace()
    qloss = diff[:, :qdim].sum(-1)
    vloss = diff[:, qdim:].sum(-1)
    info = {}
    info['qloss'] = qloss.mean().detach().item()
    info['vloss'] = vloss.mean().detach().item()
    return l2_loss, info


def train_epoch(model, opt, train_data, batch_size, dt, grad_norm):
    train_generator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    loss_ls = []
    loss_infos = []
    grad_norm_ls = []
    for (x, u, xp) in tqdm.tqdm(train_generator, desc='train', leave=True, position=2):
        opt.zero_grad()
        loss, loss_info = compute_loss(model, x, u, xp, dt) 
        print(f'loss : {loss}, loss_info: {loss_info}')
        #추가해줌
        loss.backward()
        gnorm = clip_grad_norm_(model.parameters(), grad_norm)
        opt.step()
        
        loss_ls.append(loss.detach().numpy())
        loss_infos.append(loss_info)
        grad_norm_ls.append(gnorm)

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad mean: {param.grad.mean().item()}, std: {param.grad.std().item()}")
            else:
                print(f"{name} has no gradient.")
    metrics = {}
    metrics['loss/mean']= np.mean(loss_ls)
    metrics['loss/std'] = np.std(loss_ls)
    metrics['loss/mean_log10'] = np.log10(np.mean(loss_ls))

    metrics['gnorm/mean'] = np.mean(grad_norm_ls)
    metrics['gnorm/std'] = np.std(grad_norm_ls)
    metrics['gnorm/max'] = np.max(grad_norm_ls)
    metrics['gnorm/min'] = np.min(grad_norm_ls)
    metrics['gnorm/mean_log10'] = np.log10(np.mean(grad_norm_ls))

    loss_infos = nested.zip(*loss_infos)
    for k, v in loss_infos.items():
        metrics[f'{k}/mean'] = np.mean(v)
        metrics[f'{k}/std'] = np.std(v)
        metrics[f'{k}/mean_log10'] = np.log10(np.mean(v))

    return metrics

def validate(model, valid_data, batch_size, dt):
    valid_generator = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    val_losses = []
    val_loss_infos = []
    for (x, u, xp) in tqdm.tqdm(valid_generator, desc='valid', leave=True, position=3):
        with torch.no_grad():
            loss, loss_info = compute_loss(model, x, u, xp, dt) 

        val_losses.append(loss.detach().item())
        val_loss_infos.append(loss_info)

    metrics = {}
    metrics['loss/mean'] = np.mean(val_losses)
    metrics['loss/std'] = np.std(val_losses)
    metrics['loss/mean_log10'] = np.log10(np.mean(val_losses))

    val_loss_infos = nested.zip(*val_loss_infos)
    for k, v in val_loss_infos.items():
        metrics[f'{k}/mean'] = np.mean(v)
        metrics[f'{k}/std'] = np.std(v)
        metrics[f'{k}/mean_log10'] = np.log10(np.mean(v))

    return metrics

def save_checkpoint(model, epoch, logdir, name):
    model_path = os.path.join(logdir, name)
    model_state = model.state_dict()
    torch.save(model_state, model_path)


def train(model, train_data, validation_data, hparams, loss_log_path=None):
    writer = None
    train_loss={'epoch':[], 'train_loss': []}
    if hparams.logdir is not None:
        os.makedirs(hparams.logdir, exist_ok=True)
        os.makedirs(os.path.join(hparams.logdir, 'logs'), exist_ok=True)
        writer = SummaryWriter(os.path.join(hparams.logdir, 'logs'))
    
    torch.set_default_dtype(torch.float32)
    (x, u, xp) = train_data
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(u), torch.from_numpy(xp))
    
    (x, u, xp) = validation_data
    validation_data = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(u), torch.from_numpy(xp))
    opt = torch.optim.AdamW(model.parameters(), lr=hparams.lr, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=hparams.scheduler_step_size, gamma=0.5)

    metric_tracker = MetricTracker(hparams.patience, True)
    for epoch in tqdm.trange(hparams.nepochs, position=1):
        with utils.Timer() as train_t:
            train_metrics = train_epoch(model, opt, train_data, hparams.batch_size, hparams.dt, hparams.gradnorm)
        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f'training/{k}', v, epoch)

            writer.add_scalar('training/time', train_t.dt, epoch)
        #내가 추가한 코드
        train_loss['epoch'].append(epoch)
        train_loss['train_loss'].append(train_metrics['loss/mean'])
        ##
        if validation_data is not None:
            with utils.Timer() as valid_t:
                valid_metrics = validate(model, validation_data, hparams.batch_size, hparams.dt)

            this_epoch_valid_metric = valid_metrics['loss/mean']
            metric_tracker.add_metric(this_epoch_valid_metric)

            if writer is not None:
                for k, v in valid_metrics.items():
                    writer.add_scalar(f'validation/{k}', v, epoch)

                writer.add_scalar('validation/time', valid_t.dt, epoch)
        scheduler.step()

        if validation_data is not None:
            if metric_tracker.should_stop_early():
                print("Ran out of patience. Stopping training!")
                break

            if metric_tracker.is_best_so_far():
                metric_tracker.best_epoch_metrics = valid_metrics
                if hparams.logdir is not None:
                    save_checkpoint(model, epoch, hparams.logdir, f"best.th")

            if writer is not None:
                writer.add_scalar('training/best_epoch', metric_tracker.best_epoch, epoch)

        if epoch % 20 == 0 or (epoch+1) == hparams.nepochs:
            if hparams.logdir is not None:
                save_checkpoint(model, epoch, hparams.logdir, f"model_state_epoch_{epoch}.th")
    #내가 추가한 코드
    sio.savemat(loss_log_path,  train_loss)
    ##
    if hparams.logdir is not None and validation_data  is not None:
        model.load_state_dict(torch.load(os.path.join(hparams.logdir, "best.th")))
    return model.get_params()


HParams = collections.namedtuple('HParams', ['logdir', 'nepochs', 'lr', 'batch_size', 'dt', 'scheduler_step_size', 'patience', 'gradnorm'])

if __name__ == '__main__':
    potential = models.DelanZeroPotential(2)
    mod = rigidbody.DeLan(2, 64, 3, torch.tensor([1, 1, 0, 0]), udim=1, potential=potential)
    tdata = (np.random.rand(128, 4), np.random.rand(128, 1), np.random.rand(128, 4))
    vdata = (np.random.rand(128, 4), np.random.rand(128, 1), np.random.rand(128, 4))

    train(mod, tdata, vdata, HParams(None, 2, 0.001, 32, 0.01, 50, 50, 50.))
