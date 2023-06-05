# coding:utf-8

import os
import sys
import logging
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import axisartist
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import torch

from neural_dynamics import *
from dynamics_model import HeatDynamics, BiochemicalDynamics, BirthDeathDynamics, EpidemicDynamics
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def dx_dt_figure_heat_result(filename, init_random=True, T=60, time_tick=1200):
    logging.info("============================================")
    pic_path = '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] + "_dx-dt_" + str(T) + "_" + str(time_tick)
    logging.info(pic_path)

    results = torch.load(filename)
    seed = results['seed'][0]

    if seed <= 0:
        seed = random.randint(0, 2022)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info("seed=" + str(seed))
    logging.info("init_random=" + str(init_random))

    n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    # Initial Value
    x0 = torch.zeros(N, N)
    if init_random:
        x0 = 25 * torch.rand(N, N)
    else:
        x0[int(0.05 * N):int(0.25 * N),
        int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N),
        int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N),
        int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)

    # results = torch.load(filename)
    A = results['A'][0].to(device)
    # A = torch.zeros(400, 400)
    '''n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20'''
    input_size = 1
    hidden_A_str_list = results['args']['hidden_A_list']
    hidden_A_list = []
    for item in hidden_A_str_list:
        hidden_A_list.append(int(item))
    hidden_str_list = results['args']['hidden_list']
    hidden_list = []
    for item in hidden_str_list:
        hidden_list.append(int(item))
    rtol = results['args']['rtol']
    atol = results['args']['atol']
    method = results['args']['method']
    activation_function = results['args']['activation_function']
    model = DNND(activation_function=activation_function, input_size=input_size, hidden_A_list=hidden_A_list,
                 hidden_list=hidden_list, A=A, rtol=rtol, atol=atol, method=method)
    model.load_state_dict(results['model_state_dict'][-1])
    model.to(device)

    sampled_time = 'irregular'
    T = T
    time_tick = time_tick
    logging.info("T=" + str(T))
    logging.info("time_tick=" + str(time_tick))
    # equally-sampled time
    if sampled_time == 'equal':
        print('Build Equally-sampled -time dynamics')
        t = torch.linspace(0., T, time_tick)  # time_tick) # 100 vector
    elif sampled_time == 'irregular':
        print('Build irregularly-sampled -time dynamics')
        # irregular time sequence
        sparse_scale = 10
        t = torch.linspace(0., T, time_tick * sparse_scale)  # 100 * 10 = 1000 equally-sampled tick
        t = np.random.permutation(t)[:int(time_tick * 1.2)]
        t = torch.tensor(np.sort(t))
        t[0] = 0

    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        logging.info("choice HeatDynamics")
        print(solution_numerical.shape)
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)  # 400 * 1

    integration_time_vector = t.type_as(true_y0)
    integration_time_vector.to(device)
    Xht_output = ode.odeint(model.neural_dynamic_layer.odefunc, true_y0, integration_time_vector, rtol=0.01, atol=0.001, method='euler')
    output = model.neural_dynamic_layer(t, true_y0)
    dx_dt_model = torch.zeros(Xht_output.shape)
    for i in range(Xht_output.shape[0]):
        hx_output = model.neural_dynamic_layer.odefunc(t, Xht_output[i, :, :])
        dx_dt_model[i, :, :] = hx_output
    dx_dt_model = dx_dt_model.detach().numpy()

    sample = results['t'][results['id_train'][0]]

    dx_dt_model_figure(output=output, dx_dt_model=dx_dt_model, t=t, sample=sample, T=T, time_tick=time_tick)

    dx_dt_true_figure(true_y=true_y, A=A, t=t, sample=sample, T=T, time_tick=time_tick)

    dx_dt_difference_figure(true_y=true_y, dx_dt_model=dx_dt_model, A=A, t=t, sample=sample, T=T, time_tick=time_tick)

def dx_dt_true_figure(true_y, A, t, sample, T, time_tick):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -8)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([-8, 8])
    dx_dt_true = np.zeros(true_y.shape)
    for i in range(true_y.shape[1]):
        hx_output = HeatDynamics(A)(t, true_y[:, i].unsqueeze(0).t())
        dx_dt_true[:, i] = hx_output.detach().numpy().squeeze()
    for i in range(0, 100, 10):
        # ax.plot(true_y[i, :].detach().numpy(), dx_dt_true[i, :], label='$True \enspace of \enspace dx/dt-x$')
        ax.plot(t.detach().numpy(), dx_dt_true[i, :], alpha=0.7, label='$True \enspace of \enspace dx/dt-x$')
        # ax.arrow(true_y[i, 0].detach().numpy(), dx_dt_true[i, 0],
        #          true_y[i, 4].detach().numpy()-true_y[i, 0].detach().numpy(), dx_dt_true[i, 4]-dx_dt_true[i, 0],
        #          width=0.1, facecolor='white')
        ax.plot(sample, [-7] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$dx/dt$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_derivative_true_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def dx_dt_model_figure(output, dx_dt_model, t, sample, T, time_tick):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -8)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([-8, 8])
    num = 1440
    for i in range(0, 100, 10):
        ax.plot(t.detach().numpy(), dx_dt_model[:, i].squeeze(), alpha=0.7,
                label='$model \enspace of \enspace dx{%d}/dt-t$ % i')
        ax.plot(sample, [-7] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$\hat{dx/dt}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_derivative_t_model_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)


def dx_dt_difference_figure(true_y, dx_dt_model, A, t, sample, T, time_tick):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -0.5)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([-0.5, 0.5])
    dx_dt_true = np.zeros(true_y.shape)
    for i in range(true_y.shape[1]):
        hx_output = HeatDynamics(A)(t, true_y[:, i].unsqueeze(0).t())
        dx_dt_true[:, i] = hx_output.detach().numpy().squeeze()
    for i in range(0, 100, 10):
        # ax.plot(true_y[i, :].detach().numpy(), dx_dt_true[i, :], label='$True \enspace of \enspace dx/dt-x$')
        ax.plot(t.detach().numpy(), dx_dt_model[:, i].squeeze()-dx_dt_true[i, :], alpha=0.7, label='$True \enspace of \enspace dx/dt-x$')
        # ax.arrow(true_y[i, 0].detach().numpy(), dx_dt_true[i, 0],
        #          true_y[i, 4].detach().numpy()-true_y[i, 0].detach().numpy(), dx_dt_true[i, 4]-dx_dt_true[i, 0],
        #          width=0.1, facecolor='white')
        ax.plot(sample, [-0.45] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$error \enspace of \enspace dx/dt$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_difference_true_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)


if __name__ == '__main__':
    if (not os.path.exists(r'.\figure_dx-dt_t')):
        makedirs(r'.\figure_dx-dt_t')

    log_filename = r'figure_dx-dt_t/heat_dx-dt_t.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'.\results/HeatDynamics/grid/result_HeatDynamics_grid_0128-011809_1.pth'
    dx_dt_figure_heat_result(filename, init_random=True, T=60, time_tick=1200)


