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

def inconsistency_in_definition(filename, init_random=True, T=60, time_tick=1200):
    logging.info("============================================")
    pic_path = '_'.join(strtmp for strtmp in filename.split('/')[3:]).split('.')[0] + "_inconsistency_" + str(T) + "_" + str(time_tick)
    logging.info(pic_path)

    results = torch.load(filename)
    seed = results['args']['seed']
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
        x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)

    A = results['A'][0].to(device)
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

    torch.set_printoptions(threshold=np.inf)
    t = torch.cat((t, torch.tensor([5., 6.])))
    t, _ = t.sort()
    t1_index = torch.nonzero(t == 5.)[0][0]
    t2_index = torch.nonzero(t == 6.)[0][0]
    print(t1_index)  # tensor(132)
    print(t2_index)  # tensor(157)

    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        logging.info("choice HeatDynamics")
        print(solution_numerical.shape)  # torch.Size([99, 400, 1])
    true_y = solution_numerical.squeeze().t().to(device)
    true_y0 = x0.to(device)

    true_t1 = true_y[:, t1_index]
    t_second = t[t1_index:] - t[t1_index]
    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDynamics(A), true_t1.unsqueeze(dim=-1), t_second, method='dopri5') # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        logging.info("choice HeatDynamics")
    true_y_second = solution_numerical.squeeze().t().to(device)

    output = model(t, true_y0).squeeze().t()
    output_t1 = output[:, t1_index]
    output_second = model(t_second, output_t1.unsqueeze(dim=-1)).squeeze().t()
    print(output_second.shape)  # torch.Size([400, 1310])

    sample = t
    x_t_true_figure(t=t, true_y=true_y, sample=sample, T=T, time_tick=time_tick, figure_name='dircet')
    x_t_true_figure(t=t_second + t[t1_index], true_y=true_y_second, sample=sample, T=T, time_tick=time_tick, figure_name='second')

    x_t_model_figure(t, output, sample, T, time_tick, 'dircet')
    x_t_model_figure(t_second + t[t1_index], output_second, sample, T, time_tick, 'second')

    # x_t_difference_figure(t+np.ones_like(t.shape)*2, pred_y, results_true_y, sample, T, time_tick)


def x_t_true_figure(t, true_y, sample, T, time_tick, figure_name):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([0, 26])
    for i in range(0, 100, 10):
        ax.plot(t, true_y[i, :].cpu().detach().numpy(), alpha=0.7, label='$True \enspace of \enspace x_{%d}$' % i)
        # ax.plot(sample, [1] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$x_{i}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\inconsistency_in_definition\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_extra_true_10_" + str(T) + "_" + str(time_tick) +"_" + figure_name
    plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def x_t_model_figure(t, pred_y, sample, T, time_tick, figure_name):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([0, 26])
    for i in range(0, 100, 10):
        ax.plot(t, pred_y[i, :].squeeze().cpu().detach().numpy(), alpha=0.7, label='$model \enspace of \enspace x_{%d}$' % i)
        # ax.plot(sample, [1] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$\hat{x_{i}}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\inconsistency_in_definition\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_extra_model_10_" + str(T) + "_" + str(time_tick) +"_" + figure_name
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def x_t_difference_figure(t, pred_y, true_y, sample, T, time_tick):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -0.22)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    for i in range(0, 100, 10):
        ax.plot(t, pred_y[i, :].squeeze().cpu().detach().numpy()-true_y[i, :].cpu().detach().numpy(), alpha=0.7, label='$difference \enspace of \enspace x_{%d}$' % i)
        ax.plot(sample, [1] * len(sample), '|', linewidth=0.001, color='k')
        # ax.plot(sample, [-0.20] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$error \enspace of \enspace x_{i}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    pic_path = r'.\\inconsistency_in_definition\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_difference_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

if __name__ == '__main__':
    if (not os.path.exists(r'.\inconsistency_in_definition')):
        makedirs(r'.\inconsistency_in_definition')

    log_filename = r'inconsistency_in_definition/inconsistency_in_definition_heat.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'.\results/HeatDynamics/grid/result_HeatDynamics_grid_0128-011309_0.pth'
    inconsistency_in_definition(filename, init_random=True, T=60, time_tick=1200)

