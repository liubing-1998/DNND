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
import scipy.io

from neural_dynamics import *
from dynamics_model import HeatDynamics, BiochemicalDynamics, BirthDeathDynamics, EpidemicDynamics
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def create_mat_data_figure_heat_result(filename, init_random=True, T=60, time_tick=1200):
    logging.info("============================================")
    pic_path = ''.join(filename.split('\\')[-1]).split('.')[0] + "_mat_" + str(T) + "_" + str(time_tick) + ".mat"
    logging.info(pic_path)
    save_mat_path = os.path.join('create_mat_data', pic_path)

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

    criterion = F.l1_loss
    pred_y = model(t, true_y0).squeeze().t()  
    loss = criterion(pred_y, true_y)  # [:, id_]
    relative_loss = criterion(pred_y, true_y) / true_y.mean()
    print('RESULT Test Loss {:.6f}({:.6f} Relative))'.format(loss.item(), relative_loss.item()))
    logging.info('RESULT Test Loss {:.6f}({:.6f} Relative)'.format(loss.item(), relative_loss.item()))
    prefix_name = pic_path.split('_')[4]
    scipy.io.savemat(save_mat_path, mdict={prefix_name + '_true_data': true_y.t().cpu().detach().numpy(),
                                           prefix_name + '_model_data': pred_y.t().cpu().detach().numpy()})

def read_file_name(file_dir):
    root = ""
    files = ""
    for root, dirs, files in os.walk(file_dir):
        pass
    return root, files

if __name__ == '__main__':
    if (not os.path.exists(r'.\create_mat_data')):
        makedirs(r'.\create_mat_data')

    dynamic = 'HeatDynamics'
    # network_list = ['grid', 'random', 'power_law', 'small_world', 'community']
    network = 'grid'
    file_dir = os.path.join(r'.\results', dynamic, network)
    root, file_name_lists = read_file_name(file_dir)
    file_name_list = []
    for item in file_name_lists:
        if ".pdf" in item:
            pass
        elif ".png" in item:
            pass
        elif '_' not in item:
            pass
        else:
            file_name_list.append(os.path.join(root, item))

    log_filename = r'create_mat_data/create_mat_data.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    for filepath in file_name_list:
        create_mat_data_figure_heat_result(filepath, init_random=True, T=60, time_tick=1200)



