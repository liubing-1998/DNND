# coding:utf-8

import os
import sys
import logging
import random
import argparse
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import torch
import mpl_toolkits.axisartist as axisartist
import matplotlib as mpl

from neural_dynamics import *
from dynamics_model import HeatDynamics, BiochemicalDynamics, BirthDeathDynamics, EpidemicDynamics
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def different_initial_interpolation_extrapolation_longtime(result_filename, initial_filename, init_random=True):
    logging.info("============================================")
    pic_path = result_filename.split('\\')[-1][:-4] + "__with_initial_from__" + initial_filename.split('\\')[-1][:-4]
    logging.info(pic_path)

    # 取模型参数
    results_model = torch.load(result_filename)
    results_initial = torch.load(initial_filename)

    seed = results_initial['seed'][0]
    # args.seed<=0,随机出一个种子，否则使用其值作为种子
    if seed <= 0:
        seed = random.randint(0, 2022)
    # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使结果是确定的
    torch.manual_seed(seed)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info("seed=" + str(seed))
    logging.info("init_random=" + str(init_random))

    # 生成新初始化
    n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    # Initial Value
    x0 = torch.zeros(N, N)
    # 随机生成数据
    if init_random:
        x0 = 25 * torch.rand(N, N)  # 种子固定以后，随机生成的初始化和从文件中读出来的x0值是一样的
        # x0 = results_initial['true_y'][0][:, 0].unsqueeze(dim=-1)
        logging.info("x0从对应文件中读出")
    else:
        x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)  # torch.Size([400, 1])

    # 模型参数初始化
    input_size = 1
    hidden_A_str_list = results_model['args']['hidden_A_list']
    hidden_A_list = []
    for item in hidden_A_str_list:
        hidden_A_list.append(int(item))
    hidden_str_list = results_model['args']['hidden_list']
    hidden_list = []
    for item in hidden_str_list:
        hidden_list.append(int(item))
    A = results_model['A'][0].to(device)
    rtol = results_model['args']['rtol']
    atol = results_model['args']['atol']
    method = results_model['args']['method']
    activation_function = results_model['args']['activation_function']
    model = DNND(activation_function=activation_function, input_size=input_size, hidden_A_list=hidden_A_list, hidden_list=hidden_list, A=A, rtol=rtol, atol=atol, method=method)
    model.load_state_dict(results_model['model_state_dict'][-1])
    model.to(device)

    # 时间从对应初始化的文件中读出
    t = results_initial['t']
    logging.info("t从对应文件中读出")

    with torch.no_grad():
        # ['HeatDynamics', 'BiochemicalDynamics', 'BirthDeathDynamics', ]
        if dynamic=='HeatDynamics':
            solution_numerical = ode.odeint(HeatDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
            logging.info("choice HeatDynamics")
        elif dynamic=='BiochemicalDynamics':
            solution_numerical = ode.odeint(BiochemicalDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
            logging.info("choice BiochemicalDynamics")
        elif dynamic=='BirthDeathDynamics':
            solution_numerical = ode.odeint(BirthDeathDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
            logging.info("choice BirthDeathDynamics")
        else:
            logging.info("No dynamics")
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120

    # true_y = results['true_y'][0].to(device)
    true_y0 = x0.to(device)

    # 训练集测试集划分从对应初始化的文件中读出
    id_test = results_initial['id_test'][0]
    id_test2 = results_initial['id_test2'][0]
    id_test3 = results_initial['id_test3'][0]
    logging.info("id_test, id_test2, id_test3从对应文件中读出")

    criterion = F.l1_loss
    pred_y = model(t, true_y0).squeeze().t()
    loss1 = criterion(pred_y[:, id_test], true_y[:, id_test])
    relative_loss1 = criterion(pred_y[:, id_test], true_y[:, id_test]) / true_y[:, id_test].mean()
    loss2 = criterion(pred_y[:, id_test2], true_y[:, id_test2])
    relative_loss2 = criterion(pred_y[:, id_test2], true_y[:, id_test2]) / true_y[:, id_test2].mean()
    loss3 = criterion(pred_y[:, id_test3], true_y[:, id_test3])
    relative_loss3 = criterion(pred_y[:, id_test3], true_y[:, id_test3]) / true_y[:, id_test3].mean()

    print('RESULT Test Loss1 {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative))'
          .format(loss1.item(), relative_loss1.item(), loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item()))
    logging.info('RESULT Test Loss1 {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative))'
          .format(loss1.item(), relative_loss1.item(), loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item()))


def read_file_name(file_dir):
    root = ""
    files = ""
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        pass
    return root, files

if __name__ == '__main__':
    if (not os.path.exists(r'.\different_initial_interpolation_extrapolation_longtime')):
        makedirs(r'.\different_initial_interpolation_extrapolation_longtime')

    log_filename = r'different_initial_interpolation_extrapolation_longtime/different_initial_all.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # 遍历所有结果
    dynamics = ['HeatDynamics', 'BiochemicalDynamics', 'BirthDeathDynamics', ]
    network_list = ['grid', 'random', 'power_law', 'small_world', 'community']
    for dynamic in dynamics:
        for network in network_list:
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

            for item in file_name_list[1:]:
                different_initial_interpolation_extrapolation_longtime(file_name_list[0], item, True)

