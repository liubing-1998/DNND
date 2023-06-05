# coding:utf-8

import os
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from mpl_toolkits import axisartist

from neural_dynamics import *


def read_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)
        # print(dirs)
        # print(files)
        pass
    return root, files

def x_truef_mlpf_figure(x, true_y_f, mlp_y_f, dynmaic):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -1.5)
    ax.axis["y"] = ax.new_floating_axis(1, 0.)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)  # , size=1.0
    ax.axis["y"].set_axisline_style("->", size=0.3)
    ax.set_xlim([0, 26])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_ylim([-1.5, 1.1])
    # ax.set_yticks([i for i in np.arange(-1.5, 1.1, 0.5)])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0])
    ax.plot(x, true_y_f, label='$True \enspace f$', alpha=0.7)
    ax.plot(x, mlp_y_f, label='$Predict \enspace f$', alpha=0.7)
    ax.axis["x"].label.set_text(r"$x_{i}$")
    ax.axis["y"].label.set_text(r"$f(x_{i})$")
    ax.axis["x"].label.set_size(28)
    ax.axis["y"].label.set_size(28)
    ax.legend()
    pic_path = '.\picture\\' + dynamic + '\\' + filename
    # plt.tick_params(width=1.5)
    plt.tight_layout()
    plt.savefig(pic_path + "_f.png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + "_f.pdf", bbox_inches='tight', dpi=1000)
    # plt.show()
    plt.close()

def x_trueg_mlpg_figure(x, true_y_g, mlp_y_g, X, Y, dynamic):
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig = plt.figure(constrained_layout=True, figsize=(9, 4))
    axsnest = fig.subplots(1, 2, sharey=True)
    # print(axsnest)
    # sys.exit()
    true_y_g_max = true_y_g.max()
    true_y_g_min = true_y_g.min()
    levels = np.arange(true_y_g_min, true_y_g_max, (true_y_g_max-true_y_g_min)/100)
    ctf1 = axsnest[0].contourf(X, Y, true_y_g, levels=levels, cmap=plt.cm.hot)
    axsnest[0].set_ylabel("$x_{i}$", fontsize=28)  # , fontsize=14
    axsnest[0].set_xlabel("$x_{j}$", fontsize=28)  # , fontsize=14
    axsnest[0].set_xlim([0, 25])
    axsnest[0].set_xlim([0, 25])
    ctf2 = axsnest[1].contourf(X, Y, mlp_y_g.cpu().detach().numpy(), levels=levels, cmap=plt.cm.hot)
    axsnest[1].set_ylabel("$x_{i}$", fontsize=28)  # , fontsize=14
    axsnest[1].set_xlabel("$x_{j}$", fontsize=28)  # , fontsize=14
    axsnest[1].set_xlim([0, 25])
    axsnest[1].set_xlim([0, 25])
    kwargs = {'format': '%.2f'}
    cb = fig.colorbar(ctf2, ax=axsnest, **kwargs)
    cb.locator = ticker.MaxNLocator(nbins=6)
    cb.update_ticks()

    pic_path = '.\\picture\\' + dynamic + '\\' + filename
    plt.savefig(pic_path + "_g.png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + "_g.pdf", bbox_inches='tight', dpi=1000)
    # plt.show()
    plt.close()

def x_truef_mlpf_figure_birthdeath(x, true_y_f, mlp_y_f, dynamic):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -62.5)
    ax.axis["y"] = ax.new_floating_axis(1, 0.)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)  # , size=1.0
    ax.axis["y"].set_axisline_style("->", size=0.3)
    # 设置坐标轴范围
    ax.set_xlim([0, 26])
    # ax.set_xlim([5, 16])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    # ax.set_xticks(range(0, 25, 5))
    ax.set_ylim([-62.5, 1])
    # ax.set_yticks([i for i in np.arange(-1.5, 1.1, 0.5)])
    # ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0])
    ax.plot(x, true_y_f, label='$True \enspace f$', alpha=0.7)
    ax.plot(x, mlp_y_f, label='$Predict \enspace f$', alpha=0.7)
    ax.axis["x"].label.set_text(r"$x_{i}$")
    ax.axis["y"].label.set_text(r"$f(x_{i})$")
    ax.axis["x"].label.set_size(28)
    ax.axis["y"].label.set_size(28)
    ax.legend()
    pic_path = '.\\picture\\' + dynamic + '\\' + filename
    # plt.tick_params(width=1.5)
    plt.tight_layout()
    plt.savefig(pic_path + "_f.png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + "_f.pdf", bbox_inches='tight', dpi=1000)
    # plt.show()
    plt.close()

def read_heat_result(file_path, filename, k=0.1):
    results = torch.load(file_path)
    input_size = 1
    hidden_A_str_list = results['args']['hidden_A_list']
    hidden_A_list = []
    for item in hidden_A_str_list:
        hidden_A_list.append(int(item))
    hidden_str_list = results['args']['hidden_list']
    hidden_list = []
    for item in hidden_str_list:
        hidden_list.append(int(item))
    A = results['A'][0]
    rtol = results['args']['rtol']
    atol = results['args']['atol']
    method = results['args']['method']
    activation_function = results['args']['activation_function']
    model = DNND(activation_function=activation_function, input_size=input_size, hidden_A_list=hidden_A_list,
                 hidden_list=hidden_list, A=A, rtol=rtol, atol=atol, method=method)
    model.load_state_dict(results['model_state_dict'][-1])
    model.cuda()
    x = torch.arange(0, 25, 0.1)
    x = torch.unsqueeze(x, dim=-1)  # torch.Size([250, 1])
    n = x.shape[0]
    xi = x.repeat(1, n)
    xj = x.T.repeat(n, 1)
    X, Y = np.meshgrid(x, x)

    '''
    # HeatDynamics
    # k = 1
    # g_func = -k * (xi - xj)
    '''
    print('k=', k)
    true_y_g = -k * (xi - xj)

    x_a = torch.stack((xi, xj), dim=-1)
    x_a = x_a.cuda()
    mlp_y_g = model.neural_dynamic_layer.odefunc.MLP_A(x_a)  # N*N*1
    mlp_y_g = torch.squeeze(mlp_y_g, dim=-1)  # N*N
    mlp_y_g = mlp_y_g + torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1)
    plt.rc('font', family='Times New Roman', size=24)
    plt.rc('lines', linewidth=2)
    makedirs(r'.\picture\HeatDynamics')
    x_trueg_mlpg_figure(x, true_y_g, mlp_y_g, X, Y, 'HeatDynamics')


def read_biochemical_result(file_path, filename, f=1, b=0.1, r=0.01):
    results = torch.load(file_path)
    ###################################################### 重定义模型
    # 取参数
    input_size = 1
    hidden_A_str_list = results['args']['hidden_A_list']
    hidden_A_list = []
    for item in hidden_A_str_list:
        hidden_A_list.append(int(item))
    hidden_str_list = results['args']['hidden_list']
    hidden_list = []
    for item in hidden_str_list:
        hidden_list.append(int(item))
    A = results['A'][0]
    rtol = results['args']['rtol']
    atol = results['args']['atol']
    method = results['args']['method']
    activation_function = results['args']['activation_function']
    model = DNND(activation_function = activation_function, input_size=input_size, hidden_A_list=hidden_A_list,
                 hidden_list=hidden_list, A=A, rtol=rtol, atol=atol, method=method)

    model.load_state_dict(results['model_state_dict'][-1])
    model.cuda()
    ######################################################
    # 生成x1和x2序列
    x = torch.arange(0, 25, 0.1)
    x = torch.unsqueeze(x, dim=-1)  # torch.Size([100, 1])
    n = x.shape[0]
    xi = x.repeat(1, n)  # 复制一列为n列  shape = N*N
    xj = x.T.repeat(n, 1)  # x.T先转换为一行，再复制一行为n行  shape = N*N
    # 为等高线图生成坐标轴数据
    X, Y = np.meshgrid(x, x)

    '''
    # BiochemicalDynamics公式动态f计算，g计算
    # dxi(t)/dt = f-b*xi - \sum_{j=1}^{N}Aij r*xi *xj
    # f_func = f - b * x
    # g_func = r*xi *xj
    '''
    # f = 1
    # b = 0.1
    # r = 1
    print('f=', f, ' b=', b, ' r=', r)
    true_y_f = f - b * x
    true_y_g = - r * xi * xj

    # 取出真实的x值，用于绘制散点
    real_x = results['true_y'][0]  # torch.Size([400, 120])
    real_f = f - b * real_x

    # 计算MLP计算出的线
    x_mlp = x.cuda()
    mlp_y_f = model.neural_dynamic_layer.odefunc.MLP(x_mlp) + model.neural_dynamic_layer.odefunc.linear_f(x_mlp)
    mlp_y_f = mlp_y_f.cpu().detach()

    # 计算MLP_A计算出的等高线图
    x_a = torch.stack((xi, xj), dim=-1)  # 新增维度拼接x1和x2  shape=N*N*2
    x_a = x_a.cuda()
    mlp_y_g = model.neural_dynamic_layer.odefunc.MLP_A(x_a)  # N*N*1
    # print(mlp_y_g.shape)
    # print(torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1).shape)

    mlp_y_g = torch.squeeze(mlp_y_g, dim=-1) + torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1)  # N*N
    # print(mlp_y_g.shape)
    # sys.exit()

    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    # 设置全局字体，字体大小（好像只对text生效）
    plt.rc('font', family='Times New Roman', size=24)
    plt.rc('lines', linewidth=2)  # 设置全局线宽

    makedirs(r'.\picture\BiochemicalDynamics')
    # 绘制true_y_f, mlp_y_f, 和real_y散点于ax0图
    x_truef_mlpf_figure(x, true_y_f, mlp_y_f, 'BiochemicalDynamics')

    # 绘制true_y_g等高线图
    x_trueg_mlpg_figure(x, true_y_g, mlp_y_g, X, Y, 'BiochemicalDynamics')


'''
self.B = 0.1  # 原1
self.R = 0.2  # 原0.1
self.b = 2  # 原1
self.a = 1  # 原1
'''
def read_birthdeath_result(file_path, filename, B=0.1, R=0.2, b=2, a=1):
    results = torch.load(file_path)
    ###################################################### 重定义模型
    # 取参数
    input_size = 1
    hidden_A_str_list = results['args']['hidden_A_list']
    hidden_A_list = []
    for item in hidden_A_str_list:
        hidden_A_list.append(int(item))
    hidden_str_list = results['args']['hidden_list']
    hidden_list = []
    for item in hidden_str_list:
        hidden_list.append(int(item))
    A = results['A'][0]
    rtol = results['args']['rtol']
    atol = results['args']['atol']
    method = results['args']['method']
    activation_function = results['args']['activation_function']
    model = DNND(activation_function=activation_function, input_size=input_size, hidden_A_list=hidden_A_list,
                 hidden_list=hidden_list, A=A, rtol=rtol, atol=atol, method=method)
    model.load_state_dict(results['model_state_dict'][-1])
    model.cuda()
    ######################################################
    # 生成x1和x2序列
    x = torch.arange(0, 25, 0.1)
    x = torch.unsqueeze(x, dim=-1)  # torch.Size([100, 1])
    n = x.shape[0]
    xi = x.repeat(1, n)  # 复制一列为n列  shape = N*N
    xj = x.T.repeat(n, 1)  # x.T先转换为一行，再复制一行为n行  shape = N*N
    # 为等高线图生成坐标轴数据
    X, Y = np.meshgrid(x, x)

    '''
    # BirthdeathDynamics公式动态f计算，g计算
    # dxi(t)/dt = -B*xi^b - \sum_{j=1}^{N}Aij R*xi^a
    # f_func = -B*xi^b
    # g_func = R*xi^a
    '''
    # B = 0.1,
    # R = 0.2,
    # b = 2,
    # a = 1
    print('B=', B, ' R=', R, ' b=', b, 'a=', a)
    true_y_f =  - B * x * x
    true_y_g = - R * xj

    # 计算MLP计算出的线
    x_mlp = x.cuda()
    mlp_y_f = model.neural_dynamic_layer.odefunc.MLP(x_mlp) + model.neural_dynamic_layer.odefunc.linear_f(x_mlp)
    mlp_y_f = mlp_y_f.cpu().detach()

    # 计算MLP_A计算出的等高线图
    x_a = torch.stack((xi, xj), dim=-1)  # 新增维度拼接x1和x2  shape=N*N*2
    x_a = x_a.cuda()
    mlp_y_g = model.neural_dynamic_layer.odefunc.MLP_A(x_a)  # N*N*1
    # print(mlp_y_g.shape)
    # print(torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1).shape)

    mlp_y_g = torch.squeeze(mlp_y_g, dim=-1) + torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1)  # N*N
    # print(mlp_y_g.shape)
    # sys.exit()

    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    # 设置全局字体，字体大小（好像只对text生效）
    plt.rc('font', family='Times New Roman', size=24)
    plt.rc('lines', linewidth=2)  # 设置全局线宽

    makedirs(r'.\picture\BirthDeathDynamics')
    # 绘制true_y_f, mlp_y_f, 和real_y散点于ax0图
    x_truef_mlpf_figure_birthdeath(x, true_y_f, mlp_y_f, 'BirthDeathDynamics')

    # 绘制true_y_g等高线图
    x_trueg_mlpg_figure(x, true_y_g, mlp_y_g, X, Y, 'BirthDeathDynamics')


if __name__ == '__main__':
    '''
    r'.\results'
    \GeneDynamics\BiochemicalDynamics\HeatDynamics\MutualDynamics\
    \grid\random\power_law\small_world\community
    # file_dir = r'.\results\BiochemicalDynamics\community'
    # file_dir = r'.\results\GeneDynamics\community'
    # file_dir = r'.\results\HeatDynamics\grid'
    '''
    dynamic_list = ['HeatDynamics', 'BiochemicalDynamics', 'BirthDeathDynamics', ]
    network_list = ['grid', 'random', 'power_law', 'small_world', 'community', ]

    for dynamic in dynamic_list:
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

            for filepath in file_name_list:
                filename = filepath.split('\\')[-1].split('.')[0]
                if dynamic == 'BiochemicalDynamics':
                    read_biochemical_result(filepath, filename)
                elif dynamic == 'HeatDynamics':
                    read_heat_result(filepath, filename)
                elif dynamic == 'BirthDeathDynamics':
                    read_birthdeath_result(filepath, filename)



