# coding:utf-8

import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_dynamics import *
from sklearn.linear_model import LassoCV, Lasso

'''
self.B = 0.1  # 原1
self.R = 0.2  # 原0.1
self.b = 2  # 原1
self.a = 1  # 原1
'''
def lasso_birthDeath_G(file_path,  B=0.1, R =0.2, b=2, a=1):
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
    model = DNND(activation_function=activation_function, input_size=input_size, hidden_A_list=hidden_A_list, hidden_list=hidden_list, A=A, rtol=rtol, atol=atol,
                 method=method)

    model.load_state_dict(results['model_state_dict'][-1])
    model.cuda()
    ######################################################
    # 生成x1和x2序列
    x = torch.arange(0, 25, 0.2)  # 控制生成数据的量
    x = torch.unsqueeze(x, dim=-1)  # torch.Size([250, 1])
    n = x.shape[0]
    xi = x.repeat(1, n)  # 复制一列为n列  shape = N*N
    xj = x.T.repeat(n, 1)  # x.T先转换为一行，再复制一行为n行  shape = N*N
    # print(x1.shape, x2.shape)  # torch.Size([250, 250]) torch.Size([250, 250])

    '''
    # BirthDeathDynamics公式动态f计算，g计算
    # dxi(t)/dt = -B*xi^b - \sum_{j=1}^{N}Aij R*xi^a
    # f_func = -self.B * (x ** self.b)
    # g_func = torch.mm(self.A, self.R * (x ** self.a))
    '''
    # A = A  # 邻接矩阵
    print('B=', B, ' R=', R, ' b=', b, ' a=', a)
    true_y_f = - B * (x ** b)
    true_y_g = R * (xj ** a)

    # 计算出MLP_A得到的值，相当于G的导数值
    x_a = torch.stack((xi, xj), dim=-1)  # 新增维度拼接x1和x2  shape=N*N*2
    x_a = x_a.cuda()
    mlp_y_g = model.neural_dynamic_layer.odefunc.MLP_A(x_a)  # N*N*1
    mlp_y_g = torch.squeeze(mlp_y_g, dim=-1) + torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1)   # N*N
    # print(xi.shape, xj.shape, mlp_y_g.shape)  # torch.Size([250, 250]) torch.Size([250, 250]) torch.Size([250, 250])
    xi = xi.reshape(-1)
    xj = xj.reshape(-1)
    mlp_y_g = mlp_y_g.reshape(-1)
    # print(xi.shape, xj.shape, mlp_y_g.shape)  # torch.Size([62500]) torch.Size([62500]) torch.Size([62500])
    ##########################构建基库
    column_item, matrix = coupled_Polynomial_functions(xi, xj)

    X_all = matrix.cpu().detach().numpy()
    y_g_all = mlp_y_g.cpu().detach().numpy()
    # y_g_all = true_y_g.reshape(-1).cpu().numpy()
    # reg1 = LassoCV(cv=5, fit_intercept=True, n_jobs=-1, max_iter=10000, normalize=False).fit(X_all, y_g_all)
    reg1 = Lasso(alpha=0.3, fit_intercept=False).fit(X_all, y_g_all) # , normalize=True  0.31
    print(reg1.score(X_all, y_g_all))
    # print('Best threshold: %.3f' % reg1.alpha_)
    print(column_item)
    print(reg1.coef_)
    print(reg1.intercept_)

    # 计算出MLP得到的值，相当于F的导数值
    x_mlp = x.cuda()
    mlp_y_f = model.neural_dynamic_layer.odefunc.MLP(x_mlp) + model.neural_dynamic_layer.odefunc.linear_f(x_mlp)
    y_f_all = mlp_y_f.cpu().detach()
    # print(x_mlp.shape, y_f_all.shape)
    ##########################构建基库
    column_item_f, matrix_f = polynomial_functions(x_mlp.squeeze())
    x_all = matrix_f.cpu().detach().numpy()
    # print(x_all.shape, y_f_all.shape)
    # reg2 = LassoCV(cv=5, fit_intercept=True, n_jobs=-1, max_iter=10000, normalize=False).fit(x_all, y_f_all)
    reg2 = Lasso(alpha=2, fit_intercept=False).fit(x_all, y_f_all) # , normalize=True 0.5
    print(reg2.score(x_all, y_f_all))  # true_y_f, y_f_all
    # print('Best threshold: %.3f' % reg2.alpha_)
    print(column_item_f)
    print(reg2.coef_)
    print(reg2.intercept_)



def coupled_Polynomial_functions(xi, xj):
    # column_values = ['1', 'x1i', 'x1j', 'x1jMinusx1i', 'x1jAddx1i', 'x1ix1j', 'x1i^2', 'x1j^2']
    column_values = ['x1j', 'x1jMinusx1i', 'x1jAddx1i', 'x1ix1j', 'x1i^2', 'x1j^2']
    column_values = ['x1j', 'x1ix1j', 'x1i^2', 'x1j^2']
    tmp_constant = torch.ones_like(xi)
    tmp_x1i = xi
    tmp_x1j = xj
    tmp_x1jMinusx1j = xj - xi
    tmp_x1jAddx1i = xj + xi
    tmp_x1ix1j = xi * xj
    tmp_x2i = xi ** 2
    tmp_x2j = xj ** 2
    # matrix = torch.stack((tmp_constant, tmp_x1i, tmp_x1j, tmp_x1jMinusx1j, tmp_x1jAddx1i, tmp_x1ix1j, tmp_x2i, tmp_x2j), 1)
    # matrix = torch.stack((tmp_x1j, tmp_x1jMinusx1j, tmp_x1jAddx1i, tmp_x1ix1j, tmp_x2i, tmp_x2j), 1)
    matrix = torch.stack((tmp_x1j, tmp_x1ix1j, tmp_x2i, tmp_x2j), 1)
    return column_values, matrix

def polynomial_functions(xi):
    # column_values = ['1', 'x1i', 'x1i^2', 'x1i^3', ]
    column_values = ['x1i', 'x1i^2', 'x1i^3', ]
    tmp_constant = torch.ones_like(xi)
    tmp_x1i = xi
    tmp_x2i = xi ** 2
    tmp_x3i = xi * tmp_x2i
    # matrix = torch.stack((tmp_constant, tmp_x1i, tmp_x2i, tmp_x3i), 1)
    matrix = torch.stack((tmp_x1i, tmp_x2i, tmp_x3i), 1)
    print(matrix.shape)
    return column_values, matrix


if __name__ == '__main__':
    # lasso稀疏回归求G公式部分
    # file_path = r'.\results/BirthDeathDynamics/community/result_BirthDeathDynamics_community_0526-235611_3.pth'
    # # file_path = r'.\results/BirthDeathDynamics/community/result_BirthDeathDynamics_community_0527-001314_7.pth'
    # file_path = r'.\results/BirthDeathDynamics/grid/result_BirthDeathDynamics_grid_0526-210527_5.pth'
    # file_path = r'.\results/BirthDeathDynamics/grid/result_BirthDeathDynamics_grid_0526-211439_7.pth'
    # file_path = r'.\results/BirthDeathDynamics/power_law/result_BirthDeathDynamics_power_law_0526-221813_1.pth'
    # file_path = r'.\results/BirthDeathDynamics/power_law/result_BirthDeathDynamics_power_law_0526-224931_8.pth'
    # file_path = r'.\results/BirthDeathDynamics/random/result_BirthDeathDynamics_random_0526-212846_0.pth'
    # file_path = r'.\results/BirthDeathDynamics/random/result_BirthDeathDynamics_random_0526-213316_1.pth'
    # file_path = r'.\results/BirthDeathDynamics/small_world/result_BirthDeathDynamics_small_world_0526-230315_1.pth'
    # file_path = r'.\results/BirthDeathDynamics/small_world/result_BirthDeathDynamics_small_world_0526-233856_9.pth'

    # test
    # file_path = r'.\results/BirthDeathDynamics/community/result_BirthDeathDynamics_community_0526-235611_3.pth'
    # file_path = r'.\results\BirthDeathDynamics\grid\result_BirthDeathDynamics_grid_0526-212404_9.pth'
    # network = ['grid', 'random', 'power_law', 'small_world', 'community']

    # file_path = r'results/BirthDeathDynamics/community/result_BirthDeathDynamics_community_0111-231636_DD0.pth'
    file_path = r'results/BirthDeathDynamics/community/result_BirthDeathDynamics_community_0128-143117_0.pth'
    lasso_birthDeath_G(file_path)
