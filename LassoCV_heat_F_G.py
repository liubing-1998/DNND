# coding:utf-8

import os
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_dynamics import *
from sklearn.linear_model import LassoCV, Lasso

def lasso_heat_G(file_path, k=0.1):
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
    x = torch.arange(0, 25, 0.2)
    x = torch.unsqueeze(x, dim=-1)  # torch.Size([250, 1])
    n = x.shape[0]
    xi = x.repeat(1, n)
    xj = x.T.repeat(n, 1)

    '''
    # HeatDynamics
    # k = 1
    # g_func = -k * (xi - xj)
    '''
    true_y_g = -k * (xi - xj)

    x_a = torch.stack((xi, xj), dim=-1)
    x_a = x_a.cuda()
    mlp_y_g = model.neural_dynamic_layer.odefunc.MLP_A(x_a)  # N*N*1
    mlp_y_g = torch.squeeze(mlp_y_g, dim=-1)  # N*N
    mlp_y_g = mlp_y_g + torch.squeeze(model.neural_dynamic_layer.odefunc.linear_A(x_a), dim=-1)
    # print(xi.shape, xj.shape, mlp_y_g.shape)  # torch.Size([250, 250]) torch.Size([250, 250]) torch.Size([250, 250])
    xi = xi.reshape(-1)
    xj = xj.reshape(-1)
    mlp_y_g = mlp_y_g.reshape(-1)
    # print(xi.shape, xj.shape, mlp_y_g.shape)  # torch.Size([62500]) torch.Size([62500]) torch.Size([62500])

    column_item, matrix = coupled_Polynomial_functions(xi, xj)
    # column_item, matrix = polynomial_functions(xi, xj)

    X_all = matrix.cpu().detach().numpy()
    y_all = mlp_y_g.cpu().detach().numpy()
    # y_all = true_y_g.reshape(-1).cpu().numpy()
    # reg1 = LassoCV(cv=4, fit_intercept=False, n_jobs=-1, max_iter=10000, normalize=False).fit(X_all, y_all)
    reg1 = Lasso(alpha=1.7, normalize=True, fit_intercept=False).fit(X_all, y_all)
    print(reg1.score(X_all, y_all))
    # print('Best threshold: %.3f' % reg1.alpha_)
    print(column_item)
    print(reg1.coef_)
    print(reg1.intercept_)

def coupled_Polynomial_functions(xi, xj):
    column_values = ['x1j', 'x1ix1j', 'x1jMinusx1i', 'x1jAddx1i']
    tmp_constant = torch.ones_like(xi)
    tmp_x1j = xj
    tmp_x1ix1j = xi * xj
    tmp_x1jMinusx1j = xj - xi
    tmp_x1jAddx1i = xj + xi
    matrix = torch.stack((tmp_x1j, tmp_x1ix1j, tmp_x1jMinusx1j, tmp_x1jAddx1i), 1)
    return column_values, matrix

def polynomial_functions(xi, xj):
    column_values = ['x1i', 'x1j', 'x1i^2', 'x1j^2']
    tmp_x1i = xi
    tmp_x1j = xj
    tmp_x2i = xi ** 2
    tmp_x2j = xj ** 2
    # print(tmp_x1i.shape, tmp_x1j.shape, tmp_x2i.shape, tmp_x2j.shape)
    matrix = torch.stack((tmp_x1i, tmp_x1j, tmp_x2i, tmp_x2j), 1)
    # print(matrix.shape)
    return column_values, matrix


if __name__ == '__main__':
    file_path = r'results/HeatDynamics/community/result_HeatDynamics_community_0128-042640_0.pth'

    lasso_heat_G(file_path)
