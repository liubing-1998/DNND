# coding:utf-8

import os.path
import sys
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq as ode
from utils import *


class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, activation_function, hidden_A_list, hidden_list, A):
        super(ODEFunc, self).__init__()
        self.hidden_A_list = hidden_A_list
        self.hidden_list = hidden_list
        self.A = A

        self.MLP_A = nn.Sequential()
        self.MLP = nn.Sequential()

        self.activation_function = activation_function

        for id in range(len(self.hidden_A_list)-1):
            self.MLP_A.add_module("Linear_layer_%d" % id, nn.Linear(self.hidden_A_list[id], self.hidden_A_list[id+1]))
            if id < len(self.hidden_A_list)-2:
                if self.activation_function == "ReLU":
                    self.MLP_A.add_module("Relu_layer_%d" % id, nn.ReLU())
                elif self.activation_function == "Tanh":
                    self.MLP_A.add_module("Tanh_layer_%d" % id, nn.Tanh())

        for id in range(len(self.hidden_list)-1):
            self.MLP.add_module("Linear_layer_%d" % id, nn.Linear(self.hidden_list[id], self.hidden_list[id+1]))
            if id < len(self.hidden_list)-2:
                self.MLP.add_module("Relu_layer_%d" % id, nn.ReLU())

        self.MLP_A.apply(self.init_weights)
        self.MLP.apply(self.init_weights)

        self.linear_A = nn.Linear(2, 1, bias=False)
        self.linear_A.apply(self.init_weights)
        self.linear_f = nn.Linear(1, 1, bias=False)
        self.linear_f.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """

        if self.hidden_list[0] == 0:
            x_N = torch.zeros_like(x)
        else:
            x_N = self.MLP(x)
            x_N += self.linear_f(x)

        n, d = x.shape
        x1 = x.repeat(1, n)
        x2 = x.T.repeat(n, 1)
        x_a = torch.stack((x1, x2), dim=-1)
        x_e = torch.squeeze(self.MLP_A(x_a))
        x_1_2 = torch.squeeze(self.linear_A(x_a))
        x_e = x_e + x_1_2
        x_A = torch.mul(self.A, x_e)
        x_A = torch.sum(x_A, dim=1)
        x_A = torch.unsqueeze(x_A, dim=-1)
        return x_N + x_A


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt, :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint  # false
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        return out[-1] if self.terminal else out  # 100 * 400 * 10


class DNND(nn.Module):  # myModel
    def __init__(self, activation_function, input_size, hidden_A_list, hidden_list, A, rtol=.01, atol=.001, method='dopri5'):
        super(DNND, self).__init__()
        self.input_size = input_size
        self.hidden_A_list = hidden_A_list
        self.hidden_list = hidden_list
        self.A = A  # N_node * N_node

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.neural_dynamic_layer = ODEBlock(ODEFunc(activation_function, hidden_A_list, hidden_list, A), rtol=rtol, atol=atol, method=method)  # t is like  continuous depth

    def forward(self, vt, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        hvx = self.neural_dynamic_layer(vt, x)
        return hvx
