# coding:utf-8

import torch
import torch.nn as nn

class HeatDynamics(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self, A, k=0.1):
        super(HeatDynamics, self).__init__()
        self.L = -(torch.diag(A.sum(1)) - A)  # Diffusion operator
        self.k = k   # heat capacity  1

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f


class BiochemicalDynamics(nn.Module):  # MAK first
    def __init__(self, A):
        super(BiochemicalDynamics, self).__init__()
        self.A = A
        self.f = 1
        self.b = 0.1
        self.r = 0.01
    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = f-b*xi - \sum_{j=1}^{N}Aij r*xi *xj
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        f = self.f - self.b * x
        outer = torch.mm(self.A, self.r * torch.mm(x, x.t()))
        outer = torch.diag(outer).view(-1, 1)
        f -= outer
        return f

class BirthDeathDynamics(nn.Module):  # PD second
    def __init__(self, A):
        super(BirthDeathDynamics, self).__init__()
        self.A = A
        self.B = 0.1
        self.R = 0.2
        self.b = 2
        self.a = 1
    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -B*xi^b - \sum_{j=1}^{N}Aij R*xi^a
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        f = -self.B * (x ** self.b) + torch.mm(self.A, self.R * (x ** self.a))
        return f


class EpidemicDynamics(nn.Module):
    def __init__(self, A):
        super(EpidemicDynamics, self).__init__()
        self.A = A
        self.b = 2
        self.r = 0.1

    def forward(self, t, x):
        f = -self.b * x
        outer = torch.mm(self.A, self.r * torch.mm((1 - x), x.t()))
        outer = torch.diag(outer).view(-1, 1)
        f = f + outer
        return outer

