# coding:utf-8

import os
import sys
import logging
import random
import mpl_toolkits.axisartist as axisartist
import matplotlib as mpl

from neural_dynamics import *
from dynamics_model import HeatDynamics, BiochemicalDynamics, BirthDeathDynamics, EpidemicDynamics
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def steady_extrapolation_heat_result(filename, T, time_tick, n, init_random):
    logging.info("============================================")
    pic_path = '_'.join(strtmp for strtmp in filename.split('/')[3:]).split('.')[0] + "_extra_" + str(T) + "_" + str(time_tick)
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

    # # Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
    n = n  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20

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

    # Initial Value
    if init_random:
        x0 = 25 * torch.rand(N, N)
    else:
        x0 = torch.zeros(N, N)
        x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case

    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)

    # equally-sampled time
    sampled_time = 'equal'
    # sampled_time = 'irregular'
    logging.info("T=" + str(T))
    logging.info("time_tick=" + str(time_tick))
    if sampled_time == 'equal':
        print('Build Equally-sampled -time dynamics')
        t = torch.linspace(0., T, time_tick+1)  # time_tick) # 100 vector

    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDynamics(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        logging.info("choice HeatDynamics")
        print(solution_numerical.shape)
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)  # 400 * 1

    criterion = F.l1_loss
    pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
    loss = criterion(pred_y, true_y)  # [:, id_]
    relative_loss = criterion(pred_y, true_y) / true_y.mean()

    print('RESULT Test Loss {:.6f}({:.6f} Relative))'.format(loss.item(), relative_loss.item()))
    logging.info('RESULT Test Loss {:.6f}({:.6f} Relative)'.format(loss.item(), relative_loss.item()))

    x = t
    predY = pred_y[1].cpu().detach().numpy()
    trueY = true_y[1].cpu().detach().numpy()

    np.savez(r'.\\steady_extrapolation_draw_equal\\' + 'model_true_pred_t_50_100', model_predY=predY, model_trueY=trueY, model_t=t.numpy())

    plt.rc('font', family='Times New Roman', size=16)
    plt.rc('lines', linewidth=2)

    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)
    ax.plot(x, trueY, color="red", linestyle="solid", label='$True \enspace of \enspace x_{i}$')
    ax.plot(x, predY, color="blue", linestyle="dashed", label='$Predict \enspace of \enspace x_{i}$')
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$x_{i}$")
    ax.axis["x"].label.set_size(16)
    ax.axis["y"].label.set_size(16)
    ax.legend(loc="best")
    pic_path = r'.\\steady_extrapolation_draw_equal\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_one_node"
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

    true_y_all = true_y[0:50].cpu().detach().numpy()
    pred_y_all = pred_y[0:50].cpu().detach().numpy()

    fig = plt.figure()
    ax0 = fig.add_subplot(111, projection='3d')
    zmin = true_y[:, 0].min()
    zmax = true_y[:, 0].max()
    X = t
    Y = np.arange(50)
    X, Y = np.meshgrid(X, Y)
    surf = ax0.plot_surface(X, Y, true_y_all, cmap='rainbow', linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$i$")
    ax0.set_zlabel("$x_{i}(t)$")
    pic_path = r'.\\steady_extrapolation_draw_equal\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_50_node_true"
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    surf2 = ax1.plot_surface(X, Y, pred_y_all, cmap='rainbow', linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$i$")
    ax1.set_zlabel("$x_{i}(t)$")
    pic_path = r'.\\steady_extrapolation_draw_equal\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_50_node_pred"
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close()

    y = np.arange(0, 400, 1)
    x = t.numpy()
    X, Y = np.meshgrid(x, y)
    Z = true_y.cpu().detach().numpy() - pred_y.cpu().detach().numpy()
    fig, ax = plt.subplots()
    levels = np.arange(Z.min(), Z.max(), (Z.max() - Z.min()) / 200)
    ctf = ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.bwr)
    ax.set_ylabel("$x_{i}$", fontsize=16)
    ax.set_xlabel("$t$", fontsize=16)
    cb = fig.colorbar(ctf, ax=ax)
    pic_path = r'.\\steady_extrapolation_draw_equal\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_difference_value"
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close()



if __name__ == '__main__':
    if (not os.path.exists(r'.\steady_extrapolation_draw_equal')):
        makedirs(r'.\steady_extrapolation_draw_equal')

    log_filename = r'steady_extrapolation_draw_equal/heat_extrapolation_draw.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'.\results/HeatDynamics/grid/result_HeatDynamics_grid_0128-011809_1.pth'
    steady_extrapolation_heat_result(filename, T=50, time_tick=100, n=400, init_random=False)

