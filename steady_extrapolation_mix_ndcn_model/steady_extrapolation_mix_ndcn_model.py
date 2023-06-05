# coding:utf-8
import array
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axisartist
from matplotlib.ticker import NullFormatter, FixedLocator
import math

model_true_pred_t = np.load(r'E:\code_and_data_package\PaperExperimentation20221216D\HeatDynamicsD\steady_extrapolation_draw_equal\model_true_pred_t_5_100.npz')
ndcn_true_pred_t = np.load(r'E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD\steady_extrapolation_draw_equal_heat\NDCN_true_pred_t_5_100.npz')

ndcn_predY = ndcn_true_pred_t['ndcn_predY']
ndcn_trueY = ndcn_true_pred_t['ndcn_trueY']
ndcn_t = ndcn_true_pred_t['ndcn_t']

model_predY = model_true_pred_t['model_predY']
model_trueY = model_true_pred_t['model_trueY']
model_t = model_true_pred_t['model_t']

# 设置全局字体，字体大小（好像只对text生效）
plt.rc('font', family='Times New Roman', size=24)
plt.rc('lines', linewidth=2)  # 设置全局线宽

fig = plt.figure()
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
ax.axis[:].set_visible(False)
# ax.axis["x"] = ax.new_floating_axis(0, -1.5)
ax.axis["x"] = ax.new_floating_axis(0, -1000000)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["x"].set_axis_direction('bottom')
ax.axis["y"].set_axis_direction('left')
ax.axis["x"].set_axisline_style("->", size=0.3)
ax.axis["y"].set_axisline_style("->", size=0.3)
# 设置坐标轴范围
ax.set_xlim([0, 52])
ax.set_xticks([i for i in np.arange(0, 51, 10)])
ax.set_ylim([-1000000, 200000])
ax.set_yscale("symlog", base=10)
# ax.set_yscale('function', functions=(forward, inverse))
ax.yaxis.set_minor_formatter(NullFormatter())
# yticks = [-1000000, -100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000, ]
yticks = [-100000, -1000, -10, 0, 10, 1000, 100000]
ax.yaxis.set_major_locator(FixedLocator(yticks))
# ax.plot(ndcn_t, ndcn_trueY, color="#1f77b4", linestyle="solid", label='$True \enspace of \enspace x_{1}$')  # , alpha=0.7
# ax.plot(ndcn_t, ndcn_predY, color="#2ca02c", linestyle="dashed", label='$NDCN \enspace Predict \enspace of \enspace x_{1}$')  # dashed
# # ax.plot(model_t, model_trueY, color="#d62728", linestyle="solid", label='$True \enspace of \enspace x_{i}$')
# ax.plot(model_t, model_predY, color="#d62728", linestyle="dashed", label='$DNND \enspace Predict \enspace of \enspace x_{1}$')
ax.plot(ndcn_t, ndcn_trueY, color="#1f77b4", linestyle="solid", label='$True$')  # , alpha=0.7
ax.plot(ndcn_t, ndcn_predY, color="#2ca02c", linestyle="dashed", label='$NDCN$')  # dashed
ax.plot(model_t, model_predY, color="#d62728", linestyle="dashed", label='$DNND$')
# #ff7f0e #2ca02c
ax.axis["x"].label.set_text(r"$t$")
ax.axis["y"].label.set_text(r"$log(x_{1})$")
# ax.axis["x"].label.set_size(20)
# ax.axis["y"].label.set_size(20)
ax.legend(loc="best", prop={'size': 19})
pic_path = "steady_extrapolation_mix_ndcn_model_head_grid_D1_5_100"
plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
plt.close(fig)