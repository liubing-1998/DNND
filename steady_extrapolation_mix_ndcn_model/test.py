from matplotlib import pyplot as plt, font_manager as fm
import numpy as np
from matplotlib.ticker import NullFormatter, FixedLocator


def forward(x):
    x = 1 / (frac_b - x)
    return x


def inverse(x):
    x = frac_b - 1 / x
    return x


data = [  0.9733,   0.    ,   0.0566,  -9.654 ,   0.1291,  -0.0926,  -0.0661,  -2.3085,   0.    , -10.63  ,   0.,  -3.797 ,-7.592 ,   0.    ]
x = np.arange(14)
plt.rc('font', family='Times New Roman', size=15)
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=20
)
font_text = {'size': 20}
yticks = [-11, -2.0, -0.5, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

colors = '#ff9999'
ylims = [-2000, 1.01]
bar_width = 0.4
frac_b = 1.6
text_skip = 0.03 # 标注的数据与柱状图顶（底）端间距
fig, ax = plt.subplots(tight_layout=True)


ax.bar(x, data[:, ind + 1], facecolor=colors[ind], width=bar_width)
ax.set_xticks(x)
ax.set_xlabel('Model No.', labelpad=18, fontdict=font_text)
ax.set_ylabel(r'$R^2$', fontproperties=font_formula)
ax.set_yscale('function', functions=(forward, inverse))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator(yticks[ind]))
ax.set_ylim(ylims[ind])
# 标注数据
for i in range(14):
    cur_r2 = data[i, ind + 1]
    cur_skip = frac_b - cur_r2 - 1 / (text_skip + 1 / (frac_b - cur_r2)) # 实际间距与图上间距转换
    if cur_r2 > 0:
        ax.text(x[i], cur_r2 + cur_skip, f'{cur_r2:.4}', ha='center')
    elif cur_r2 == 0:
        ax.text(x[i], cur_r2 + cur_skip, 'Divergence' if i == 8 else 'Unfitted', ha='center')
    else:
        ax.text(x[i], cur_r2 - cur_skip, f'{cur_r2:.4}', ha='center', va='top')
fig.set_size_inches([15.36, 7.57])