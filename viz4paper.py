from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox


def simple_exp_smooth(d, extra_periods=1, alpha=0.4):
    d = np.array(d)  # Transform the input into a numpy array
    cols = len(d)  # Historical period length
    d = np.append(d, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods
    f = np.full(cols + extra_periods, np.nan)  # Forecast array
    f[1] = d[0]  # initialization of first forecast
    # Create all the t+1 forecasts until end of historical period
    for t in range(2, cols + 1):
        f[t] = alpha * d[t - 1] + (1 - alpha) * f[t - 1]
        f[cols + 1:] = f[t]  # Forecast for all extra periods
    df = pd.DataFrame.from_dict({'Demand': d, 'Forecast': f, 'Error': d - f})
    return df['Forecast'][:-1]


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def mean_ext(a):
    b = []
    max_len = max(len(item) for item in a)
    for item in a:
        if len(item) < max_len:
            item[len(a):] += [None] * (max_len - len(item))
        tmp = item.copy()
        b.append(tmp)
    for item in range(len(b)):
        for i in range(len(b[item])):
            if b[item][i] is None:
                b[item][i] = b[(item + 1) % 2][i]
    return [np.mean(i) for i in np.asarray(b, dtype=object).T]


path_poisson = 'poisson'
path_uniform = 'uniform/'
plt.rcParams.update({'font.size': 22, 'figure.facecolor': (1, 1, 1, 1)})

parser = ArgumentParser(description='Визуализация результатов')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

data_dir = args.data_dir if args.data_dir else './exp_res'
save_dir = args.save_dir if args.save_dir else './exp_res_plots'

print('data_dir: {}\nsave_dir: {}'.format(data_dir, save_dir))

p = 0
for path in [path_poisson, path_uniform]:
    dirs_distr = [x for x in Path('./{}/{}'.format(data_dir, path)).iterdir() if x.is_dir()]
    for num_agents_dirs in dirs_distr:
        # print(num_agents_dirs)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(54, 12))
        # fig, ax1 = plt.subplots(1, 1, figsize=(18, 12))
        dirs_edge = [x for x in Path(num_agents_dirs).iterdir() if x.is_dir()]
        # print(sorted(dirs_edge))
        for num_edges_dirs in sorted(dirs_edge, key=lambda x: int(str(x).split('/')[3])):
            all_vec_drop = []
            all_vec_fin = []
            all_vec_rew = []
            exps = [x for x in Path(num_edges_dirs).iterdir() if (x.is_dir() and 'PPO' in str(x))]
            for txt_path in exps:
                p = pd.read_csv(txt_path / 'progress.csv')
                fin_max = p['custom_metrics/finished_max'].to_list()
                drop_max = p['custom_metrics/drop_max'].to_list()
                rew_tmp = (p.episode_reward_mean * -1).to_list()
                all_vec_fin.append(fin_max)
                all_vec_drop.append(drop_max)
                all_vec_rew.append(rew_tmp)
            all_vec_fin = np.asarray(all_vec_fin, dtype=object)
            mean_fin = mean_ext(all_vec_fin)
            all_vec_drop = np.asarray(all_vec_drop, dtype=object)
            mean_drop = mean_ext(all_vec_drop)
            all_vec_rew = np.asarray(all_vec_rew, dtype=object)
            mean_rew = mean_ext(all_vec_rew)

            fig.suptitle(' DISTRIBUTION, '.join(num_agents_dirs.__str__().upper().split('/')[1:]) + ' AGENTS')

            legend = sorted(map(lambda x: '{} Edge Agents'.format(int(str(x).split('/')[3])), dirs_edge))

            x = [(i + 1) for i in range(len(mean_drop))]
            ax1.plot(x, mean_fin)
            ax1.legend(legend)
            ax1.title.set_text('Finished tasks ratio')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Percentage from all tasks')
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.yaxis.set_major_locator(MaxNLocator(11))

            ax2.plot(x, mean_drop)
            ax2.legend(legend)
            ax2.title.set_text('Dropped tasks ratio')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Percentage from all tasks')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.yaxis.set_major_locator(MaxNLocator(11))

            x = [(i + 1) for i in range(len(mean_rew))]
            ax3.plot(x, simple_exp_smooth(mean_rew, alpha=0.8))
            ax3.legend(legend)
            ax3.title.set_text('Objective function')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Objective function')
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

            plots_path = str(num_edges_dirs).split('/')[1:]
            if eval('{} - {}'.format(plots_path[1], plots_path[2])) <= 1:
                extent_ax1 = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig('./{}/{}_{}_{}_fin.png'.format(save_dir, *plots_path),
                            bbox_inches=extent_ax1, dpi=300)
                extent_ax2 = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig('./{}/{}_{}_{}_drop.png'.format(save_dir, *plots_path),
                            bbox_inches=extent_ax2, dpi=300)
                extent_ax3 = full_extent(ax3).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig('./{}/{}_{}_{}_obj.png'.format(save_dir, *plots_path),
                            bbox_inches=extent_ax3, dpi=300)
                plt.savefig('./{}/{}_{}_{}_all_plots.png'.format(save_dir, *plots_path), dpi=300)
