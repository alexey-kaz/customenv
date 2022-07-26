from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ray.tune import ExperimentAnalysis


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
    return df


class Viz:
    def __init__(self, exp_name, num_steps):
        self.exp_name = exp_name
        self.num_steps = num_steps

    def plot_drop(self, mode='Train'):
        path = './exp_res/{}/{}/'.format(self.exp_name, mode)
        all_vec = []
        for txt_path in Path(path).glob("drop*"):
            all_vec.append(np.load(txt_path))
        all_vec = np.asarray(all_vec)
        maxT = np.asarray([np.max(i) for i in all_vec.T]) / self.num_steps * 0.9
        minT = np.asarray([np.min(i) for i in all_vec.T]) / self.num_steps * 0.9
        return maxT, minT

    def plot_mean_reward(self):
        path = './exp_res/{}'.format(self.exp_name)
        analysis = ExperimentAnalysis(path)
        all_vec = []
        for i in analysis.trial_dataframes.values():
            episode_reward_mean = i.episode_reward_mean.to_list()
            all_vec.append(episode_reward_mean)
        all_vec = np.asarray(all_vec)
        maxT = np.asarray([np.max(i) for i in all_vec.T])
        minT = np.asarray([np.min(i) for i in all_vec.T])
        return maxT, minT

    def plot_anything(self, type_plot):
        if type_plot == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            maxT, minT = self.plot_drop()
            mean_vec = (minT + maxT) / 2
            x = [(i + 1) for i in range(len(maxT))]
            ax1.fill_between(x, minT, maxT, alpha=0.4)
            ax1.plot(x, mean_vec)
            ax1.set_xlabel('Эпизод')
            ax1.set_ylabel('Отношение просроченных заданий к общему числу заданий')
            ax1.legend(['Среднее значение', 'Вариация'])
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            maxT, minT = self.plot_mean_reward()
            mean_vec = (minT + maxT) / 2
            ax2.fill_between(x, minT, maxT, alpha=0.4)
            ax2.plot(x, mean_vec)
            ax2.set_xlabel('Эпизод')
            ax2.set_ylabel('Значение целевой функции')
            ax2.legend(['Среднее значение', 'Вариация'])
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig('./exp_res/{}/both_variance.png'.format(self.exp_name), dpi=300)
        else:
            if type_plot == 'reward':
                maxT, minT = self.plot_mean_reward()
                mean_vec = (minT + maxT) / 2
                x = [(i + 1) for i in range(len(maxT))]
                y_label = 'Значение целевой функции'
                fig_name = 'mean_rew_variance'
            else:
                maxT, minT = self.plot_drop('Train')
                mean_vec = (minT + maxT) / 2
                x = [(i + 1) for i in range(len(maxT))]
                y_label = 'Отношение просроченных заданий к общему числу заданий'
                fig_name = 'drop_variance'
            fig, ax = plt.subplots()
            ax.fill_between(x, minT, maxT, alpha=0.4)
            ax.plot(x, mean_vec)
            ax.xlabel('Эпизод')
            ax.ylabel(y_label)
            ax.legend(['Среднее значение', 'Вариация'])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.savefig('./exp_res/{}/{}.png'.format(self.exp_name, fig_name), dpi=300)
            plt.cla()

    def plot_all(self):
        self.plot_anything('both')
        self.plot_anything('reward')
        self.plot_anything('mean')

