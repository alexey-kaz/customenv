import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Error": d - f})
    return df


def plot_drop(exp_name, num_steps, num_agents, num_rcv, mode='Train'):
    path = "./exp_res/{}/{}/".format(exp_name, mode)
    path_drop = path + 'drop_{}_{}_{}'.format(num_steps, num_agents, num_rcv)
    drop_vec = np.load(path_drop + '.npy')
    x = [num_steps * (i + 1) for i in range(len(drop_vec))]
    plt.plot(x, drop_vec, alpha=0.1)
    plt.plot(x, simple_exp_smooth(drop_vec, 1, 0.3).Forecast[1:])
    plt.title('{} шагов в итерации, {} агентов, {} принимающих'.format(num_steps, num_agents,
                                                                       num_rcv))
    plt.xlabel('Число шагов')
    plt.ylabel('Число дропов')
    plt.legend(['Без сглаживания', 'Экспоненциальное сглаживание'])
    plt.savefig(path_drop + '.png', dpi=300)


def plot_drop_double(exp_name, num_steps, num_agents, num_rcv):
    path_train = "./exp_res/{}/{}/".format(exp_name, 'Train')
    path_drop_train = path_train + 'drop_{}_{}_{}'.format(num_steps, num_agents, num_rcv)
    drop_vec_train = np.load(path_drop_train + '.npy')
    x = [num_steps * (i + 1) for i in range(len(drop_vec_train))]
    plt.plot(x, drop_vec_train, "g")
    path_test = "./exp_res/{}/{}/".format(exp_name, 'Test')
    path_drop_test = path_test + 'drop_{}_{}_{}'.format(num_steps, num_agents, num_rcv)
    drop_vec_test = np.load(path_drop_test + '.npy')
    x = [len(drop_vec_train) + num_steps * (i + 1) for i in range(len(drop_vec_test))]
    plt.plot(x, drop_vec_test)
    plt.title('{} шагов в итерации, {} агентов, {} принимающих'.format(num_steps, num_agents,
                                                                       num_rcv))
    plt.xlabel('Число шагов')
    plt.ylabel('Число дропов')
    plt.legend(['Без сглаживания', 'Экспоненциальное сглаживание'])
    plt.savefig("./exp_res/{}/train_test".format(exp_name) + '.png', dpi=300)
