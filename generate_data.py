import os
import numpy as np
import pandas as pd
from scipy.stats import erlang


class DataGen:
    def __init__(self, rcv_v, num_steps=500, num_agents=5, max_run_time=10, avg_gap=5, distribution='uniform'):
        self.rcv_vec = rcv_v
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.max_run_time = max_run_time
        self.max_cpu_usage = 5
        self.min_route = 1
        self.max_route = 4
        self.min_queue = 10
        self.max_queue = 25
        self.min_cpu = 100
        self.max_cpu = 200
        self.distribution = distribution
        self.avg_gap = avg_gap

    def gen_relations(self):
        b = np.random.randint(self.min_route, self.max_route, size=(self.num_agents, self.num_agents))
        res = (b + b.T) // 2
        np.fill_diagonal(res, 0)
        return res

    def gen_cd_info(self):
        c1 = np.random.randint(self.min_queue, self.max_queue, size=self.num_agents)
        c2 = np.random.randint(self.min_cpu, self.max_cpu, size=self.num_agents)
        return pd.DataFrame({'queue_avail': c1, 'queue_max': c1, 'cpu_avail': c2, 'cpu_max': c2})

    def gen_tasks_distribution(self):
        def generate_task(last_time):
            duration = 1
            if self.distribution == 'uniform':
                duration = int(np.random.uniform(0, self.avg_gap * 2 - 1) + 1)
            elif self.distribution == 'poisson':
                duration = int(np.random.poisson(lam=self.avg_gap - 1) + 1)
            elif self.distribution == 'erlang':
                duration = int(erlang.rvs(1, scale=self.avg_gap - 1) + 1)
            end_time = last_time + duration
            return end_time

        def generate_single_resource_schedule(length, avg_spread):
            tasks = []
            last_time = np.random.poisson(lam=avg_spread)
            while last_time < length:
                task_end_time = generate_task(last_time)
                if task_end_time < length:
                    tasks.append(task_end_time)
                    last_time = task_end_time + np.random.poisson(lam=avg_spread)
                else:
                    break
            return tasks

        def generate_random_schedule_list(avg_spread):
            return [
                generate_single_resource_schedule(self.num_steps * 0.8, avg_spread=avg_spread)
                for _ in self.rcv_vec
            ]
        lst_sched = generate_random_schedule_list(1)
        column_names = ['time', 'rcv']
        df = pd.DataFrame(columns=column_names)
        for i, j in enumerate(self.rcv_vec):
            tmp_df = pd.DataFrame({'time': lst_sched[i]})
            tmp_df['rcv'] = j
            df = pd.concat([df, tmp_df], ignore_index=True)
        df['life_time'] = np.random.randint(self.max_run_time*2, self.max_run_time*4, size=df.shape[0])
        df['life_time_global'] = df['time'] + df['life_time']
        df['run_time_vec'] = np.random.randint(1, self.max_run_time, size=(df.shape[0], self.num_agents)).tolist()
        df['run_time'] = df.apply(lambda x: x.get(key='run_time_vec')[x.get('rcv')], axis=1)
        df['cpu_usage_vec'] = np.random.randint(1, self.max_cpu_usage, size=(df.shape[0], self.num_agents)).tolist()
        df['cpu_usage'] = df.apply(lambda x: x.get(key='cpu_usage_vec')[x.get('rcv')], axis=1)
        df = df.sort_values(by=['time', 'rcv'], ignore_index=True)
        df['snd'] = -1
        df['queued'] = False
        df['active'] = False
        df['time_activated'] = None
        return df


if not os.path.exists('./data'):
    os.makedirs('./data')

num_agents_vec = [5, 10, 15, 20]
num_steps_vec = [500]

for distr in ['poisson', 'uniform']:
    if not os.path.exists('./data/{}'.format(distr)):
        os.makedirs('./data/{}'.format(distr))
    for i in num_steps_vec:
        for j in num_agents_vec:
            for k in list(range(5, j+1, 2)):
                print(j, k)
                if not os.path.exists('./data/{}/{}_{}_{}'.format(distr, i, j, k)):
                    os.makedirs('./data/{}/{}_{}_{}'.format(distr, i, j, k))
                rcv_vec = np.random.choice(range(j), k, replace=False)
                for m in ['Train']:
                    datagen = DataGen(rcv_vec, i, j, i // 20, 2, distr)
                    tasks_df = datagen.gen_tasks_distribution()
                    relations = datagen.gen_relations()
                    cd_info = datagen.gen_cd_info()
                    tasks_df.to_csv(r'./data/{}/{}_{}_{}/{}_tasks_df.csv'.format(distr, i, j, k, m), index=False)
                    cd_info.to_csv(r'./data/{}/{}_{}_{}/{}_cd_info.csv'.format(distr, i, j, k, m), index=False)
                    np.save('./data/{}/{}_{}_{}/{}_relations.npy'.format(distr, i, j, k, m), relations)
