import os
import numpy as np
import scipy
import pandas as pd


class DataGen:
    def __init__(self, rcv_v, num_steps=500, num_agents=5, max_run_time=20):
        self.rcv_vec = rcv_v
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.max_run_time = max_run_time
        self.max_cpu_usage = 5
        self.min_route = 1
        self.max_route = 4
        self.min_queue = 10
        self.max_queue = 25
        self.min_cpu = 50
        self.max_cpu = 80

    def gen_relations(self):
        b = np.random.randint(self.min_route, self.max_route, size=(self.num_agents, self.num_agents))
        res = (b + b.T) // 2
        np.fill_diagonal(res, 0)
        return res

    def gen_cd_info(self):
        c1 = np.random.randint(self.min_queue, self.max_queue, size=self.num_agents)
        c2 = np.random.randint(self.min_cpu, self.max_cpu, size=self.num_agents)
        return pd.DataFrame({'queue_avail': c1, 'queue_max': c1, 'cpu_avail': c2, 'cpu_max': c2})

    def gen_tasks_random(self):
        tmp_del = 4
        c1 = np.random.randint(0, self.num_steps // 1.5, size=self.num_steps // tmp_del * self.num_agents)
        c2 = np.random.choice(self.rcv_vec, size=self.num_steps // tmp_del * self.num_agents)
        c3 = np.random.randint(self.max_run_time, self.num_steps, size=self.num_steps // tmp_del * self.num_agents)
        df = pd.DataFrame({'time': c1, 'rcv': c2, 'life_time': c3, 'life_time_global': c1 + c3})
        df['run_time_vec'] = np.random.randint(1, self.max_run_time,
                                               size=(self.num_steps // tmp_del * self.num_agents,
                                                     self.num_agents)).tolist()
        df['run_time'] = df.apply(lambda x: x.get(key='run_time_vec')[x.get('rcv')], axis=1)
        df['cpu_usage_vec'] = np.random.randint(1, self.max_cpu_usage,
                                                size=(self.num_steps // tmp_del * self.num_agents,
                                                      self.num_agents)).tolist()
        df['cpu_usage'] = df.apply(lambda x: x.get(key='cpu_usage_vec')[x.get('rcv')], axis=1)
        df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'rcv'], ignore_index=True)
        df['snd'] = -1
        df['queued'] = False
        df['active'] = False
        df['ttl'] = 0
        df['time_activated'] = None
        return df

    def gen_tasks_poisson(self):
        def generate_task(last_time, avg_gap):
            duration = np.random.poisson(lam=avg_gap - 1) + 1
            end_time = last_time + duration
            return end_time

        def generate_single_resource_schedule(length, avg_spread, avg_gap):
            tasks = []
            last_time = np.random.poisson(avg_spread)
            while last_time < length:
                task = generate_task(last_time, avg_gap)
                if task < length:
                    tasks.append(task)
                    last_time = task + np.random.poisson(lam=avg_spread)
                else:
                    break
            return tasks

        def generate_random_schedule_list(avg_spread, avg_gap):
            return [
                generate_single_resource_schedule(self.num_steps*0.8, avg_spread=avg_spread, avg_gap=avg_gap)
                for _ in range(self.num_agents)
            ]

        lst_sched = generate_random_schedule_list(1, 4)
        column_names = ['time', 'rcv']
        df = pd.DataFrame(columns=column_names)
        for i in range(self.num_agents):
            tmp_df = pd.DataFrame({'time': lst_sched[i]})
            tmp_df['rcv'] = i
            df = pd.concat([df, tmp_df], ignore_index=True)
        df['life_time'] = np.random.randint(self.max_run_time, self.num_steps, size=df.shape[0])
        df['life_time_global'] = df['time'] + df['life_time']
        df['run_time_vec'] = np.random.randint(1, self.max_run_time, size=(df.shape[0], 5)).tolist()
        df['run_time'] = df.apply(lambda x: x.get(key='run_time_vec')[x.get('rcv')], axis=1)
        df['cpu_usage_vec'] = np.random.randint(1, self.max_cpu_usage, size=(df.shape[0], 5)).tolist()
        df['cpu_usage'] = df.apply(lambda x: x.get(key='cpu_usage_vec')[x.get('rcv')], axis=1)
        df = df.sort_values(by=['time', 'rcv'], ignore_index=True)
        df['snd'] = -1
        df['queued'] = False
        df['active'] = False
        df['ttl'] = 0
        df['time_activated'] = None
        return df


if not os.path.exists('./data'):
    os.makedirs('./data')

num_agents_vec = [5, 10, 15]
num_steps_vec = [250 * 2 ** (i + 1) for i in range(3)]

for i in num_steps_vec:
    for j in num_agents_vec:
        for k in np.linspace(j // 2, j, 3, dtype=int):
            os.makedirs('./data/{}_{}_{}'.format(i, j, k))
            rcv_vec = np.random.choice(range(j), k, replace=False)
            for m in ['Train', 'Test']:
                datagen = DataGen(rcv_vec, i, j, i / 10)
                tasks_df = datagen.gen_tasks_random()
                relations = datagen.gen_relations()
                cd_info = datagen.gen_cd_info()
                tasks_df.to_csv(r'./data/{}_{}_{}/{}_tasks_df.csv'.format(i, j, k, m), index=False)
                cd_info.to_csv(r'./data/{}_{}_{}/{}_cd_info.csv'.format(i, j, k, m), index=False)
                np.save('./data/{}_{}_{}/{}_relations.npy'.format(i, j, k, m), relations)
