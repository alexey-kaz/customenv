import os
import sys
import numpy as np
import pandas as pd


class DataGen:
    def __init__(self):
        self.max_cpu_usage = 5
        self.max_run_time = 20

        if len(sys.argv) == 3:
            self.num_agents = int(sys.argv[1])
            self.num_steps = int(sys.argv[2])
        else:
            self.num_agents = 5
            self.num_steps = 500
        self.min_route = 1
        self.max_route = 4
        self.min_queue = 10
        self.max_queue = 25
        self.min_cpu = 50
        self.max_cpu = 80
        self.gen_agents = [0, 1, 2]

    def gen_relations(self):
        b = np.random.randint(self.min_route, self.max_route, size=(self.num_agents, self.num_agents))
        res = (b + b.T) // 2
        np.fill_diagonal(res, 0)
        return res

    def gen_cd_info(self):
        c1 = np.random.randint(self.min_queue, self.max_queue, size=self.num_agents)
        c2 = np.random.randint(self.min_cpu, self.max_cpu, size=self.num_agents)
        return pd.DataFrame({'queue_avail': c1, 'queue_max': c1, 'cpu_avail': c2, 'cpu_max': c2})

    def gen_tasks(self):
        tmp_del = 4
        c1 = np.random.randint(0, self.num_steps/2, size=self.num_steps // tmp_del * self.num_agents)
        # c2 = np.random.randint(0, self.num_agents, size=self.num_steps // tmp_del * self.num_agents)
        c2 = np.random.choice(self.gen_agents, size=self.num_steps // tmp_del * self.num_agents)
        c3 = np.random.randint(self.num_steps/20, self.num_steps, size=self.num_steps // tmp_del * self.num_agents)
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


if not os.path.exists('./data'):
    os.makedirs('./data')
datagen = DataGen()
tasks_df = datagen.gen_tasks()
relations = datagen.gen_relations()
cd_info = datagen.gen_cd_info()
tasks_df.to_csv(r'./data/tasks_df_5_500.csv', index=False)
cd_info.to_csv(r'./data/cd_info_5_500.csv', index=False)
np.save('./data/relations_5_500.npy', relations)
