import os
import sys
import numpy as np
import pandas as pd

if len(sys.argv) == 3:
    num_agents = int(sys.argv[1])
    num_steps = int(sys.argv[2])
else:
    num_agents = 5
    num_steps = 500


def gen_relations(min_route, max_route):
    b = np.random.randint(min_route, max_route, size=(num_agents, num_agents))
    res = (b + b.T) // 2
    np.fill_diagonal(res, 0)
    return res


def gen_cd_info(min_queue, max_queue, min_cpu, max_cpu):
    c1 = np.random.randint(min_queue, max_queue, size=num_agents)
    c2 = np.random.randint(min_cpu, max_cpu, size=num_agents)
    return pd.DataFrame({'queue_avail': c1, 'queue_max': c1, 'cpu_avail': c2, 'cpu_max': c2})


def gen_tasks(max_run_time, max_cpu_usage):
    tmp_del = 4
    c1 = np.random.randint(0, num_steps, size=num_steps // tmp_del * num_agents)
    c2 = np.full(num_steps // tmp_del * num_agents, -1)
    c3 = np.random.randint(0, num_agents, size=num_steps // tmp_del * num_agents)
    c4 = np.random.randint(1, max_run_time, size=num_steps // tmp_del * num_agents)
    c5 = np.random.randint(max_run_time, num_steps, size=num_steps // tmp_del * num_agents)
    c6 = np.random.randint(1, max_cpu_usage, size=num_steps // tmp_del * num_agents)
    df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'run_time': c4, 'life_time': c5, 'cpu_usage': c6})
    df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'snd', 'rcv'], ignore_index=True)
    df['queued'] = False
    df['active'] = False
    df['ttl'] = 0
    return df


if not os.path.exists('./data'):
    os.makedirs('./data')
tasks_df = gen_tasks(5, 20)
relations = gen_relations(1, 4)
cd_info = gen_cd_info(10, 25, 50, 80)
tasks_df.to_csv(r'./data/tasks_df_5_500.csv', index=False)
cd_info.to_csv(r'./data/cd_info_5_500.csv', index=False)
np.save('./data/relations_5_500.npy', relations)
