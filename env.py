import gym
import numpy as np
import pandas as pd

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Diploma_Env(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.num_steps = env_config['num_steps']
        self.num_agents = env_config['num_agents']  # number of computing devices (CDs)
        self.is_JSSP = env_config['is_JSSP']  # make True for all send times between CDs to be 0
        self.alpha, self.gamma = env_config['alpha'], env_config['gamma']
        self.state_space_shape = 3
        low = np.array([0] * self.state_space_shape, dtype=np.float)
        high = np.array([1] * self.state_space_shape, dtype=np.float)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float)
        self.action_space = gym.spaces.Discrete(self.num_agents)
        self.time = 0
        self.tasks_df_main = self.gen_tasks(5, 20)
        self.relations_main = self.gen_relations(1, 4)
        self.cd_info_main = self.gen_cd_info(10, 25, 50, 80)
        self.tasks_df = None
        self.relations = None
        self.cd_info = None
        self.all_cpu_usage = None
        self.all_queue_usage = None
        self.exp_num = 0

    def gen_relations(self, min_route, max_route):
        if not self.is_JSSP:
            b = np.random.randint(min_route, max_route, size=(self.num_agents, self.num_agents))
            res = (b + b.T) // 2
            np.fill_diagonal(res, 0)
        else:
            res = np.full((self.num_agents, self.num_agents), 0)
        return res

    def gen_cd_info(self, min_queue, max_queue, min_cpu, max_cpu):
        c1 = np.random.randint(min_queue, max_queue, size=self.num_agents)
        c2 = np.random.randint(min_cpu, max_cpu, size=self.num_agents)
        return pd.DataFrame({'queue_avail': c1, 'queue_max': c1, 'cpu_avail': c2, 'cpu_max': c2})

    def gen_tasks(self, max_run_time, max_cpu_usage):
        tmp_del = 4
        c1 = np.random.randint(0, self.num_steps, size=self.num_steps // tmp_del * self.num_agents)
        c2 = np.full(self.num_steps // tmp_del * self.num_agents, -1)
        c3 = np.random.randint(0, self.num_agents, size=self.num_steps // tmp_del * self.num_agents)
        c4 = np.random.randint(1, max_run_time, size=self.num_steps // tmp_del * self.num_agents)
        c5 = np.random.randint(max_run_time, self.num_steps, size=self.num_steps // tmp_del * self.num_agents)
        c6 = np.random.randint(1, max_cpu_usage, size=self.num_steps // tmp_del * self.num_agents)
        df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'run_time': c4, 'life_time': c5, 'cpu_usage': c6})
        df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'snd', 'rcv'], ignore_index=True)
        df['queued'] = False
        df['active'] = False
        df['ttl'] = 0
        return df

    def send_task(self, task, rcv):
        self.tasks_df.drop(self.tasks_df[(self.tasks_df.time == task.time.values[0])
                                         & (self.tasks_df.snd == task.snd.values[0])
                                         & (self.tasks_df.rcv == task.rcv.values[0])].index, inplace=True)
        # new time
        srs = task.copy(deep=True).squeeze()
        srs['time'] = task.time.values[0] + self.relations[rcv][task.rcv.values[0]] + 1
        # not yet in queue after being sent
        srs['queued'] = False
        # new snd and rcv
        srs['snd'], srs['rcv'] = srs['rcv'], rcv
        srs['ttl'] += 1
        # add new row to tasks_df and sort by time & rcv
        self.tasks_df = pd.concat([self.tasks_df, pd.DataFrame(srs).transpose()], ignore_index=True)
        self.tasks_df.sort_values(by=['time', 'rcv'], inplace=True)
        self.tasks_df.reset_index(drop=True, inplace=True)

    def free_agent(self, n):
        dict_cd = {m: self.cd_info.queue_avail[m] / self.cd_info.queue_max[m]
                   for m in range(self.num_agents) if m != n}
        return max(dict_cd, key=dict_cd.get) if max(dict_cd.values()) else np.random.randint(0, self.num_agents)

    def reset(self):
        if not self.time:
            # print("EXPERIMENT", self.exp_num)
            self.tasks_df = self.tasks_df_main.copy(deep=True)
            self.relations = self.relations_main.copy()
            self.cd_info = self.cd_info_main.copy(deep=True)
            self.all_cpu_usage = {i: 0 for i in range(self.num_agents)}
            self.all_queue_usage = {i: 0 for i in range(self.num_agents)}
            self.exp_num += 1
        state = {}
        for i in range(self.num_agents):
            state[i] = np.asarray([0.] * self.state_space_shape)
        return state

    def step(self, action_dict):
        active_col_index = self.tasks_df.columns.get_loc('active')
        state, rew, done, info = {}, {i: 0 for i in range(self.num_agents)}, {}, {}
        # all new tasks from this time step are labeled as queued
        self.tasks_df.loc[(self.tasks_df.time == self.time), 'queued'] = True
        # drop tasks with tll > 2
        self.tasks_df.drop(self.tasks_df[self.tasks_df.ttl > 2].index, inplace=True)
        self.tasks_df.reset_index(drop=True, inplace=True)
        for i in range(self.num_agents):
            # if there are queued tasks on i-th agent
            if not self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].empty:
                # if several tasks were sent to i-th agent
                if self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].shape[0] > 1:
                    # choose all new queued except for the first one
                    tmp_queued_time_not_first = self.tasks_df[(self.tasks_df.queued == True)
                                                              & (self.tasks_df.rcv == i)
                                                              & (self.tasks_df.time == self.time)].iloc[1:]
                    # move them to next time step
                    for j, k in tmp_queued_time_not_first.iterrows():
                        tmp_free_agent = self.free_agent(i)
                        self.send_task(k.to_frame().transpose(), tmp_free_agent)
                # finished active task on i-th agent
                active_runtime_finished = self.tasks_df[(self.tasks_df.active == True) & (self.tasks_df.rcv == i)
                                                        & (self.tasks_df.run_time + self.tasks_df.time == self.time)]
                # increase available queue and cpu of i-th agent
                self.cd_info.queue_avail[i] += active_runtime_finished.shape[0]
                self.cd_info.cpu_avail[i] += sum(active_runtime_finished.cpu_usage)
                # reward for each finished active task
                rew[i] += active_runtime_finished.shape[0] * 0.5
                # delete finished task from dataframe
                self.tasks_df.drop(active_runtime_finished.index, inplace=True)

                queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)
                                              & (self.tasks_df.time == self.time)]
                # if there are queued tasks on i-th agent
                tmp_queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
                tmp_queued_other = self.tasks_df[(self.tasks_df.rcv == action_dict[i]) & (self.tasks_df.queued == True)]
                if not queued_i.empty:
                    if self.cd_info.queue_avail[action_dict[i]] > 0 and self.cd_info.cpu_avail[action_dict[i]] > queued_i.cpu_usage.values[-1]:
                        # append action to its own queue
                        if action_dict[i] == i:
                            # all queued tasks run_times
                            tmp_reward = np.sum(tmp_queued_i.run_time)
                            # negative reward if lifetime ends before all tasks in queue are done
                            rew[i] += - 1 * (tmp_reward > tmp_queued_i.life_time.values[-1])
                            # reduce available queue space of i-th agent
                            self.cd_info.queue_avail[i] -= 1
                            # add this task to queue usage of i-th agent
                            self.all_queue_usage[i] += 1
                        # every other number N means "send to N"
                        else:
                            # all runtimes on new agent
                            tmp_reward = np.sum(tmp_queued_other.run_time)
                            # new task's runtime
                            tmp_reward += tmp_queued_i.run_time.values[-1]
                            # time to send between agents
                            tmp_reward += self.relations[i][action_dict[i]]
                            # negative reward if lifetime ends before task is received and all tasks in new queue are done
                            rew[i] += -1 * (tmp_reward > tmp_queued_i.life_time.values[-1])
                            # reduce available queue space of receiving agent
                            self.cd_info.queue_avail[action_dict[i]] -= 1
                            # add this task to queue usage of i-th agent
                            self.all_queue_usage[action_dict[i]] += 1
                            # send from i-th agent to chosen by action
                            self.send_task(queued_i, action_dict[i])
                    else:
                        tmp_free_agent = self.free_agent(i)
                        self.send_task(queued_i, tmp_free_agent)
            done[i], info[i] = True, {}
        for i in range(self.num_agents):
            # get all queued tasks of i-th agent (that aren't active)
            queued_i = self.tasks_df[(self.tasks_df.queued == True) & (self.tasks_df.rcv == i)
                                     & (self.tasks_df.active == False)]
            # make queued tasks active on i-th agent until it runs out of cpu
            for j, k in queued_i.iterrows():
                if self.cd_info.cpu_avail[i] < k.cpu_usage:
                    break
                self.tasks_df.iloc[k.name, active_col_index] = True
                self.cd_info.cpu_avail[i] -= k.cpu_usage
                self.all_cpu_usage[i] += k.cpu_usage

        delta = sum(self.all_cpu_usage.values()) / (self.time + 1) / self.num_agents
        num_tasks_state, tasks_ratio_state, total_run_time_state = [], [], []

        for i in range(self.num_agents):

            cpu_avg_i = self.all_cpu_usage[i] / (self.time + 1)
            cpu_div = 0. if not self.cd_info.cpu_avail[i] else cpu_avg_i / self.cd_info.cpu_avail[i]
            rew[i] -= self.alpha * cpu_div + self.gamma * (cpu_div - delta) ** 2

            queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
            if not queued_i.empty:
                # number of queued tasks
                num_tasks_state.append(queued_i.shape[0])
                # ratio of queued tasks to max_queue
                tasks_ratio_state.append(queued_i.shape[0] / self.cd_info.queue_max[i])
                # runtimes of all queued tasks
                total_run_time_state.append(np.sum(queued_i.run_time.values))
            else:
                num_tasks_state.append(0.)
                tasks_ratio_state.append(0.)
                total_run_time_state.append(0.)
        non_normalized_state = [num_tasks_state, tasks_ratio_state, total_run_time_state]
        normalized_state = [None] * len(non_normalized_state)
        # normalize state from 0 to 1
        for i in range(len(normalized_state)):
            if np.max(non_normalized_state[i]):
                normalized_state[i] = (non_normalized_state[i] - np.min(non_normalized_state[i])) / np.ptp(non_normalized_state[i])
            else:
                normalized_state[i] = [0.] * self.num_agents

        # state encoding
        state = {i: np.asarray(j, dtype=object) for i, j in enumerate(zip(*normalized_state))}

        done["__all__"] = True

        self.tasks_df.drop(self.tasks_df[self.tasks_df.time + self.tasks_df.life_time == self.time].index, inplace=True)
        self.time = (self.time + 1) % self.num_steps
        # available_actions = np.where(self.cd_info.queue_avail > 0, 1, 0)
        return state, rew, done, info
