import gym
import numpy as np
import pandas as pd

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Diploma_Env(MultiAgentEnv):
    def __init__(self, num_steps=200, num_agents=5, is_JSSP=False, return_agent_actions=False, part=False):
        super().__init__()
        self.num_steps = num_steps
        self.num_agents = num_agents  # number of computing devices (CDs)
        self.tasks_df = None  # database of all jobs
        self.relations = None  # time to send tasks between CDs
        self.cd_info = None  # CDs queues and cpu usage
        self.is_JSSP = is_JSSP  # make True for all send times between CDs to be 0
        low = np.array([0, 0, 0], dtype=np.float)
        high = np.array([1, 1, 1], dtype=np.float)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float)
        self.action_space = gym.spaces.Discrete(self.num_agents)
        self.time = 0
        self.exp_num = -1
        self.all_cpu_usage = None
        self.all_queue_usage = None

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
        c1 = np.random.randint(self.num_steps, size=self.num_steps // 4 * self.num_agents)
        c2 = np.full(self.num_steps // 4 * self.num_agents, -1)
        c3 = np.random.randint(self.num_agents, size=self.num_steps // 4 * self.num_agents)
        c4 = np.random.randint(1, max_run_time, size=self.num_steps // 4 * self.num_agents)
        c5 = np.random.randint(max_run_time, self.num_steps, size=self.num_steps // 4 * self.num_agents)
        c6 = np.random.randint(max_cpu_usage, self.num_steps, size=self.num_steps // 4 * self.num_agents)
        df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'run_time': c4, 'life_time': c5, 'cpu_usage': c6})
        df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'snd', 'rcv'], ignore_index=True)
        df['queued'] = False
        df['active'] = False
        return df

    def send_task(self, task, rcv):
        row = self.tasks_df.loc[(self.tasks_df.time == task.time.values[0])
                                & (self.tasks_df.snd == task.snd.values[0])
                                & (self.tasks_df.rcv == task.rcv.values[0]), 'time'].index[0]
        col_time = self.tasks_df.columns.get_loc('time')
        col_queued = self.tasks_df.columns.get_loc('queued')
        # new time
        self.tasks_df.iloc[row, col_time] = task.time.values[0] + self.relations[rcv][task.rcv.values[0]]
        # not yet in queue after being sent
        self.tasks_df.iloc[row, col_queued] = False

    def reset(self):
        if not self.time:
            self.tasks_df = self.gen_tasks(5, 10)
            self.relations = self.gen_relations(1, 4)
            self.cd_info = self.gen_cd_info(5, 15, 20, 50)
            self.exp_num += 1
            self.all_cpu_usage = {i: 0 for i in range(self.num_agents)}
            self.all_queue_usage = {i: 0 for i in range(self.num_agents)}
            print('Experiment', self.exp_num)
        obs = {}
        for i in range(self.num_agents):
            obs[i] = np.asarray([0., 0., 0.])
        return obs

    def step(self, action_dict):
        run_time_col_index = self.tasks_df.columns.get_loc('run_time')
        queued_col_index = self.tasks_df.columns.get_loc('queued')
        active_col_index = self.tasks_df.columns.get_loc('active')
        # print('timestep', self.time)
        obs, rew, done, info = {}, {i: 0 for i in range(self.num_agents)}, {}, {}
        num_tasks, tasks_ratio, total_run_time = [], [], []
        # all new tasks from this time step are labeled as queued
        self.tasks_df.loc[(self.tasks_df.time == self.time), 'queued'] = True
        # drop tasks if queue on all agents is full
        if not self.cd_info.queue_avail.any():
            self.tasks_df.drop(self.tasks_df[(self.tasks_df.queued == True)
                                             & (self.tasks_df.time == self.time)], inplace=True)
        for i in range(self.num_agents):
            # if there are queued tasks on i-th agent
            if not self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].empty:
                # if several tasks were sent to i-th agent
                if self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].shape[0] > 1:
                    # choose all new queued except for the first one
                    tmp_queued_time_not_first = self.tasks_df.loc[(self.tasks_df.queued == True)
                                                                  & (self.tasks_df.rcv == i)
                                                                  & (self.tasks_df.time == self.time),
                                                                  'run_time'].index[1:]
                    # move them to next time step
                    self.tasks_df.iloc[tmp_queued_time_not_first, run_time_col_index] = \
                        self.tasks_df.iloc[tmp_queued_time_not_first, run_time_col_index].apply(lambda x: x + 1)
                    # change back their queued status
                    self.tasks_df.iloc[tmp_queued_time_not_first, queued_col_index] = False
                # get the first actually queued task
                tmp_queued_time_first = self.tasks_df[(self.tasks_df.queued == True)
                                                      & (self.tasks_df.rcv == i)].index[0]
                # make it active
                self.tasks_df.iloc[tmp_queued_time_first, active_col_index] = True
                # finished (runtime == 0) active task on i-th agent
                tmp_active_runtime = self.tasks_df[(self.tasks_df.active == True) & (self.tasks_df.rcv == i)
                                                   & (self.tasks_df.run_time == 0)]
                # if there are active finished tasks on i-th agent
                if not tmp_active_runtime.empty:
                    # increase available queue space of i-th agent
                    self.cd_info.queue_avail[i] += 1

                    # # reward for finished task
                    # reward += 100

                    # delete finished task from dataframe
                    self.tasks_df.drop(self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.active == True)
                                                     & (self.tasks_df.run_time == 0)].index[0], inplace=True)

                tmp_queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
                tmp_queued_other = self.tasks_df[(self.tasks_df.rcv == action_dict[i]) & (self.tasks_df.queued == True)]
                # if there are queued tasks on i-th agent
                if not tmp_queued_i.empty:
                    # append action to its own queue
                    if self.cd_info.cpu_avail[action_dict[i]] < tmp_queued_i.cpu_usage.values[0]:
                        rew[i] -= 10
                    elif action_dict[i] == i:
                        # reduce available queue space of i-th agent
                        self.cd_info.queue_avail[i] -= 1

                        # add this task's cpu to i-th agent
                        self.all_cpu_usage[i] += tmp_queued_i.cpu_usage.values[0]
                        # add this task to queue usage of i-th agent
                        self.all_queue_usage[i] += 1

                        # # all queued tasks run_times (except for the new one)
                        # tmp_reward = np.sum(tmp_queued_i.run_time[:-1])
                        # # negative reward if lifetime ends before all tasks in queue are done
                        # reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[-1])

                    # every other number N means "send to N"
                    else:
                        # reduce available queue space of receiving agent
                        self.cd_info.queue_avail[action_dict[i]] -= 1

                        # add this task's cpu to i-th agent
                        self.all_cpu_usage[action_dict[i]] += tmp_queued_i.cpu_usage.values[0]
                        # add this task to queue usage of i-th agent
                        self.all_queue_usage[action_dict[i]] += 1

                        # # all runtimes on new agent
                        # tmp_reward = np.sum(tmp_queued_other.run_time)
                        # # new task's runtime
                        # tmp_reward += tmp_queued_i.run_time.values[-1]
                        # # time to send between agents
                        # tmp_reward += self.relations[i][action_dict[i]]
                        # # negative reward if lifetime ends before task is received and all tasks in new queue are done
                        # reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[-1])

                        # send from i-th agent to chosen by action
                        self.send_task(tmp_queued_i, action_dict[i])
            done[i], info[i] = True, {}
            # rew[i], done[i], info[i] = reward, True, {}

        alpha, beta, gamma = 0.5, 0.5, 0.5
        delta = sum(self.all_cpu_usage.values()) / (self.time + 1)
        theta = sum(self.all_queue_usage.values()) / (self.time + 1)
        for i in range(self.num_agents):
            cpu_avg_i = self.all_cpu_usage[i] / (self.time + 1)
            queue_avg_i = self.all_queue_usage[i] / (self.time + 1)
            rew[i] += alpha * cpu_avg_i / self.cd_info.cpu_avail[i]
            rew[i] += gamma * (cpu_avg_i / self.cd_info.cpu_avail[i] - delta) ** 2
            if self.cd_info.queue_avail[i]:
                rew[i] += beta * queue_avg_i / self.cd_info.queue_avail[i]
                rew[i] += gamma * (queue_avg_i / self.cd_info.queue_avail[i] - theta) ** 2

        for i in range(self.num_agents):
            tmp_queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
            if not tmp_queued_i.empty:
                # number of queued tasks
                num_tasks.append(tmp_queued_i.shape[0])
                # ratio of queued tasks to max_queue
                tmp_task_ratio = tmp_queued_i.shape[0] / self.cd_info.queue_avail[i] if self.cd_info.queue_avail[
                    i] else 0
                tasks_ratio.append(tmp_task_ratio)
                # runtimes of all queued tasks
                total_run_time.append(np.sum(tmp_queued_i.run_time.values))
            else:
                num_tasks.append(0.)
                tasks_ratio.append(0.)
                total_run_time.append(0.)
        # normalize obs from 0 to 1
        if np.max(num_tasks):
            result_num_tasks = (num_tasks - np.min(num_tasks)) / np.ptp(num_tasks)
        else:
            result_num_tasks = [0.] * self.num_agents
        if np.max(tasks_ratio):
            result_tasks_ratio = (tasks_ratio - np.min(tasks_ratio)) / np.ptp(tasks_ratio)
        else:
            result_tasks_ratio = [0.] * self.num_agents
        if np.max(total_run_time):
            result_total_run_time = (total_run_time - np.min(total_run_time)) / np.ptp(total_run_time)
        else:
            result_total_run_time = [0.] * self.num_agents

        # observation (visible state) encoding
        obs = {i: np.asarray(j, dtype=object) for i, j in enumerate(zip(result_num_tasks,
                                                                        result_tasks_ratio, result_total_run_time))}

        done["__all__"] = True

        for i in range(self.num_agents):
            tmp_active_runtime = self.tasks_df.loc[(self.tasks_df.rcv == i)
                                                   & (self.tasks_df.active == True), 'run_time']
            # if there are active tasks on i-th agent
            if not tmp_active_runtime.empty:
                # runtime -1
                self.tasks_df.iloc[tmp_active_runtime.index[0], run_time_col_index] = tmp_active_runtime.values[0] - 1

        mask = (self.tasks_df['queued'] == True)
        tasks_queued = self.tasks_df[mask]
        self.tasks_df.loc[mask, 'life_time'] = tasks_queued['life_time'] - 1
        self.tasks_df.drop(self.tasks_df[self.tasks_df.life_time < 0].index, inplace=True)
        self.tasks_df.reset_index(drop=True, inplace=True)
        self.time = (self.time + 1) % self.num_steps
        available_actions = np.where(self.cd_info.queue_avail > 0, 1, 0)
        return obs, rew, done, info
