import gym
import numpy as np
import pandas as pd

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Diploma_Env(MultiAgentEnv):
    def __init__(self, return_agent_actions=False, part=False):
        super().__init__()
        self.num_steps = 200
        self.num_agents = 5
        self.tasks_df = None
        self.relations = None
        self.max_queues = None
        self.time = 0
        self.is_JSSP = False  # make True for all send times between agents to be 0
        low = np.array([0, 0], dtype=np.float)
        high = np.array([1, 1], dtype=np.float)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float)
        self.action_space = gym.spaces.Discrete(self.num_agents)

    def gen_relations(self, min_route, max_route):
        if not self.is_JSSP:
            b = np.random.randint(min_route, max_route, size=(self.num_agents, self.num_agents))
            res = (b + b.T) // 2
            np.fill_diagonal(res, 0)
        else:
            res = np.full((self.num_agents, self.num_agents), 0)
        return res

    def gen_max_queue(self, min_queue, max_queue):
        return np.random.randint(min_queue, max_queue, size=self.num_agents)

    def gen_tasks(self, max_run_time):
        c1 = np.random.randint(self.num_steps, size=self.num_steps // 2 * self.num_agents)
        c2 = np.full(self.num_steps // 2 * self.num_agents, -1)
        c3 = np.random.randint(self.num_agents, size=self.num_steps // 2 * self.num_agents)
        c4 = np.random.randint(1, max_run_time, size=self.num_steps // 2 * self.num_agents)
        c5 = np.random.randint(max_run_time, self.num_steps, size=self.num_steps // 2 * self.num_agents)
        df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'run_time': c4, 'life_time': c5})
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
        self.tasks_df.iloc[row, col_time] = task.time.values[0] + self.relations[rcv][task.rcv.values[0]]  # new time
        self.tasks_df.iloc[row, col_queued] = False  # not yet in queue after being sent

    def reset(self):
        if not self.time:
            print('reset tasks_df & relations')
            self.tasks_df = self.gen_tasks(5)
            self.relations = self.gen_relations(1, 4)
            self.max_queues = self.gen_max_queue(5, 15)
        obs = {}
        for i in range(self.num_agents):
            obs[i] = np.asarray([0., 0.])
        return obs

    def step(self, action_dict):
        run_time_col_index = self.tasks_df.columns.get_loc('run_time')
        queued_col_index = self.tasks_df.columns.get_loc('queued')
        active_col_index = self.tasks_df.columns.get_loc('active')
        print('timestep', self.time)
        obs, rew, done, info = {}, {}, {}, {}
        num_tasks, tasks_ratio, total_run_time = [], [], []
        self.tasks_df.loc[
            (
                        self.tasks_df.time == self.time), 'queued'] = True  # all new tasks from this time step are labeled as queued
        for i in range(self.num_agents):
            reward = 0
            # for complete graph
            if not self.tasks_df[(self.tasks_df.rcv == i) & (
                    self.tasks_df.queued == True)].empty:  # if there are queued tasks on i-th agent
                if self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].shape[
                    0] > 1:  # if several tasks were sent to i-th agent
                    # leave only the first one in queue, every other task is moved to be dealt with on another time step
                    tmp_queued_time_not_first = self.tasks_df.loc[
                                                    (self.tasks_df.queued == True) & (self.tasks_df.rcv == i)
                                                    & (self.tasks_df.time == self.time), 'run_time'].index[1:]
                    self.tasks_df.iloc[tmp_queued_time_not_first, run_time_col_index] = \
                        self.tasks_df.iloc[tmp_queued_time_not_first, run_time_col_index].apply(lambda x: x + 1)
                    self.tasks_df.iloc[tmp_queued_time_not_first, queued_col_index] = False
                tmp_queued_time_first = self.tasks_df[(self.tasks_df.queued == True) & (self.tasks_df.rcv == i)].index[
                    0]  # get the first actually queued task
                self.tasks_df.iloc[tmp_queued_time_first, active_col_index] = True  # and make it active
                tmp_active_runtime = self.tasks_df[(self.tasks_df.active == True) & (self.tasks_df.rcv == i)
                                                   & (
                                                               self.tasks_df.run_time == 0)]  # finished (runtime == 0) active task on i-th agent
                if not tmp_active_runtime.empty:  # if there are active finished tasks on i-th agent
                    reward += 50
                    self.tasks_df.drop(self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.active == True)
                                                     & (self.tasks_df.run_time == 0)].index[0],
                                       inplace=True)  # delete finished task from dataframe
                tmp_queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
                tmp_queued_other = self.tasks_df[(self.tasks_df.rcv == action_dict[i]) & (self.tasks_df.queued == True)]
                if not tmp_queued_i.empty:  # if there are queued tasks on i-th agent
                    if action_dict[i] == i:  # append action to its own queue
                        tmp_reward = np.sum(
                            tmp_queued_i.run_time[:-1])  # all queued tasks run_times (except for the new one)
                        reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[
                            -1])  # negative reward if lifetime ends before all tasks in queue are done
                    elif not tmp_queued_other.empty:  # every other number N means "send to N"
                        tmp_reward = np.sum(tmp_queued_other.run_time)  # all runtimes on new agent
                        tmp_reward += tmp_queued_i.run_time.values[-1]  # new task's runtime
                        tmp_reward += self.relations[i][action_dict[i]]  # time to send between agents
                        reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[
                            -1])  # negative reward if lifetime ends before task is received and all tasks in new queue are done
                        self.send_task(tmp_queued_i, action_dict[i])  # send from i-th agent to chosen by action
            rew[i], done[i], info[i] = reward, True, {}
        for i in range(self.num_agents):
            tmp_queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
            if not tmp_queued_i.empty:
                num_tasks.append(tmp_queued_i.shape[0])  # number of queued tasks
                tasks_ratio.append(tmp_queued_i.shape[0] / self.max_queues[i])
                total_run_time.append(np.sum(tmp_queued_i.run_time.values))  # runtimes of all queued tasks
            else:
                num_tasks.append(0.)
                tasks_ratio.append(0.)
                total_run_time.append(0.)
        # normalize from 0 to 1
        if np.max(num_tasks) != 0:
            result_num_tasks = (num_tasks - np.min(num_tasks)) / np.ptp(num_tasks)
        else:
            result_num_tasks = [0.] * self.num_agents
        # if np.max(tasks_ratio) != 0:
        #     result_tasks_ratio = (tasks_ratio - np.min(tasks_ratio)) / np.ptp(tasks_ratio)
        # else:
        #     result_tasks_ratio = [0.] * self.num_agents
        if np.max(total_run_time) != 0:
            result_total_run_time = (total_run_time - np.min(total_run_time)) / np.ptp(total_run_time)
        else:
            result_total_run_time = [0.] * self.num_agents

        # obs = {i: np.asarray(j) for i, j in enumerate(zip(result_num_tasks, result_tasks_ratio, result_total_run_time))}  # observation (visible state) encoding
        obs = {i: np.asarray(j) for i, j in
               enumerate(zip(result_num_tasks, result_total_run_time))}  # observation (visible state) encoding

        done["__all__"] = True
        for i in range(self.num_agents):
            tmp_active_runtime = self.tasks_df.loc[(self.tasks_df.rcv == i) & (
                        self.tasks_df.active == True), 'run_time']  # if there are active tasks on i-th agent
            if not tmp_active_runtime.empty:
                self.tasks_df.iloc[tmp_active_runtime.index[0], run_time_col_index] = tmp_active_runtime.values[
                                                                                          0] - 1  # runtime -1

        mask = (self.tasks_df['queued'] == True)
        tasks_queued = self.tasks_df[mask]
        self.tasks_df.loc[mask, 'life_time'] = tasks_queued['life_time'] - 1
        # self.tasks_df.iloc[tmp_queued_lifetime, life_time_col_index] = \
        #     self.tasks_df.iloc[tmp_queued_lifetime, life_time_col_index].apply(lambda x: x - 1)  # life_time of all queued tasks -1
        # self.tasks_df.apply(lambda x: print(x))
        self.tasks_df.drop(self.tasks_df[self.tasks_df.life_time < 0].index, inplace=True)
        self.tasks_df.reset_index(drop=True, inplace=True)
        self.time = (self.time + 1) % self.num_steps
        return obs, rew, done, info
