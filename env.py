import gym
import numpy as np
import pandas as pd

from ray.rllib.env.multi_agent_env import MultiAgentEnv


def gen_jobs(n_ag, max_time_steps, max_run_time):
    c1 = np.random.randint(max_time_steps, size=max_time_steps * n_ag)
    c2 = np.full(max_time_steps * n_ag, -1)
    c3 = np.random.randint(n_ag, size=max_time_steps * n_ag)
    c4 = np.random.randint(1, max_run_time, size=max_time_steps * n_ag)
    c5 = np.random.randint(max_run_time, max_time_steps, size=max_time_steps * n_ag)
    df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'run_time': c4, 'life_time': c5})
    df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'snd', 'rcv'], ignore_index=True)
    df['queued'] = False
    df['active'] = False
    return df


class Diploma_Env(MultiAgentEnv):
    def __init__(self, return_agent_actions=False, part=False):
        self.num_steps = 200
        self.num_agents = 5
        self.jobs_df = None
        self.relations = None
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

    def send_task(self, job, rcv):
        row = self.jobs_df.loc[(self.jobs_df.time == job.time.values[0])
                               & (self.jobs_df.snd == job.snd.values[0])
                               & (self.jobs_df.rcv == job.rcv.values[0]), 'time'].index[0]
        col_time = self.jobs_df.columns.get_loc('time')
        col_queued = self.jobs_df.columns.get_loc('queued')
        self.jobs_df.iloc[row, col_time] = job.time.values[0] + self.relations[rcv][job.rcv.values[0]]  # new time
        self.jobs_df.iloc[row, col_queued] = False  # not yet in queue after being sent

    def reset(self):
        if not self.time:
            print('reset jobs_df & relations')
            self.jobs_df = gen_jobs(self.num_agents, self.num_steps, 5)
            self.relations = self.gen_relations(1, 4)
        obs = {}
        for i in range(self.num_agents):
            obs[i] = np.asarray([0., 0.])
        return obs

    def step(self, action_dict):
        run_time_col_index = self.jobs_df.columns.get_loc('run_time')
        queued_col_index = self.jobs_df.columns.get_loc('queued')
        active_col_index = self.jobs_df.columns.get_loc('active')
        life_time_col_index = self.jobs_df.columns.get_loc('life_time')
        print('timestep', self.time)
        obs, rew, done, info = {}, {}, {}, {}
        num_jobs, total_run_time = [], []
        self.jobs_df.loc[
            (self.jobs_df.time == self.time), 'queued'] = True  # all new jobs from this time step are labeled as queued
        for i in range(self.num_agents):
            reward = 0
            # for complete graph
            if not self.jobs_df[(self.jobs_df.rcv == i) & (
                    self.jobs_df.queued == True)].empty:  # if there are queued jobs on i-th agent
                if self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].shape[0] > 1:  # if several jobs were sent to i-th agent
                    # leave only the first one in queue, every other task is moved to be dealt with on another time step
                    tmp_queued_time_not_first = self.jobs_df.loc[(self.jobs_df.queued == True) & (self.jobs_df.rcv == i)
                                                                 & (self.jobs_df.time == self.time), 'run_time'].index[1:]
                    self.jobs_df.iloc[tmp_queued_time_not_first, run_time_col_index] = \
                        self.jobs_df.iloc[tmp_queued_time_not_first, run_time_col_index].apply(lambda x: x + 1)
                    self.jobs_df.iloc[tmp_queued_time_not_first, queued_col_index] = False
                tmp_queued_time_first = self.jobs_df[(self.jobs_df.queued == True) & (self.jobs_df.rcv == i)].index[0]  # get the first actually queued job
                self.jobs_df.iloc[tmp_queued_time_first, active_col_index] = True  # and make it active
                tmp_active_runtime = self.jobs_df[(self.jobs_df.active == True) & (self.jobs_df.rcv == i)
                                                  & (self.jobs_df.run_time == 0)]  # finished (runtime == 0) active job on i-th agent
                if not tmp_active_runtime.empty:  # if there are active finished jobs on i-th agent
                    reward += 50
                    self.jobs_df.drop(self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.active == True)
                                                   & (self.jobs_df.run_time == 0)].index[0], inplace=True)  # delete finished job from dataframe
                tmp_queued_i = self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)]
                tmp_queued_other = self.jobs_df[(self.jobs_df.rcv == action_dict[i]) & (self.jobs_df.queued == True)]
                if not tmp_queued_i.empty:  # if there are queued jobs on i-th agent
                    if action_dict[i] == i:  # append action to its own queue
                        tmp_reward = np.sum(tmp_queued_i.run_time[:-1])  # all queued jobs run_times (except for the new one)
                        reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[-1])  # negative reward if lifetime ends before all jobs in queue are done
                    elif not tmp_queued_other.empty:  # every other number N means "send to N"
                        tmp_reward = np.sum(tmp_queued_other.run_time)  # all runtimes on new agent
                        tmp_reward += tmp_queued_i.run_time.values[-1]  # new job's runtime
                        tmp_reward += self.relations[i][action_dict[i]]  # time to send between agents
                        reward += -20 * (tmp_reward > tmp_queued_i.life_time.values[-1])  # negative reward if lifetime ends before job is received and all jobs in new queue are done
                        self.send_task(tmp_queued_i, action_dict[i])  # send from i-th agent to chosen by action
            rew[i], done[i], info[i] = reward, True, {}
        for i in range(self.num_agents):
            tmp_queued_i = self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)]
            if not tmp_queued_i.empty:
                num_jobs.append(tmp_queued_i.shape[0])  # number of queued jobs
                total_run_time.append(np.sum(tmp_queued_i.run_time.values))  # runtimes of all queued jobs
            else:
                num_jobs.append(0.)
                total_run_time.append(0.)

        # normalize from 0 to 1
        if np.max(num_jobs) != 0:
            result_num_jobs = (num_jobs - np.min(num_jobs)) / np.ptp(num_jobs)
        else:
            result_num_jobs = [0.] * self.num_agents
        if np.max(total_run_time) != 0:
            result_total_run_time = (total_run_time - np.min(total_run_time)) / np.ptp(total_run_time)
        else:
            result_total_run_time = [0.] * self.num_agents

        obs = {i: np.asarray(j) for i, j in enumerate(zip(result_num_jobs, result_total_run_time))}  # observation (visible state) encoding

        done["__all__"] = True
        for i in range(self.num_agents):
            tmp_active_runtime = self.jobs_df.loc[(self.jobs_df.rcv == i) & (self.jobs_df.active == True), 'run_time']  # if there are active jobs on i-th agent
            if not tmp_active_runtime.empty:
                self.jobs_df.iloc[tmp_active_runtime.index[0], run_time_col_index] = tmp_active_runtime.values[0] - 1  # runtime -1

        tmp_queued_lifetime = self.jobs_df.loc[self.jobs_df.queued == True, 'life_time'].index
        self.jobs_df.iloc[tmp_queued_lifetime, life_time_col_index] = \
            self.jobs_df.iloc[tmp_queued_lifetime, life_time_col_index].apply(lambda x: x - 1)  # life_time of all queued jobs -1
        self.jobs_df.drop(self.jobs_df[self.jobs_df.life_time < 0].index, inplace=True)
        self.jobs_df.reset_index(drop=True, inplace=True)
        self.time = (self.time + 1) % self.num_steps
        return obs, rew, done, info
