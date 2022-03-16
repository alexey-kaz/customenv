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
    df['done'] = False
    df['queued'] = False
    df['running'] = False
    return df


def gen_relations(n_ag, min_route, max_route):
    b = np.random.randint(min_route, max_route, size=(n_ag, n_ag))
    res = (b + b.T) // 2
    np.fill_diagonal(res, 0)
    return res


class JSSPEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions=False, part=False):
        self.num_agents = 5
        self.jobs_df = None
        # self.DDDDDDDDDELETE_THISSSSSSSSS = None
        self.relations = None
        self.time = None
        low = np.array([0, 0], dtype=np.float)
        high = np.array([1, 1], dtype=np.float)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float)
        # gym.spaces.Box(low=0, high=self.num_agents, shape=(1,))
        self.action_space = gym.spaces.Discrete(self.num_agents)

    def send_task(self, job, rcv):
        # tmp = [job.time + self.relations[rcv][job.rcv], job.rcv, rcv, False, job.run_time, job.life_time, False]
        # tmp = pd.DataFrame({'time': job.time + self.relations[rcv][job.rcv],
        #                     'snd': job.rcv,
        #                     'rcv': rcv,
        #                     'run_time': job.run_time,
        #                     'life_time': job.life_time,
        #                     'done': False,
        #                     'queued': True,
        #                     'running': False})
        # # print(job.values, tmp.values)
        # self.jobs_df = pd.concat([self.jobs_df, tmp], ignore_index=True)
        # self.DDDDDDDDDELETE_THISSSSSSSSS = pd.concat([self.DDDDDDDDDELETE_THISSSSSSSSS, tmp], ignore_index=True)
        # self.jobs_df.loc[len(self.jobs_df)] = tmp
        # print(job.time)
        row = self.jobs_df.loc[(self.jobs_df.time == job.time.values[0])
                               & (self.jobs_df.snd == job.snd.values[0])
                               & (self.jobs_df.rcv == job.rcv.values[0]), 'time'].index[0]
        col = self.jobs_df.columns.get_loc('time')
        # print(self.jobs_df.shape)
        # print(row, col)
        # print(self.jobs_df.iloc[row, col])
        self.jobs_df.iloc[row, col] = job.time + self.relations[rcv][job.rcv]

        # self.jobs_df.drop(self.jobs_df.index[(self.jobs_df.time == job.time.values[0])
        #                                      & (self.jobs_df.snd == job.snd.values[0])
        #                                      & (self.jobs_df.rcv == job.rcv.values[0])], inplace=True)

    def reset(self):
        self.jobs_df = gen_jobs(self.num_agents, 10, 5)
        self.relations = gen_relations(self.num_agents, 1, 4)
        # self.DDDDDDDDDELETE_THISSSSSSSSS = pd.DataFrame(columns=self.jobs_df.columns)
        # print('reset self.DDDDDDDDDELETE_THISSSSSSSSS\n', self.DDDDDDDDDELETE_THISSSSSSSSS)
        self.time = -1
        obs = {}
        for i in range(self.num_agents):
            obs[i] = np.asarray([0., 0.])
        return obs

    def step(self, action_dict):
        run_time_col_index = self.jobs_df.columns.get_loc('run_time')
        time_col_index = self.jobs_df.columns.get_loc('time')
        queued_col_index = self.jobs_df.columns.get_loc('queued')
        running_col_index = self.jobs_df.columns.get_loc('running')
        # max_time_step = 0
        self.time += 1
        obs, rew, done, info = {}, {}, {}, {}
        reward = 0
        num_jobs, total_run_time = [], []
        self.jobs_df.loc[(self.jobs_df.time == self.time), 'queued'] = True
        # print(self.jobs_df.loc[(self.jobs_df.queued == True) & (self.jobs_df.time == self.time), 'time'].index[0])
        # self.jobs_df.iloc[self.jobs_df.loc[(self.jobs_df.queued == True) & (self.jobs_df.time == self.time),
        #                                    'running'].index[0], running_col_index] = True
        # self.DDDDDDDDDELETE_THISSSSSSSSS = pd.concat([self.DDDDDDDDDELETE_THISSSSSSSSS,
        #                                  self.jobs_df[self.jobs_df.time == self.time]],
        #                                 ignore_index=True).sort_values(['time', 'rcv'],
        #                                                                ascending=False).drop_duplicates()
        # -1 to all first jobs runtimes in all queues
        # print('loc_obs before\n', self.DDDDDDDDDELETE_THISSSSSSSSS.columns, '\n', self.DDDDDDDDDELETE_THISSSSSSSSS)
        for i in range(self.num_agents):
            # loc_obs = self.jobs_df[self.jobs_df.rcv == i]
            # for non-complete graph
            # loc_obs = [self.obs[x] for x in range(self.num_agents) if self.relations[x][i] or x == i]
            # add tasks of this time step and agent
            # self.obs[i] = self.jobs_df[(self.jobs_df.time <= self.time) & (self.jobs_df.rcv == i)]
            # print(loc_obs[(loc_obs.rcv == i) & (loc_obs.time == self.time)].empty)
            if not self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].empty:
                self.jobs_df.iloc[self.jobs_df.loc[(self.jobs_df.queued == True) & (self.jobs_df.time == self.time),
                                                   'running'].index[0], running_col_index] = True
                if self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].shape[0] > 1:
                    self.jobs_df.iloc[self.jobs_df.loc[(self.jobs_df.queued == True) &
                                                       (self.jobs_df.rcv == i) & (self.jobs_df.time == self.time),
                                                       'run_time'].index[1:], run_time_col_index] = \
                        self.jobs_df.iloc[self.jobs_df.loc[(self.jobs_df.queued == True) & (self.jobs_df.rcv == i),
                                                           'run_time'].index[1:],
                                          run_time_col_index].apply(lambda x: x + 1)
                    self.jobs_df.iloc[self.jobs_df.loc[(self.jobs_df.queued == True) &
                                                       (self.jobs_df.rcv == i) & (self.jobs_df.time == self.time),
                                                       'time'].index[1:], queued_col_index] = False
                    # print('step in')
                print('IDIOT: time {}, i {}\n'.format(self.time, i), self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.running == True)].run_time.values[0])
                if self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.running == True)].run_time.values[0] == 0:
                    print('YAY I AM A GOOD BOY')
                    reward += 200
                    self.jobs_df.loc[(self.jobs_df.rcv == i) & (self.jobs_df.running == True), 'done'] = True
                if action_dict[i] == i:  # append action is the same as agent number
                    tmp = np.sum(self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].run_time[:-1])
                    # print('action_dict[{}] == {}'.format(i, i))
                    reward += -50 * (tmp > self.jobs_df[(self.jobs_df.rcv == i)
                                                        & (self.jobs_df.queued == True)].life_time.values[-1])
                    # np.append(self.obs[i], self.jobs_df[(self.jobs_df.time == self.time) & (self.jobs_df.rcv == i)])
                else:  # every other number N means "send to N"
                    # print(action_dict[i])
                    # print(self.jobs_df[self.jobs_df.rcv == action_dict[i]])
                    tmp = np.sum(self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].run_time)
                    tmp += self.jobs_df[(self.jobs_df.rcv == i)
                                             & (self.jobs_df.queued == True)].run_time.values[0]
                    tmp += self.relations[i][action_dict[i]]
                    # max_time_step = max(max_time_step, self.relations[i][action_dict[i] - 1])
                    # print(loc_obs[loc_obs.rcv == i].life_time.values)
                    reward += -50 * (tmp > self.jobs_df[(self.jobs_df.rcv == i)
                                                             & (self.jobs_df.queued == True)].life_time.values[0])
                    self.send_task(self.jobs_df[(self.jobs_df.time == self.time) & (self.jobs_df.queued == True)
                                                & (self.jobs_df.rcv == i)], action_dict[i])
                # loc_obs = self.jobs_df[(self.jobs_df.time <= self.time) & (self.jobs_df.rcv == i)]
            rew[i], done[i], info[i] = reward, True, {}

            # num_jobs.append(loc_obs[loc_obs.rcv == i].shape[0])
            # print('loc_obs[loc_obs.rcv == {}].run_time.values'.format(i), loc_obs[loc_obs.rcv == {}].run_time.values)
            # total_run_time.append(np.sum(loc_obs[loc_obs.rcv == {}].run_time.values))

            # print('num_jobs, total_run_time: {}, {}'.format(num_jobs, total_run_time))
            rew[i], done[i], info[i] = reward, True, {}
            # print('done {}\n'.format(i))
        self.jobs_df.drop(self.jobs_df.index[self.jobs_df.done == True], inplace=True)
        # print('self.DDDDDDDDDELETE_THISSSSSSSSS after\n', self.DDDDDDDDDELETE_THISSSSSSSSS)
        for i in range(self.num_agents):
            if self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].empty:
                num_jobs.append(0.)
                total_run_time.append(0.)
            else:
                # print('loc_obs[loc_obs.rcv == {}].shape[0]'.format(i), loc_obs[loc_obs.rcv == i].shape[0])
                num_jobs.append(self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].shape[0])
                # print('self.DDDDDDDDDELETE_THISSSSSSSSS[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == {}].run_time.values'.format(i),
                #       self.DDDDDDDDDELETE_THISSSSSSSSS[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i].run_time.values)
                total_run_time.append(
                    np.sum(self.jobs_df[(self.jobs_df.rcv == i) & (self.jobs_df.queued == True)].run_time.values))
        # print('step out')
        # print("time: {}".format(self.time))
        # print('result self.DDDDDDDDDELETE_THISSSSSSSSS:\n', self.DDDDDDDDDELETE_THISSSSSSSSS)
        # print('num_jobs, total_run_time: {}, {}'.format(num_jobs, total_run_time))
        if np.max(num_jobs) != 0:
            result_num_jobs = (num_jobs - np.min(num_jobs)) / np.ptp(num_jobs)
        else:
            result_num_jobs = [0.] * self.num_agents
        if np.max(total_run_time) != 0:
            result_total_run_time = (total_run_time - np.min(total_run_time)) / np.ptp(total_run_time)
        else:
            result_total_run_time = [0.] * self.num_agents
        # print('result_num_jobs, result_total_run_time: {}, {}'.format(result_num_jobs, result_total_run_time))
        # print(result_num_jobs, result_total_run_time)
        obs = {i: np.asarray(j) for i, j in enumerate(zip(result_num_jobs, result_total_run_time))}

        done["__all__"] = True
        print('obs, rew, done, info: {}, {}, {}, {}'.format(obs, rew, done, info))
        # col_index = self.DDDDDDDDDELETE_THISSSSSSSSS.columns.get_loc('run_time')
        # for i in range(5):
        #     print('empty', self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].empty)
        #     if not self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].empty:
        #         print('all', self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'])
        #         print('values', self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].values)
        #         print('index[0]', self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].index[0])
        #         print('iloc', self.DDDDDDDDDELETE_THISSSSSSSSS.iloc[self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].index[0]])
        #         print('iloc column', self.DDDDDDDDDELETE_THISSSSSSSSS.iloc[self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].index[0], col_index])
        #         self.DDDDDDDDDELETE_THISSSSSSSSS.iloc[self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].index[0], col_index] \
        #             = self.DDDDDDDDDELETE_THISSSSSSSSS.loc[self.DDDDDDDDDELETE_THISSSSSSSSS.rcv == i, 'run_time'].values[0] - 1
        #     if not self.jobs_df.loc[self.jobs_df.rcv == i, 'run_time'].empty:
        #         self.jobs_df.iloc[self.jobs_df.loc[self.jobs_df.rcv == i, 'run_time'].index[0], col_index] \
        #             = self.jobs_df.loc[self.jobs_df.rcv == i, 'run_time'].values[0] - 1
        self.jobs_df['life_time'] = self.jobs_df['life_time'].apply(lambda x: x - 1)
        # print(self.observation_space)
        return obs, rew, done, info
