import gym
import numpy as np
import pandas as pd
from ast import literal_eval

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Diploma_Env(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.num_steps = env_config['num_steps']
        self.num_agents = env_config['num_agents']  # number of computing devices (CDs)
        self.is_JSSP = env_config['is_JSSP']  # make True for all send times between CDs to be 0
        self.queue_rew_toggle = env_config['queue_rew_toggle']  # make True for all send times between CDs to be 0
        self.alpha, self.beta, self.gamma = env_config['alpha'], env_config['beta'], env_config['gamma']
        self.state_space_shape = 5
        low = np.array([0] * self.state_space_shape, dtype=np.float)
        high = np.array([1] * self.state_space_shape, dtype=np.float)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float)
        self.action_space = gym.spaces.Discrete(self.num_agents)
        self.time = 0
        self.tasks_df_main = pd.read_csv('{}/tasks_df_{}_{}.csv'.format(env_config['data_path'], self.num_agents, self.num_steps), converters={'run_time_vec': literal_eval, 'cpu_usage_vec': literal_eval})
        self.relations_main = np.load('{}/relations_{}_{}.npy'.format(env_config['data_path'], self.num_agents, self.num_steps))
        self.cd_info_main = pd.read_csv('{}/cd_info_{}_{}.csv'.format(env_config['data_path'], self.num_agents, self.num_steps))
        self.tasks_df = None
        self.relations = None
        self.cd_info = None
        self.all_cpu_usage = None
        self.all_queue_usage = None
        self.drop = None
        self.exp_num = 0

    def send_task(self, task, rcv):
        self.tasks_df.drop(self.tasks_df[(self.tasks_df.time == task.time.values[0])
                                         & (self.tasks_df.snd == task.snd.values[0])
                                         & (self.tasks_df.rcv == task.rcv.values[0])].index, inplace=True)
        # new time
        srs = task.copy(deep=True).squeeze()
        # print(srs)
        srs['time'] = task.time.values[0] + self.relations[rcv][task.rcv.values[0]] + 1
        # not yet in queue after being sent
        srs['queued'] = False
        # new snd and rcv
        srs['snd'], srs['rcv'] = srs['rcv'], rcv
        srs['ttl'] += 1
        # print(srs.get(key='run_time_vec'), srs.get(key='cpu_usage_vec'))
        srs['run_time'] = srs.get(key='run_time_vec')[rcv]
        srs['cpu_usage'] = srs.get(key='cpu_usage_vec')[rcv]
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
            print('drop', self.drop)
            self.drop = 0
            self.tasks_df = self.tasks_df_main.copy(deep=True)
            print('df shape', self.tasks_df.shape[0])
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
        time_activated_col_index = self.tasks_df.columns.get_loc('time_activated')
        state, rew, done, info = {}, {i: 0 for i in range(self.num_agents)}, {}, {}
        bad_action = [0 for i in range(self.num_agents)]
        # all new tasks from this time step are labeled as queued
        self.tasks_df.loc[(self.tasks_df.time == self.time), 'queued'] = True
        # drop tasks with tll > 2
        self.drop += self.tasks_df[self.tasks_df.ttl > 10].shape[0]
        self.tasks_df.drop(self.tasks_df[self.tasks_df.ttl > 10].index, inplace=True)
        self.tasks_df.reset_index(drop=True, inplace=True)
        for i in range(self.num_agents):
            # if there are queued tasks on i-th agent
            if not self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].empty:
                # if several tasks were sent to i-th agent
                if self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)].shape[0] > 1:
                    # choose all new queued except for the first one
                    queued_i_time_not_first = self.tasks_df[(self.tasks_df.queued == True)
                                                              & (self.tasks_df.rcv == i)
                                                              & (self.tasks_df.time == self.time)].iloc[1:]
                    # move them to next time step
                    for j, k in queued_i_time_not_first.iterrows():
                        tmp_free_agent = self.free_agent(i)
                        self.send_task(k.to_frame().transpose(), tmp_free_agent)
                # finished active task on i-th agent
                active_runtime_finished = self.tasks_df[(self.tasks_df.active == True) & (self.tasks_df.rcv == i)
                                                        & (self.tasks_df.run_time + self.tasks_df.time_activated == self.time)]
                # increase available queue and cpu of i-th agent
                self.cd_info.queue_avail[i] += active_runtime_finished.shape[0]
                self.cd_info.cpu_avail[i] += sum(active_runtime_finished.cpu_usage)
                # reward for each finished active task
                rew[i] += active_runtime_finished.shape[0] * 0.5
                # delete finished task from dataframe
                self.tasks_df.drop(active_runtime_finished.index, inplace=True)

                queued_i_time = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)
                                              & (self.tasks_df.time == self.time)]
                # if there are queued tasks on i-th agent
                queued_i = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True) & (self.tasks_df.active == False)]
                queued_other = self.tasks_df[(self.tasks_df.rcv == action_dict[i]) & (self.tasks_df.queued == True) & (self.tasks_df.active == False)]
                if not queued_i_time.empty:
                    if self.cd_info.queue_avail[action_dict[i]] > 0 and self.cd_info.cpu_avail[action_dict[i]] > queued_i_time.cpu_usage.values[-1]:
                        # append action to its own queue
                        if action_dict[i] == i:
                            # all queued tasks run_times
                            tmp_reward = np.sum(queued_i.run_time)
                            # new task's runtime
                            tmp_reward += queued_i_time.run_time.values[-1]
                            # negative reward if lifetime ends before all tasks in queue are done
                            rew[i] += -0.5 * (tmp_reward > queued_i_time.life_time.values[-1])
                            # reduce available queue space of i-th agent
                            self.cd_info.queue_avail[i] -= 1
                            # add this task to queue usage of i-th agent
                            self.all_queue_usage[i] += 1
                        # every other number N means "send to N"
                        else:
                            # all runtimes on new agent
                            tmp_reward = np.sum(queued_other.run_time)
                            # new task's runtime
                            tmp_reward += queued_i_time.run_time.values[-1]
                            # time to send between agents
                            tmp_reward += self.relations[i][action_dict[i]]
                            # negative reward if lifetime ends before task is received and all tasks in new queue are done
                            rew[i] += -0.5 * (tmp_reward > queued_i_time.life_time.values[-1])
                            # reduce available queue space of receiving agent
                            self.cd_info.queue_avail[action_dict[i]] -= 1
                            # add this task to queue usage of i-th agent
                            self.all_queue_usage[action_dict[i]] += 1
                            # send from i-th agent to chosen by action
                            self.send_task(queued_i_time, action_dict[i])
                    else:
                        rew[i] += -0.25
                        bad_action[i] = 1
                        tmp_free_agent = self.free_agent(i)
                        self.send_task(queued_i_time, tmp_free_agent)
            done[i], info[i] = True, {}
        min_time_left_state = []
        for i in range(self.num_agents):
            # get all queued tasks of i-th agent (that aren't active)
            queued_i = self.tasks_df[(self.tasks_df.queued == True) & (self.tasks_df.rcv == i)]
            queued_i_not_active = queued_i[queued_i.active == False]
            # make queued tasks active on i-th agent until it runs out of cpu
            for j, k in queued_i_not_active.iterrows():
                if self.cd_info.cpu_avail[i] < k.cpu_usage:
                    break
                self.tasks_df.iloc[k.name, active_col_index] = True
                self.tasks_df.iloc[k.name, time_activated_col_index] = self.time
                self.cd_info.cpu_avail[i] -= k.cpu_usage
                self.all_cpu_usage[i] += k.cpu_usage
            tmp_min_time_left = 0. if queued_i.empty else min(queued_i.life_time_global - (self.time + queued_i.run_time))
            min_time_left_state.append(tmp_min_time_left)

        delta = sum(self.all_cpu_usage.values()) / (self.time + 1) / self.num_agents
        theta = sum(self.all_queue_usage.values()) / (self.time + 1) / self.num_agents
        num_tasks_state, tasks_ratio_state, total_run_time_state = [], [], []

        for i in range(self.num_agents):
            cpu_avg_i = self.all_cpu_usage[i] / (self.time + 1)
            cpu_div = 0. if not self.cd_info.cpu_avail[i] else cpu_avg_i / self.cd_info.cpu_avail[i]
            queue_avg_i = self.all_queue_usage[i] / (self.time + 1)
            queue_div = 0. if not self.cd_info.queue_avail[i] else queue_avg_i / self.cd_info.queue_avail[i]
            queue_factor = self.beta * queue_div + self.gamma * (queue_div - theta) ** 2
            rew[i] -= self.alpha * cpu_div + self.gamma * (cpu_div - delta) ** 2
            if self.queue_rew_toggle:
                rew[i] += queue_factor
            queued_i_time = self.tasks_df[(self.tasks_df.rcv == i) & (self.tasks_df.queued == True)]
            if queued_i_time.empty:
                num_tasks_state.append(0.)
                tasks_ratio_state.append(0.)
                total_run_time_state.append(0.)
            else:
                # number of queued tasks
                num_tasks_state.append(queued_i_time.shape[0])
                # ratio of queued tasks to max_queue
                tasks_ratio_state.append(queued_i_time.shape[0] / self.cd_info.queue_max[i])
                # runtimes of all queued tasks
                total_run_time_state.append(np.sum(queued_i_time.run_time.values))
        non_normalized_state = [num_tasks_state, tasks_ratio_state, total_run_time_state, min_time_left_state]
        print(non_normalized_state)
        normalized_state = [None] * len(non_normalized_state)
        # normalize state from 0 to 1
        for i in range(len(normalized_state)):
            if np.max(non_normalized_state[i]):
                normalized_state[i] = (non_normalized_state[i] - np.min(non_normalized_state[i])) / np.ptp(non_normalized_state[i])
            else:
                normalized_state[i] = [0.] * self.num_agents
        normalized_state.append(bad_action)
        # state encoding
        state = {i: np.asarray(j, dtype=object) for i, j in enumerate(zip(*normalized_state))}
        done["__all__"] = True
        self.drop += self.tasks_df[self.tasks_df.life_time_global == self.time].shape[0]
        self.tasks_df.drop(self.tasks_df[self.tasks_df.life_time_global == self.time].index, inplace=True)
        self.time = (self.time + 1) % self.num_steps
        # available_actions = np.where(self.cd_info.queue_avail > 0, 1, 0)
        return state, rew, done, info
