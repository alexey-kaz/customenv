import gym
import numpy as np
import pandas as pd
from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.tf import TFModelV2, FullyConnectedNetwork


def gen_jobs(n_ag, max_time_steps, max_run_time):
    c1 = np.random.randint(max_time_steps, size=max_time_steps * n_ag)
    c2 = np.full(max_time_steps * n_ag, -1)
    c3 = np.random.randint(n_ag, size=max_time_steps * n_ag)
    c4 = np.full(max_time_steps * n_ag, False)
    c5 = np.random.randint(1, max_run_time, size=max_time_steps * n_ag)
    c6 = np.random.randint(max_run_time * 2, max_time_steps, size=max_time_steps * n_ag)
    df = pd.DataFrame({'time': c1, 'snd': c2, 'rcv': c3, 'sent': c4, 'run_time': c5, 'life_time': c6})
    df = df.sort_values(by=['time', 'rcv']).drop_duplicates(subset=['time', 'snd', 'rcv'])
    return df


def gen_relations(n_ag, min_route, max_route):
    b = np.random.randint(min_route, max_route, size=(n_ag, n_ag))
    res = (b + b.T) // 2
    np.fill_diagonal(res, 0)
    return res


class JSSPEnv(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name, true_obs_shape=(11,),
                 action_embed_size=5, *args, **kwargs):
        super(JSSPEnv, self).__init__(obs_space,
                                      action_space, num_outputs, model_config, name)
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape),
            action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
        self.time = None
        self.obs = None
        self.num_agents = 5
        self.observation_space = gym.spaces.Box(low=0, high=800, shape=(1,))
        self.action_space = gym.spaces.Discrete(3)
        n_agents = 5
        self.jobs_df = gen_jobs(n_agents, 100, 8)
        self.relations = gen_relations(n_agents, 0, 4)

    def send_task(self, job, rcv):
        tmp = [job.time + self.relations[rcv][job.rcv], job.rcv, rcv, False, job.run_time, job.life_time]
        self.jobs_df.loc[len(self.jobs_df)] = tmp

    def reset(self):
        self.time = 0
        self.obs = {}
        queue = np.array([])
        for i in range(self.num_agents):
            self.obs[i] = np.array([queue])
        return self.obs

    def step(self, action_dict):
        self.time += 1
        # obs, rew, done, info = {}, {}, {}, {}
        reward = 0
        # -1 to all first jobs runtimes in all queues
        for i in range(self.num_agents):
            loc_obs = [self.obs[x] for x in range(self.num_agents) if self.relations[x][i] or x == i]
            if not loc_obs[i][0].run_time.values[0] == 0:
                reward += 20
                np.delete(self.obs[i], 0)
            elif action_dict[i][0] == i:  # append action is the same as agent number
                tmp = np.sum([j.run_time for j in self.obs[i][:-2]])
                reward += -50 * (tmp > self.obs[i][-1].life_time.values[0])
                np.append(loc_obs[i], self.jobs_df[(self.jobs_df.time == self.time) & (self.jobs_df.rcv == i - 1)])
            else:  # every other number N means "send to N"
                tmp = np.sum([j.run_time for j in self.obs[action_dict[i][0]]])
                tmp += self.obs[i][-1].run_time.values[0] + self.relations[i][action_dict[i][0] - 1]
                reward += -50 * (tmp > self.obs[i][-1].life_time)
                self.send_task(self.jobs_df[(self.jobs_df.time == self.time) & (self.jobs_df.snd == -1)
                                            & (self.jobs_df.rcv == i - 1)], action_dict[i][0])
            # obs[i], rew[i], done[i], info[i] = np.array([self.curr_water]), reward, True, {}

        # done["__all__"] = True
        # print(obs)
        # print(self.observation_space)
        # return obs, rew, done, info
