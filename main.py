import os
import sys

import numpy as np
import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from datetime import datetime

# Import environment definition
from env import Diploma_Env
from viz import Viz

ray.init()
tf = try_import_tf()


class Experiment:
    def __init__(self):
        self.distr = distribution
        self.env_name = 'Diploma_Env'
        self.exp_name = 'exp_{}_{}'.format(time, distribution)
        self.env_conf = None
        self.multi_env = None

    # Driver code for training
    def setup_and_train(self):

        self.env_conf = {
            'distr': self.distr,
            'num_steps': num_steps,
            'num_agents': num_ag,
            'num_rcv': num_rcv,
            'queue_rew_toggle': True,
            'alpha': 0.3,
            'beta': 0.3,
            'gamma': 0.3,
            'data_path': os.path.abspath('./data/{}/'.format(self.distr)),
            'datetime': time
        }
        self.multi_env = Diploma_Env(self.env_conf)

        # Create a single environment and register it
        def env_creator(env_config):
            return Diploma_Env(env_config)

        register_env(self.env_name, lambda conf: env_creator(conf))

        # Get environment obs, action spaces and number of agents
        obs_space = self.multi_env.observation_space
        act_space = self.multi_env.action_space
        num_agents = self.multi_env.num_agents

        # Create a policy mapping
        def gen_policy():
            return None, obs_space, act_space, {}

        policy_graphs = {}
        for i in range(num_agents):
            policy_graphs['agent-' + str(i)] = gen_policy()

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return 'agent-{}'.format(agent_id)

        # Define configuration with hyperparam and training details
        train_config = {
            'use_critic': True,
            'lambda': 0.9,
            'kl_coeff': 0.65,
            'kl_target': 0.01,
            'num_workers': 1,
            'shuffle_sequences': True,
            'num_cpus_per_worker': 3,
            'num_sgd_iter': 30,
            'sgd_minibatch_size': 128,
            'train_batch_size': self.env_conf['num_steps'],
            # 'rollout_fragment_length': num_steps / num_workers,
            'lr': 3e-4,
            'model': {
                'vf_share_layers': False,
                'fcnet_hiddens': [256, 256],
                'fcnet_activation': 'tanh'},
            'multiagent': {
                'policies': policy_graphs,
                'policy_mapping_fn': policy_mapping_fn,
            },

            'clip_param': 0.2,

            'simple_optimizer': True,

            'env': self.env_name,
            'env_config': self.env_conf
        }

        # Define experiment details
        stop = int(max_n_steps / self.env_conf['num_steps'])
        exp_dict = {
            'name': self.exp_name,
            'run_or_experiment': 'PPO',
            'stop': {
                'training_iteration': stop
            },
            'config': train_config,
            'num_samples': 2,
            'local_dir': './exp_res',
            'mode': 'max',
            'verbose': 0,
            'checkpoint_freq': 2
        }
        # Initialize ray and run
        print(exp_dict)
        tune.run(**exp_dict)
        v = Viz(self.exp_name, self.env_conf['num_steps'])
        v.plot_anything('both')


time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
num_steps_vec = [250 * 2 ** (i + 1) for i in range(3)]
num_agents_vec = [5, 10, 15]
num_rcv_vec = {i: np.linspace(i // 2, i, 3, dtype=int) for i in num_agents_vec}
distribution_types = ['erlang', 'poisson', 'uniform']
s, a, r, d = map(int, sys.argv[1:])
num_steps = num_steps_vec[s]
num_ag = num_agents_vec[a]
num_rcv = num_rcv_vec[num_agents_vec[a]][r]
distribution = distribution_types[d]
print('num_steps: {}\nnum_agents: {}\nnum_rcv: {}\ndistribution: {}'.format(num_steps, num_ag,
                                                                            num_rcv, distribution))
max_n_steps = num_steps * 80

exp = Experiment()
print('Train', distribution.upper())
exp.setup_and_train()
print('Done', distribution.upper())
