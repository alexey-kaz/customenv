import glob
import os
import sys

import numpy as np
import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from datetime import datetime

# Import environment definition
from env import Diploma_Env
from viz import plot_drop, plot_drop_double

ray.init()
tf = try_import_tf()
n_workers = 0
# max_n_steps = 20000
max_n_steps = 1000
time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class Experiment:
    def __init__(self):
        num_steps_vec = [250 * 2 ** (i + 1) for i in range(3)]
        num_agents_vec = [5, 15]
        num_rcv_vec = {i: np.linspace(i // 2, i, 3, dtype=int) for i in num_agents_vec}
        s, a, r = map(int, sys.argv[1:])
        print('num_steps: {}\nnum_agents: {}\nnum_rcv: {}'.format(num_steps_vec[a],
                                                                  num_agents_vec[s],
                                                                  num_rcv_vec[num_agents_vec[s]][r]))
        self.env_conf = {
            'mode': 'Train',
            'num_steps': num_steps_vec[a],
            'num_agents': num_agents_vec[s],
            'num_rcv': num_rcv_vec[num_agents_vec[s]][r],
            'is_JSSP': False,
            'queue_rew_toggle': True,
            'alpha': 0.3,
            'beta': 0.3,
            'gamma': 0.3,
            'data_path': os.path.abspath('./data/'),
            'datetime': time
        }
        self.train_config = None
        self.env_name = "Diploma_Env"
        self.multi_env = Diploma_Env(self.env_conf)
        self.exp_name = 'exp_{}'.format(time)

    # Driver code for training
    def setup_and_train(self):
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
        self.train_config = {
            # 'num_training_iterations': 20,
            "use_critic": True,
            "lambda": 1.0,
            "kl_coeff": 0.1,
            'num_workers': 1,
            "shuffle_sequences": True,
            "num_cpus_per_worker": 3,
            "num_sgd_iter": 30,
            "sgd_minibatch_size": 128,
            "train_batch_size": self.env_conf['num_steps'],
            # "rollout_fragment_length": num_steps / num_workers,
            "lr": 3e-4,
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh"},
            "multiagent": {
                "policies": policy_graphs,
                "policy_mapping_fn": policy_mapping_fn,
            },

            "clip_param": 0.3,
            "vf_clip_param": 3,

            "simple_optimizer": True,

            "env": self.env_name,
            "env_config": self.env_conf
        }

        # Define experiment details
        exp_dict = {
            'name': self.exp_name,
            'run_or_experiment': 'PPO',
            "stop": {
                "training_iteration": int(max_n_steps / self.env_conf['num_steps'])
            },
            "config": self.train_config,
            # 'num_samples': 6,
            'local_dir': './exp_res',
            'mode': 'max',
            "verbose": 0,
            'checkpoint_at_end': True
        }

        # Initialize ray and run
        tune.run(**exp_dict)
        plot_drop(self.exp_name, self.env_conf['num_steps'], self.env_conf['num_agents'], self.env_conf['num_rcv'])

    def test(self, test_num_eps=2):
        test_env_config = self.env_conf.copy()
        test_env_config['mode'] = 'Test'
        test_config = self.train_config.copy()
        test_config["env_config"] = test_env_config
        path = "./exp_res/{}".format(self.exp_name)
        print(path)
        checkpoint_path = glob.glob(path + "/PPO_Diploma_Env*/checkpoint*/checkpoint-?")[0]
        tune.run(ppo.PPOTrainer,
                 name='Test_' + self.exp_name,
                 config=test_config,
                 restore=checkpoint_path,
                 local_dir='./exp_res',
                 stop={"episodes_total": test_num_eps},
                 checkpoint_at_end=True)
        plot_drop(self.exp_name, self.env_conf['num_steps'], self.env_conf['num_agents'],
                  self.env_conf['num_rcv'], 'Test')

    def viz_train_test(self):
        plot_drop_double(self.exp_name, self.env_conf['num_steps'],
                         self.env_conf['num_agents'], self.env_conf['num_rcv'])


exp = Experiment()
exp.setup_and_train()
exp.test()
exp.viz_train_test()
