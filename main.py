import os

from argparse import ArgumentParser
import ray
from ray import tune
from ray.tune.registry import register_env
from datetime import datetime

# Import environment definition
from env import Diploma_Env
from viz import Viz

# from ray.rllib.utils import try_import_torch
# torch = try_import_torch()

from ray.rllib.utils import try_import_tf

tf = try_import_tf()

ray.init()


class Experiment:
    def __init__(self):
        self.distr = distribution
        self.env_name = 'Diploma_Env'
        self.exp_name = 'exp_{}_{}'.format(time, distribution)
        self.env_conf = None
        self.multi_env = None
        self.exp_drops = {}
        self.exp_finished = {}

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

        def stopper(trial_id, result):
            if trial_id in self.exp_drops.keys():
                self.exp_drops[trial_id].append(result['custom_metrics']['drop_max'])
                self.exp_finished[trial_id].append(result['custom_metrics']['finished_max'])
            else:
                self.exp_drops[trial_id] = [result['custom_metrics']['drop_max']]
                self.exp_finished[trial_id] = [result['custom_metrics']['finished_max']]
            num = 3
            if len(self.exp_drops[trial_id]) >= num * 3:
                tmp_drop = self.exp_drops[trial_id][-num:]
                div_tmp_drop = max(tmp_drop) / min(tmp_drop) if min(tmp_drop) else 0
                print('drops', tmp_drop, max(tmp_drop), min(tmp_drop), abs(1 - div_tmp_drop))
                tmp_finished = self.exp_finished[trial_id][-num:]
                div_tmp_finished = max(tmp_finished) / min(tmp_finished) if min(tmp_finished) else 0
                print('finished', tmp_finished, max(tmp_finished), min(tmp_finished), abs(1 - div_tmp_finished))
                # return (abs(1 - div_tmp_finished) <= 1 and min(tmp_finished) > 0.9) or min(tmp_finished) > 0.94
                return min(tmp_finished) > 0.9
                # return (abs(1 - div_tmp_finished) <= 1 and min(tmp_finished) > 0.7) \
                #     or result['training_iteration'] == num_episodes

        def on_episode_end(info):
            if info['env'] in self.exp_drops.keys():
                self.exp_drops[info['env']].append(info['episode'].last_info_for(0)['all_drops'])
                self.exp_finished[info['env']].append(info['episode'].last_info_for(0)['all_finished'])
            else:
                self.exp_drops[info['env']] = [info['episode'].last_info_for(0)['all_drops']]
                self.exp_finished[info['env']] = [info['episode'].last_info_for(0)['all_finished']]
            info['episode'].custom_metrics['drop'] = info['episode'].last_info_for(0)['all_drops']
            info['episode'].custom_metrics['finished'] = info['episode'].last_info_for(0)['all_finished']

        # Define configuration with hyperparam and training details
        train_config = {
            'use_critic': True,
            'lambda': 0.95,
            'kl_coeff': 0.65,
            'kl_target': 0.01,
            # 'shuffle_sequences': True,
            'num_workers': 1,
            'num_cpus_per_worker': 4,
            # 'num_gpus_per_worker': 0.4,
            'num_sgd_iter': 40,
            'sgd_minibatch_size': 256,
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
            'env_config': self.env_conf,
            'callbacks': {
                "on_episode_end": on_episode_end,
            },
        }

        # Define experiment details
        exp_dict = {
            'name': self.exp_name,
            'run_or_experiment': 'PPO',
            'stop': stopper,
            'config': train_config,
            'num_samples': num_samples,
            'local_dir': './exp_res',
            'mode': 'max',
            'verbose': 0,
            'checkpoint_freq': 5
        }
        # Initialize ray and run
        tune.run(**exp_dict)
        v = Viz(self.exp_name, self.env_conf['num_steps'])
        v.plot_anything('both')


time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

parser = ArgumentParser(description='Запуск экспериментов')
parser.add_argument('--n_steps', type=int)
parser.add_argument('--n_agents', type=int)
parser.add_argument('--n_rcv', type=int)
parser.add_argument('--distr', type=str)
args = parser.parse_args()

num_steps = args.n_steps
num_ag = args.n_agents
num_rcv = args.n_rcv
distribution = args.distr
print('num_steps: {}\nnum_agents: {}\nnum_rcv: {}\ndistribution: {}'.format(num_steps, num_ag,
                                                                            num_rcv, distribution))
num_episodes = 40
num_samples = 2

exp = Experiment()
print('Train', distribution.upper())
exp.setup_and_train()
print('Done', distribution.upper())
