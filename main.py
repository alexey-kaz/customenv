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

ray.init()
tf = try_import_tf()
n_workers = 0
max_n_steps = 20000
time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

num_steps_vec = [250 * 2 ** (i + 1) for i in range(3)]
num_agents_vec = [5, 15]
num_rcv_vec = {i: np.linspace(i // 2, i, 3, dtype=int) for i in num_agents_vec}
s, a, r = map(int, sys.argv[1:])
env_conf = {
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


# Driver code for training
def setup_and_train():
    # Create a single environment and register it
    def env_creator(env_config):
        return Diploma_Env(env_config)

    multi_env = Diploma_Env(env_conf)
    env_name = "Diploma_Env"
    register_env(env_name, lambda conf: env_creator(conf))

    # Get environment obs, action spaces and number of agents
    obs_space = multi_env.observation_space
    act_space = multi_env.action_space
    num_agents = multi_env.num_agents

    # Create a policy mapping
    def gen_policy():
        return None, obs_space, act_space, {}

    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return 'agent-{}'.format(agent_id)

    # Define configuration with hyperparam and training details
    config = {
        # 'num_training_iterations': 20,
        "use_critic": True,
        "lambda": 1.0,
        "kl_coeff": 0.1,
        'num_workers': 1,
        "shuffle_sequences": True,
        "num_cpus_per_worker": 3,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "train_batch_size": env_conf['num_steps'],
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

        "env": env_name,
        "env_config": env_conf
    }

    # Define experiment details
    exp_name = 'exp_{}'.format(time)
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": int(max_n_steps / env_conf['num_steps'])
        },
        "config": config,
        # 'num_samples': 6,
        'local_dir': './exp_res',
        'mode': 'max',
        "verbose": 0,
    }

    # Initialize ray and run
    tune.run(**exp_dict)


setup_and_train()
