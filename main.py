import ray
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from datetime import datetime

# Import environment definition
from env import Diploma_Env

tf = try_import_tf()

n_steps = 500
n_agents = 5
n_workers = 0
max_num_steps = 20000
env_conf = {
    'num_steps': n_steps,
    'num_agents': n_agents,
    'is_JSSP': False,
    'alpha': 0.3,
    'gamma': 0.3,
}


# Driver code for training
def setup_and_train(num_steps, num_agents, num_workers):
    # Create a single environment and register it
    def env_creator(env_config):
        return Diploma_Env(env_config)

    multi_env = Diploma_Env(env_conf)
    env_name = "Diploma_Env"
    register_env(env_name, env_creator)

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

    def policy_mapping_fn(agent_id, **kwargs):
        return 'agent-{}'.format(agent_id)

    # Define configuration with hyperparam and training details
    config = {
        # 'num_training_iterations': 20,
        "use_critic": True,
        "lambda": 1.0,
        "kl_coeff": 0.1,
        "shuffle_sequences": True,
        "log_level": "INFO",
        # "num_cpus_per_worker": 8 / num_workers,
        "num_workers": num_workers,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "train_batch_size": num_steps,
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
    exp_name = 'exp_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": max_num_steps/num_steps/10
        },
        'checkpoint_freq': 25,
        "config": config,
        'num_samples': 6,
        'local_dir': './exp_res',
        'mode': 'max'
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)


if __name__ == '__main__':
    setup_and_train(n_steps, n_agents, n_workers)
