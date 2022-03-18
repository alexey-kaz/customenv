import ray
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env

# Import environment definition
from env import Diploma_Env

tf = try_import_tf()


# Driver code for training
def setup_and_train():
    # Create a single environment and register it
    def env_creator(_):
        return Diploma_Env()

    multi_env = Diploma_Env()
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
        "log_level": "INFO",
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 64,
        "train_batch_size": 200,
        # "rollout_fragment_length": 100,
        'horizon':  20,
        'no_done_at_end': False,
        'soft_horizon': True,
        "lr": 5e-3,
        "model": {"fcnet_hiddens": [8, 8]},
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "simple_optimizer": True,
        "env": "Diploma_Env"}

    # Define experiment details
    exp_name = 'my_exp'
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": 50
        },
        'checkpoint_freq': 5,
        "config": config,
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)


if __name__ == '__main__':
    setup_and_train()
