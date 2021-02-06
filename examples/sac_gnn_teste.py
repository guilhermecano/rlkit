#from larocs_sim.envs.ar_drone.ar_drone import ARDroneEnv
from larocs_sim.envs.ar_drone.ar_drone_graph import ARDroneGraphEnv
from rlkit.samplers.rollout_functions import torch_geometric_rollout
from rlkit.torch.networks.gnns.geometric.networks import GNN
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.gnn_replay_buffer import GNNEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import GNNGaussianPolicy, MakeGNNDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import GNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from torch_geometric.nn import GATConv

my_env = ARDroneGraphEnv

def experiment(variant):

    expl_env = NormalizedBoxEnv(my_env(headless=True,
                 init_strategy='gaussian',
                 clipped = False,
                 scaled = False))
    eval_env = expl_env
    num_node_features = expl_env.observation_space.low.size
    action_dim = 1

    M = variant['layer_size']
    qf1 = GNN(
        hidden_sizes=[M, M],
        num_node_features=num_node_features,
        graph_propagation=GATConv,
        readout = None,
        num_edge_features = 0,
        output_size=1,
        output_activation=None,
        layer_norm=False,
        layer_norm_kwargs=None,
        gp_kwargs=None,
        readout_kwargs=None
    )

    qf2 = GNN(
        hidden_sizes=[M, M],
        num_node_features=num_node_features,
        graph_propagation=GATConv,
        readout = None,
        num_edge_features = 0,
        output_size=1,
        output_activation=None,
        layer_norm=False,
        layer_norm_kwargs=None,
        gp_kwargs=None,
        readout_kwargs=None
    )

    target_qf1 = GNN(
        hidden_sizes=[M, M],
        num_node_features=num_node_features,
        graph_propagation=GATConv,
        readout = None,
        num_edge_features = 0,
        output_size=1,
        output_activation=None,
        layer_norm=False,
        layer_norm_kwargs=None,
        gp_kwargs=None,
        readout_kwargs=None
    )

    target_qf2 = GNN(
        hidden_sizes=[M, M],
        num_node_features=num_node_features,
        graph_propagation=GATConv,
        readout = None,
        num_edge_features = 0,
        output_size=1,
        output_activation=None,
        layer_norm=False,
        layer_norm_kwargs=None,
        gp_kwargs=None,
        readout_kwargs=None
    )

    policy = GNNGaussianPolicy(
        num_node_features=num_node_features,
        action_size=action_dim,
        hidden_sizes=[M, M],
        graph_propagation=GATConv
    )

    eval_policy = MakeGNNDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        rollout_fn=torch_geometric_rollout
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        save_env_in_snapshot=False,
        rollout_fn=torch_geometric_rollout
    )
    replay_buffer = GNNEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    # algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=32,
        replay_buffer_size=int(10),
        algorithm_kwargs=dict(
            num_epochs=5000,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=900,
            min_num_steps_before_training=0,
            max_path_length=300,
            batch_size=800,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('teste_ardrone_gnn', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
