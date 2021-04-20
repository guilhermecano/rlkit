from larocs_sim.envs.ar_drone.ar_drone_graph import ARDroneGNNEnv
from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.torch.relational.networks import ConcatObsActionGAT
from rlkit.torch.sac.policies.gaussian_policy import TanhGATGaussianPolicy

from torch.nn import Linear
my_env = ARDroneGraphEnv


def experiment(variant):

    expl_env = NormalizedBoxEnv(my_env(headless=True,
                 init_strategy='gaussian',
                 clipped = False,
                 scaled = False))
    eval_env = expl_env
    num_node_features = expl_env.observation_space.low.size
    action_dim = 5
    action_feature_dim = 1

    q_kwargs = {"num_of_layers":2,
        "num_heads_per_layer":[2,1],
        "num_features_per_layer":[18 + action_feature_dim, 64, 64],
        "add_skip_connection":True,
        "bias":True,
        "dropout":0.6,
        "layer_type":LayerType.IMP3,
        "log_attention_weights":False,
        "readout":nn.Linear,
        "readout_sizes":[32, 1]}

    qf1 = ConcatObsActionGNN(**q_kwargs)
    qf2 = ConcatObsActionGNN(**q_kwargs)
    target_qf1 = ConcatObsActionGNN(**q_kwargs)
    target_qf2 = ConcatObsActionGNN(**q_kwargs)


    policy = TanhGNNGaussianPolicynum_of_layers=2,
             action_size=1, 
             std=None,
             activation = nn.Tanh,
             readout=nn.Linear,
             readout_activation=nn.Tanh,
             readout_sizes=[64, 1],
             num_heads_per_layer=[3, 1],
             num_features_per_layer=[18, 64, 64],
             add_skip_connection = True,
             bias=True,
             dropout=0.6,
             layer_type=LayerType.IMP3,
             log_attention_weights=False)

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        rollout_fn=rollout
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        save_env_in_snapshot=False,
        rollout_fn=rollout
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACGNNTrainer(
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
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=5,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=900,
            min_num_steps_before_training=1000,
            max_path_length=300,
            batch_size=8000,
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
    setup_logger('ardrone_gnn_teste', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
