import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from rlkit.policies.base import Policy, ExplorationPolicy
from rlkit.torch.core import PyTorchModule
from rlkit.torch import pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm
from rlkit.torch.pytorch_util import activation_from_string


class GNN(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            num_node_features,
            graph_propagation,
            readout = None,
            num_edge_features = 0,
            output_size=1,
            hidden_activation=F.relu,
            output_activation=None,
            layer_norm=False,
            layer_norm_kwargs=None,
            gp_kwargs=None,
            readout_kwargs=None
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        if gp_kwargs is None:
            gp_kwargs = dict()
        if readout_kwargs is None:
            readout_kwargs = dict()
        
        # Using PyTorch Geometric Data Structures
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.readout = readout
        self.graph_propagation = graph_propagation
        self.readout = readout
        self.layer_norm = layer_norm
        self.gpls = []
        self.layer_norms = []
        self.output_activation = output_activation
        self.gp_kwargs = gp_kwargs
        self.readout_kwargs = readout_kwargs
        node_size = self.num_node_features

        for i, next_size in enumerate(hidden_sizes):
            gpl = self.graph_propagation(node_size, next_size, **self.gp_kwargs)
            node_size = next_size
            self.__setattr__("gp{}".format(i), gpl)
            self.gpls.append(gpl)
            
            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
        
        self.last_gp = self.graph_propagation(node_size, output_size, **self.gp_kwargs)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, gpl in enumerate(self.gpls):
            x = gpl(x, edge_index)
            if self.layer_norm and i < len(self.gpls) - 1:
                x = self.layer_norms[i](x)
            x = self.hidden_activation(x)
        output = self.last_gp(x, edge_index)
        if self.output_activation is not None:
            output = self.output_activation(output)
        if self.readout is not None:
            output = self.readout(output, **self.readout_kwargs)
        return output


# class PolicyGAT(PyTorchModule, ExplorationPolicy):
#     """
#     Used for policy network
#     """

#     def __init__(self,
#                  graph_propagation,
#                  readout,
#                  *args,
#                  input_module=FetchInputPreprocessing,
#                  input_module_kwargs=None,
#                  mlp_class=FlattenTanhGaussianPolicy,
#                  composite_normalizer=None,
#                  batch_size=None,
#                  **kwargs):

#         self.save_init_params(locals())
#         super().__init__()
#         self.composite_normalizer = composite_normalizer

#         # Internal modules
#         self.graph_propagation = graph_propagation
#         self.selection_attention = readout

#         self.mlp = mlp_class(**kwargs['mlp_kwargs'])
#         self.input_module = input_module(**input_module_kwargs)

#     def forward(self,
#                 obs,
#                 mask=None,
#                 demo_normalizer=False,
#                 **mlp_kwargs):
#         assert mask is not None
#         vertices = self.input_module(obs, mask=mask)
#         response_embeddings = self.graph_propagation.forward(vertices, mask=mask)

#         selected_objects = self.selection_attention(
#             vertices=response_embeddings,
#             mask=mask
#         )
#         selected_objects = selected_objects.squeeze(1)
#         return self.mlp(selected_objects, **mlp_kwargs)

#     def get_action(self,
#                    obs_np,
#                    **kwargs):
#         assert len(obs_np.shape) == 1
#         actions, agent_info = self.get_actions(obs_np[None], **kwargs)
#         assert isinstance(actions, np.ndarray)
#         return actions[0, :], agent_info

#     def get_actions(self,
#                     obs_np,
#                     **kwargs):
#         mlp_outputs = self.eval_np(obs_np, **kwargs)
#         assert len(mlp_outputs) == 8
#         actions = mlp_outputs[0]

#         agent_info = dict()
#         return actions, agent_info