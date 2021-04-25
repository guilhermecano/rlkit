import torch.nn as nn
import torch.nn.functional as F
from rlkit.policies.base import ExplorationPolicy
import torch
from rlkit.torch.networks import Mlp
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import numpy as np
from rlkit.torch.networks.gat.utils import LayerType
from rlkit.torch.networks.gat.gat import get_layer_type, GATLayer, GATLayerImp1, GATLayerImp2, GATLayerImp3

class GAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.
    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.
    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, activation=nn.ELU, log_attention_weights=False):
        super().__init__()
        self.num_of_layers = num_of_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.num_features_per_layer = num_features_per_layer
        self.dropout = dropout
        self.layer_type = layer_type
        self.bias = bias
        self.add_skip_connection = add_skip_connection
        self.log_attention_weights = log_attention_weights
        
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation= activation() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class ReadoutGAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.
    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.
    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False, readout=None, readout_sizes=[], readout_kwargs={}):
        super().__init__()
        self.num_of_layers = num_of_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.num_features_per_layer = num_features_per_layer
        self.dropout = dropout
        self.layer_type = layer_type
        self.bias = bias
        self.add_skip_connection = add_skip_connection
        self.log_attention_weights = log_attention_weights
        
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)
        
        self.gat_net = nn.Sequential(
                *gat_layers,
            )
        last_size = num_features_per_layer[-1]
        if readout is not None:
            readout_lst = []
            for n in readout_sizes:
                readout_lst.append(readout(last_size, n, **readout_kwargs))
                last_size = n
            self.readout = nn.Sequential(
                *readout_lst
            )
            
    def forward(self, data):
        x, edge_index = self.gat_net(data)
        x = self.readout(x)
        return x

class ConcatObsActionGAT(ReadoutGAT):
    """
    Concatenate actions into state Data object.
    """
    
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
    
    def forward(self, data, actions=None, **kwargs):
        if actions is not None:
            concat_node_features = torch.cat((data[0], actions), dim=self.dim)
            concat_node_features
            concat_data = (concat_node_features, data[1])
        else:
            concat_data = data
        return super().forward(concat_data, **kwargs)


# class PolicyReNN(PyTorchModule, ExplorationPolicy):
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


# class ValueReNN(PyTorchModule):
#     def __init__(self,
#                  graph_propagation,
#                  readout,
#                  input_module=FetchInputPreprocessing,
#                  input_module_kwargs=None,
#                  state_preprocessing_fnx=fetch_preprocessing,
#                  *args,
#                  value_mlp_kwargs=None,
#                  composite_normalizer=None,
#                  **kwargs):
#         self.save_init_params(locals())
#         super().__init__()
#         self.input_module = input_module(**input_module_kwargs)
#         self.graph_propagation = graph_propagation
#         self.readout = readout
#         self.composite_normalizer = composite_normalizer

#     def forward(self,
#                 obs,
#                 mask=None,
#                 return_stacked_softmax=False):
#         vertices = self.input_module(obs, mask=mask)
#         new_vertices = self.graph_propagation.forward(vertices, mask=mask)
#         pooled_output = self.readout(new_vertices, mask=mask)
#         return pooled_output


# class QValueReNN(PyTorchModule):
#     """
#     Used for q-value network
#     """

#     def __init__(self,
#                  graph_propagation,
#                  readout,
#                  input_module=FetchInputPreprocessing,
#                  input_module_kwargs=None,
#                  state_preprocessing_fnx=fetch_preprocessing,
#                  *args,
#                  composite_normalizer=None,
#                  **kwargs):
#         self.save_init_params(locals())
#         super().__init__()
#         self.graph_propagation = graph_propagation
#         self.state_preprocessing_fnx = state_preprocessing_fnx
#         self.readout = readout
#         self.composite_normalizer = composite_normalizer
#         self.input_module = input_module(**input_module_kwargs)

#     def forward(self, obs, actions, mask=None, return_stacked_softmax=False):
#         assert mask is not None
#         vertices = self.input_module(obs, actions=actions, mask=mask)
#         relational_block_embeddings = self.graph_propagation.forward(vertices, mask=mask)
#         pooled_output = self.readout(relational_block_embeddings, mask=mask)
#         assert pooled_output.size(-1) == 1
#         return pooled_output


