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
from copy import deepcopy


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

    def forward(self, data, **kwargs):
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
            output = self.readout(output, **kwargs)
        return output

class ConcatObsActionGNN(GNN):
    """
    Concatenate actions into state Data object.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, data, actions=None, **kwargs):
        if actions is not None:
            concat_node_features = torch.cat((data.x, actions), dim=self.dim)
            concat_data = deepcopy(data) #TODO: Evaluate impact
            concat_data.x = concat_node_features
        else:
            concat_data = data
        return super().forward(concat_data, **kwargs)