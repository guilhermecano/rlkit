import abc

import numpy as np
import torch
from torch import nn as nn
from torch_geometric.data import Batch, Data

from rlkit.torch import pytorch_util as ptu


class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass

@profile
def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    return elem_or_tuple_to_numpy(outputs)

@profile
def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return ptu.from_geom_dataset(np_array_or_other)

@profile
def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other

@profile
def _elem_or_tuple_to_variable(elem_or_tuple): # TODO: This can compromise conventional algorithms, create one specially for GNNs later
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    if elem_or_tuple.dtype != np.dtype('O'):
        tensor_elem = ptu.from_numpy(elem_or_tuple).float()
        # Workaround for making the graph action shape compatible to others
        if len(tensor_elem.shape) == 3:
            dim = tensor_elem.shape # "Actions" is the only 3D tensor here
            tensor_elem = tensor_elem.view(dim[0]*dim[1], dim[2])
        return tensor_elem
    else:
        batch_lst = []
        for v in elem_or_tuple: #TODO: Remover essa aberração quando eu descobrir de onde vem Data em cpu, sendo que era pra estar todo mundo em GPU.
            batch_lst.append(ptu.from_geom_dataset(v))
        b = Batch()
        geom_batch = b.from_data_list(batch_lst)
        # geom_batch = ptu.from_datalist_to_batch(elem_or_tuple)
        return geom_batch

@profile
def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)

@profile
def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

@profile
def np_to_pytorch_batch(np_batch): # TODO: This can compromise conventional algorithms, create one specially for GNNs later
    if isinstance(np_batch, dict):
        return {
            k: _elem_or_tuple_to_variable(x)
            for k, x in _filter_batch(np_batch)
            # if x.dtype != np.dtype('O')  # no longer ignore object (e.g. dictionaries)
        }
    else:
        _elem_or_tuple_to_variable(np_batch)
