import copy
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data._utils.collate import collate as _collate, default_collate_fn_map

import numpy as np


# pad
PaddedObject = namedtuple("PaddedObject", ["obj"])


def pad(obj):
    return PaddedObject(obj)


def track_mask(input: Union[torch.Tensor, np.ndarray]):
    return pad(torch.ones((input.shape[0]), dtype=torch.bool))


def pad_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # todo this will be more optimal than any code we'll write? it's in C++
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


pad_collate_fn_map = copy.deepcopy(default_collate_fn_map)
pad_collate_fn_map[torch.Tensor] = pad_collate_tensor_fn


def pad_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate([e.obj for e in batch], collate_fn_map=pad_collate_fn_map)


# pad8
Padded8Object = namedtuple("Padded8Object", ["obj"])


def pad8(obj):
    return Padded8Object(obj)


def track_mask8(input: Union[torch.Tensor, np.ndarray]):
    return pad8(torch.ones((input.shape[0]), dtype=torch.bool))


def pad8_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    max_len = max([elem.shape[0] for elem in batch])

    if max_len % 8 == 0:
        return pad_collate_tensor_fn(batch)

    elem = batch[0]
    batch.append(
        torch.zeros(
            (max_len + 8 - (max_len % 8), *elem.shape[1:]), dtype=batch[0].dtype
        )
    )

    return pad_collate_tensor_fn(batch)[:-1]


pad8_collate_fn_map = copy.deepcopy(default_collate_fn_map)
pad8_collate_fn_map[torch.Tensor] = pad8_collate_tensor_fn


def pad8_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate([e.obj for e in batch], collate_fn_map=pad8_collate_fn_map)


# chain
ChainObject = namedtuple("ChainObject", ["obj"])
ChainBatchTrackerObject = namedtuple("ChainBatchTrackerObject", ["obj"])


def chain(obj):
    return ChainObject(obj)


def track_batch(input: Union[torch.Tensor, np.ndarray]):
    return ChainBatchTrackerObject(torch.ones((input.shape[0]), dtype=torch.long))


def chain_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return torch.cat(batch, dim=0)


def chain_batch_tracker_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate(
        [i * e.obj for i, e in enumerate(batch)],
        collate_fn_map=chain_collate_fn_map,
    )


chain_collate_fn_map = copy.deepcopy(default_collate_fn_map)
chain_collate_fn_map[torch.Tensor] = chain_collate_tensor_fn
chain_collate_fn_map[ChainBatchTrackerObject] = chain_batch_tracker_collate_tensor_fn


def chain_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate([elem.obj for elem in batch], collate_fn_map=chain_collate_fn_map)


# add all new types to collate fn map
# note that once a recipe is selected for a given object it cannot be overwritten
# this is to avoid the following example scenario where pad(obj) is called but obj a
# dict and one of its values was already wrapped in chain.
collate_fn_map = copy.deepcopy(default_collate_fn_map)
collate_fn_map[PaddedObject] = pad_collate_object_fn
collate_fn_map[Padded8Object] = pad8_collate_object_fn
collate_fn_map[ChainObject] = chain_collate_object_fn
collate_fn_map[ChainBatchTrackerObject] = chain_batch_tracker_collate_tensor_fn


def collate(batch):
    r"""Extension of PyTorch's `default_collate` function to enable more advanced
    collation of samples of variable lengths.

    To specify how the collation recipe, wrap the objects using `pad` or `chain`.
    If the wrapped object is an Iterable or Mapping, all its elements will inherite
    the collation recipe. All objects that are already supported by `default_collate`
    can be wrapped.

    If an object is not wrapped, the default collation recipe will be used. i.e. the
    outcome will be identical to `default_collate`.

    `pad` or `chain` do not track any padding masks or batch index, since that might not
    always be needed. Use `track_mask` or `track_batch` to track masks or batch index
    for a particular array or tensor.

    Args:
        batch: a single batch to be collated
    """
    return _collate(batch, collate_fn_map=collate_fn_map)
