from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch
from torch._six import string_classes
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
from torch_geometric.data import Batch, Data, HeteroData


class Collater:
    def __call__(self, batch, *, outer=True):
        """
        ref: torch.utils.data._utils.collate.default_collate
        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, Mapping):
            return {key: self([d[key] for d in batch], outer=False) for key in elem}

        if outer:
            if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
                return elem_type(*(self(samples, outer=False) for samples in zip(*batch)))
            elif isinstance(elem, Sequence) and not isinstance(elem, string_classes):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError("each element in list of batch should be of equal size")
                transposed = zip(*batch)
                return [self(samples, outer=False) for samples in transposed]

        if isinstance(elem, (Data, HeteroData)):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self([torch.as_tensor(b) for b in batch], outer=False)
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return self(elem_type(*(self(samples, outer=False) for samples in batch)), outer=False)
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                return batch
            return self([self(samples, outer=False) for samples in batch], outer=False)

        raise TypeError(default_collate_err_msg_format.format(elem_type))


class DataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        def wrap(batch):
            return ori_collate_fn(Collater(follow_batch, exclude_keys)(batch))

        if "collate_fn" in kwargs:
            ori_collate_fn = kwargs.pop("collate_fn")
            collate_fn = wrap
        else:
            collate_fn = Collater(follow_batch, exclude_keys)

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)
