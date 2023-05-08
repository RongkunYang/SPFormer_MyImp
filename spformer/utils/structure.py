import torch
from numpy import ndarray
from typing import Any, Dict, Union


class Instances3D:
    def __init__(self, num_points: int, gt_instances: ndarray = None, **kwargs: Any):
        self._num_points = num_points
        self._gt_instances = gt_instances
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def gt_instances(self) -> ndarray:
        return self._gt_instances

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            super(Instances3D, self).__setattr__(key, value)
        else:
            self.set(key, value)

    def __getattr__(self, name: str) -> Any:
        if name == '_fields' or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, 'Adding a field of length {} to a Instances of length {}'.format(
                data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_field(self) -> Dict[str, Any]:
        return self._fields

    def to(self, *args: Any, **kwargs: Any) -> 'Instances3D':
        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def cuda(self, *args: Any, **kwargs: Any) -> 'Instances3D':
        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            if hasattr(v, 'cuda'):
                v = v.cuda(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> 'Instances3D':
        if type(item) == int:
            if item >= len(self) or item <= -len(self):
                raise IndexError('Instances index out of range!')
            else:
                item = slice(item, None, len(self))

        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError('`Empty Instances does not support __len__!`')

    def __iter__(self):
        raise NotImplementedError('`Instances` object is not iterable!')

    def __str__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self))
        s += 'num_points={}'.format(self._num_points)
        s += 'fields=[{}]'.format(', '.join((f'{k}:{v}' for k, v in self._fields.items())))
        return s

    __repr__ = __str__
