from copy import deepcopy

from torch.utils.data.dataset import Dataset


class DictDataset(Dataset):
    def __init__(self, data):
        self.data = [deepcopy(i) for i in data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
