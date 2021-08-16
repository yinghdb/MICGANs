import torch
from clusterers import base_clusterer


class Clusterer(base_clusterer.BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_y(self, batch_size):
        return torch.randint(low=0, high=self.num_k, size=[batch_size]).long().cuda()

    def get_labels(self, x, y):
        return torch.randint(low=0, high=self.num_k, size=y.shape).long().cuda()

    def get_one_label(self):
        return  torch.randint(low=0, high=self.num_k, size=[1]).long().cuda()