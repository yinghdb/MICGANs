import torch

class RndClusterer():
    def __init__(self, num_k):
        super().__init__()
        self.num_k = num_k

    def sample_y(self, batch_size):
        return torch.randint(low=0, high=self.num_k, size=[batch_size]).long().cuda()

    def get_labels(self, x, y):
        return torch.randint(low=0, high=self.num_k, size=y.shape).long().cuda()

    def get_one_label(self):
        return  torch.randint(low=0, high=self.num_k, size=[1]).long().cuda()