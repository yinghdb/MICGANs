import copy

import torch
import numpy as np

class BaseClusterer():
    def __init__(self,
                 num_k=-1,
                 batch_size=100,
                 **kwargs):
        ''' requires that self.x is not on the gpu, or else it hogs too much gpu memory ''' 
        self.num_k = num_k
        self.batch_size = batch_size

    def get_labels(self, x, y):
        return y


    def print_label_distribution(self, x=None):
        print(self.get_label_distribution(x))
