import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import numpy as np
import torch.nn as nn

import os

def compute_purity(y_pred, y_true):
        """
        Calculate the purity, a measurement of quality for the clustering 
        results.
        
        Each cluster is assigned to the class which is most frequent in the 
        cluster.  Using these classes, the percent accuracy is then calculated.
        
        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0 
          while a perfect clustering has a purity of 1.

        """

        # get the set of unique cluster ids
        clusters = set(y_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)
        
        return float(correct) / len(y_pred)

def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5  # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    for x_next, y_next, _ in data_loader:
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        if n > N:
            break
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def get_most_recent(d, ext):
    if not os.path.exists(d):
        print('Directory', d, 'does not exist')
        return -1 
    its = []
    for f in os.listdir(d):
        try:
            it = int(f.split(ext + "_")[1].split('.pt')[0])
            its.append(it)
        except Exception as e:
            pass
    if len(its) == 0:
        print('Found no files with extension \"%s\" under %s' % (ext, d))
        return -1
    return max(its)

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d or type(m) == nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        nn.init.constant_(m.running_mean, 0.0)
        nn.init.constant_(m.running_var, 1.0)