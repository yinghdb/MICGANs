import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F



class LatentEmbeddingConcat(nn.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def forward(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        return yz

class LatentEmbeddingAdd(nn.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim, norm=True):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)
        self.norm = norm

    def forward(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        assert (z.size(1) == yembed.size(1))
        if self.norm:
            z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        yz = z + yembed
        return yz

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inp, *args, **kwargs):
        return inp


class LinearConditionalMaskLogits(nn.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number'''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, nlabels)

    def forward(self, inp, y=None, take_best=False, get_features=False):
        out = self.fc(inp)
        if get_features: return out

        if not take_best:
            y = y.view(-1)
            index = Variable(torch.LongTensor(range(out.size(0))))
            if y.is_cuda:
                index = index.cuda()
            return out[index, y]
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits


class LinearUnconditionalLogits(nn.Module):
    ''' standard discriminator logit layer '''

    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(nc, 1)

    def forward(self, inp, y, take_best=False):
        assert (take_best == False)

        out = self.fc(inp)
        return out.view(out.size(0))


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(*((batch_size, ) + self.shape))


class ConditionalBatchNorm2d(nn.Module):
    ''' from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775 '''

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialize scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_(
        )  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out


class BatchNorm2d(nn.Module):
    ''' identical to nn.BatchNorm2d but takes in y input that is ignored '''

    def __init__(self, nc, nchannels, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(nc)

    def forward(self, x, y):
        return self.bn(x)
