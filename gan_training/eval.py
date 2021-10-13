import numpy as np
import torch
from torch.nn import functional as F

from gan_training.utils import compute_purity

class Evaluator(object):
    def __init__(self,
                 generator,
                 encoder,
                 multi_gauss,
                 train_loader,
                 batch_size=64,
                 device=None):
        self.generator = generator
        self.encoder = encoder
        self.multi_gauss = multi_gauss
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.device = device

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            raise NotImplementedError()
        elif isinstance(y, int):
            y = torch.full((batch_size, ),
                           y,
                           device=self.device,
                           dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x

    def compute_purity_score(self):
        predicted_classes = []
        gt_labels = []

        with torch.no_grad():
            for x_real, y_gt, _ in self.train_loader:
                x_real = x_real.cuda()
                embeddings = self.encoder(x_real)
                probs, log_probs = self.multi_gauss.compute_embed_probs(embeddings)
                max_indexes = torch.argmax(log_probs, dim=1).cpu()
                predicted_classes.append(max_indexes)
                gt_labels.append(y_gt)

        predicted_classes = torch.cat(predicted_classes).numpy()
        gt_labels = torch.cat(gt_labels).numpy()

        score = compute_purity(predicted_classes, gt_labels)

        return score
