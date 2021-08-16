import torch
from torch import nn

import numpy as np
from sklearn.decomposition import PCA

class MultiGaussian(nn.Module):
    def __init__(self, num_k, embed_dim, fix_mean, sigma_scalor, dim_reduce, reduce_ratio):
        super(MultiGaussian, self).__init__()

        # initialize means
        if num_k == 3 and embed_dim == 2:
            mean = torch.zeros(num_k, embed_dim)
            mean[0, 0] = 1.0
            mean[0, 1] = 0.0
            mean[1, 0] = -0.5
            mean[1, 1] = 0.866
            mean[2, 0] = -0.5
            mean[2, 1] = -0.866
        else:
            mean = torch.zeros(num_k, embed_dim)
            for i in range(num_k):
                mean[i, i] = 1.0
        self.mean = nn.Parameter(mean)
        self.mean.requires_grad = False

        # initialize sigma and related parameters
        self.sigma = nn.Parameter(torch.eye(embed_dim).unsqueeze(0).repeat(num_k, 1, 1) * 0.1)
        self.sigma_det_recip = nn.Parameter(1.0 / (torch.sqrt(torch.det(self.sigma))))
        self.sigma_det_recip_scalor = nn.Parameter(torch.zeros(num_k))
        self.sigma_inverse = nn.Parameter(torch.inverse(self.sigma))

        # no update in training
        self.sigma.requires_grad = False
        self.sigma_det_recip.requires_grad = False
        self.sigma_det_recip_scalor.requires_grad = False
        self.sigma_inverse.requires_grad = False

        self.num_k = num_k
        self.embed_dim = embed_dim
        self.removed_list = []
        self.sigma_scalor = sigma_scalor
        self.fix_mean = fix_mean

        self.reduce = dim_reduce
        self.reduce_ratio = reduce_ratio
        self.mean_reduce = None


    def update_sigma(self, sigma):
        self.sigma.data = sigma
        self.sigma_det_recip.data = 1.0 / (torch.sqrt(torch.det(self.sigma)))
        self.sigma_inverse.data = torch.inverse(self.sigma)

    def compute_embed_probs_reduce(self, embed):
        if self.mean_reduce is None:
            return self.compute_embed_probs(embed)

        embed_array = embed.cpu().numpy()
        embed = torch.tensor(self.pca.transform(embed_array)).cuda()
    
        embed_diff = (embed[:, None, :] - self.mean_reduce[None, :, :])
        diff_1 = embed_diff.unsqueeze(2)
        diff_2 = embed_diff.unsqueeze(3)

        prob_1 = 1.0 / np.power(2 * np.pi, 0.5 * self.embed_dim) * self.sigma_det_recip_reduce
        diff_mat = -0.5*(torch.matmul(torch.matmul(diff_1, self.sigma_inverse_reduce), diff_2)).squeeze(3).squeeze(2)
        # diff_scale = diff_mat.max(dim=1)[0] # avoid too 0.0 prob 
        # prob_2 = torch.exp(diff_mat - diff_scale[:, None])
        # prob_2 = torch.exp(diff_mat + self.sigma_det_recip_scalor_reduce)
        prob_2 = diff_mat + self.sigma_det_recip_scalor_reduce
        prob_2[prob_2>80] = 80
        prob_2 = torch.exp(prob_2)
        prob = prob_1 * prob_2

        if torch.isnan(prob_1.max()) or torch.isinf(prob_1.max()):
            print("prob 1 nan!")
        if torch.isnan(prob_2.max()) or torch.isinf(prob_2.max()):
            print("prob 2 nan!")
        if torch.isnan(prob.max()) or torch.isinf(prob.max()):
            print("prob nan!")
            
        log_prob_1 = torch.log(prob_1)
        log_prob_2 = diff_mat + self.sigma_det_recip_scalor_reduce
        log_prob = log_prob_1 + log_prob_2

        return prob, log_prob

    def compute_embed_probs(self, embed):
        embed_diff = (embed[:, None, :] - self.mean[None, :, :])
        diff_1 = embed_diff.unsqueeze(2)
        diff_2 = embed_diff.unsqueeze(3)

        prob_1 = 1.0 / np.power(2 * np.pi, 0.5 * self.embed_dim) * self.sigma_det_recip
        diff_mat = -0.5*(torch.matmul(torch.matmul(diff_1, self.sigma_inverse), diff_2)).squeeze(3).squeeze(2)
        # diff_scale = diff_mat.max(dim=1)[0] # avoid too 0.0 prob 
        # prob_2 = torch.exp(diff_mat - diff_scale[:, None])
        prob_2 = torch.exp(diff_mat + self.sigma_det_recip_scalor)
        prob = prob_1 * prob_2

        if torch.isnan(prob_1.max()) or torch.isinf(prob_1.max()):
            print("prob 1 nan!")
        if torch.isnan(prob_2.max()) or torch.isinf(prob_2.max()):
            print("prob 2 nan!")
        if torch.isnan(prob.max()) or torch.isinf(prob.max()):
            print("prob nan!")
            
        log_prob_1 = torch.log(prob_1)
        log_prob_2 = diff_mat + self.sigma_det_recip_scalor
        log_prob = log_prob_1 + log_prob_2

        return prob, log_prob
    
    def get_means(self, labels=None):
        if labels is None:
            return self.mean
        else:
            return self.mean[labels]

    def update_reduce(self, embeddings_list):
        mean_reduce = self.pca.transform(self.mean.cpu().numpy())
        self.mean_reduce = torch.tensor(mean_reduce).cuda()

        self.sigma_reduce = torch.eye(self.reduce_dim).unsqueeze(0).repeat(self.num_k, 1, 1).cuda()
        self.sigma_det_recip_reduce = 1.0 / (torch.sqrt(torch.det(self.sigma_reduce)))
        self.sigma_det_recip_scalor_reduce = torch.zeros(self.num_k).cuda()
        self.sigma_inverse_reduce = torch.inverse(self.sigma_reduce)

        for k, embeddings in enumerate(embeddings_list):
            if k in self.removed_list:
                continue
            if len(embeddings) <= (self.embed_dim+1) * 10:
                self.mean[k] = self.mean[k] - 10000 # remove the mode
                self.removed_list.append(k)
                continue

            embeddings_array = embeddings.cpu().numpy()
            embeddings = torch.tensor(self.pca.transform(embeddings_array)).cuda()

            mean = embeddings.mean(dim=0)
            embedding_diff = (embeddings[:, :] - mean) * self.sigma_scalor
            embedding_diff_t = embedding_diff.transpose(0,1)
            sigma = torch.matmul(embedding_diff_t, embedding_diff)
            sigma = sigma / (len(embedding_diff) - 1)
            # to avoid sigma_det_recip being inf, use a scalor reduce its value
            det_scalor = torch.tensor(0.0)
            sigma_det = torch.det(sigma * torch.exp(det_scalor))
            while sigma_det < 1e-20:
                det_scalor += 1
                sigma_det = torch.det(sigma * torch.exp(det_scalor))
            sigma_det_recip = 1.0 / (torch.sqrt(sigma_det))
            sigma_det_recip_scalor = det_scalor * self.embed_dim / 2
            # if sigma_det_recip_scalor > 80:
            #     continue    # fix the distribution

            sigma_inverse = torch.inverse(sigma)

            if torch.isnan(sigma_det_recip) or torch.isnan(sigma.sum()) or torch.isnan(sigma_inverse.sum()) or \
                torch.isinf(sigma_det_recip) or torch.isinf(sigma.sum()) or torch.isinf(sigma_inverse.sum()):
                print(f"{k}: distribution fixed for nan!")
                continue    # fix the distribution

            if sigma_det_recip > 1e20:
                print(f"{k}: distribution fixed for sigma det!")
                continue    # fix the distribution

            # end
            self.sigma_reduce[k] = sigma
            self.sigma_det_recip_reduce[k] = sigma_det_recip
            self.sigma_det_recip_scalor_reduce[k] = sigma_det_recip_scalor
            self.sigma_inverse_reduce[k] = sigma_inverse
            self.mean_reduce[k] = mean

            mean_ori = torch.tensor(self.pca.inverse_transform(np.array([mean.cpu().numpy()]))).cuda()
            self.mean[k] = mean_ori

    def update(self, embeddings_list):
        for k, embeddings in enumerate(embeddings_list):
            if k in self.removed_list:
                continue
            if len(embeddings) <= (self.embed_dim+1) * 10:
                self.mean[k] = self.mean[k] - 10000 # remove the mode
                self.removed_list.append(k)
                continue

            if self.fix_mean:
                mean = self.mean[k]
            else:
                mean = embeddings.mean(dim=0)
            embedding_diff = (embeddings[:, :] - mean) * self.sigma_scalor
            embedding_diff_t = embedding_diff.transpose(0,1)
            sigma = torch.matmul(embedding_diff_t, embedding_diff)
            sigma = sigma / (len(embedding_diff) - 1)
            # to avoid sigma_det_recip being inf, use a scalor reduce its value
            det_scalor = torch.tensor(0.0)
            sigma_det = torch.det(sigma * torch.exp(det_scalor))
            while sigma_det < 1e-20:
                det_scalor += 1
                sigma_det = torch.det(sigma * torch.exp(det_scalor))
            sigma_det_recip = 1.0 / (torch.sqrt(sigma_det))
            sigma_det_recip_scalor = det_scalor * self.embed_dim / 2
            if sigma_det_recip_scalor > 80:
                continue    # fix the distribution

            sigma_inverse = torch.inverse(sigma)

            if torch.isnan(sigma_det_recip) or torch.isnan(sigma.sum()) or torch.isnan(sigma_inverse.sum()) or \
                torch.isinf(sigma_det_recip) or torch.isinf(sigma.sum()) or torch.isinf(sigma_inverse.sum()):
                print(f"{k}: distribution fixed for nan!")
                continue    # fix the distribution

            if sigma_det_recip > 1e20:
                print(f"{k}: distribution fixed for sigma det!")
                continue    # fix the distribution

            # end
            self.sigma[k] = sigma
            self.sigma_det_recip[k] = sigma_det_recip
            self.sigma_det_recip_scalor[k] = sigma_det_recip_scalor
            self.sigma_inverse[k] = sigma_inverse
            if not self.fix_mean:
                self.mean[k] = mean

    def get_remove_list(self):
        return self.removed_list

    def pca_transform(self, embeddings):
        X = embeddings.cpu().numpy()
        pca = PCA(n_components=self.embed_dim)
        pca.fit(X)
        ratios = pca.explained_variance_ratio_

        print("pca ratios:", ratios)
        reduce_dim = 0
        r = 0
        for ratio in ratios:
            reduce_dim += 1
            r += ratio
            if r > self.reduce_ratio:
                break
        self.reduce_dim = reduce_dim
        self.pca = PCA(n_components=reduce_dim)
        self.pca.fit(X)