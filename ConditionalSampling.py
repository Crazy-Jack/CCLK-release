from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA
import torch.nn.functional as F


class ConditionalSamplingLoss(nn.Module):
    """Continuous version for conditional sampling loss"""
    def __init__(self, temperature=0.1, mode='hardnegatives',
                 temp_z=0.1, scale=1, lambda_=0.1, 
                 weight_clip_threshold=1e-6, distance_mode='cosine', inverse_device='cpu', inverse_gradient=False):
        super(ConditionalSamplingLoss, self).__init__()
        self.temp_z = temp_z
        self.lambda_ = lambda_
        self.ce_loss = nn.CrossEntropyLoss()
        self.mode = mode
        self.scale = scale
        self.cosinesim = nn.CosineSimilarity(dim=-1)
        self.weight_clip_threshold=weight_clip_threshold
        self.inverse_device = inverse_device
        self.distance_mode = distance_mode
        self.inverse_gradient = inverse_gradient


    def forward(self, raw_score, condition1, condition2, high_threshold=0.8, low_threshold=0.2, device='cuda:0', warmup=False):
        """
        raw_score: [2n, 2n],
        condition: [n, z_dim]

        1) Compute M = K_XY (K_Z + lambda I)^-1 K_Z
        2) build conditional sampling loss

        return loss (scalar)
        """
        n = int(raw_score.shape[0] / 2)

        if warmup:
            # use simclr to warmup for all cases
            targets = torch.arange(2 * n, dtype=torch.long, device=raw_score.device)
            loss = self.ce_loss(raw_score, targets)
            return loss

        Kxy = torch.exp(raw_score[:n, :n])
        Kxx = torch.exp(raw_score[:n, n:])
        Kyy = torch.exp(raw_score[n:, :n])
        Kyx = torch.exp(raw_score[n:, n:])


        if self.distance_mode == 'cosine':
            distance1 = self.cosinesim(condition1.unsqueeze(-2), condition1.unsqueeze(-3))
            distance2 = self.cosinesim(condition2.unsqueeze(-2), condition2.unsqueeze(-3))
        elif self.distance_mode == 'RBF':
            X_norm = torch.sum(condition1 ** 2, axis=-1)
            distance1 = 1 * torch.exp(- 1 / self.temp_z * (X_norm[:,None] + X_norm[None,:] - 2 * torch.matmul(condition1, condition1.T)))
            X_norm = torch.sum(condition2 ** 2, axis=-1)
            distance2 = 1 * torch.exp(- 1 / self.temp_z * (X_norm[:,None] + X_norm[None,:] - 2 * torch.matmul(condition2, condition2.T)))
        elif self.distance_mode == 'linear':
            distance1 = torch.matmul(condition1, condition1.T) / self.temp_z
            distance2 = torch.matmul(condition2, condition2.T) / self.temp_z
        elif self.distance_mode == 'polynomial':
            distance1 = torch.matmul(condition1, condition1.T)
            distance1 = distance1 ** 3 / self.temp_z
            distance2 = torch.matmul(condition2, condition2.T)
            distance2 = distance2 ** 3 / self.temp_z
        elif self.distance_mode == 'laplacian':
            X_diff_norm = torch.sqrt(((condition1[:, None] - condition1[None, :])**2).sum(-1))
            distance1 = 1 * torch.exp(- 1 / self.temp_z * (X_diff_norm))
            X_diff_norm = torch.sqrt(((condition2[:, None] - condition2[None, :])**2).sum(-1))
            distance2 = 1 * torch.exp(- 1 / self.temp_z * (X_diff_norm))
        else:
            raise NotImplementedError

        n = distance1.shape[0]
        distance1[range(n), range(n)] = 1.
        K_Z_x = distance1

        n = distance2.shape[0]
        distance2[range(n), range(n)] = 1.
        K_Z_y = distance2

        # compute weights
        weight_x = self.compute_weight(K_Z_x)
        weight_y = self.compute_weight(K_Z_y)

        # reweighting K matrix by weight
        Mxy = torch.matmul(Kxy, weight_x) # n, n
        Mxx = torch.matmul(Kxx, weight_x) # n, n
        Myx = torch.matmul(Kyx, weight_y) # n, n
        Myy = torch.matmul(Kyy, weight_y) # n, n


        # loss
        if self.mode in ['hardnegatives', 'weac-infonce']:

            # pos
            pos = torch.diagonal(raw_score[:n, :n], 0) # n,

            # negatives
            deno = torch.clamp(torch.exp(pos) + (n - 1) * (torch.diagonal(Mxy, 0) + torch.diagonal(Mxx, 0)), 1e-7, 1e+20)
            log_negatives = torch.log(deno) # n

            loss_x = - (pos - log_negatives).mean()

            pos = torch.diagonal(raw_score[n:, n:], 0) # n,
            deno = torch.clamp(torch.exp(pos) + (n - 1) * (torch.diagonal(Myx, 0) + torch.diagonal(Myy, 0)), 1e-7, 1e+20)
            log_negatives = torch.log(deno) # n

            loss_y = - (pos - log_negatives).mean()

            loss = (loss_x + loss_y) / 2

        elif self.mode in ['cl-infonce']:

            pos = torch.clamp(torch.diagonal(Mxx, 0) + torch.diagonal(Mxy, 0), 1e-7, 1e+20) # n,
            deno = torch.clamp(pos + Kxy.sum(1) + Kxx.sum(1), 1e-7, 1e+20)
            log_negatives = torch.log(deno)
            loss_x = - ( torch.log(pos) - log_negatives).mean()

            pos = torch.clamp(torch.diagonal(Myy, 0) + torch.diagonal(Myx, 0), 1e-7, 1e+20) # n,
            deno = torch.clamp(pos + Kyx.sum(1) + Kyy.sum(1), 1e-7, 1e+20)
            log_negatives = torch.log(deno)
            loss_y = - ( torch.log(pos) - log_negatives).mean()

            loss = (loss_x + loss_y) / 2

        return loss

    def compute_weight(self, K_Z, device='cuda'):
        """
        K_Z is numpy array [n, n]
        return weight in cuda: [n, n]
        tricks includes: 1) weight diag=0 to avoid recompute f(xi, yi),
                         2) remove very small values for numerical stability
                         3) clamp negative weight to avoid negative loss
        """

        n = K_Z.shape[0]

        
        if self.inverse_device == 'gpu':
            if self.inverse_gradient:
                inverse = torch.inverse(K_Z + self.lambda_ * torch.eye(n).to(device))
            else:
                with torch.no_grad():
                    K_Z = K_Z.detach()
                    inverse = torch.inverse(K_Z + self.lambda_ * torch.eye(n).to(device))
        else:
            K_Z = K_Z.detach().cpu().numpy()
            inverse = np.linalg.inv(K_Z + self.lambda_ * np.eye(n)).astype(np.float32)
            # to cuda to speed up
            inverse = torch.from_numpy(inverse).to(device)
            K_Z = torch.from_numpy(K_Z.astype(np.float32)).to(device)

        # calculate (Kz + lamda I)^-1 @ Kz
        weight = torch.matmul(inverse, K_Z)
        
        weight[range(n), range(n)] = 0. # avoid reconsider positive pairs
        #print(weight)
        weight[weight < self.weight_clip_threshold] = 0. # get rid of negatives

        return weight

    def normalization(self, distance):
        #return (distance - distance.mean()) / distance.std()
        return F.softmax(distance, dim=-1)




