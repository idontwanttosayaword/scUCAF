import os
from typing import Any, Dict, Optional
import torch
import random
import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
import torch.nn.functional as F
from torch import nn


def square_euclid_distance(Z, center):
    ZZ = torch.sum(Z * Z, dim=1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = torch.sum(center * center, dim=1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = torch.mm(Z, center.T)
    distance = ZZ_CC - 2 * ZC

    return distance


def high_confidence(Z, center, tao=0.7):
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values

    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tao)))
    index = torch.where(distance_norm <= value[-1],
                        torch.ones_like(distance_norm),
                        torch.zeros_like(distance_norm))

    high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
    H = high_conf_index_v1

    H_cpu = H.cpu().numpy()
    H_mat = np.ix_(H_cpu, H_cpu)

    return H, H_mat


def comprehensive_similarity(p1, p2):
    sim_matrix = F.normalize(p1) @ F.normalize(p2).T
    return sim_matrix


def hard_sample_aware_infoNCE(S, Mask, pos_neg_weight, pos_weight, node_num):
    pos_neg = Mask * torch.exp(S * pos_neg_weight)
    pos = torch.cat([torch.diag(S, node_num), torch.diag(S, -node_num)], dim=0)
    pos = torch.exp(pos * pos_weight)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
    return infoNEC


def pseudo_matrix(P, S):
    Q = (P == P.unsqueeze(1)).float()
    S_norm = (S - S.min()) / (S.max() - S.min())
    M_mat = (1 - Q) * (-torch.log(1 - S_norm + 1e-8))
    return M_mat
