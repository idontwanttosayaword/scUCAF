import h5py
import numpy as np
import scanpy as sc
import scipy
import seaborn as sns
import pandas as pd
import scipy as sp
from statsmodels.nonparametric.smoothers_lowess import lowess


class CreateDataset:
    def __init__(self, config):
        self.adata1 = sc.read_h5ad(config.h5ad_file1)
        self.adata2 = sc.read_h5ad(config.h5ad_file2)

        self.x1 = self.adata1.X
        self.x2 = self.adata2.X
        self.y = self.adata1.obs['y']

        unique_classes = np.unique(self.y)
        num_classes = len(unique_classes)

   