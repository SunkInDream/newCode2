import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import queue
import time
from sklearn.cluster import KMeans
from models_CAUSAL import *
from models_TCDF import ADDSTCN
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from models_impute import process_single_matrix
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
def compute_single_causal(args):
    mat, params, gpu = args
    return compute_causal_matrix(mat, params, gpu)[0]

class MyDataset(Dataset):
    def __init__(self, file_paths, tag_file=None, tag_name=None, id_name=None): 
        self.file_paths = file_paths  
        self.file_names = []  # 文件名列表
        self.original_data = []        # 含nan的原始数据 (list of ndarray)
        self.mask_data = []            # 掩码矩阵，1表示有值，0表示缺失 (list of ndarray)
        self.initial_filled = []       # 初次填补的结果 (list of ndarray)
        self.final_filled = []         # 最终填补结果 (list of ndarray, 默认为None)
        self.center_repre = []       # 中心表示 (list of ndarray)
        self.labels = []          # 标签 (list)
        self.total_causal_matrix = None  # 总因果矩阵 (ndarray, 默认为None)
        files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
        for file in files:
            basename = os.path.basename(file)  # 获取文件名: 229831.csv
            number_str = os.path.splitext(basename)[0]  # 去除扩展名: 229831
            self.file_names.append(number_str)  # 存储纯数字部分
            df = pd.read_csv(file)
            data = df.values.astype(np.float32)
            self.original_data.append(data)
            first_prepro_data =  FirstProcess(file)
            mask = (~first_prepro_data.isna()).values.astype(np.float32)
            self.mask_data.append(mask)
            second_prepro_data = SecondProcess(first_prepro_data)
            second_prepro_data = second_prepro_data.values.astype(np.float32)
            self.initial_filled.append(second_prepro_data)
            self.final_filled.append(None)
        # 修改__init__方法末尾部分
        if tag_file is not None and tag_name is not None and id_name is not None:
            # 读取tag文件
            tag_df = pd.read_csv(tag_file)
            
            # 创建ID到标签的映射字典
            id_to_tag = dict(zip(tag_df[id_name].astype(str), tag_df[tag_name]))
            
            # 将标签映射到文件名上
            self.labels = [id_to_tag.get(file_id, None) for file_id in self.file_names]
        else:
            self.labels = [None] * len(self.file_names)
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return {
            'original': self.original_data[idx],
            'mask': self.mask_data[idx],
            'initial_filled': self.initial_filled[idx],
            'final_filled':self.final_filled[idx],
            'file_names': self.file_names[idx],
            'labels': self.labels[idx]
            #'total_causal_matrix': torch.tensor(self.total_causal_matrix) if self.total_causal_matrix is not None else None
        }
    def agr(self, n_cluster, params=None):
        # 仅执行聚类，返回中心表示
        data = np.array([np.nanmean(x, axis=0) for x in self.initial_filled])
        data = np.nan_to_num(data, nan=0.0)
        km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
        labels = km.fit_predict(data)

        self.center_repre = []
        
        for k in range(n_cluster):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:  # 处理空聚类
                continue
            cluster_data = data[idxs]
            dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
            best_idx = idxs[np.argmin(dists)]
            self.center_repre.append(self.initial_filled[best_idx])
            
        return self.center_repre  # 返回中心表示，不进行因果发现  
