import os
import numpy as np
import pandas as pd
from models_CAUSAL import *
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file_paths, tag_file=None, tag_name=None, id_name=None): 
        self.file_paths = file_paths  
        self.file_names = []             # 文件名列表
        self.original_data = []          # 含nan的原始数据 (list of ndarray)
        self.mask_data = []              # 掩码矩阵，1表示有值，0表示缺失 (list of ndarray)
        self.initial_filled = []         # 初次填补的结果 (list of ndarray)
        self.final_filled = []           # 最终填补结果 (list of ndarray, 默认为None)
        self.center_repre = []           # 中心表示 (list of ndarray)
        self.labels = []                 # 标签 (list)
        self.total_causal_matrix = None  # 总因果矩阵 (ndarray, 默认为None)
        files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
        for file in files:
            basename = os.path.basename(file)           # 获取文件名: 229831.csv
            number_str = os.path.splitext(basename)[0]  # 去除扩展名: 229831
            self.file_names.append(number_str)          # 存储纯数字部分
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
        if tag_file is not None and tag_name is not None and id_name is not None: #读取标签文件
            tag_df = pd.read_csv(tag_file)
            id_to_tag = dict(zip(tag_df[id_name].astype(str), tag_df[tag_name]))
            self.labels = [id_to_tag.get(file_id, None) for file_id in self.file_names]
        else:
            self.labels = [None] * len(self.file_names)
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return {
            'file_names': self.file_names[idx],
            'original': self.original_data[idx],
            'mask': self.mask_data[idx],
            'initial_filled': self.initial_filled[idx],
            'final_filled':self.final_filled[idx],
            'labels': self.labels[idx],
            'total_causal_matrix': self.total_causal_matrix,
        }
    def agregate(self, n_cluster):
        self.center_repre = []
        data = np.array([np.nanmean(x, axis=0) for x in self.initial_filled])
        km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
        labels = km.fit_predict(data)

        for k in range(n_cluster):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0: 
                continue
            cluster_data = data[idxs]
            dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
            best_idx = idxs[np.argmin(dists)]
            self.center_repre.append(int(best_idx))
            
        return self.center_repre  
