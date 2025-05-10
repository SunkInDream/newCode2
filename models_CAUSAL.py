import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.impute import KNNImputer
import random
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from models_TCDF import *
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.fftpack import fft, ifft
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from contextlib import redirect_stdout
#from models_runTCDF import run_tcdf_analysis
from sklearn.preprocessing import StandardScaler
import time
import glob
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from io import StringIO
import io
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn as nn
import concurrent.futures
def FirstProcess(file):
    df = pd.read_csv(file)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
         # 对于全空的列，填充为-1
            df[column] = -1
        else:
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts()
                if not value_counts.empty:
                    mode_value = value_counts.index[0]
                    mode_count = value_counts.iloc[0]
                    # 使用有效数据数量判断是否超过阈值
                    if mode_count >= 0.8 * len(non_null_data):
                        df[column] = col_data.fillna(mode_value)
    return df

def SecondProcess(file, perturbation_prob=0.1, perturbation_scale=0.1):
    df_copy = file.copy()
    
    for column in df_copy.columns:
        series = df_copy[column]
        missing_mask = series.isna()

        if not missing_mask.any():
            continue  # 如果没有缺失值，跳过该列

        # 后面的代码保持不变
        missing_segments = []
        start_idx = None

        # 查找缺失值的连续段
        for i, is_missing in enumerate(missing_mask):
            if is_missing and start_idx is None:
                start_idx = i
            elif not is_missing and start_idx is not None:
                missing_segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            missing_segments.append((start_idx, len(missing_mask) - 1))

        # 对每个缺失段进行填补
        for start, end in missing_segments:
            left_value, right_value = None, None
            left_idx, right_idx = start - 1, end + 1

            # 找到前后最近的非缺失值
            while left_idx >= 0 and np.isnan(series.iloc[left_idx]):
                left_idx -= 1
            if left_idx >= 0:
                left_value = series.iloc[left_idx]

            while right_idx < len(series) and np.isnan(series.iloc[right_idx]):
                right_idx += 1
            if right_idx < len(series):
                right_value = series.iloc[right_idx]

            # 如果前后都没有非缺失值，使用均值填充
            if left_value is None and right_value is None:
                fill_value = series.dropna().mean()
                df_copy.loc[missing_mask, column] = fill_value
                continue

            # 如果只有一个方向有非缺失值，使用另一个方向的值填充
            if left_value is None:
                left_value = right_value
            elif right_value is None:
                right_value = left_value

            # 使用等差数列填补缺失值
            segment_length = end - start + 1
            step = (right_value - left_value) / (segment_length + 1)
            values = [left_value + step * (i + 1) for i in range(segment_length)]

            # 添加扰动
            value_range = np.abs(right_value - left_value) or (np.abs(left_value) * 0.1 if left_value != 0 else 1.0)
            for i in range(len(values)):
                if random.random() < perturbation_prob:
                    perturbation = random.uniform(-1, 1) * perturbation_scale * value_range
                    values[i] += perturbation

            # 将填补后的值赋回数据框
            for i, value in enumerate(values):
                df_copy.iloc[start + i, df_copy.columns.get_loc(column)] = value

    return df_copy
def task(args):
    file, params, gpu = args
    matrix, columns = compute_causal_matrix(file, params, gpu)
    print(f"\nResult for {os.path.basename(file)}:")
    print(np.array(matrix))
    return matrix   