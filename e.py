import os
import shutil
from typing import Optional
import os
import torch
import random
import numpy as np
import pandas as pd
from models_TCDF import ADDSTCN
import torch.nn.functional as F
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
def copy_files(src_dir: str, dst_dir: str, num_files: int = -1, file_ext: Optional[str] = None):
    """
    复制 src_dir 下的指定数量文件到 dst_dir。
    
    参数:
        src_dir (str): 源目录路径。
        dst_dir (str): 目标目录路径。
        num_files (int): 要复制的文件数量。如果为 -1，复制所有文件。
        file_ext (str, optional): 只复制指定扩展名的文件，例如 '.txt'。默认复制所有文件。
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = os.listdir(src_dir)
    files = [f for f in files if os.path.isfile(os.path.join(src_dir, f))]

    if file_ext:
        files = [f for f in files if f.lower().endswith(file_ext.lower())]

    if num_files != -1:
        files = files[:num_files]

    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        shutil.copy2(src_path, dst_path)
        print(f"已复制: {f}")
from scipy.integrate import odeint
from omegaconf import OmegaConf

opt = OmegaConf.load("opt/lorenz_example.yaml")
opt_data = opt.data
def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt 
def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:, :], GC
def generate_multiple_lorenz_datasets(num_datasets, p, T, seed_start=0):
    datasets = []
    for i in range(num_datasets):
        X, GC = simulate_lorenz_96(p=p, T=T, seed=seed_start+i)
        datasets.append((X, GC))
    return datasets
def save_lorenz_datasets_to_csv(datasets, output_dir):
    """
    将Lorenz-96数据集保存为CSV文件
    
    参数:
    datasets -- 数据集列表，每个元素为(X, GC)元组
    output_dir -- 保存CSV文件的目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有数据集
    for i, (X, GC) in enumerate(datasets):
        # 构造文件名
        X_filename = os.path.join(output_dir, f"lorenz_dataset_{i}_timeseries.csv")
        #GC_filename = os.path.join(output_dir, f"lorenz_dataset_{i}_causality.csv")
        
        # 保存时间序列数据
        np.savetxt(X_filename, X, delimiter=',')
        
        # 保存因果关系矩阵
        #np.savetxt(GC_filename, GC, delimiter=',', fmt='%d')  # 使用%d格式因为GC是整数矩阵
        
    print(f"已保存 {len(datasets)} 个数据集到 {output_dir} 目录")

def generate_and_save_lorenz_datasets(num_datasets, p, T, output_dir, seed_start=0):
    """
    生成多个Lorenz-96数据集并保存为CSV文件
    
    参数:
    num_datasets -- 要生成的数据集数量
    p -- Lorenz-96模型的变量数量
    T -- 每个数据集的时间步数
    output_dir -- 保存CSV文件的目录路径
    seed_start -- 随机种子的起始值，默认为0
    """
    # 生成数据集
    datasets = generate_multiple_lorenz_datasets(num_datasets, p, T, seed_start)
    
    # 保存数据集为CSV
    save_lorenz_datasets_to_csv(datasets, output_dir)
    
    return datasets
# 示例用法
copy_files("./ICU_Charts", "./data", 500, file_ext=".csv")
# copy_files("source_folder", "destination_folder", -1, file_ext=".txt")


# 使用示例
#generate_and_save_lorenz_datasets(num_datasets=10, p=10, T=30, output_dir="./data")
