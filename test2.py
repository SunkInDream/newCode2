import os
from concurrent.futures import ProcessPoolExecutor
from models_dataset import *
from models_CAUSAL import *
from models_TCDF import *
import torch
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
params = {
    'epochs': 30,
    'kernel_size': 3,
    'layers': 3,
    'dilation_c': 2,
    'lr': 0.01,
    'optimizername': 'Adam',
    'significance': 0.5
}

def task(args):
    file, params, gpu = args
    matrix, columns = compute_causal_matrix(file, params, gpu)
    print(f"\nResult for {os.path.basename(file)}:")
    print(np.array(matrix))
    return matrix

if __name__ == "__main__":
    dataset = MyDataset('./data')
    print(len(dataset))
    print(dataset[0]['original'])
    print(dataset[0]['mask'])
    print(dataset[0]['initial_filled'])
    
    # 只使用dataset.causal_dis进行聚类，不做并行计算
    centers = dataset.causal_dis(3)
    files = [os.path.join(dataset.file_paths, f) for f in os.listdir(dataset.file_paths) if f.endswith(".csv")]
    gpus = list(range(torch.cuda.device_count())) or ['cpu']
    tasks = [(f, params, gpus[i % len(gpus)]) for i, f in enumerate(files)]

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(task, tasks))

    if results:
        # 初始化与第一个矩阵相同形状的零矩阵
        total_matrix = np.zeros_like(results[0])
        
        # 将所有矩阵相加
        for matrix in results:
            total_matrix += matrix
            
        # 将总和矩阵保存到dataset对象中
        dataset.total_causal_matrix = total_matrix
    if hasattr(dataset, 'total_causal_matrix') and dataset.total_causal_matrix is not None:
        mat = dataset.total_causal_matrix.copy()
        new_matrix = np.zeros_like(mat)

        for col in range(mat.shape[1]):
            temp_col = mat[:, col].copy()
            temp_col[col] = 0  # 忽略对角线
            if np.count_nonzero(temp_col) < 3:
                new_matrix[:, col] = 1
            else:
                top3 = np.argsort(temp_col)[-3:]
                new_matrix[top3, col] = 1

        # 更新total_causal_matrix
        dataset.total_causal_matrix = new_matrix
        print("\n处理后的总因果矩阵:")
        print(dataset.total_causal_matrix)
        print(f"矩阵形状: {dataset.total_causal_matrix.shape}")
        print(f"非零元素数量: {np.count_nonzero(dataset.total_causal_matrix)}")