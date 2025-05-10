from models_dataset import *
from models_CAUSAL import *
import os
from concurrent.futures import ProcessPoolExecutor
from models_TCDF import *
import torch
import numpy as np
import multiprocessing

def task(args):
    array, params, gpu = args
    matrix, columns = compute_causal_matrix(array, params, gpu)
    print(f"\nResult for array shape {array.shape}:")
    print(np.array(matrix))
    return matrix

params = {
    'epochs': 30,
    'kernel_size': 3,
    'layers': 3,
    'dilation_c': 2,
    'lr': 0.01,
    'optimizername': 'Adam',
    'significance': 0.5
}


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    dataset = MyDataset('./data')
    print(len(dataset))
    print(dataset[0]['original'])
    print(dataset[0]['mask'])
    print(dataset[0]['initial_filled'])
    
    # 只使用dataset.causal_dis进行聚类，不做并行计算
    centers = dataset.causal_dis(3)
    
    # 在main.py中进行并行计算，与test.py相同的方式
    gpus = list(range(torch.cuda.device_count())) or ['cpu']
    tasks = [(arr, params, gpus[i % len(gpus)]) for i, arr in enumerate(dataset.initial_filled)]
    
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        results = list(executor.map(task, tasks))
        print(f"处理完成 {len(results)} 个数组")
    
    # 将所有因果矩阵相加
    if results:
        # 初始化与第一个矩阵相同形状的零矩阵
        total_matrix = np.zeros_like(results[0])
        
        # 将所有矩阵相加
        for matrix in results:
            total_matrix += matrix
            
        # 将总和矩阵保存到dataset对象中
        dataset.total_causal_matrix = total_matrix
    if hasattr(dataset, 'total_causal_matrix') and dataset.total_causal_matrix is not None:
        num_rows, num_cols = dataset.total_causal_matrix.shape
        
        # 创建新矩阵
        new_matrix = np.zeros_like(dataset.total_causal_matrix)
        
        # 处理每一列
        for col in range(num_cols):
            column = dataset.total_causal_matrix[:, col]
            non_zero_indices = np.nonzero(column)[0]
            
            if len(non_zero_indices) <= 3:
                # 如果非零元素不足3个，所有非零元素位置置1
                new_matrix[non_zero_indices, col] = 1
            else:
                # 找出值最大的前三个位置
                top_indices = np.argsort(column)[-3:]  # 取整列中最大的三个
                new_matrix[top_indices, col] = 1
        
        # 更新total_causal_matrix
        dataset.total_causal_matrix = new_matrix
        print("\n处理后的总因果矩阵:")
        print(dataset.total_causal_matrix)
        print(f"矩阵形状: {dataset.total_causal_matrix.shape}")
        print(f"非零元素数量: {np.count_nonzero(dataset.total_causal_matrix)}")
    dataset.impute_with_tcn(params)
    print("填补完成")
    print(dataset.final_filled)
