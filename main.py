import os
from concurrent.futures import ProcessPoolExecutor
from models_dataset import *
from models_CAUSAL import *
from models_TCDF import *
from models_impute import *
import torch
import numpy as np
import multiprocessing

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
    if gpu != 'cpu':
        # 设置环境变量限制可见GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        # 在环境变量设置后，子进程中的GPU编号总是从0开始
        device = 'cuda:0'  # 重要！设置环境变量后，可见的GPU总是0
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
        
    # 将正确映射后的device传递给函数，而不是原始gpu编号
    matrix, columns = compute_causal_matrix(file, params, device)
    print(f"\nResult for {os.path.basename(file)}: 使用设备 {device}")
    print(np.array(matrix))
    return matrix

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset = MyDataset('./data')
    print(len(dataset))
    print(dataset[0]['original'])
    print(dataset[0]['mask'])
    print(dataset[0]['initial_filled'])
    
    centers = dataset.agr(8)
    files = [os.path.join(dataset.file_paths, f) for f in os.listdir(dataset.file_paths) if f.endswith(".csv")]
    gpus = list(range(torch.cuda.device_count())) or ['cpu']
    tasks = [(f, params, gpus[i % len(gpus)]) for i, f in enumerate(files)]

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
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
    model_params = {
    'num_levels': 3,
    'kernel_size': 6,
    'dilation_c': 4,
    }
    train_all_features_parallel(dataset, model_params, epochs=30, lr=0.01, evaluate=True,
                                point_ratio=0, block_ratio=0,
                                block_min_w=10, block_max_w=15,
                                block_min_h=10, block_max_h=15)

