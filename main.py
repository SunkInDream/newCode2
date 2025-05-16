import os
from concurrent.futures import ProcessPoolExecutor
from models_dataset import *
from models_CAUSAL import *
from models_TCDF import *
from models_impute import *
from models_downstream import *
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
    
    # 添加处理NumPy数组的代码
    if isinstance(file, np.ndarray):
        file_name = f"array_data_{id(file)}"  # 为数组创建唯一标识符
    else:
        file_name = os.path.basename(file)
        
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
    print(f"\nResult for {file_name}: 使用设备 {device}")
    print(np.array(matrix))
    return matrix

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset = MyDataset('./data', tag_file='./static_tag.csv', tag_name='DIEINHOSPITAL', id_name='ICUSTAY_ID')
    print(len(dataset))
    print(dataset[0]['original'])
    print(dataset[0]['mask'])
    print(dataset[0]['initial_filled'])
    print(dataset[0]['file_names'])
    print(dataset[4]['labels'])
     # 1. 执行聚类并获取中心点表示
    centers = dataset.agr(20)
     # 2. 只使用聚类中心的代表文件计算因果矩阵
    center_files = []
    files = [os.path.join(dataset.file_paths, f) for f in os.listdir(dataset.file_paths) if f.endswith('.csv')]
    
    # 如果 center_repre 是数组，将其转换为实际文件路径
    if isinstance(dataset.center_repre, np.ndarray):
        # 使用索引获取文件
        for idx in dataset.center_repre:
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(files):
                center_files.append(files[idx])
    else:
        # 如果已经是文件路径列表，直接使用
        center_files = dataset.center_repre
    
    print(f"使用{len(center_files)}个聚类中心代表进行因果矩阵计算")
    
    # 3. 为聚类中心分配GPU资源
    gpus = list(range(torch.cuda.device_count()-1)) or ['cpu']
    tasks = [(f, params, gpus[i % len(gpus)]) for i, f in enumerate(center_files)]
    
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
    'num_levels': 6,
    'kernel_size': 6,
    'dilation_c': 4,
    }
    train_all_features_parallel(dataset, model_params, epochs=150, lr=0.02, evaluate=True,
                                point_ratio=0.1, block_ratio=0.4,
                                block_min_w=20, block_max_w=30,
                                block_min_h=5, block_max_h=10)
    evaluate_downstream_methods(dataset)