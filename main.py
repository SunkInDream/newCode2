import os
import sys
from concurrent.futures import ProcessPoolExecutor
from models_dataset import *
from models_CAUSAL import *
from models_TCDF import *
from models_impute import *
from models_downstream import *
import numpy as np
import multiprocessing
import warnings
from models_dataset import *  # Ensure MyDataset is imported
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
        device = 'cuda:0'  
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
        
    # 将正确映射后的device传递给函数，而不是原始gpu编号
    matrix, columns = compute_causal_matrix(file, params, device)
    print(f"\nResult for {file_name}: 使用设备 {device}")
    print(np.array(matrix))
    return matrix
params = {
    'layers': 6,
    'kernel_size': 6,
    'dilation_c': 4,
    'optimizername': 'Adam',
    'lr': 0.02,
    'epochs': 150,
    'significance': 0.8,
}
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    multiprocessing.set_start_method('spawn', force=True)
    dataset = MyDataset('./data', tag_file='./static_tag.csv', tag_name='DIEINHOSPITAL', id_name='ICUSTAY_ID')
    print(len(dataset))
    print(dataset[0]['original'])
    print(dataset[0]['mask'])
    print(dataset[0]['initial_filled'])
    print(dataset[0]['file_names'])
    print(dataset[0]['labels'])
     # 1. 执行聚类并获取中心点表示
    centers = dataset.agregate(20)
     # 2. 只使用聚类中心的代表文件计算因果矩阵
    center_files = []
    files = [os.path.join(dataset.file_paths, f) for f in os.listdir(dataset.file_paths) if f.endswith('.csv')]
    
    for idx in dataset.center_repre:
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(files):
            center_files.append(files[idx])
    
    print(f"使用{len(dataset.center_repre)}个聚类中心代表进行因果矩阵计算")
    
    # 3. 为聚类中心分配GPU资源
    gpus = list(range(torch.cuda.device_count()-1)) or ['cpu']
    #gpus = ['cpu']
    tasks = [(f, params, gpus[i % len(gpus)]) for i, f in enumerate(center_files)]
    
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        results = list(executor.map(task, tasks))
    
    if results:
        total_matrix = np.zeros_like(results[0])
        for matrix in results:
            total_matrix += matrix
        mat = total_matrix.copy()
        new_matrix = np.zeros_like(mat)

        for col in range(mat.shape[1]):
            temp_col = mat[:, col].copy()
            temp_col[col] = 0  # 忽略对角线
            if np.count_nonzero(temp_col) < 3:
                new_matrix[:, col] = 1
            else:
                top3 = np.argsort(temp_col)[-3:]
                new_matrix[top3, col] = 1
        dataset.total_causal_matrix = new_matrix
    model_params = {
    'num_levels': 6,
    'kernel_size': 6,
    'dilation_c': 4,
    }
    train_all_features_parallel(dataset, model_params, epochs=150, lr=0.02, evaluate=True,
                                point_ratio=0.1, block_ratio=0.6,
                                block_min_w=20, block_max_w=30,
                                block_min_h=5, block_max_h=10)
    evaluate_downstream_methods(dataset,k_folds=5)