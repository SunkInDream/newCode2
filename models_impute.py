import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp 
from models_TCDF import ADDSTCN
import torch.nn.functional as F
from pygrinder import (
    mcar,
    mar_logistic,
    mnar_x,
    mnar_t,
    mnar_nonuniform,
    rdo,
    seq_missing,
    block_missing,
    calc_missing_rate
)
from sklearn.cluster import KMeans
from models_CAUSAL import *
from models_TCDF import *
from baseline import *
from models_downstream import *
def impute(original, causal_matrix, model_params, epochs=100, lr=0.01, gpu_id=None):
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    mask = (~np.isnan(original)).astype(int)
    initial_filled = initial_process(original)
    sequence_len, total_features = initial_filled.shape
    final_filled = initial_filled.copy()
    for target in range(total_features):
        inds = list(np.where(causal_matrix[:, target] == 1)[0])
        if target not in inds:
            inds.append(target)
        else:
            inds.remove(target)
            inds.append(target)
        inds = inds[:3] + [target]

        inp = initial_filled[:, inds].T[np.newaxis, ...]
        y_np = initial_filled[:, target][np.newaxis, :, None]
        m_np = (mask[:, target] == 1)[np.newaxis, :, None]

        x = torch.tensor(inp, dtype=torch.float32).to(device)
        y = torch.tensor(y_np, dtype=torch.float32).to(device)
        m = torch.tensor(m_np, dtype=torch.float32).to(device)

        model = ADDSTCN(target, input_size=len(inds),
                            cuda=(device == 'cuda:0'),
                            **model_params).to(device)
            # 使用更适合时间序列的学习率调度
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5, verbose=False)
            
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            pred = model(x)
            loss = F.mse_loss(pred * m, y * m) + 0.001 * sum(p.abs().sum() for p in model.parameters())
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step(loss)
                
                # 保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 使用最佳模型进行预测
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask[:, target] == 0) # 原始缺失才填补
            final_filled[to_fill, target] = out[to_fill]
    return final_filled
def impute_worker(task_queue, causal_matrix, result_queue, model_params, epochs, lr, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    while True:
        item = task_queue.get()
        if item is None:
            break
        file_path = item
        try:
            filename = os.path.basename(file_path)
            data = pd.read_csv(file_path).values.astype(np.float32)

            result = impute(data, causal_matrix, model_params, epochs=epochs, lr=lr, gpu_id=0)
            result_queue.put((filename, result))  # ⬅ 将结果返回，而不是保存文件
        except Exception as e:
            result_queue.put((file_path, None))  # 标记失败
def parallel_impute_folder(causal_matrix, input_dir, model_params, epochs=100, lr=0.01):
    num_gpus = torch.cuda.device_count()
    file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for file_path in file_list:
        task_queue.put(file_path)

    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=impute_worker, args=(
            task_queue, causal_matrix, result_queue, model_params, epochs, lr, gpu_id
        ))
        p.start()
        workers.append(p)

    results = []

    # ✅ 加进度条：每个任务完成后更新一次
    for _ in tqdm(range(len(file_list)), desc="批量填补中"):
        filename, result = result_queue.get()
        if result is not None:
            results.append(result)
            pd.DataFrame(result).to_csv(f"./data_imputed/my_model/{filename}", index=False)
        else:
            print(f"[错误] {filename} 填补失败")

    for p in workers:
        p.join()

    return results
def agregate(initial_filled, n_cluster):
    # Step 1: 每个样本按列取均值，构造聚类输入
    data = np.array([np.nanmean(x, axis=0) for x in initial_filled])

    # Step 2: KMeans 聚类
    km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
    labels = km.fit_predict(data)

    # Step 3: 逐类找代表样本，带进度条
    idx_arr = []
    for k in tqdm(range(n_cluster), desc="选择每簇代表样本"):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        cluster_data = data[idxs]
        dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
        best_idx = idxs[np.argmin(dists)]
        idx_arr.append(int(best_idx))

    return idx_arr
def causal_worker(task_queue, result_queue, initial_matrix_arr, params, gpu_id):
    """
    每个进程运行的 worker，绑定指定 GPU，处理 task_queue 中的任务
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # 每个子进程看的是 CUDA_VISIBLE_DEVICES 下的 cuda:0

    while True:
        item = task_queue.get()
        if item is None:
            break
        task_id, i = item
        try:
            matrix = compute_causal_matrix(initial_matrix_arr[i], params=params, gpu_id=0)
            result_queue.put((task_id, matrix))
        except Exception as e:
            print(f"[GPU {gpu_id}] 任务 {task_id} 失败: {e}")
            result_queue.put((task_id, None))
def causal_discovery(original_matrix_arr, n_cluster=5, isStandard=False, standard_cg=None,
                     params={
                         'layers': 6,
                         'kernel_size': 6,
                         'dilation_c': 4,
                         'optimizername': 'Adam',
                         'lr': 0.02,
                         'epochs': 100,
                         'significance': 1.2,
                     }):

    if isStandard:
        if standard_cg is None:
            raise ValueError("standard_cg must be provided when isStandard is True")
        else:
            return standard_cg

    # Step 1: 预处理数据
    initial_matrix_arr = original_matrix_arr.copy()
    for i in tqdm(range(len(initial_matrix_arr)), desc="预处理样本"):
        initial_matrix_arr[i] = initial_process(initial_matrix_arr[i])

    # Step 2: 聚类并获取每组索引
    idx_arr = agregate(initial_matrix_arr, n_cluster)

    # Step 3: 多 GPU 并行计算 causal_matrix
    num_gpus = torch.cuda.device_count()
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for task_id, i in enumerate(idx_arr):
        task_queue.put((task_id, i))
    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=causal_worker, args=(task_queue, result_queue, initial_matrix_arr, params, gpu_id))
        p.start()
        workers.append(p)

    results = [None] * len(idx_arr)

    # ✅ 用 tqdm 包裹 result_queue.get() 获取进度条
    for _ in tqdm(range(len(idx_arr)), desc="因果发现中"):
        task_id, matrix = result_queue.get()
        results[task_id] = matrix

    for p in workers:
        p.join()

    # Step 4: 合并结果
    cg_total = None
    for matrix in results:
        if matrix is None:
            continue
        if cg_total is None:
            cg_total = matrix.copy()
        else:
            cg_total += matrix

    if cg_total is None:
        raise RuntimeError("所有任务都失败，未能得到有效的因果矩阵")

    # Step 5: 选 top3 作为 final causal graph
    np.fill_diagonal(cg_total, 0)
    new_matrix = np.zeros_like(cg_total)
    for col in range(cg_total.shape[1]):
        temp_col = cg_total[:, col].copy()
        if np.count_nonzero(temp_col) < 3:
            new_matrix[:, col] = 1
        else:
            top3 = np.argsort(temp_col)[-3:]
            new_matrix[top3, col] = 1

    return new_matrix
def mse_evaluate(mx, causal_matrix, gpu_id=None):
    ground_truth = mx.copy()
    X_block_3d = block_missing(mx[np.newaxis, ...], factor=0.1, block_width=3, block_len=3)
    X_block = X_block_3d[0]

    imputed = impute(
        X_block, causal_matrix,
        model_params={'num_levels': 6, 'kernel_size': 6, 'dilation_c': 4},
        epochs=100, lr=0.01, gpu_id=gpu_id
    )

    def mse(a, b):
        return np.mean((a - b) ** 2)

    return {
        'my_model':    mse(imputed,      ground_truth),
        'zero_impu':   mse(zero_impu(X_block), ground_truth),
        'mean_impu':   mse(mean_impu(X_block), ground_truth),
        'median_impu': mse(median_impu(X_block), ground_truth),
        'mode_impu':   mse(mode_impu(X_block), ground_truth),
        'random_impu': mse(random_impu(X_block), ground_truth),
        'knn_impu':    mse(knn_impu(X_block), ground_truth),
        'ffill_impu':  mse(ffill_impu(X_block), ground_truth),
        'bfill_impu':  mse(bfill_impu(X_block), ground_truth),
        'miracle_impu': mse(miracle_impu(X_block), ground_truth),
        'saits_impu':  mse(saits_impu(X_block), ground_truth),
        'timemixerpp_impu': mse(timemixerpp_impu(X_block), ground_truth),
        'tefn_impu':   mse(tefn_impu(X_block), ground_truth)
    }
def mse_worker(task_queue, result_queue, causal_matrix, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  

    print(f"Worker启动，物理GPU ID={gpu_id}，可见设备={os.environ['CUDA_VISIBLE_DEVICES']}")

    while True:
        item = task_queue.get()
        if item is None:
            break

        idx, item_data = item
        try:
            if isinstance(item_data, tuple) and len(item_data) > 1:
                filename, matrix = item_data
                print(f"[Worker GPU {gpu_id}] 任务{idx}: 处理文件 {filename}")
            else:
                matrix = item_data

            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"期望numpy数组，得到{type(matrix)}")
            if matrix.ndim != 2:
                raise ValueError(f"期望二维矩阵，得到{matrix.ndim}维")

            result = mse_evaluate(matrix, causal_matrix, gpu_id=0)
            result_queue.put((idx, result))
            print(f"[Worker GPU {gpu_id}] 完成任务 {idx}")
        except Exception as e:
            import traceback
            print(f"[Worker GPU {gpu_id}] 评估任务 {idx} 失败: {e}")
            print(traceback.format_exc())
            result_queue.put((idx, None))
def parallel_mse_evaluate(res_list, causal_matrix):
    num_gpus = torch.cuda.device_count()
    print(f"发现 {num_gpus} 个GPU设备")
    
    # 没有GPU就单线程运行
    if num_gpus == 0:
        results = []
        for i, (fname, matrix) in enumerate(res_list):
            try:
                result = mse_evaluate(matrix, causal_matrix)
                results.append(result)
            except Exception as e:
                print(f"评估任务 {i} 失败: {e}")
                results.append(None)
        return results
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 加入任务，传入元组
    for idx, item in enumerate(res_list):
        task_queue.put((idx, item))
    
    # 结束标记
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 创建工作进程
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=mse_worker, args=(task_queue, result_queue, causal_matrix, gpu_id))
        p.start()
        workers.append(p)
    
    # 收集结果
    results = [None] * len(res_list)
    for _ in range(len(res_list)):
        try:
            idx, result = result_queue.get()
            results[idx] = result
        except Exception as e:
            print(f"获取结果时出错: {e}")
    
    # 等待所有进程结束
    for p in workers:
        p.join()
    
    valid_mse_dicts = [d for d in results if d is not None]
    if not valid_mse_dicts:
        print("错误: 所有MSE评估任务均失败!")
    else:
        print(f"成功完成 {len(valid_mse_dicts)}/{len(results)} 个MSE评估")
        
        avg_mse = {}
        for method in tqdm(valid_mse_dicts[0], desc="计算平均 MSE"):
            vals = [d[method] for d in valid_mse_dicts if d is not None]
            if vals:
                avg_mse[method] = sum(vals) / len(vals)
            else:
                avg_mse[method] = float('nan')
        
        print("各方法平均 MSE:")
        for method, v in sorted(avg_mse.items()):
            print(f"{method:12s}: {v:.6f}")
            
        results_df = pd.DataFrame([
            {'Method': method, 'Average_MSE': v} 
            for method, v in sorted(avg_mse.items())
        ])
        results_df.to_csv('mse_evaluation_results.csv', index=False)
        print(f"结果已保存到: mse_evaluation_results.csv")
        
        return avg_mse