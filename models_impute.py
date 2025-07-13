import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
# import tensorflow as tf
import multiprocessing as mp 
from models_TCDF import *
import torch.nn.functional as F
from pygrinder import block_missing
from sklearn.cluster import KMeans
from models_TCDF import *
from baseline import *
from models_downstream import *
from multiprocessing import Process, Queue

def FirstProcess(matrix, threshold=0.8):
    df = pd.DataFrame(matrix)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
            df[column] = -1
        else:
            non_nan_data = col_data.dropna()
            value_counts = non_nan_data.value_counts()
            mode_value = value_counts.index[0]
            mode_count = value_counts.iloc[0]
            if mode_count >= threshold * len(non_nan_data):
                df[column] = col_data.fillna(mode_value)
    return df.values

def SecondProcess(matrix, perturbation_prob=0.1, perturbation_scale=0.1):
    df_copy = pd.DataFrame(matrix)
    for column in df_copy.columns:
        series = df_copy[column]
        missing_mask = series.isna()

        if not missing_mask.any():
            continue  # 如果没有缺失值，跳过该列
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

    return df_copy.values.astype(np.float32)

def initial_process(matrix, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    matrix = FirstProcess(matrix, threshold)
    matrix = SecondProcess(matrix, perturbation_prob, perturbation_scale)
    return matrix

def impute(original, causal_matrix, model_params, epochs=150, lr=0.02, gpu_id=None):
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    original_copy = original.copy()
    # 阶段1: 填补空列 + 高重复列
    first_stage_initial_filled = FirstProcess(original_copy)
    # 阶段2: 数值扰动增强
    initial_filled = SecondProcess(first_stage_initial_filled)

    mask = (~np.isnan(first_stage_initial_filled)).astype(int)
    sequence_len, total_features = initial_filled.shape
    final_filled = initial_filled.copy()

    for target in range(total_features):
        # 选择因果特征
        inds = list(np.where(causal_matrix[:, target] == 1)[0])
        if target not in inds:
            inds.append(target)
        else:
            inds.remove(target)
            inds.append(target)
        inds = inds[:3] + [target]  # 保留

        # 构造滞后目标变量
        target_shifted = np.roll(initial_filled[:, target], 1)
        target_shifted[0] = 0.0
        x_data = np.concatenate([initial_filled[:, inds], target_shifted[:, None]], axis=1)

        x = torch.tensor(x_data.T[np.newaxis, ...], dtype=torch.float32).to(device)
        y = torch.tensor(initial_filled[:, target][np.newaxis, :, None], dtype=torch.float32).to(device)
        m = torch.tensor((mask[:, target] == 1)[np.newaxis, :, None], dtype=torch.float32).to(device)

        # 构建模型
        input_dim = x.shape[1]
        model = ADDSTCN(target, input_size=input_dim, cuda=(device != torch.device('cpu')), **model_params).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 编译加速
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except:
                pass

        # 训练
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred * m, y * m)
            loss.backward()
            optimizer.step()

        # 推理
        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask[:, target] == 0)
            to_fill_filtered = to_fill[0]
            if len(to_fill_filtered) > 0:
                final_filled[to_fill_filtered, target] = out[to_fill_filtered]

    return final_filled


def impute_single_file(file_path, causal_matrix, model_params, epochs=100, lr=0.01, gpu_id=None):
    """单文件填补函数，用于进程池"""
    # # 设置GPU
    # if gpu_id != 'cpu' and torch.cuda.is_available():
    #     torch.cuda.set_device(gpu_id)
    #     device = torch.device(f'cuda:{gpu_id}')
    # else:
    #     device = torch.device('cpu')
    
    # 读取数据
    data = pd.read_csv(file_path).values.astype(np.float32)
    filename = os.path.basename(file_path)
        
    # 调用优化后的impute函数
    result = impute(data, causal_matrix, model_params, epochs=epochs, lr=lr, gpu_id=gpu_id)
    return filename, result

def parallel_impute(file_path, causal_matrix, model_params, epochs=150, lr=0.02, simultaneous_per_gpu=2, max_workers=None):
    """使用进程池的并行填补"""
    # 获取文件列表
    file_list = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".csv")]
    
    # 确定工作进程数和GPU分配
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        max_workers = max_workers or os.cpu_count()
        gpu_ids = ['cpu'] * len(file_list)
    else:
        # 每个GPU运行simultaneous_per_gpu个进程以提高利用率
        max_workers = max_workers or (num_gpus * simultaneous_per_gpu)
        gpu_ids = [i % num_gpus for i in range(len(file_list))]
    
    print(f"使用 {max_workers} 个进程并行处理 {len(file_list)} 个文件")
    
    # 准备参数列表
    args_list = [(file_path, causal_matrix, model_params, epochs, lr, gpu_id) 
                 for file_path, gpu_id in zip(file_list, gpu_ids)]
    
    # 并行执行
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.starmap(impute_single_file, args_list),
            total=len(file_list),
            desc="批量填补中",
            ncols=80
        ))
    
    # 保存结果
    os.makedirs("./data_imputed/my_model/mimic-iii", exist_ok=True)
    successful_results = []
    
    for filename, result in results:
        successful_results.append(result)
        pd.DataFrame(result).to_csv(f"./data_imputed/my_model/mimic-iii/{filename}", index=False)
    
    print(f"成功填补 {len(successful_results)}/{len(file_list)} 个文件")
    return successful_results

def agregate(initial_filled_array, n_cluster):
    # Step 1: 每个样本按列取均值，构造聚类输入
    data = np.array([np.nanmean(x, axis=0) for x in initial_filled_array])

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
        return pd.read_csv(standard_cg).values

    # Step 1: 预处理
    initial_matrix_arr = original_matrix_arr.copy()
    for i in tqdm(range(len(initial_matrix_arr)), desc="预处理样本"):
        initial_matrix_arr[i] = initial_process(initial_matrix_arr[i])

    # Step 2: 聚类并提取代表样本
    idx_arr = agregate(initial_matrix_arr, n_cluster)
    data_list = [initial_matrix_arr[idx] for idx in idx_arr]
    params_list = [params] * len(data_list)

    # Step 3: 多 GPU 并行因果发现
    results = parallel_compute_causal_matrices(data_list, params_list)

    # Step 4: 汇总结果
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

    # # Step 5: 选 Top-3 构建最终因果图
    # np.fill_diagonal(cg_total, 0)
    # new_matrix = np.zeros_like(cg_total)
    # for col in range(cg_total.shape[1]):
    #     col_values = cg_total[:, col]
    #     if np.count_nonzero(col_values) < 3:
    #         new_matrix[:, col] = 1
    #     else:
    #         top3 = np.argsort(col_values)[-3:]
    #         new_matrix[top3, col] = 1
       # Step 5: 选 Top-5 构建最终因果图
    np.fill_diagonal(cg_total, 0)
    new_matrix = np.zeros_like(cg_total)
    for col in range(cg_total.shape[1]):
        col_values = cg_total[:, col]
        if np.count_nonzero(col_values) < 5:
            new_matrix[:, col] = 1
        else:
            top5 = np.argsort(col_values)[-5:]
            new_matrix[top5, col] = 1
    return new_matrix

# ================================
# 1. 单文件评估函数
# ================================
def mse_evaluate_single_file(mx, causal_matrix, gpu_id=0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gt = mx.copy()
    X = block_missing(mx[np.newaxis, ...], factor=0.2, block_width=15, block_len=10)[0]

    def mse(a, b): 
        a = torch.as_tensor(a, dtype=torch.float32, device=device) 
        b = torch.as_tensor(b, dtype=torch.float32, device=device) 
        return F.mse_loss(a, b).item()

    res = {}
    res['my_model'] = mse(
        impute(X, causal_matrix,
               model_params={'num_levels':10, 'kernel_size': 8, 'dilation_c': 2},
               epochs=150, lr=0.02, gpu_id=gpu_id),
        gt
    )

    if gpu_id == 0:  # 只在处理第一个数据表时保存
        zero_result = zero_impu(X)
        my_model_result = impute(X, causal_matrix,
                           model_params={'num_levels':10, 'kernel_size': 8, 'dilation_c': 2},
                           epochs=150, lr=0.02, gpu_id=gpu_id)
        pd.DataFrame(gt).to_csv("gt_matrix.csv", index=False)
        pd.DataFrame(my_model_result).to_csv("my_model_matrix.csv", index=False)
        pd.DataFrame(zero_result).to_csv("zero_impu_matrix.csv", index=False)
        print("✅ 已保存 gt_matrix.csv, my_model_matrix.csv, zero_impu_matrix.csv")

    def is_reasonable_mse(mse_value, threshold=10000.0):
        """检查MSE值是否合理"""
        return (not np.isnan(mse_value) and 
                not np.isinf(mse_value) and 
                0 <= mse_value <= threshold)

    baseline = [
        ('zero_impu', zero_impu), ('mean_impu', mean_impu),
        ('median_impu', median_impu), ('mode_impu', mode_impu),
        ('random_impu', random_impu), ('knn_impu', knn_impu),
        ('ffill_impu', ffill_impu), ('bfill_impu', bfill_impu),
        ('miracle_impu', miracle_impu), ('saits_impu', saits_impu),
        ('timemixerpp_impu', timemixerpp_impu), 
        ('tefn_impu', tefn_impu),
    ]

    for name, fn in baseline:
        try:
            print(f"开始执行 {name}...")
            result = fn(X)
            
            if result is None:
                print(f"❌ {name}: 返回了 None")
                res[name] = float('nan')
            elif not isinstance(result, np.ndarray):
                print(f"❌ {name}: 返回类型错误 {type(result)}")
                res[name] = float('nan')
            elif result.shape != X.shape:
                print(f"❌ {name}: 形状不匹配 {result.shape} vs {X.shape}")
                res[name] = float('nan')
            else:
                # 检查填补结果是否包含异常值
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    print(f"❌ {name}: 填补结果包含 NaN 或 Inf")
                    res[name] = float('nan')
                elif np.any(np.abs(result) > 1e6):  # 检查是否有异常大值
                    print(f"❌ {name}: 填补结果包含异常大值 (max: {np.max(np.abs(result)):.2e})")
                    res[name] = float('nan')
                else:
                    mse_value = mse(result, gt)
                    
                    # 检查MSE是否合理
                    if is_reasonable_mse(mse_value, threshold=10000.0):
                        res[name] = mse_value
                        print(f"✅ {name}: {mse_value:.6f}")
                    else:
                        print(f"❌ {name}: MSE异常 ({mse_value:.2e})")
                        res[name] = float('nan')
                
        except Exception as e:
            print(f"❌ {name}: 执行失败 - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            res[name] = float('nan')

    print(f"所有结果: {res}")

    # for name, fn in baseline:
    #     res[name] = mse(fn(X), gt)

    return res

# ================================
# 2. 用于 Pool 的包装函数（每个任务）
# ================================
def worker_wrapper(args):
    idx, mx, causal_matrix, gpu_id = args
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 确保PyTorch使用正确的设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 在子进程中，GPU 0就是分配给的GPU
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # 传递正确的设备
    res = mse_evaluate_single_file(mx, causal_matrix, gpu_id=0, device=device)  # ← 这里改为0
    return idx, res

# ================================
# 3. 并行调度函数（进程池实现）
# ================================
def parallel_mse_evaluate(res_list, causal_matrix, simultaneous_per_gpu=3):
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("[INFO] 没有可用 GPU，使用 CPU 顺序评估")
        all_res = [mse_evaluate_single_file(x, causal_matrix)
                   for x in tqdm(res_list, desc="CPU")]
        return {k: float(np.mean([d[k] for d in all_res]))
                for k in all_res[0]}

    # 根据 GPU 数量和每张卡可并行任务数，设置进程池大小
    max_workers = num_gpus * simultaneous_per_gpu
    print(f"[INFO] 使用 {num_gpus} 个 GPU，每个 GPU 最多并行 {simultaneous_per_gpu} 个任务，总进程数: {max_workers}")

    # 为每个任务分配对应的 GPU
    gpu_ids = [i % num_gpus for i in range(len(res_list))]
    args_list = [(i, res_list[i], causal_matrix, gpu_ids[i]) for i in range(len(res_list))]

    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(worker_wrapper, args_list), total=len(args_list), desc="All‑tasks"))

    results.sort(key=lambda x: x[0])  # 恢复顺序
    only_result_dicts = [res for _, res in results]

    avg = {}
    for k in only_result_dicts[0]:
        values = [r[k] for r in only_result_dicts]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if len(valid_values) > 0:
            avg[k] = float(np.nanmean(values))
            print(f"📊 {k}: {len(valid_values)}/{len(values)} 个有效值，平均: {avg[k]:.6f}")
        else:
            avg[k] = float('nan')
            print(f"❌ {k}: 所有值都是 NaN")
    pd.DataFrame([{'Method': k, 'Average_MSE': v} for k, v in avg.items()]) \
        .to_csv("mse_evaluation_results.csv", index=False)

    return avg
