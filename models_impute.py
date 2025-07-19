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
from pygrinder import (
    mcar,
    mar_logistic,
    mnar_x,
)

from sklearn.cluster import KMeans
from baseline import *
from scipy.stats import wasserstein_distance
from models_downstream import *
from multiprocessing import Process, Queue
from models_TCN import MultiADDSTCN, ParallelFeatureADDSTCN, ADDSTCN

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

def SecondProcess(matrix, perturbation_prob=0.3, perturbation_scale=0.3):
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

def impute(original, causal_matrix, model_params, epochs=150, lr=0.02, gpu_id=None, ifGt=False, gt=None):
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print('missing_count', np.isnan(original).sum())
    # 预处理
    first = FirstProcess(original.copy())
    # print('missing_count', np.isnan(first).sum())
    mask = (~np.isnan(first)).astype(int)
    # pd.DataFrame(mask).to_csv("mask_qian.csv", index=False)
    initial_filled = SecondProcess(first)
    initial_filled_copy = initial_filled.copy()
    # 构造张量 (1, T, N)
    x = torch.tensor(initial_filled[None, ...], dtype=torch.float32, device=device)
    y = torch.tensor(initial_filled[None, ...], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, ...], dtype=torch.float32, device=device)
    gt = gt
    # print("Y.shape", y.shape)
    # 创建模型
    print("causal_matrix.shape", causal_matrix.shape)
    model = ParallelFeatureADDSTCN(
        causal_matrix=causal_matrix,
        model_params=model_params
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ✅ 创建学习率调度器
    def lr_lambda(epoch):
        if epoch < 50:
            return 1.0  # 前50个epoch使用原始学习率
        else:
            return 1.0  # 后100个epoch使用0.3倍学习率 (0.01 * 0.3 = 0.003)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_loss = float('inf')
    best_imputed = None

    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(x)  # (1, T, N)

        # Loss1: 观测值的预测误差
        loss_1 = F.mse_loss(pred * m, y * m)

        # # ✅ 修复 Loss2: 针对缺失位置与gt的一致性
        # if ifGt and gt is not None:
        #     # 构造 gt 张量，与 pred 格式一致
        #     gt_tensor = torch.tensor(gt[None, ...], dtype=torch.float32, device=device)  # (1, T, N)
           
        #     missing_mask = 1 - m  # 缺失位置的 mask
            
        #     # 只计算缺失位置的损失
        #     missing_count = missing_mask.sum()
        #     if missing_count > 0:
        #         loss_2 = F.mse_loss(pred * missing_mask, gt_tensor * missing_mask)
        #         print('pred * missing_mask',pred* missing_mask)
        #         print('gt_tensor * missing_mask',gt_tensor * missing_mask)
        #     else: 
        #         loss_2 = torch.tensor(0.0, device=device, requires_grad=True)
        # else:
        #     loss_2 = torch.tensor(0.0, device=device, requires_grad=True)


        # Loss4: 统计分布
        def statistical_alignment_loss(pred, y):
            """
            基于统计分布对齐的损失函数
            """
            pred_features = pred.squeeze(0)  # (T, N)
            y_features = y.squeeze(0)       # (T, N)
            
            losses = []
            
            # 1. 均值对齐
            pred_mean = pred_features.mean(dim=0)  # (N,)
            y_mean = y_features.mean(dim=0)        # (N,)
            mean_loss = F.mse_loss(pred_mean, y_mean)
            losses.append(mean_loss)
            
            # 2. 标准差对齐
            pred_std = pred_features.std(dim=0)    # (N,)
            y_std = y_features.std(dim=0)          # (N,)
            std_loss = F.mse_loss(pred_std, y_std)
            losses.append(std_loss)
            
            # 3. 分位数对齐
            quantiles = [0.25, 0.5, 0.75]
            for q in quantiles:
                pred_q = torch.quantile(pred_features, q, dim=0)
                y_q = torch.quantile(y_features, q, dim=0)
                q_loss = F.mse_loss(pred_q, y_q)
                losses.append(q_loss)
            
            return sum(losses) / len(losses)

        loss_3 = statistical_alignment_loss(pred, y)

        # 加权总损失（如需调整，修改权重）
        total_loss = 0.6 * loss_1 + 0.4 * loss_3 
        print(f"[Epoch {epoch+1}/{epochs}] Total Loss: {total_loss.item():.6f}")
        total_loss.backward()
        opt.step()
        
        # ✅ 更新学习率
        scheduler.step()

        # 保存最佳预测结果
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            with torch.no_grad():
                best_imputed = model(x).cpu().squeeze(0).numpy()

    # 用最优结果进行填补
    imputed = best_imputed
    res = initial_filled.copy()
    res[mask == 0] = imputed[mask == 0]
    pd.DataFrame(res).to_csv("result_1.csv", index=False)
     
    return res, mask, initial_filled_copy


# def impute(original, causal_matrix, model_params, epochs=150, lr=0.02, gpu_id=None, ifGt=False, gt=None):
#     device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')

#     first = FirstProcess(original.copy())
#     mask = (~np.isnan(first)).astype(int)
#     filled = SecondProcess(first)

#     # 构造张量
#     x = torch.tensor(filled.T[None, ...], dtype=torch.float32, device=device)
#     y = torch.tensor(filled[None, ...], dtype=torch.float32, device=device).transpose(1, 2)
#     m = torch.tensor(mask[None, ...], dtype=torch.float32, device=device).transpose(1, 2)

#     model = MultiADDSTCN(
#         causal_mask=causal_matrix,
#         cuda=device.type == 'cuda',
#         **model_params
#     ).to(device)

#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     best_loss = float('inf')
#     best_imputed = None

#     for epoch in range(epochs):
#         opt.zero_grad()
#         pred = model(x)  # (1, T, N)

#         # Loss1: 观测值的预测误差
#         loss_1 = F.mse_loss(pred * m, y * m)

#         # Loss2: 针对缺失位置与gt的一致性（与评估函数保持一致）
#         if ifGt and gt is not None:
#             mask_np = (~np.isnan(original)).astype(int)
#             missing_mask_np = (1 - mask_np).astype(bool)

#             pred_np = pred.detach().cpu().squeeze(0).numpy()
#             gt_np = gt.copy()

#             pred_missing = pred_np[missing_mask_np]
#             gt_missing = gt_np[missing_mask_np]

#             pred_tensor = torch.tensor(pred_missing, dtype=torch.float32, device=device, requires_grad=True)
#             gt_tensor = torch.tensor(gt_missing, dtype=torch.float32, device=device)

#             loss_2 = F.mse_loss(pred_tensor, gt_tensor)
#         else:
#             loss_2 = ((pred * m - y * m) ** 2).sum() / m.sum()

#         # Loss3: 协方差矩阵对齐
#         pred_np = pred.squeeze(0).T
#         y_np = y.squeeze(0).T
#         if pred_np.shape[1] > 1:
#             try:
#                 cov_pred = torch.cov(pred_np)
#                 cov_y = torch.cov(y_np)
#                 loss_3 = F.mse_loss(cov_pred, cov_y)
#             except:
#                 loss_3 = torch.tensor(0.0, device=device, requires_grad=True)
#         else:
#             loss_3 = torch.tensor(0.0, device=device, requires_grad=True)

#         # Loss4: RBF核映射对齐
#         def rbf_kernel(mat, sigma=1.0):
#             if mat.shape[0] <= 1:
#                 return torch.zeros((1, 1), device=mat.device, requires_grad=True)
#             dist = torch.cdist(mat, mat, p=2) ** 2
#             return torch.exp(-dist / (2 * sigma ** 2))

#         try:
#             K_pred = rbf_kernel(pred_np)
#             K_y = rbf_kernel(y_np)
#             loss_4 = F.mse_loss(K_pred, K_y)
#         except:
#             loss_4 = torch.tensor(0.0, device=device, requires_grad=True)

#         # 加权总损失
#         total_loss = loss_1 + 0 * loss_2 + 0 * loss_3 + 0 * loss_4
#         print(f"[Epoch {epoch+1}/{epochs}] Total Loss: {total_loss.item():.6f}")
#         total_loss.backward()
#         opt.step()

#         # 保存最佳预测结果
#         if total_loss.item() < best_loss:
#             best_loss = total_loss.item()
#             with torch.no_grad():
#                 best_imputed = model(x).cpu().squeeze(0).numpy()

#     # 用最优结果进行填补
#     imputed = best_imputed
#     filled[mask == 0] = imputed[mask == 0]
#     return filled




# def impute(original, causal_matrix, model_params, epochs=150, lr=0.02, gpu_id=None):
#     if gpu_id is not None and torch.cuda.is_available():
#         device = torch.device(f'cuda:{gpu_id}')
#     else:
#         device = torch.device('cpu')
#     original_copy = original.copy()
#     # 阶段1: 填补空列 + 高重复列
#     first_stage_initial_filled = FirstProcess(original_copy)
#     # 阶段2: 数值扰动增强
#     initial_filled = SecondProcess(first_stage_initial_filled)

#     mask = (~np.isnan(first_stage_initial_filled)).astype(int)
#     sequence_len, total_features = initial_filled.shape
#     final_filled = initial_filled.copy()

#     for target in range(total_features):
#         # 选择因果特征
#         inds = list(np.where(causal_matrix[:, target] == 1)[0])
#         if target not in inds:
#             inds.append(target)
#         else:
#             inds.remove(target)
#             inds.append(target)
#         inds = inds[:3] + [target]  # 保留

#         # 构造滞后目标变量
#         target_shifted = np.roll(initial_filled[:, target], 1)
#         target_shifted[0] = 0.0
#         x_data = np.concatenate([initial_filled[:, inds], target_shifted[:, None]], axis=1)

#         x = torch.tensor(x_data.T[np.newaxis, ...], dtype=torch.float32).to(device)
#         y = torch.tensor(initial_filled[:, target][np.newaxis, :, None], dtype=torch.float32).to(device)
#         m = torch.tensor((mask[:, target] == 1)[np.newaxis, :, None], dtype=torch.float32).to(device)

#         # 构建模型
#         input_dim = x.shape[1]
#         model = ADDSTCN(target, input_size=input_dim, cuda=(device != torch.device('cpu')), **model_params).to(device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#         # 编译加速
#         if hasattr(torch, 'compile'):
#             try:
#                 model = torch.compile(model)
#             except:
#                 pass

#         # 训练
#         for epoch in range(1, epochs + 1):
#             model.train()
#             optimizer.zero_grad()
#             pred = model(x)
#             loss = F.mse_loss(pred * m, y * m)
#             loss.backward()
#             optimizer.step()

#         # 推理
#         model.eval()
#         with torch.no_grad():
#             out = model(x).squeeze().cpu().numpy()
#             to_fill = np.where(mask[:, target] == 0)
#             to_fill_filtered = to_fill[0]
#             if len(to_fill_filtered) > 0:
#                 final_filled[to_fill_filtered, target] = out[to_fill_filtered]

#     return final_filled


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
    """聚类并提取代表样本"""
    data = np.array([matrix.flatten() for matrix in initial_filled_array])
    
    # ✅ 添加数据清理步骤
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据范围: [{np.nanmin(data):.6e}, {np.nanmax(data):.6e}]")
    
    # 1. 检查并处理无穷大值
    inf_mask = np.isinf(data)
    if inf_mask.any():
        print(f"发现 {inf_mask.sum()} 个无穷大值，将替换为有限值")
        data[inf_mask] = np.nan
    
    # 2. 检查并处理 NaN 值
    nan_mask = np.isnan(data)
    if nan_mask.any():
        print(f"发现 {nan_mask.sum()} 个 NaN 值，将用中位数填充")
        # 用中位数填充 NaN
        median_val = np.nanmedian(data)
        data[nan_mask] = median_val
    
    # 3. 数据范围限制（避免数值过大）
    # 使用分位数来限制极值
    q1, q99 = np.percentile(data, [1, 99])
    print(f"1%分位数: {q1:.6e}, 99%分位数: {q99:.6e}")
    
    # 将极值限制在合理范围内
    data = np.clip(data, q1, q99)
    
    # 4. 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    print(f"清理后数据形状: {data_scaled.shape}")
    print(f"清理后数据范围: [{data_scaled.min():.6f}, {data_scaled.max():.6f}]")
    
    # 5. 聚类
    km = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    try:
        labels = km.fit_predict(data_scaled)
    except Exception as e:
        print(f"聚类失败: {e}")
        # 如果还是失败，使用随机采样
        print("使用随机采样作为备选方案")
        return np.random.choice(len(initial_filled_array), n_cluster, replace=False).tolist()
    
    print(f"聚类完成，标签分布: {np.bincount(labels)}")
    
    # 6. 选择每个簇的代表样本
    idx_arr = []
    for k in tqdm(range(n_cluster), desc="选择每簇代表样本"):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        
        cluster_data = data_scaled[idxs]
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
        return pd.read_csv(standard_cg, header=None).values

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
       # Step 5: 选 Top-4 构建最终因果图
    np.fill_diagonal(cg_total, 0)
    new_matrix = np.zeros_like(cg_total)
    for col in range(cg_total.shape[1]):
        col_values = cg_total[:, col]
        if np.count_nonzero(col_values) < 4:
            new_matrix[:, col] = 1
        else:
            top5 = np.argsort(col_values)[-4:]
            new_matrix[top5, col] = 1
    return new_matrix

# ================================
# 1. 单文件评估函数
# ================================
def mse_evaluate_single_file(mx, causal_matrix, gpu_id=0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ground truth
    gt = mx.copy()
    gt2 = gt.copy()
    pd.DataFrame(gt).to_csv("1.csv", index=False)
    # 随机 mask 生成缺失
    X = mar_logistic(mx, obs_rate=0.2, missing_rate=0.1)
    X = X[np.newaxis, ...]  # 增加一个维度
    X = mnar_x(X, offset=0.05)
    X = mcar(X, p=0.05)
    X = X.squeeze(0)  # 去掉多余的维度
    Mask = (~np.isnan(X)).astype(int)
    pd.DataFrame(X).to_csv("2.csv", index=False)
    # # mask: 观测为 1，缺失为 0
    # M = (~np.isnan(X)).astype(int)
    # missing_place = 1 - M
    Mask = (~np.isnan(X)).astype(int)
    # 掩码版 MSE，只在缺失位置评估
    def mse(a, b, mask):
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        b = torch.as_tensor(b, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        mask = 1- mask
        element_wise_error = (a - b) ** 2  
        
        pd.DataFrame((element_wise_error*mask).cpu().numpy()).to_csv("element_wise_error.csv", index=False)
        pd.DataFrame(a.cpu().numpy()).to_csv("a.csv", index=False)
        pd.DataFrame(b.cpu().numpy()).to_csv("b.csv", index=False)
        pd.DataFrame(mask.cpu().numpy()).to_csv("mask.csv", index=False)
        pd.DataFrame((a*mask).cpu().numpy()).to_csv("a*c.csv", index=False)
        pd.DataFrame((b*mask).cpu().numpy()).to_csv("b*c.csv", index=False)
        
        # 计算 masked MSE
        masked_error = F.mse_loss(a * mask, b * mask).item()

        return masked_error

    res = {}

    # 我的模型评估
    # print("开始执行 my_model...")
    imputed_result, mask, initial_processed = impute(X, causal_matrix,
                            model_params={'num_levels':10, 'kernel_size': 8, 'dilation_c': 2},
                            epochs=100, lr=0.02, gpu_id=gpu_id, ifGt=True, gt=gt)
    # print("imputed_result.shape", imputed_result.shape, "gt2.shape", gt2.shape, "mask.shape", mask.shape)
    res['my_model'] = mse(imputed_result, gt2, mask)
    def is_reasonable_mse(mse_value, threshold=10000.0):
        return (not np.isnan(mse_value) and 
                not np.isinf(mse_value) and 
                0 <= mse_value <= threshold)

    # baseline 方法
    baseline = [
        ('initial_process', initial_process),
        ('zero_impu', zero_impu),
        ('mean_impu', mean_impu),
        ('median_impu', median_impu),
        ('mode_impu', mode_impu),
        ('random_impu', random_impu), ('knn_impu', knn_impu),
        ('ffill_impu', ffill_impu), ('bfill_impu', bfill_impu),
        ('miracle_impu', miracle_impu), ('saits_impu', saits_impu),
        ('timemixerpp_impu', timemixerpp_impu), 
        ('tefn_impu', tefn_impu),
    ]

    for name, fn in baseline:
        print(f"开始执行 {name}...")
        result = fn(X)
        if np.any(np.abs(result) > 1e6):
            print(f"❌ {name}: 填补结果包含异常大值 (max: {np.max(np.abs(result)):.2e})")
            res[name] = float('nan')
        else:
            mse_value = mse(result, gt, Mask)
            if is_reasonable_mse(mse_value):
                res[name] = mse_value
                print(f"✅ {name}: {mse_value:.6f}")
            else:
                print(f"❌ {name}: MSE异常 ({mse_value:.2e})")
                res[name] = float('nan')

    print(f"所有结果: {res}")
    return res


# ================================
# 2. 用于 Pool 的包装函数（每个任务）
# ================================
def worker_wrapper(args):
    idx, mx, causal_matrix, gpu_id = args
    gt = mx.copy()
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
