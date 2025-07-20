import os
import torch
import random
import gc
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
from multiprocessing import set_start_method
from scipy.stats import wasserstein_distance
from models_downstream import *
from multiprocessing import Process, Queue
from models_TCN import MultiADDSTCN, ParallelFeatureADDSTCN, ADDSTCN

def FirstProcess(matrix, threshold=0.8):
    matrix = np.array(matrix, dtype=np.float32)
    
    # 第一阶段：处理空列和高重复列
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]
        
        if np.isnan(col_data).all():
            matrix[:, col_idx] = -1
            continue
            
        valid_mask = ~np.isnan(col_data)
        if not valid_mask.any():
            continue
            
        valid_data = col_data[valid_mask]
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        max_count_idx = np.argmax(counts)
        mode_value = unique_vals[max_count_idx]
        mode_count = counts[max_count_idx]
        
        if mode_count >= threshold * len(valid_data):
            matrix[np.isnan(col_data), col_idx] = mode_value
    return matrix

def SecondProcess(matrix, perturbation_prob=0.3, perturbation_scale=0.3):
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]
        missing_mask = np.isnan(col_data)
        
        if not missing_mask.any():
            continue
        
        series = pd.Series(col_data)
        interpolated = series.interpolate(method='linear', limit_direction='both').values
        
        if np.isnan(interpolated).any():
            interpolated[np.isnan(interpolated)] = np.nanmean(col_data)
        
        # 添加扰动
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0 and perturbation_prob > 0:
            n_perturb = int(len(missing_indices) * perturbation_prob)
            if n_perturb > 0:
                perturb_indices = np.random.choice(missing_indices, n_perturb, replace=False)
                value_range = np.ptp(col_data[~missing_mask]) or 1.0
                perturbations = np.random.uniform(-1, 1, n_perturb) * perturbation_scale * value_range
                interpolated[perturb_indices] += perturbations
        
        matrix[:, col_idx] = interpolated
    
    return matrix.astype(np.float32)  # ✅ 修复：移到循环外面

def initial_process(matrix, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    matrix = FirstProcess(matrix, threshold)
    matrix = SecondProcess(matrix, perturbation_prob, perturbation_scale)
    return matrix

def impute(original, causal_matrix, model_params, epochs=100, lr=0.02, gpu_id=None, ifGt=False, gt=None):
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print('missing_count', np.isnan(original).sum())
    
    # 预处理
    first = FirstProcess(original.copy())
    mask = (~np.isnan(first)).astype(int)
    initial_filled = SecondProcess(first)
    initial_filled_copy = initial_filled.copy()
    
    # 使用float32确保兼容性
    x = torch.tensor(initial_filled[None, ...], dtype=torch.float32, device=device)
    y = torch.tensor(initial_filled[None, ...], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, ...], dtype=torch.float32, device=device)
    
    # 创建模型
    print("causal_matrix.shape", causal_matrix.shape)
    model = ParallelFeatureADDSTCN(
        causal_matrix=causal_matrix,
        model_params=model_params
    ).to(device)

    # 编译加速
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except:
            pass

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None

    # 早停机制
    best_loss = float('inf')
    best_imputed = None
    patience = 15
    no_improve_count = 0
    
    # 预计算统计量
    y_mean = y.mean(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)
    quantiles = [0.25, 0.5, 0.75]
    y_quantiles = [torch.quantile(y.float(), q, dim=1, keepdim=True) for q in quantiles]  # ✅ 确保float32

    for epoch in range(epochs):
        opt.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(x)
                
                # Loss1: 观测值的预测误差
                loss_1 = F.mse_loss(pred * m, y * m)
                
                # ✅ 修复：确保pred是float32类型用于统计计算
                pred_float = pred.float()  # 明确转换为float32
                
                pred_mean = pred_float.mean(dim=1, keepdim=True)
                pred_std = pred_float.std(dim=1, keepdim=True)
                
                mean_loss = F.mse_loss(pred_mean, y_mean)
                std_loss = F.mse_loss(pred_std, y_std)
                
                # ✅ 修复：使用float32版本计算分位数
                quantile_losses = []
                for i, q in enumerate(quantiles):
                    pred_q = torch.quantile(pred_float, q, dim=1, keepdim=True)
                    quantile_losses.append(F.mse_loss(pred_q, y_quantiles[i]))
                
                loss_3 = (mean_loss + std_loss + sum(quantile_losses)) / (2 + len(quantiles))
                total_loss = 0.6 * loss_1 + 0.4 * loss_3
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            # 普通训练
            pred = model(x)
            
            loss_1 = F.mse_loss(pred * m, y * m)
            
            # ✅ 确保pred是float32类型
            pred_float = pred.float()
            
            pred_mean = pred_float.mean(dim=1, keepdim=True)
            pred_std = pred_float.std(dim=1, keepdim=True)
            
            mean_loss = F.mse_loss(pred_mean, y_mean)
            std_loss = F.mse_loss(pred_std, y_std)
            
            quantile_losses = []
            for i, q in enumerate(quantiles):
                pred_q = torch.quantile(pred_float, q, dim=1, keepdim=True)  # ✅ 使用float版本
                quantile_losses.append(F.mse_loss(pred_q, y_quantiles[i]))
            
            loss_3 = (mean_loss + std_loss + sum(quantile_losses)) / (2 + len(quantiles))
            total_loss = 0.6 * loss_1 + 0.4 * loss_3
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        
        scheduler.step()

        # 早停检查
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve_count = 0
            with torch.no_grad():
                best_imputed = model(x).float().cpu().squeeze(0).numpy()  # ✅ 确保转换为float32
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 减少打印频率和显存清理
        if epoch % 2 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {current_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 用最优结果进行填补
    res = initial_filled.copy()
    if best_imputed is not None:
        res[mask == 0] = best_imputed[mask == 0]
    
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

def impute_wrapper(task_queue, result_queue, causal_matrix, model_params, epochs, lr, output_dir):
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, file_path, gpu_id = task

        try:
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            df = pd.read_csv(file_path)
            data = df.values.astype(np.float32)

            imputed, mask, initial = impute(data, causal_matrix, model_params, epochs, lr, gpu_id=0)

            # 构造保存路径
            filename = os.path.basename(file_path).replace('.csv', '_imputed.csv')
            save_path = os.path.join(output_dir, filename)
            pd.DataFrame(imputed).to_csv(save_path, index=False)

            result_queue.put((idx, file_path, save_path))
        except Exception as e:
            result_queue.put((idx, file_path, f"Error: {e}"))


def parallel_impute(
    file_paths,  # 这个参数应该改名为 data_dir 更清楚
    causal_matrix,
    model_params,
    epochs=150,
    lr=0.02,
    simultaneous_per_gpu=2,
    max_workers=None,
    output_dir="imputed_results"
):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    os.makedirs(output_dir, exist_ok=True)

    # ✅ 修复：正确处理输入路径
    if isinstance(file_paths, str) and os.path.isdir(file_paths):
        # 如果传入的是目录路径，获取所有CSV文件
        file_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
        print(f"从目录 '{file_paths}' 找到 {len(file_list)} 个CSV文件")
    elif isinstance(file_paths, list):
        # 如果传入的是文件列表
        file_list = file_paths
        print(f"收到文件列表，共 {len(file_list)} 个文件")
    else:
        raise ValueError("file_paths must be a directory path or list of file paths")

    if len(file_list) == 0:
        print("❌ 错误: 没有找到任何CSV文件")
        return {}

    # ✅ 添加调试信息
    print(f"输出目录: {output_dir}")
    print(f"前3个文件: {file_list[:3]}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("⚠️ 警告: 未检测到GPU，使用CPU模式")
        # CPU模式下也可以运行，不要抛出错误
        total_workers = min(4, len(file_list))  # CPU模式限制进程数
    else:
        total_workers = num_gpus * simultaneous_per_gpu
        
    if max_workers:
        total_workers = min(total_workers, max_workers)

    print(f"总工作进程数: {total_workers}")

    task_queue = Queue()
    result_queue = Queue()

    # 添加任务 - 使用正确的文件列表
    for idx, file_path in enumerate(file_list):  # ✅ 使用 file_list 而不是 file_paths
        assigned_gpu = idx % num_gpus if num_gpus > 0 else None
        task_queue.put((idx, file_path, assigned_gpu))
        
        # ✅ 添加调试信息
        if idx < 5:
            print(f"任务 {idx}: {file_path} -> GPU {assigned_gpu}")

    # 添加终止信号
    for _ in range(total_workers):
        task_queue.put(None)

    # 启动 workers
    workers = []
    for worker_id in range(total_workers):
        p = Process(
            target=impute_wrapper,
            args=(task_queue, result_queue, causal_matrix, model_params, epochs, lr, output_dir)
        )
        p.start()
        workers.append(p)
        print(f"启动工作进程 {worker_id+1}/{total_workers}")

    results = {}
    completed_count = 0
    
    # ✅ 修复进度条 - 使用正确的总数
    with tqdm(total=len(file_list), desc="Imputing and Saving") as pbar:
        for _ in range(len(file_list)):  # ✅ 使用 file_list 的长度
            idx, path, result = result_queue.get()
            results[path] = result
            completed_count += 1
            
            # ✅ 添加详细的结果信息
            if isinstance(result, str) and result.startswith("Error"):
                print(f"❌ 文件 {os.path.basename(path)} 失败: {result}")
            else:
                print(f"✅ 文件 {os.path.basename(path)} 完成: {result}")
            
            pbar.update(1)

    # 等待所有进程完成
    for p in workers:
        p.join()

    print(f"总体完成情况: {completed_count}/{len(file_list)} 个文件")
    
    # ✅ 检查输出目录
    if os.path.exists(output_dir):
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        print(f"输出目录 '{output_dir}' 中有 {len(output_files)} 个文件")
        if len(output_files) > 0:
            print(f"前3个输出文件: {output_files[:3]}")
    else:
        print(f"❌ 输出目录 '{output_dir}' 不存在")

    return results
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
        return pd.read_csv(standard_cg, header=None).values

    # Step 1: 批量预处理
    initial_matrix_arr = []
    batch_size = 100
    
    for i in tqdm(range(0, len(original_matrix_arr), batch_size), desc="批量预处理"):
        batch = original_matrix_arr[i:i+batch_size]
        batch_results = [initial_process(matrix) for matrix in batch]
        initial_matrix_arr.extend(batch_results)
        
        if i % (batch_size * 5) == 0:
            gc.collect()

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
        ('tefn_impu', tefn_impu),('timesnet_impu', timesnet_impu),
        ('tsde_impu', tsde_impu)
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