import os
import torch
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
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
from e import pre_checkee
from sklearn.preprocessing import StandardScaler
from multiprocessing import set_start_method
from scipy.stats import wasserstein_distance
from models_downstream import *
from multiprocessing import Process, Queue
from models_TCN import MultiADDSTCN, ParallelFeatureADDSTCN, ADDSTCN
import subprocess
import time
from multiprocessing import Pool, get_context
from functools import partial

def set_seed_all(seed=42):
    """设置所有随机种子以确保可重复性"""
    import random
    import numpy as np
    import torch
    import os
    
    # Python random
    random.seed(seed)
    
    # Numpy random (pygrinder依赖这个)
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置Python hash种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🎲 设置全局随机种子: {seed}")

def wait_for_gpu_free(threshold_mb=500, sleep_time=10):
    """
    等待所有 GPU 显存占用都小于阈值（单位：MiB），再返回。
    默认等待所有GPU显存小于500MB。
    """
    print(f"⏳ 正在等待GPU空闲 (显存占用 < {threshold_mb}MiB)...")
    while True:
        try:
            output = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", 
                shell=True
            )
            used_memory = [int(x) for x in output.decode().strip().split('\n')]
            if all(mem < threshold_mb for mem in used_memory):
                print("✅ 所有GPU空闲，可开始执行 miracle_impu。")
                break
            else:
                print(f"🚧 显存使用情况: {used_memory} MiB，不满足要求，等待 {sleep_time}s...")
                time.sleep(sleep_time)
        except Exception as e:
            print(f"检测 GPU 显存失败: {e}")
            time.sleep(sleep_time)

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

def impute(original, causal_matrix, model_params, epochs=100, lr=0.02, gpu_id=None, ifGt=False, gt=None, ablation=0, seed=42):
    """添加seed参数"""
    
    # ✅ 设置种子确保训练过程可重复
    set_seed_all(seed)
    
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print('missing_count', np.isnan(original).sum())
    
    # 预处理
    first = FirstProcess(original.copy())
    mask = (~np.isnan(first)).astype(int)
    initial_filled = SecondProcess(first)
    initial_filled_copy = initial_filled.copy()
    
    # 标准化
    scaler = StandardScaler()
    initial_filled_scaled = scaler.fit_transform(initial_filled)
    
    # 使用标准化后的数据创建张量
    x = torch.tensor(initial_filled_scaled[None, ...], dtype=torch.float32, device=device)
    y = torch.tensor(initial_filled_scaled[None, ...], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, ...], dtype=torch.float32, device=device)

    # ✅ 创建模型前再次设置种子
    set_seed_all(seed)
    if ablation==0:
        ablation_causal = causal_matrix.copy()
        ablation_causal = ablation_causal[...]==1
        model = ParallelFeatureADDSTCN(
            causal_matrix=ablation_causal,
            model_params=model_params
        ).to(device)
    elif ablation==1:
        model = ParallelFeatureADDSTCN(
            causal_matrix=causal_matrix,
            model_params=model_params
        ).to(device)
    elif ablation==2:
        model = MultiADDSTCN(
            causal_mask=causal_matrix,
            num_levels=4,
            cuda=True
        ).to(device)

    # 编译加速
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except:
            pass

    # ✅ 优化器初始化前设置种子
    set_seed_all(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    grad_scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None

    # 早停机制
    best_loss = float('inf')
    best_imputed = None
    patience = 15
    no_improve_count = 0
    
    # 预计算统计量
    y_mean = y.mean(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)
    quantiles = [0.25, 0.5, 0.75]
    y_quantiles = [torch.quantile(y.float(), q, dim=1, keepdim=True) for q in quantiles]

    for epoch in range(epochs):
        opt.zero_grad()
        
        if grad_scaler:  # ✅ 使用 grad_scaler
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
            
            grad_scaler.scale(total_loss).backward()  # ✅ 使用 grad_scaler
            grad_scaler.unscale_(opt)                 # ✅ 使用 grad_scaler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(opt)                     # ✅ 使用 grad_scaler
            grad_scaler.update()                      # ✅ 使用 grad_scaler
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
                torch.cuda.ipc_collect()

    # 用最优结果进行填补并反标准化
    res = initial_filled.copy()
    if best_imputed is not None:
        best_imputed_rescaled = scaler.inverse_transform(best_imputed)  # ✅ 反标准化
        res[mask == 0] = best_imputed_rescaled[mask == 0]
    
    pd.DataFrame(res).to_csv("result_1.csv", index=False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return res, mask, initial_filled_copy



def impute_wrapper(args):
        import torch
        import os

        # ✅ 解包新增的 skip_existing 参数
        if len(args) == 11:  # 新版本有10个参数
            idx, mx, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr, skip_existing = args
        else:  # 兼容旧版本（9个参数）
            idx, mx, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr = args
            skip_existing = False

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')

        try:
            # ✅ 构造输出文件路径
            filename = os.path.basename(file_path).replace('.csv', '_imputed.csv')
            save_path = os.path.join(output_dir, filename)
            
            # ✅ 在worker级别再次检查是否跳过（双重保险）
            if skip_existing and os.path.exists(save_path):
                print(f"⏩ Worker级跳过已存在文件: {filename}")
                return idx, save_path

            # 执行填补
            imputed_result, mask, initial_processed = impute(
                mx,
                causal_matrix,
                model_params=model_params,
                epochs=epochs, lr=lr, gpu_id=gpu_id
            )

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            # 保存结果
            pd.DataFrame(imputed_result).to_csv(save_path, index=False)

            print(f"✅ 完成填补: {os.path.basename(file_path)} → {filename}")
            return idx, save_path
            
        except Exception as e:
            print(f"❌ 填补失败: {os.path.basename(file_path)}, 错误: {e}")
            return idx, f"Error: {e}"


def parallel_impute(
    file_paths,             # str 目录路径（如 ./data/mimic-iii）
    causal_matrix,          # 因果图 cg
    model_params,           # 填补模型参数
    epochs=100,
    lr=0.02,
    simultaneous_per_gpu=2,
    output_dir="imputed_results",
    skip_existing=False     # ✅ 新增参数：是否跳过已存在的文件
):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[INFO] 没有可用 GPU，使用 CPU 顺序处理")
        num_gpus = 1

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 使用 {num_gpus} 个 GPU，每个 GPU 最多并行 {simultaneous_per_gpu} 个任务，总进程数: {num_gpus * simultaneous_per_gpu}")

    # ✅ 获取文件路径列表
    file_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
    print(f"[INFO] 找到 {len(file_list)} 个待处理文件")

    # ✅ 检查跳过逻辑
    if skip_existing:
        print(f"🔍 启用跳过已存在文件模式，检查输出目录: {output_dir}")
        existing_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
        
        # 过滤出需要处理的文件
        filtered_file_list = []
        skipped_count = 0
        
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename in existing_files:
                skipped_count += 1
                print(f"⏩ 跳过已存在文件: {filename}")
            else:
                filtered_file_list.append(file_path)
        
        file_list = filtered_file_list
        print(f"📊 跳过统计: {skipped_count} 个已存在，{len(file_list)} 个待处理")
        
        if len(file_list) == 0:
            print("✅ 所有文件都已存在，无需处理")
            return {}

    args_list = []
    for idx, file_path in enumerate(file_list):
        df = pd.read_csv(file_path)
        data = df.values.astype(np.float32)
        gpu_id = idx % num_gpus
        # ✅ 传递 skip_existing 参数到 worker
        args_list.append((idx, data, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr, skip_existing))
    
    with mp.Pool(processes=num_gpus * simultaneous_per_gpu) as pool:
        results = list(tqdm(pool.imap(impute_wrapper, args_list), total=len(args_list), desc="Filling"))

    results.sort(key=lambda x: x[0])
    output_paths = {file_list[idx]: result for idx, result in results}
    return output_paths

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

def causal_discovery(original_matrix_arr, n_cluster=5, isStandard=False, standard_cg=None,met='lorenz',
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
    pd.DataFrame(new_matrix).to_csv(f'./causality_matrices/{met}_causality_matrix.csv', index=False, header=False)
    return new_matrix

# ================================
# 1. 单文件评估函数
# ================================
def mse_evaluate_single_file(mx, causal_matrix, gpu_id=0, device=None, met='lorenz', missing='mar', seed=42, ablation=0):
    """添加seed参数控制随机性"""
    
    # ✅ 每次调用都重新设置种子，确保挖洞过程可重复
    set_seed_all(seed)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ground truth
    gt = mx.copy()
    gt2 = gt.copy()
    pd.DataFrame(gt).to_csv("gt_matrix.csv", index=False)  # 改个名避免冲突
    
    # ✅ 挖洞过程 - 在设置种子后立即执行
    try:
        print(f"🔍 开始挖洞过程 (seed={seed})...")
        
        # 先设置一次种子确保mar_logistic的确定性
        set_seed_all(seed)
        if missing == 'mar':
            X = mar_logistic(mx, obs_rate=0.1, missing_rate=0.6)
        
        # 后续步骤也需要保持确定性
        if missing == 'mnar':
            X = mx.copy()
            X = X[np.newaxis, ...]  
            X = mnar_x(X, offset=0.6)
            X = X.squeeze(0)
        
        if missing == 'mcar':
            X = mx.copy()
            X = X[np.newaxis, ...]  
            X = mcar(X, p=0.5)
            X = X.squeeze(0)
        pre_checkee(X, met)
        print(f"✅ 挖洞完成，缺失率: {np.isnan(X).sum() / X.size:.2%}")
        
    except (ValueError, RuntimeError) as e:
        print(f"⚠️ mar_logistic失败，跳过此文件: {e}")
        return None
    
    pd.DataFrame(X).to_csv("missing_matrix.csv", index=False)
    
    # mask: 观测为 1，缺失为 0
    Mask = (~np.isnan(X)).astype(int)
    
    # 掩码版 MSE，只在缺失位置评估
    def mse(a, b, mask):
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        b = torch.as_tensor(b, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        mask = 1 - mask  # 反转mask，1表示缺失位置
        
        # 保存调试信息
        pd.DataFrame((a * mask).cpu().numpy()).to_csv("pred_missing.csv", index=False)
        pd.DataFrame((b * mask).cpu().numpy()).to_csv("gt_missing.csv", index=False)
        pd.DataFrame(mask.cpu().numpy()).to_csv("missing_mask.csv", index=False)
        
        # 计算 masked MSE
        masked_error = F.mse_loss(a * mask, b * mask).item()
        return masked_error

    res = {}

    # ✅ 我的模型评估 - 传递种子
    print("开始执行 my_model...")
    set_seed_all(seed)  # 确保模型训练也是确定的
    imputed_result, mask, initial_processed = impute(
        X, causal_matrix,
        model_params={'num_levels':10, 'kernel_size': 8, 'dilation_c': 2},
        epochs=100, lr=0.02, gpu_id=gpu_id, ifGt=True, gt=gt, seed=seed, ablation=ablation
    )
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    if ablation==1:
        res['ablation1'] = mse(imputed_result, gt2, mask)
    elif ablation==2:
        res['ablation2'] = mse(imputed_result, gt2, mask)

    def is_reasonable_mse(mse_value, threshold=1000000.0):
        return (not np.isnan(mse_value) and 
                not np.isinf(mse_value) and 
                0 <= mse_value <= threshold)

    # ✅ baseline 方法 - 每个方法执行前都设置种子
    baseline = [
        ('initial_process', initial_process),
        ('zero_impu', zero_impu),
        # ('mean_impu', mean_impu),
        # ('knn_impu', knn_impu),
        # ('mice_impu', mice_impu),
        # ('ffill_impu', ffill_impu), 
        # ('bfill_impu', bfill_impu),
        # ('miracle_impu', miracle_impu), 
        # ('saits_impu', saits_impu),
        # ('timemixerpp_impu', timemixerpp_impu), 
        # ('tefn_impu', tefn_impu),
        # ('timesnet_impu', timesnet_impu),
        # ('tsde_impu', tsde_impu),
        # ('grin_impu', grin_impu),
    ]
    if not ablation:
        res['my_model'] = mse(imputed_result, gt2, Mask)

    for name, fn in baseline:
        print(f"开始执行 {name}...")
        
        # ✅ 每个baseline方法执行前设置相同种子
        set_seed_all(seed)
        
        try:
            result = fn(X)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
        except Exception as e:
            print(f"❌ {name} 执行失败: {e}")
            res[name] = float('nan')

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
    print(f"所有结果: {res}")
    return res

# ================================
# 2. 用于 Pool 的包装函数（每个任务）
# ================================
def worker_wrapper(args):
    import torch
    import os

    idx, mx, causal_matrix, gpu_id, met, missing, seed, ablation = args  # ✅ 添加seed参数

    # ✅ 每个worker进程都设置相同的基础种子
    set_seed_all(seed + idx)  # 每个样本使用不同但确定的种子

    print(f"[Worker PID {os.getpid()}] 分配到 GPU: {gpu_id}, Seed: {seed + idx}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[Worker PID {os.getpid()}] 实际使用 device: {device}")
    else:
        device = torch.device('cpu')
        print(f"[Worker PID {os.getpid()}] 警告：未检测到 GPU，使用 CPU")

    # ✅ 传递种子到评估函数
    res = mse_evaluate_single_file(mx, causal_matrix, gpu_id=gpu_id, device=device, met=met, missing=missing, seed=seed + idx, ablation=ablation)
    return idx, res

# ================================
# 3. 并行调度函数（进程池实现）
# ================================
def parallel_mse_evaluate(res_list, causal_matrix, met, simultaneous_per_gpu=3, missing='mar', seed=42, ablation=0):
    """添加seed参数"""
    
    # ✅ 设置主进程种子
    set_seed_all(seed)
    
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("[INFO] 没有可用 GPU，使用 CPU 顺序评估")
        all_res = []
        for i, x in enumerate(tqdm(res_list, desc="CPU")):
            set_seed_all(seed + i)  # 每个样本使用确定种子
            result = mse_evaluate_single_file(x, causal_matrix, seed=seed + i)
            all_res.append(result)
        
        # 计算平均值...
        return {k: float(np.nanmean([d[k] for d in all_res if d is not None]))
                for k in all_res[0] if all_res[0] is not None}

    max_workers = num_gpus * simultaneous_per_gpu
    print(f"[INFO] 使用 {num_gpus} 个 GPU，每个 GPU 最多并行 {simultaneous_per_gpu} 个任务，总进程数: {max_workers}")
    print(f"🎲 使用基础种子: {seed}")

    # ✅ 为每个任务分配对应的 GPU 和种子
    gpu_ids = [i % num_gpus for i in range(len(res_list))]
    args_list = [(i, res_list[i], causal_matrix, gpu_ids[i], met, missing, seed, ablation) for i in range(len(res_list))]

    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(worker_wrapper, args_list), total=len(args_list), desc="All‑tasks"))

    results.sort(key=lambda x: x[0])  # 恢复顺序
    only_result_dicts = [res for _, res in results if res is not None]

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
    
    # ✅ 保存结果时包含种子信息
    pd.DataFrame([{'Method': k, 'Average_MSE': v, 'Seed': seed} for k, v in avg.items()]) \
        .to_csv(f"mse_evaluation_results_seed{seed}.csv", index=False)

    return avg