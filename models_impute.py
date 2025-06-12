import os
import torch
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
    
    # 预计算所有张量
    mask = (~np.isnan(original)).astype(int)
    initial_filled = initial_process(original)
    sequence_len, total_features = initial_filled.shape
    final_filled = initial_filled.copy()
    
    # 批处理准备数据
    all_inputs = []
    all_targets = []
    all_masks = []
    all_inds = []
    
    for target in range(total_features):
        inds = list(np.where(causal_matrix[:, target] == 1)[0])
        if target not in inds:
            inds.append(target)
        else:
            inds.remove(target)
            inds.append(target)
        inds = inds[:3] + [target]
        
        inp = torch.tensor(initial_filled[:, inds].T[np.newaxis, ...], dtype=torch.float32).to(device)
        y_np = torch.tensor(initial_filled[:, target][np.newaxis, :, None], dtype=torch.float32).to(device)
        m_np = torch.tensor((mask[:, target] == 1)[np.newaxis, :, None], dtype=torch.float32).to(device)
        
        all_inputs.append(inp)
        all_targets.append(y_np)
        all_masks.append(m_np)
        all_inds.append(inds)
    
    # 并行训练多个特征
    for target in range(total_features):
        x, y, m, inds = all_inputs[target], all_targets[target], all_masks[target], all_inds[target]
        
        model = ADDSTCN(target, input_size=len(inds), cuda=(device != torch.device('cpu')), **model_params).to(device)
        
        # 优化器设置
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # AdamW更快
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr*5, total_steps=epochs)  # 更激进的调度
        
        # 编译模型加速（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except:
                pass  # 如果编译失败，继续使用原模型
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            pred = model(x)
            loss = F.mse_loss(pred * m, y * m) + 0.001 * sum(p.abs().sum() for p in model.parameters())
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            
            # 早停策略
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 10:  # 早停
                    break
        
        # 预测
        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask[:, target] == 0)
            final_filled[to_fill, target] = out[to_fill]
    
    return final_filled

def impute_single_file(args):
    """单文件填补函数，用于进程池"""
    file_path, causal_matrix, model_params, epochs, lr, gpu_id = args
    
    # 设置GPU
    if gpu_id != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    try:
        # 读取数据
        data = pd.read_csv(file_path).values.astype(np.float32)
        filename = os.path.basename(file_path)
        
        # 调用优化后的impute函数
        result = impute(data, causal_matrix, model_params, epochs=epochs, lr=lr, gpu_id=gpu_id)
        
        return filename, result
    except Exception as e:
        return os.path.basename(file_path), None

def parallel_impute(causal_matrix, input_dir, model_params, epochs=100, lr=0.01, max_workers=None):
    """使用进程池的并行填补"""
    # 获取文件列表
    file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    # 确定工作进程数和GPU分配
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        max_workers = max_workers or os.cpu_count()
        gpu_ids = ['cpu'] * len(file_list)
    else:
        # 每个GPU运行2个进程以提高利用率
        max_workers = max_workers or (num_gpus * 2)
        gpu_ids = [i % num_gpus for i in range(len(file_list))]
    
    print(f"使用 {max_workers} 个进程并行处理 {len(file_list)} 个文件")
    
    # 准备参数列表
    args_list = [(file_path, causal_matrix, model_params, epochs, lr, gpu_id) 
                 for file_path, gpu_id in zip(file_list, gpu_ids)]
    
    # 并行执行
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(impute_single_file, args_list),
            total=len(file_list),
            desc="批量填补中",
            ncols=80
        ))
    
    # 保存结果
    os.makedirs("./data_imputed/my_model", exist_ok=True)
    successful_results = []
    
    for filename, result in results:
        if result is not None:
            successful_results.append(result)
            pd.DataFrame(result).to_csv(f"./data_imputed/my_model/{filename}", index=False)
        else:
            print(f"[错误] {filename} 填补失败")
    
    print(f"成功填补 {len(successful_results)}/{len(file_list)} 个文件")
    return successful_results

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
# def causal_worker(task_queue, result_queue, initial_matrix_arr, params, gpu_id):
#     """
#     每个进程运行的 worker，绑定指定 GPU，处理 task_queue 中的任务
#     """
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     torch.cuda.set_device(0)  # 每个子进程看的是 CUDA_VISIBLE_DEVICES 下的 cuda:0

#     while True:
#         item = task_queue.get()
#         if item is None:
#             break
#         task_id, i = item
#         try:
#             matrix = compute_causal_matrix(initial_matrix_arr[i], params=params, gpu_id=0)
#             result_queue.put((task_id, matrix))
#         except Exception as e:
#             print(f"[GPU {gpu_id}] 任务 {task_id} 失败: {e}")
#             result_queue.put((task_id, None))
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
    # num_gpus = torch.cuda.device_count()
    # task_queue = mp.Queue()
    # result_queue = mp.Queue()

    # for task_id, i in enumerate(idx_arr):
    #     task_queue.put((task_id, i))
    # for _ in range(num_gpus):
    #     task_queue.put(None)

    # workers = []
    # for gpu_id in range(num_gpus):
    #     p = mp.Process(target=causal_worker, args=(task_queue, result_queue, initial_matrix_arr, params, gpu_id))
    #     p.start()
    #     workers.append(p)

    # results = [None] * len(idx_arr)

    # # ✅ 用 tqdm 包裹 result_queue.get() 获取进度条
    # for _ in tqdm(range(len(idx_arr)), desc="因果发现中"):
    #     task_id, matrix = result_queue.get()
    #     results[task_id] = matrix

    # for p in workers:
    #     p.join()
    data_list = []
    for idx in idx_arr:  # idx_arr 是代表样本的索引列表
        data_list.append(initial_matrix_arr[idx]) 
    params_list = [params] * len(data_list)  
    results = parallel_compute_causal_matrices(data_list, params_list)
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
def mse_evaluate_single(args):
    """单个MSE评估任务函数，用于进程池"""
    task_data, causal_matrix, gpu_id = args
    
    # 在任何TensorFlow操作之前设置环境变量
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id != 'cpu' else ""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # 现在导入TensorFlow
    import tensorflow as tf
    
    # 配置TensorFlow
    if gpu_id != 'cpu':
        # 配置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # 在子进程中，只能看到一个GPU（由CUDA_VISIBLE_DEVICES设置）
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"[Worker {gpu_id}] TensorFlow配置GPU成功")
            except RuntimeError as e:
                print(f"[Worker {gpu_id}] TensorFlow GPU配置失败: {e}")
    
    # 设置PyTorch GPU
    if gpu_id != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(0)  # 在子进程中总是0（由环境变量控制）
        device = torch.device('cuda:0')
        print(f"[Worker {gpu_id}] PyTorch使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[Worker {gpu_id}] 使用CPU")
    
    try:
        # 数据解包逻辑
        if isinstance(task_data, tuple):
            if len(task_data) == 2:
                idx, data = task_data
                if isinstance(data, tuple) and len(data) == 2:
                    filename, matrix = data
                    print(f"[Worker {gpu_id}] 处理文件 {filename}")
                else:
                    matrix = data
                    print(f"[Worker {gpu_id}] 处理索引 {idx}")
            else:
                print(f"[Worker {gpu_id}] 意外的数据格式: {type(task_data)}, 长度: {len(task_data)}")
                idx = 0
                matrix = task_data[-1]
        else:
            idx = 0
            matrix = task_data
            print(f"[Worker {gpu_id}] 直接处理矩阵数据")
        
        # 验证输入
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"期望numpy数组，得到{type(matrix)}")
        if matrix.ndim != 2:
            raise ValueError(f"期望二维矩阵，得到{matrix.ndim}维")
        
        # 执行MSE评估 - 传递实际的GPU设备ID
        actual_gpu_id = 0 if gpu_id != 'cpu' else None
        result = mse_evaluate_safe(matrix, causal_matrix, device=device, tf_gpu_id=actual_gpu_id)
        
        return idx, result
        
    except Exception as e:
        import traceback
        print(f"[Worker {gpu_id}] 评估任务失败: {e}")
        print(traceback.format_exc())
        return 0, None

def mse_evaluate_safe(mx, causal_matrix, device=None, tf_gpu_id=None):
    """安全的MSE评估函数，避免TensorFlow设备冲突"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gpu_id = tf_gpu_id if tf_gpu_id is not None else 0
    
    ground_truth = mx.copy()
    X_block_3d = block_missing(mx[np.newaxis, ...], factor=0.1, block_width=3, block_len=3)
    X_block = X_block_3d[0]

    def mse_pytorch(a, b):
        """使用PyTorch计算MSE"""
        try:
            if isinstance(a, np.ndarray):
                a_tensor = torch.from_numpy(a).float()
            else:
                a_tensor = a.float()
                
            if isinstance(b, np.ndarray):
                b_tensor = torch.from_numpy(b).float()
            else:
                b_tensor = b.float()
            
            # 分批处理避免内存问题
            if a_tensor.numel() > 500000:
                batch_size = 100000
                total_loss = 0
                total_elements = 0
                
                a_flat = a_tensor.view(-1)
                b_flat = b_tensor.view(-1)
                
                for i in range(0, a_flat.numel(), batch_size):
                    end_idx = min(i + batch_size, a_flat.numel())
                    a_batch = a_flat[i:end_idx].to(device)
                    b_batch = b_flat[i:end_idx].to(device)
                    
                    batch_loss = F.mse_loss(a_batch, b_batch, reduction='sum')
                    total_loss += batch_loss.cpu().item()
                    total_elements += a_batch.numel()
                    
                    del a_batch, b_batch, batch_loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                return total_loss / total_elements
            else:
                a_tensor = a_tensor.to(device)
                b_tensor = b_tensor.to(device)
                result = F.mse_loss(a_tensor, b_tensor).cpu().item()
                
                del a_tensor, b_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                return result
                
        except Exception as e:
            print(f"MSE计算失败: {e}")
            return float('inf')

    results = {}
    
    # 1. 我们的方法
    try:
        print(f"[GPU {gpu_id}] 评估我们的方法...")
        imputed = impute(
            X_block, causal_matrix,
            model_params={'num_levels': 6, 'kernel_size': 6, 'dilation_c': 4},
            epochs=50, lr=0.01, gpu_id=gpu_id
        )
        results['my_model'] = mse_pytorch(imputed, ground_truth)
        print(f"✅ 我们的方法 MSE: {results['my_model']:.6f}")
        del imputed
    except Exception as e:
        print(f"❌ 我们的方法失败: {e}")
        results['my_model'] = float('inf')
    
    # 2. 简单基线方法
    simple_methods = [
        ('zero_impu', zero_impu),
        ('mean_impu', mean_impu), 
        ('median_impu', median_impu),
        ('mode_impu', mode_impu),
        ('random_impu', random_impu),
        ('knn_impu', knn_impu),
        ('ffill_impu', ffill_impu),
        ('bfill_impu', bfill_impu),
    ]
    
    for method_name, method_func in simple_methods:
        try:
            result = method_func(X_block)
            results[method_name] = mse_pytorch(result, ground_truth)
            del result
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            results[method_name] = float('inf')
    
    # 3. TensorFlow方法 - 使用全局配置
    tf_methods = [
        ('miracle_impu', miracle_impu),
        ('saits_impu', saits_impu),
        ('timemixerpp_impu', timemixerpp_impu),
        ('tefn_impu', tefn_impu),
    ]
    
    import tensorflow as tf
    
    # 创建单一的TensorFlow配置
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    if device.type == 'cuda':
        config.gpu_options.visible_device_list = "0"  # 子进程中总是使用第一个可见GPU
    
    for method_name, method_func in tf_methods:
        try:
            print(f"🔄 [GPU {gpu_id}] 评估 {method_name}...")
            
            # 使用统一的会话配置
            with tf.compat.v1.Session(config=config) as sess:
                result = method_func(X_block)
                results[method_name] = mse_pytorch(result, ground_truth)
                print(f"✅ {method_name} MSE: {results[method_name]:.6f}")
                del result
                
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            results[method_name] = float('inf')
        finally:
            # 清理资源
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return results

def parallel_mse_evaluate(res_list, causal_matrix, max_workers=None):
    """使用进程池的并行MSE评估，与parallel_impute相同的模式"""
    
    # 确定工作进程数和GPU分配
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        max_workers = max_workers or os.cpu_count()
        gpu_ids = ['cpu'] * len(res_list)
        print(f"使用 {max_workers} 个CPU进程处理 {len(res_list)} 个MSE评估任务")
    else:
        # 每个GPU运行2个进程以提高利用率
        max_workers = max_workers or (num_gpus * 2)
        gpu_ids = [i % num_gpus for i in range(len(res_list))]
        print(f"使用 {max_workers} 个进程，{num_gpus} 个GPU并行处理 {len(res_list)} 个MSE评估任务")
    
    # 准备参数列表 - 简化数据结构
    args_list = []
    for i, matrix in enumerate(res_list):
        # 简化：直接传递 (索引, 矩阵) 的格式
        task_data = (i, matrix)
        args_list.append((task_data, causal_matrix, gpu_ids[i]))
    
    print(f"准备处理 {len(args_list)} 个MSE评估任务")
    
    # 并行执行
    with Pool(processes=max_workers) as pool:
        results_raw = list(tqdm(
            pool.imap(mse_evaluate_single, args_list),
            total=len(res_list),
            desc="MSE评估中",
            ncols=80
        ))
    
    # 按索引排序结果
    results = [None] * len(res_list)
    for idx, result in results_raw:
        if idx < len(results):
            results[idx] = result
    
    # 处理结果统计 - 专门调试TimeMixerPP
    valid_mse_dicts = [d for d in results if d is not None and isinstance(d, dict)]
    
    if not valid_mse_dicts:
        print("错误: 所有MSE评估任务均失败!")
        return None
    else:
        print(f"成功完成 {len(valid_mse_dicts)}/{len(results)} 个MSE评估")
        
        # 特别调试TimeMixerPP的所有结果
        print("\n🔍 TimeMixerPP 详细结果分析:")
        timemixer_values = []
        for i, d in enumerate(valid_mse_dicts):
            if d is not None and 'timemixerpp_impu' in d:
                value = d['timemixerpp_impu']
                timemixer_values.append((i, value))
                if np.isinf(value):
                    print(f"  任务 {i}: {value} ⚠️ INF")
                elif value > 1e6:
                    print(f"  任务 {i}: {value:.2e} ⚠️ 极大值")
                else:
                    print(f"  任务 {i}: {value:.6f} ✅")
        
        print(f"\nTimeMixerPP 统计:")
        values_only = [v for _, v in timemixer_values]
        finite_values = [v for v in values_only if not np.isinf(v) and not np.isnan(v)]
        inf_count = sum(1 for v in values_only if np.isinf(v))
        
        print(f"  - 总任务数: {len(values_only)}")
        print(f"  - 有限值数量: {len(finite_values)}")
        print(f"  - inf 值数量: {inf_count}")
        
        if finite_values:
            print(f"  - 有限值范围: [{min(finite_values):.6f}, {max(finite_values):.6f}]")
            print(f"  - 有限值平均: {sum(finite_values)/len(finite_values):.6f}")
        
        # 计算所有方法的平均MSE
        avg_mse = {}
        for method in tqdm(valid_mse_dicts[0].keys(), desc="计算平均 MSE"):
            vals = []
            inf_count = 0
            large_count = 0  # 极大值计数
            
            for d in valid_mse_dicts:
                if d is not None and method in d:
                    value = d[method]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if np.isinf(value):
                            inf_count += 1
                        elif value > 1e10:  # 极大值阈值
                            large_count += 1
                            print(f"⚠️ {method} 发现极大值: {value:.2e}")
                        elif not np.isnan(value):
                            vals.append(float(value))
            
            # 特别处理TimeMixerPP
            if method == 'timemixerpp_impu':
                print(f"\n{method} 最终统计:")
                print(f"  - 正常值: {len(vals)} 个")
                print(f"  - inf值: {inf_count} 个")
                print(f"  - 极大值: {large_count} 个")
                
                if len(vals) > 0:
                    avg_val = sum(vals) / len(vals)
                    print(f"  - 正常值平均: {avg_val:.6f}")
                    
                    # 如果有inf或极大值，只用正常值计算平均
                    if inf_count > 0 or large_count > 0:
                        print(f"  - 忽略异常值，使用正常值平均")
                        avg_mse[method] = avg_val
                    else:
                        # 计算总平均（包括可能的极值）
                        all_finite_vals = vals.copy()
                        total_avg = sum(all_finite_vals) / len(all_finite_vals) if all_finite_vals else float('inf')
                        avg_mse[method] = total_avg
                else:
                    avg_mse[method] = float('inf')
            else:
                # 其他方法的正常处理
                if len(vals) > 0:
                    avg_mse[method] = sum(vals) / len(vals)
                else:
                    avg_mse[method] = float('inf')
            
            print(f"方法 {method}: {len(vals)} 个有效值，平均 MSE = {avg_mse[method]}")
        
        # 输出结果
        print("\n各方法平均 MSE:")
        print("-" * 40)
        for method, v in sorted(avg_mse.items()):
            if not np.isnan(v):
                print(f"{method:20s}: {v:.6f}")
            else:
                print(f"{method:20s}: 无有效结果")
        
        # 保存结果
        results_df = pd.DataFrame([
            {'Method': method, 'Average_MSE': v} 
            for method, v in sorted(avg_mse.items())
        ])
        results_df.to_csv('mse_evaluation_results.csv', index=False)
        print(f"\n结果已保存到: mse_evaluation_results.csv")
        
        return avg_mse