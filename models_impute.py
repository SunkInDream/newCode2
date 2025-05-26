import os
import torch
import random
import numpy as np
import pandas as pd
from models_TCDF import ADDSTCN
import torch.nn.functional as F
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
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
def impute(original, causal_matrix, model_params, epochs=100, lr=0.01, gpu_id=None):
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    mask = (~np.isnan(original)).astype(int)
    initial_filled = initial_process(original)
    sequence_len, total_features = initial_filled.shape
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
        final_filled = initial_filled.copy()
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
            to_fill = np.where(mask[:, target] == 0)[0]  # 原始缺失才填补
            final_filled[to_fill, target] = out[to_fill]
    return final_filled
def agregate(initial_filled, n_cluster):
        data = np.array([np.nanmean(x, axis=0) for x in initial_filled])
        km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
        labels = km.fit_predict(data)
        idx_arr = []    
        for k in range(n_cluster):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0: 
                continue
            cluster_data = data[idxs]
            dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
            best_idx = idxs[np.argmin(dists)]
            idx_arr.append(int(best_idx))
        return idx_arr
def causal_discovery(original_matrix_arr, n_cluster=5, isStandard=False, standard_cg=None, params = {
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
    else:
        initial_matrix_arr = original_matrix_arr.copy()
        for i in range(len(initial_matrix_arr)):
            initial_matrix_arr[i] = initial_process(initial_matrix_arr[i])
        idx_arr = agregate(initial_matrix_arr, n_cluster)
        cg_total = None
        columns = None
        for idx, i in enumerate(idx_arr):
            matrix = compute_causal_matrix(initial_matrix_arr[i], params=params, gpu_id=0)            # 第一次迭代时初始化cg_total
            if cg_total is None:
                cg_total = matrix.copy()
            else:
                cg_total += matrix
        np.fill_diagonal(cg_total, 0)
        new_matrix = np.zeros_like(cg_total)
        for col in range(cg_total.shape[1]):
            temp_col = cg_total[:, col].copy()
            if np.count_nonzero(temp_col) < 3:
                new_matrix[:, col] = 1
            else:
                top3 = np.argsort(temp_col)[-3:]
                new_matrix[top3, col] = 1
        top3_matrix = new_matrix
        return top3_matrix
def mse_evaluate(mx, causal_matrix):
    # (旧版代码省略)
    # mx: 原始 2D 数组，shape=(T, features)
    ground_truth = mx.copy()

    # 1) 模拟缺失：block_missing 要求 3D 输入
    X_block_3d = block_missing(
        np.expand_dims(mx, axis=0),  # (1, T, F)
        factor=0.1, block_width=3, block_len=3
    )
    # 2) 去掉 batch 维度，回到 2D
    X_block = X_block_3d[0]         # (T, F) 带 NaN

    # 3) 模型插补
    imputed = impute(
        X_block, causal_matrix,
        model_params={'num_levels': 6, 'kernel_size': 6, 'dilation_c': 4},
        epochs=100, lr=0.01, gpu_id=None
    )

    # 4) 各基线方法插补
    zero_res   = zero_impu(X_block)
    mean_res   = mean_impu(X_block)
    median_res = median_impu(X_block)
    mode_res   = mode_impu(X_block)
    random_res = random_impu(X_block)
    knn_res    = knn_impu(X_block)
    ffill_res  = ffill_impu(X_block)
    bfill_res  = bfill_impu(X_block)

    # 5) 定义 MSE 计算，全部在 2D 上
    def mse(a, b):
        return np.mean((a - b) ** 2)

    return {
        'my_model':    mse(imputed,      ground_truth),
        'zero_impu':   mse(zero_res,     ground_truth),
        'mean_impu':   mse(mean_res,     ground_truth),
        'median_impu': mse(median_res,   ground_truth),
        'mode_impu':   mse(mode_res,     ground_truth),
        'random_impu': mse(random_res,   ground_truth),
        'knn_impu':    mse(knn_res,      ground_truth),
        'ffill_impu':  mse(ffill_res,    ground_truth),
        'bfill_impu':  mse(bfill_res,    ground_truth),
    }
    
import os, torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from models_impute import impute

def _impute_worker(mx, causal_matrix, model_params, gpu_id):
    # 每个子进程里只看到一张卡 cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    return impute(mx, causal_matrix=causal_matrix, model_params=model_params)

def parallel_impute(data_list, causal_matrix, model_params):
    """把 data_list 中每个矩阵分配给所有 GPU 并行 impute；返回与 data_list 等长的结果列表"""
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        # no GPU，退化到串行
        return [impute(mx, causal_matrix=causal_matrix, model_params=model_params) for mx in data_list]

    ctx = get_context("spawn")
    results = [None] * len(data_list)
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as exe:
        futures = {
            exe.submit(_impute_worker, mx, causal_matrix, model_params, idx % n_gpus): idx
            for idx, mx in enumerate(data_list)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results

def process_single_matrix(args):
    idx, x_np, mask_np, causal_matrix, model_params, epochs, lr, gpu_id, evaluate, \
        point_ratio, block_ratio, block_min_w, block_max_w, block_min_h, block_max_h = args

    if gpu_id != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Processing matrix {idx} on gpu {gpu_id}...')

    seq_len, total_features = x_np.shape
    initial_filled = x_np.copy()
    model_cache = {}
    # === 填补原始缺失位置 ===
    for target in range(total_features):
        inds = list(np.where(causal_matrix[:, target] == 1)[0])
        if target not in inds:
            inds.append(target)
        else:
            inds.remove(target)
            inds.append(target)
        inds = inds[:3] + [target]

        inp = x_np[:, inds].T[np.newaxis, ...]
        y_np = x_np[:, target][np.newaxis, :, None]
        m_np = (mask_np[:, target] == 1)[np.newaxis, :, None]

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
                # 添加额外的L1正则化以提高对局部模式的学习能力
            loss = F.mse_loss(pred * m, y * m) + 0.001 * sum(p.abs().sum() for p in model.parameters())
            optim.zero_grad()
            loss.backward()
                # 梯度裁剪防止过拟合
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
            model_cache[target] = best_state

        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask_np[:, target] == 0)[0]  # 原始缺失才填补
            initial_filled_copy = initial_filled.copy()
            initial_filled[to_fill, target] = out[to_fill]
            ground_truth = initial_filled.copy()
          

    metrics = {}
    if evaluate:
        # === 第二步：挖去填补后任意位置，再评估 ===
        eval_mask = np.ones_like(mask_np, dtype=bool)
        rng = np.random.default_rng()
        total_pos = [(i, j) for i in range(seq_len) for j in range(total_features)]
        n_total = len(total_pos)

        # 点缺失
        n_point = int(n_total * point_ratio)
        point_remove = rng.choice(n_total, n_point, replace=False)
        for idx in point_remove:
            i, j = total_pos[idx]
            eval_mask[i, j] = 0

        # 块缺失
        area_removed = 0
        target_area = int(n_total * block_ratio)
        while area_removed < target_area:
            h = random.randint(block_min_h, block_max_h)
            w = random.randint(block_min_w, block_max_w)
            i = random.randint(0, seq_len - h)
            j = random.randint(0, total_features - w)

            block = [(r, c) for r in range(i, i + h) for c in range(j, j + w)]
            for r, c in block:
                eval_mask[r, c] = 0
            area_removed += len(block)

        # === 统一评估 ===
        reimpu = initial_filled.copy()
        for target in range(total_features):
            inds = list(np.where(causal_matrix[:, target] == 1)[0])
            if target not in inds:
                inds.append(target)
            else:
                inds.remove(target)
                inds.append(target)
            inds = inds[:3] + [target]

            inp = reimpu[:, inds].T[np.newaxis, ...]
            x = torch.tensor(inp, dtype=torch.float32).to(device)

            model = ADDSTCN(target, input_size=len(inds),
                            cuda=(device == 'cuda:0'),
                            **model_params).to(device)
            model.load_state_dict(model_cache[target])  # ✅ 从内存加载权重
            model.eval()

            with torch.no_grad():
                out = model(x).squeeze().cpu().numpy()
                to_fill = np.where(eval_mask[:, target] == 0)[0]
                reimpu[to_fill, target] = out[to_fill]
        final_mask = (mask_np==1) & (eval_mask==0) 
        ground_truth = ground_truth[final_mask]
        metrics['model'] = ((reimpu[final_mask] - ground_truth) ** 2).mean()

        z = initial_filled_copy.copy()
        z[eval_mask==0] = 0
        metrics['zero'] = ((z[final_mask] - ground_truth) ** 2).mean()

        med = np.nanmedian(np.where(eval_mask == 1, initial_filled_copy, np.nan), axis=0)
        mdf = initial_filled_copy.copy()
        for j in range(total_features):
            mdf[eval_mask[:, j] == 0, j] = med[j]
        metrics['median'] = ((mdf[final_mask] - ground_truth) ** 2).mean()

        mn = np.nanmean(np.where(eval_mask == 1, initial_filled_copy, np.nan), axis=0)
        mnf = initial_filled_copy.copy()
        for j in range(total_features):
            mnf[eval_mask[:, j] == 0, j] = mn[j]
        metrics['mean'] = ((mnf[final_mask] - ground_truth) ** 2).mean()

        df = pd.DataFrame(initial_filled_copy.copy())
        df_mask = pd.DataFrame(eval_mask.copy())
        df[df_mask == 0] = np.nan
        bfill = df.bfill().ffill().values
        ffill = df.ffill().bfill().values
        metrics['bfill'] = ((bfill[final_mask] - ground_truth) ** 2).mean()
        metrics['ffill'] = ((ffill[final_mask] - ground_truth) ** 2).mean()

        knn = KNNImputer()
        knnf = knn.fit_transform(np.where(eval_mask == 1, initial_filled_copy, np.nan))
        metrics['knn'] = ((knnf[final_mask] - ground_truth) ** 2).mean()

        mice = IterativeImputer()
        micef = mice.fit_transform(np.where(eval_mask == 1, initial_filled_copy, np.nan))
        metrics['mice'] = ((micef[final_mask] - ground_truth) ** 2).mean()
        
        # from pypots.imputation import DLinear, TimeLLM, MOMENT
        # data_for_pypots = initial_filled_copy.copy()
        # data_for_pypots[eval_mask] = np.nan
        # data_for_pypots = data_for_pypots[np.newaxis, ...]
        # train_set = {"X": data_for_pypots}
        # model = DLinear(
        #         n_steps=data_for_pypots.shape[1], 
        #         n_features=data_for_pypots.shape[2], 
        #         moving_avg_window_size=5,
        #         d_model=32,
        #     )
                    
        # model.fit(train_set)
        # imputed_data = model.impute(train_set)
        # imputed_data = imputed_data.squeeze(0)
        # metrics['DLine'] = ((imputed_data[eval_mask] - ground_truth) ** 2).mean()
        
        # data_for_pypots = initial_filled_copy.copy()
        # data_for_pypots[eval_mask] = np.nan
        # data_for_pypots = data_for_pypots[np.newaxis, ...]
        # train_set = {"X": data_for_pypots}
        # model = MOMENT(
        #     n_steps=data_for_pypots.shape[1], 
        #     n_features=data_for_pypots.shape[2],
        #     transformer_backbone="t5-small",
        #     transformer_type="encoder_only",
        #     head_dropout=0.1,
        #     finetuning_mode="linear-probing",
        #     revin_affine=False,
        #     add_positional_embedding=True,
        #     value_embedding_bias=False,
        #     orth_gain=0.1,
        #     patch_size=4,
        #     patch_stride=4,
        #     d_ffn=4,
        #     d_model=4,
        #     n_layers=2, 
        #     dropout=0.1,
        # )
                    
        # model.fit(train_set)
        # imputed_data = model.impute(train_set)
        # imputed_data = imputed_data.squeeze(0)
        # metrics['TimeLLM'] = ((imputed_data[eval_mask] - true) ** 2).mean()
    return idx, initial_filled_copy, metrics


def train_all_features_parallel(dataset, model_params, epochs=10, lr=0.001, evaluate=False,
                               point_ratio=0.1, block_ratio=0.2,
                               block_min_w=1, block_max_w=5,
                               block_min_h=1, block_max_h=5):
    # 1. 规定使用gpu的个数
    available_gpus = max(min(torch.cuda.device_count() - 1, 3),0) 
    
    # 2. 限制最大进程数
    max_workers = max(min(available_gpus * 2, 8),1)  # 每GPU最多2个进程，总共不超过8个
    print(f"使用 {available_gpus} 张GPU，最大 {max_workers} 个进程")
    
    # 3. 准备任务列表
    tasks = []
    for i, mat in enumerate(dataset.initial_filled):
        if mat is not None:
            gpu_id = i % available_gpus if available_gpus > 0 else 'cpu'
            tasks.append((i, mat, dataset.mask_data[i], dataset.total_causal_matrix, 
                      model_params, epochs, lr, gpu_id, evaluate,
                      point_ratio, block_ratio, block_min_w, block_max_w, block_min_h, block_max_h))
    
    # 4. 分批处理任务，避免一次启动太多进程
    batch_size = 10  # 每批处理10个任务
    all_tasks = tasks.copy()
    eval_results = []
    
    # 确保final_filled和initial_filled长度一致
    while len(dataset.final_filled) < len(dataset.initial_filled):
        dataset.final_filled.append(None)
    
    # 5. 批处理任务
    for batch_start in range(0, len(all_tasks), batch_size):
        batch_tasks = all_tasks[batch_start:batch_start + batch_size]
        print(f"处理批次 {batch_start//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size}，包含 {len(batch_tasks)} 个任务")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for task in batch_tasks:
                # 记录任务和索引的对应关系
                future = executor.submit(process_single_matrix, task)
                future_to_idx[future] = task[0]  # 保存索引
                
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]  # 这是我们提交任务时的原始索引
                # 获取处理结果 - 忽略返回的索引，直接使用原始索引
                _, filled_mat, metrics = future.result()
                    
                # 直接更新到dataset中的对应位置
                if filled_mat is not None:
                    dataset.final_filled[idx] = filled_mat
                    print(f"任务 {idx} 完成, 直接更新到final_filled[{idx}]")
                    #pd.DataFrame(dataset.final_filled[idx]).to_csv(f'./finalIdInitial_{idx}.csv', index=False)
                    
                # 保存评估结果
                if evaluate and metrics:
                    eval_results.append({**metrics, 'idx': idx})  # 使用原始索引

    # 计算成功更新数量
    updated_count = sum(1 for mat in dataset.final_filled if mat is not None)
    print(f"成功更新 {updated_count}/{len(dataset.initial_filled)} 个矩阵")
    
    # 7. 处理评估结果（打印平均结果）
    if evaluate and eval_results:
        df = pd.DataFrame(eval_results)
        mean_metrics = df.drop(columns=['idx']).mean()
        pd.set_option('display.float_format', '{:.6f}'.format)
        print('\nAverage Evaluation MSE across all matrices:')
        print(mean_metrics)
        # 添加下面几行代码，将结果保存到文件
        results_dir = 'evaluation_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'imputation_results.csv')
        df.to_csv(results_file, index=False)
        
        # 保存平均指标到单独的文件
        summary_file = os.path.join(results_dir, 'imputation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write('Average Evaluation MSE across all matrices:\n')
            f.write(mean_metrics.to_string())
        
        print(f'\n结果已保存到 {results_file} 和 {summary_file}')
    return eval_results if evaluate else None
