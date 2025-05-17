import os
import torch
import random
import numpy as np
import pandas as pd
from models_TCDF import ADDSTCN
import torch.nn.functional as F
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed

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

        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask_np[:, target] == 0)[0]  # 原始缺失才填补
            initial_filled_copy = initial_filled.copy()
            initial_filled[to_fill, target] = out[to_fill]

    metrics = {}
    if evaluate:
        # === 第二步：挖去填补后任意位置，再评估 ===
        mask_temp = np.ones_like(mask_np)  # 不考虑原始缺失，全部设为1
        eval_mask = np.zeros_like(mask_np, dtype=bool)
        rng = np.random.default_rng()
        total_pos = [(i, j) for i in range(seq_len) for j in range(total_features)]
        n_total = len(total_pos)

        # 点缺失
        n_point = max(1, int(n_total * point_ratio))
        point_remove = rng.choice(n_total, n_point, replace=False)
        for idx in point_remove:
            i, j = total_pos[idx]
            mask_temp[i, j] = 0
            eval_mask[i, j] = True

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
                if mask_temp[r, c] == 1:
                    mask_temp[r, c] = 0
                    eval_mask[r, c] = True
            area_removed += len(block)

        # # 再填补一次，评估用
        # re_filled = filled.copy()
        # for target in range(total_features):
        #     inds = list(np.where(causal_matrix[:, target] == 1)[0])
        #     if target not in inds:
        #         inds.append(target)
        #     else:
        #         inds.remove(target)
        #         inds.append(target)
        #     inds = inds[:3] + [target]

        #     inp = filled[:, inds].T[np.newaxis, ...]
        #     y_np = filled[:, target][np.newaxis, :, None]
        #     m_np = (mask_temp[:, target] == 1)[np.newaxis, :, None]

        #     x = torch.tensor(inp, dtype=torch.float32).to(device)
        #     y = torch.tensor(y_np, dtype=torch.float32).to(device)
        #     m = torch.tensor(m_np, dtype=torch.float32).to(device)

        #     model = ADDSTCN(target, input_size=len(inds),
        #                     cuda=(device == 'cuda:0'),
        #                     **model_params).to(device)
        #     optim = torch.optim.Adam(model.parameters(), lr=lr)

        #     for epoch in range(epochs):
        #         model.train()
        #         pred = model(x)
        #         loss = F.mse_loss(pred * m, y * m)
        #         optim.zero_grad()
        #         loss.backward()
        #         optim.step()

        #     model.eval()
        #     with torch.no_grad():
        #         out = model(x).squeeze().cpu().numpy()
        #         to_fill = np.where(mask_temp[:, target] == 0)[0]
        #         re_filled[to_fill, target] = out[to_fill]

        # === 统一评估 ===
        true = initial_filled_copy[eval_mask]
        metrics['model'] = ((initial_filled[eval_mask] - true) ** 2).mean()

        z = initial_filled_copy.copy()
        z[mask_temp != 1] = 0
        metrics['zero'] = ((z[eval_mask] - true) ** 2).mean()

        med = np.nanmedian(np.where(mask_temp == 1, initial_filled_copy, np.nan), axis=0)
        mdf = initial_filled_copy.copy()
        for j in range(total_features):
            mdf[mask_temp[:, j] != 1, j] = med[j]
        metrics['median'] = ((mdf[eval_mask] - true) ** 2).mean()

        mn = np.nanmean(np.where(mask_temp == 1, initial_filled_copy, np.nan), axis=0)
        mnf = initial_filled_copy.copy()
        for j in range(total_features):
            mnf[mask_temp[:, j] != 1, j] = mn[j]
        metrics['mean'] = ((mnf[eval_mask] - true) ** 2).mean()

        df = pd.DataFrame(initial_filled_copy.copy())
        df_mask = pd.DataFrame(mask_temp.copy())
        df[df_mask != 1] = np.nan
        bfill = df.bfill().ffill().values
        ffill = df.ffill().bfill().values
        metrics['bfill'] = ((bfill[eval_mask] - true) ** 2).mean()
        metrics['ffill'] = ((ffill[eval_mask] - true) ** 2).mean()

        knn = KNNImputer()
        knnf = knn.fit_transform(np.where(mask_temp == 1, initial_filled_copy, np.nan))
        metrics['knn'] = ((knnf[eval_mask] - true) ** 2).mean()

        mice = IterativeImputer()
        micef = mice.fit_transform(np.where(mask_temp == 1, initial_filled_copy, np.nan))
        metrics['mice'] = ((micef[eval_mask] - true) ** 2).mean()

    return idx, initial_filled_copy, metrics


def train_all_features_parallel(dataset, model_params, epochs=10, lr=0.001, evaluate=False,
                               point_ratio=0.1, block_ratio=0.2,
                               block_min_w=1, block_max_w=5,
                               block_min_h=1, block_max_h=5):
    # 1. 规定使用gpu的个数
    available_gpus = min(torch.cuda.device_count() - 1, 3)  
    
    # 2. 限制最大进程数
    max_workers = min(available_gpus * 2, 8)  # 每GPU最多2个进程，总共不超过8个
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

    return eval_results if evaluate else None