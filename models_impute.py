import os
import torch
import torch.nn.functional as F
from models_TCDF import ADDSTCN
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import random
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import torch
import torch.nn.functional as F
def process_single_matrix(args):
    idx, x_np, mask_np, causal_matrix, model_params, epochs, lr, gpu_id, evaluate, \
        point_ratio, block_ratio, block_min_w, block_max_w, block_min_h, block_max_h = args

    if gpu_id != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Processing matrix {idx} on {device}')

    seq_len, total_features = x_np.shape
    filled = x_np.copy()

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
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            pred = model(x)
            loss = F.mse_loss(pred * m, y * m)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask_np[:, target] == 0)[0]  # 原始缺失才填补
            filled[to_fill, target] = out[to_fill]

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

        # 再填补一次，评估用
        re_filled = filled.copy()
        for target in range(total_features):
            inds = list(np.where(causal_matrix[:, target] == 1)[0])
            if target not in inds:
                inds.append(target)
            else:
                inds.remove(target)
                inds.append(target)
            inds = inds[:3] + [target]

            inp = filled[:, inds].T[np.newaxis, ...]
            y_np = filled[:, target][np.newaxis, :, None]
            m_np = (mask_temp[:, target] == 1)[np.newaxis, :, None]

            x = torch.tensor(inp, dtype=torch.float32).to(device)
            y = torch.tensor(y_np, dtype=torch.float32).to(device)
            m = torch.tensor(m_np, dtype=torch.float32).to(device)

            model = ADDSTCN(target, input_size=len(inds),
                            cuda=(device == 'cuda:0'),
                            **model_params).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                pred = model(x)
                loss = F.mse_loss(pred * m, y * m)
                optim.zero_grad()
                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                out = model(x).squeeze().cpu().numpy()
                to_fill = np.where(mask_temp[:, target] == 0)[0]
                re_filled[to_fill, target] = out[to_fill]

        # === 统一评估 ===
        true = filled[eval_mask]
        metrics['model'] = ((re_filled[eval_mask] - true) ** 2).mean()

        z = filled.copy()
        z[mask_temp != 1] = 0
        metrics['zero'] = ((z[eval_mask] - true) ** 2).mean()

        med = np.nanmedian(np.where(mask_temp == 1, filled, np.nan), axis=0)
        mdf = filled.copy()
        for j in range(total_features):
            mdf[mask_temp[:, j] != 1, j] = med[j]
        metrics['median'] = ((mdf[eval_mask] - true) ** 2).mean()

        mn = np.nanmean(np.where(mask_temp == 1, filled, np.nan), axis=0)
        mnf = filled.copy()
        for j in range(total_features):
            mnf[mask_temp[:, j] != 1, j] = mn[j]
        metrics['mean'] = ((mnf[eval_mask] - true) ** 2).mean()

        df = pd.DataFrame(filled.copy())
        df_mask = pd.DataFrame(mask_temp.copy())
        df[df_mask != 1] = np.nan
        bfill = df.bfill().ffill().values
        ffill = df.ffill().bfill().values
        metrics['bfill'] = ((bfill[eval_mask] - true) ** 2).mean()
        metrics['ffill'] = ((ffill[eval_mask] - true) ** 2).mean()

        knn = KNNImputer()
        knnf = knn.fit_transform(np.where(mask_temp == 1, filled, np.nan))
        metrics['knn'] = ((knnf[eval_mask] - true) ** 2).mean()

        mice = IterativeImputer()
        micef = mice.fit_transform(np.where(mask_temp == 1, filled, np.nan))
        metrics['mice'] = ((micef[eval_mask] - true) ** 2).mean()

    return idx, filled, metrics


def train_all_features_parallel(dataset, model_params, epochs=10, lr=0.001, evaluate=False,
                              point_ratio=0.1, block_ratio=0.2,
                              block_min_w=1, block_max_w=5,
                              block_min_h=1, block_max_h=5):

    print(f"{len(dataset.initial_filled)} {len(dataset.final_filled)}")
    
    # 1. 准备任务列表
    tasks = []
    for i, mat in enumerate(dataset.initial_filled):
        if mat is not None:
            tasks.append((i, mat, dataset.mask_data[i], dataset.total_causal_matrix, 
                       model_params, epochs, lr, i % torch.cuda.device_count(), evaluate,
                       point_ratio, block_ratio, block_min_w, block_max_w, block_min_h, block_max_h))
    
    # 2. 创建结果存储（不再直接使用dataset.final_filled）
    result_matrices = [None] * len(dataset.initial_filled)
    eval_results = []
    
    # 3. 使用进程池，但不直接更新dataset
    with ProcessPoolExecutor() as executor:
        future_to_idx = {}
        for task in tasks:
            # 记录任务和索引的对应关系
            future = executor.submit(process_single_matrix, task)
            future_to_idx[future] = task[0]  # 保存索引
            
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]  # 这是我们提交任务时的原始索引
            try:
                # 获取处理结果
                ret_idx, filled_mat, metrics = future.result()
                
                # 检查索引匹配
                if ret_idx != idx:
                    print(f"警告: 返回索引 {ret_idx} 与期望索引 {idx} 不匹配，使用原始索引")
                
                # 存储到我们的临时列表
                if 0 <= idx < len(result_matrices):
                    result_matrices[idx] = filled_mat
                else:
                    print(f"错误: 索引 {idx} 超出范围 0-{len(result_matrices)-1}")
                
                # 保存评估结果
                if evaluate and metrics:
                    eval_results.append({**metrics, 'idx': idx})
            except Exception as e:
                print(f"处理任务 {idx} 出错: {str(e)}")
    
    # 4. 更新处理完成后再一次性更新dataset
    for i, mat in enumerate(result_matrices):
        if mat is not None and i < len(dataset.final_filled):
            dataset.final_filled[i] = mat
        elif mat is not None:
            # 需要扩展 final_filled 列表
            dataset.final_filled.extend([None] * (i - len(dataset.final_filled) + 1))
            dataset.final_filled[i] = mat
    
    # 5. 处理评估结果
    # 5. 处理评估结果（打印平均结果）
    if evaluate and eval_results:
        df = pd.DataFrame(eval_results)
        mean_metrics = df.drop(columns=['idx']).mean()
        pd.set_option('display.float_format', '{:.6f}'.format)
        print('\nAverage Evaluation MSE across all matrices:')
        print(mean_metrics)

    
    return eval_results if evaluate else None