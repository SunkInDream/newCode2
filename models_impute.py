import os
import torch
import torch.nn.functional as F
from models_TCDF import ADDSTCN
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def process_single_matrix(args):
    import random
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
    eval_mask = None

    if evaluate:
        rng = np.random.default_rng()
        ones = np.argwhere(mask_np == 1)
        n_total = len(ones)
        col_counts = mask_np.sum(axis=0)
        mask_temp = mask_np.copy()

        # === 点缺失 ===
        n_point = max(1, int(n_total * point_ratio))
        point_remove = ones[rng.choice(n_total, n_point, replace=False)]

        valid_remove = []
        for i, j in point_remove:
            if col_counts[j] > 1:
                valid_remove.append((i, j))
                col_counts[j] -= 1
        for i, j in valid_remove:
            mask_temp[i, j] = 2

        # === 块缺失 ===
        area_removed = 0
        target_area = int(n_total * block_ratio)
        col_counts = mask_temp.sum(axis=0)  # 每列有效值计数

        while area_removed < target_area:
            h = random.randint(block_min_h, block_max_h)
            w = random.randint(block_min_w, block_max_w)
            i = random.randint(0, seq_len - h)
            j = random.randint(0, total_features - w)

            block = []
            for r in range(i, i + h):
                for c in range(j, j + w):
                    if mask_temp[r, c] == 1 and col_counts[c] > 1:
                        block.append((r, c))

            for r, c in block:
                mask_temp[r, c] = 2
                col_counts[c] -= 1

            area_removed += len(block)

    # === 训练并填补 ===
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
            if epoch % 5 == 0:
                print(f'matrix {idx}, feat {target}/{total_features}, epoch {epoch}, loss {loss.item():.6f}')

        model.eval()
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
            to_fill = np.where(mask_np[:, target] != 1)[0]
            filled[to_fill, target] = out[to_fill]

    metrics = {}
    if evaluate:
        true = x_np[eval_mask]
        pred = filled[eval_mask]
        metrics['model'] = ((pred - true) ** 2).mean()

        z = x_np.copy()
        z[mask_np != 1] = 0
        metrics['zero'] = ((z[eval_mask] - true) ** 2).mean()

        med = np.nanmedian(np.where(mask_np == 1, x_np, np.nan), axis=0)
        mdf = x_np.copy()
        for j in range(total_features):
            mdf[mask_np[:, j] != 1, j] = med[j]
        metrics['median'] = ((mdf[eval_mask] - true) ** 2).mean()

        mn = np.nanmean(np.where(mask_np == 1, x_np, np.nan), axis=0)
        mnf = x_np.copy()
        for j in range(total_features):
            mnf[mask_np[:, j] != 1, j] = mn[j]
        metrics['mean'] = ((mnf[eval_mask] - true) ** 2).mean()

        df = pd.DataFrame(x_np.copy())
        df_mask = pd.DataFrame(mask_np.copy())
        df[df_mask != 1] = np.nan
        bfill = df.bfill().ffill().values
        ffill = df.ffill().bfill().values
        metrics['bfill'] = ((bfill[eval_mask] - true) ** 2).mean()
        metrics['ffill'] = ((ffill[eval_mask] - true) ** 2).mean()

        knn = KNNImputer()
        knnf = knn.fit_transform(np.where(mask_np == 1, x_np, np.nan))
        metrics['knn'] = ((knnf[eval_mask] - true) ** 2).mean()

        mice = IterativeImputer()
        micef = mice.fit_transform(np.where(mask_np == 1, x_np, np.nan))
        metrics['mice'] = ((micef[eval_mask] - true) ** 2).mean()

    return idx, filled, metrics

def train_all_features_parallel(dataset, model_params, epochs=300, lr=0.01, evaluate=False,
                                point_ratio=0.1, block_ratio=0.1,
                                block_min_w=2, block_max_w=5,
                                block_min_h=2, block_max_h=5):
    assert dataset.total_causal_matrix is not None
    cm = dataset.total_causal_matrix

    gpus = list(range(torch.cuda.device_count())) or ['cpu']
    print(f'Available GPUs: {gpus}')

    tasks = [
        (i,
         dataset.initial_filled[i],
         dataset.mask_data[i],
         cm,
         model_params,
         epochs,
         lr,
         gpus[i % len(gpus)],
         evaluate,
         point_ratio,
         block_ratio,
         block_min_w,
         block_max_w,
         block_min_h,
         block_max_h)
        for i in range(len(dataset.initial_filled))
    ]

    results = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for idx, filled_mat, metrics in executor.map(process_single_matrix, tasks):
            dataset.final_filled[idx] = filled_mat
            if evaluate:
                metrics['idx'] = idx
                results.append(metrics)
            print(f'Matrix {idx} done')

    if evaluate:
        df = pd.DataFrame(results).sort_values('idx').set_index('idx')
        
        # 添加以下几行代码，禁用科学计数法显示
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print('\nEvaluation MSE comparison:')
        print(df)
