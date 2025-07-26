import numpy as np
import pandas as pd
from scipy import stats
import os
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn.impute import KNNImputer
# from miracle import *
# from pypots.imputation import SAITS,TimeMixerPP,TimeLLM,MOMENT,TEFN
from typing import Optional
# from pypots.optim.adam import Adam
# from pypots.nn.modules.loss import MAE, MSE
import torch
from torch.utils.data import Dataset, DataLoader
from models_impute import *

def zero_impu(mx):
   return np.nan_to_num(mx, nan=0)
# 在baseline.py中添加调试信息
# def zero_impu(mx):
#     print(f"🔍 zero_impu 输入: 缺失值数量 = {np.isnan(mx).sum()}")
#     result = mx.copy()
    
#     # 检查是否调用了预处理
#     if hasattr(zero_impu, '_debug'):
#         print("zero_impu: 直接用0填充")
#         result[np.isnan(result)] = 0.0
#     else:
#         # 可能这里调用了其他处理函数
#         result = FirstProcess(result)  # ← 这里可能是问题所在
#         result = SecondProcess(result)
    
#     print(f"🔍 zero_impu 输出: 缺失值数量 = {np.isnan(result).sum()}")
#     print(f"🔍 zero_impu 输出: 零值数量 = {(result == 0).sum()}")
#     return result
def mean_impu(mx):
    # mx = mx.copy()
    # col_means = np.nanmean(mx, axis=0)
    # inds = np.where(np.isnan(mx))
    # mx[inds] = np.take(col_means, inds[1])
    # if np.isnan(mx).any():
    #     mx = np.nan_to_num(mx, nan=-1)
    # return mx
    mean = np.nanmean(mx)
    return np.where(np.isnan(mx), mean, mx)

def median_impu(mx):
    mx = mx.copy()
    # col_medians = np.nanmedian(mx, axis=0)
    # inds = np.where(np.isnan(mx))
    # mx[inds] = np.take(col_medians, inds[1])
    # if np.isnan(mx).any():
    #     mx = np.nan_to_num(mx, nan=-1)
    # return mx
    median = np.nanmedian(mx)
    return np.where(np.isnan(mx), median, mx)

def mode_impu(mx):
    mx = mx.copy()
    flat_values = mx[~np.isnan(mx)]  # 展平所有非NaN值
    global_mode = stats.mode(flat_values, keepdims=False).mode
    if np.isnan(global_mode):
        global_mode = 0  # 兜底
    inds = np.where(np.isnan(mx))
    mx[inds] = global_mode
    return mx

def random_impu(mx):
    mx = mx.copy()
    non_nan_values = mx[~np.isnan(mx)]  # 获取所有非缺失值（1D数组）
    
    if non_nan_values.size == 0:
        # 整张表全是 NaN，兜底填 -1
        mx[:] = -1
        return mx

    inds = np.where(np.isnan(mx))  # 找到所有 NaN 的位置
    mx[inds] = np.random.choice(non_nan_values, size=len(inds[0]), replace=True)
    return mx


# def knn_impu(mx, k=5):
#     mx = mx.copy()
#     all_nan_cols = np.all(np.isnan(mx), axis=0)

#     # 计算全局均值（不为 NaN）
#     global_mean = np.nanmean(mx)

#     # 全空列先填全局均值，避免 KNNImputer 报错
#     mx[:, all_nan_cols] = global_mean

#     imputer = KNNImputer(n_neighbors=k)
#     return imputer.fit_transform(mx)

def knn_impu(mx, k=5):
    import time
    start_time = time.time()
    
    print(f"🔍 开始KNN填补: 数据形状={mx.shape}, 缺失值={np.isnan(mx).sum()}")
    
    mx = mx.copy()
    
    # ✅ 1. 设置单线程
    import os
    print("⚙️ 设置单线程模式...")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # ✅ 2. 处理全空列
    print("🔧 检查全空列...")
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"   发现 {all_nan_cols.sum()} 个全空列，用全局均值填充")
        global_mean = np.nanmean(mx)
        if np.isnan(global_mean):
            global_mean = 0.0
        mx[:, all_nan_cols] = global_mean
    else:
        print("   无全空列")
    
    # ✅ 3. 调整k值
    print("📊 调整KNN参数...")
    valid_samples = (~np.isnan(mx)).sum(axis=0).min()
    original_k = k
    k = min(k, max(1, valid_samples - 1))
    print(f"   k值: {original_k} -> {k}")
    
    # ✅ 4. 开始KNN填补
    print("🚀 开始KNN计算...")
    try:
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=k)
        
        print("   创建KNNImputer完成")
        print("   开始fit_transform...")
        
        result = imputer.fit_transform(mx)
        
        elapsed = time.time() - start_time
        print(f"✅ KNN填补完成，耗时 {elapsed:.2f} 秒")
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ KNN填补在 {elapsed:.2f} 秒后失败: {e}")
        raise e

def mice_impu(mx, max_iter=5):
    """改进：处理全空列 + 最简版MICE填补"""
    mx = mx.copy()
    n_rows, n_cols = mx.shape

    # === Step 0: 处理全空列 ===
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        global_mean = np.nanmean(mx)
        mx[:, all_nan_cols] = global_mean

    # === Step 1: 初始均值填补 ===
    imp = SimpleImputer(strategy='mean')
    matrix_filled = imp.fit_transform(mx)

    # === Step 2: MICE主循环 ===
    for _ in range(max_iter): 
        for col in range(n_cols): 
            missing_idx = np.where(np.isnan(mx[:, col]))[0]
            if len(missing_idx) == 0:
                continue

            observed_idx = np.where(~np.isnan(mx[:, col]))[0]
            X_train = np.delete(matrix_filled[observed_idx], col, axis=1)
            y_train = mx[observed_idx, col]
            X_pred = np.delete(matrix_filled[missing_idx], col, axis=1)

            model = BayesianRidge()
            model.fit(X_train, y_train)
            matrix_filled[missing_idx, col] = model.predict(X_pred)

    return matrix_filled

def ffill_impu(mx):
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.ffill(axis=0)

    # 补全前几行未被填补的位置（例如第一行是 NaN）
    global_mean = np.nanmean(mx)
    df = df.fillna(global_mean)

    return df.values

def bfill_impu(mx):
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.bfill(axis=0)

    # 补全最后几行未被填补的位置（例如最后一行是 NaN）
    global_mean = np.nanmean(mx)
    df = df.fillna(global_mean)

    return df.values

def miracle_impu(mx: np.ndarray) -> np.ndarray:
    from miracle import MIRACLE
    mx = mx.copy().astype(np.float32)
    global_mean = np.nanmean(mx)
    if np.isnan(global_mean):
        global_mean = 0.0
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean

    n_feats = mx.shape[1]
    missing_idx = np.where(np.any(np.isnan(mx), axis=0))[0][:20]

    model = MIRACLE(
        num_inputs=n_feats,
        missing_list=missing_idx.tolist(),
        n_hidden=min(32, max(8, n_feats // 2)),
        lr=0.008,
        max_steps=50,
        window=5,
        seed=42
    )

    result = model.fit(mx)

    del model
    gc.collect()
    return result.astype(np.float32)


# ✅ 同时优化其他方法，提高整体baseline质量
def saits_impu(mx, epochs=None, d_model=None, n_layers=None, device=None):
    """动态参数的SAITS填补"""
    from pypots.imputation import SAITS
    
    mx = mx.copy()
    seq_len, n_features = mx.shape
    total_size = seq_len * n_features
    
    # 处理全空列
    global_mean = np.nanmean(mx)
    if np.isnan(global_mean):
        global_mean = 0.0
    
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    
    # ✅ 根据数据大小动态调整参数
    if epochs is None:
        if total_size > 50000:
            epochs = 20
            d_model = 64
            n_layers = 1
        elif total_size > 10000:
            epochs = 50
            d_model = 128
            n_layers = 2
        else:
            epochs = 100
            d_model = 128
            n_layers = 2
    
    if d_model is None:
        d_model = min(128, max(32, n_features * 4))
    
    if n_layers is None:
        n_layers = 2 if total_size < 20000 else 1
    
    try:
        data_3d = mx[np.newaxis, :, :]
        
        saits = SAITS(
            n_steps=seq_len,
            n_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=min(4, d_model // 32),
            d_k=d_model // 8,
            d_v=d_model // 8,
            d_ffn=d_model,
            dropout=0.1,
            epochs=epochs,
            patience=10,
            batch_size=32,
            device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        )
        
        train_set = {"X": data_3d}
        saits.fit(train_set)
        imputed_data_3d = saits.impute(train_set)
        
        return imputed_data_3d[0]
        
    except Exception as e:
        print(f"SAITS失败: {e}")
        return mean_impu(mx)


def timemixerpp_impu(mx):
    import numpy as np
    import torch
    from pypots.imputation import TimeMixerPP
    from sklearn.impute import SimpleImputer

    # Step 1: 准备输入数据 (T, N) → (1, T, N)
    mx = mx.astype(np.float32)
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用填充")
        mx[:, all_nan_cols] = global_mean
    T, N = mx.shape
    data = mx[None, ...]  # (1, T, N)

    # Step 2: 构建 mask
    missing_mask = np.isnan(data).astype(np.float32)
    indicating_mask = (~np.isnan(data)).astype(np.float32)

    # Step 3: 简单均值填补初始缺失值
    imp = SimpleImputer(strategy='mean', keep_empty_features=True)
    X_filled = imp.fit_transform(mx).astype(np.float32)
    X_filled = X_filled[None, ...]

    # Step 4: 构造数据字典
    dataset = {
        "X": X_filled,
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask,
        "X_ori": data
    }

    # Step 5: 初始化模型
    model = TimeMixerPP(
            n_steps=T,
            n_features=N,
            n_layers=1,
            d_model=64,  # ✅ 增大到64
            d_ffn=128,   # ✅ 增大到128
            top_k=T//2,  # ✅ 动态设置为时间步的一半
            n_heads=2,   # ✅ 增加到2
            n_kernels=6, # ✅ 增加到6，确保多尺度
            dropout=0.1,
            channel_mixing=True,   # ✅ 改为True
            channel_independence=False,  # ✅ 改为False
            downsampling_layers=1,    # ✅ 改为1层下采样
            downsampling_window=2,    # ✅ 改为2
            apply_nonstationary_norm=False,
            batch_size=1,
            epochs=10,
            patience=3,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    # Step 6: 训练模型
    model.fit(train_set=dataset)

    # Step 7: 使用标准预测方法
    result = model.predict(dataset)
    if isinstance(result, dict):
        imputed = result.get('imputation', list(result.values())[0])
    else:
        imputed = result

    # 移除batch维度
    if len(imputed.shape) == 3:
        imputed = imputed[0]  # (T, N)

    return imputed


def tefn_impu(mx, epoch=100, device=None):
    from pypots.imputation import TEFN
    from pypots.optim.adam import Adam
    from pypots.nn.modules.loss import MAE, MSE
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用填充")
        mx[:, all_nan_cols] = global_mean
    mx = mx.copy()
    n_steps, n_features = mx.shape

    data = mx[np.newaxis, :, :]  # shape: (1, T, F)
    missing_mask = (~np.isnan(data)).astype(np.float32)
    indicating_mask = 1 - missing_mask
    data_filled = np.nan_to_num(data, nan=0.0).astype(np.float32)
    X_ori_no_nan = np.nan_to_num(data, nan=0.0).astype(np.float32)
    class OneSampleDataset(Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return (
                idx,
                data_filled[0],
                missing_mask[0],
                X_ori_no_nan[0],
                indicating_mask[0],
            )

    dataloader = DataLoader(OneSampleDataset(), batch_size=1, shuffle=False)

    model = TEFN(
        n_steps=n_steps,
        n_features=n_features,
        n_fod=2,
        apply_nonstationary_norm=True,
        ORT_weight=1.0,
        MIT_weight=1.0,
        batch_size=1,
        epochs=epoch,
        patience=5,
        training_loss=MAE,
        validation_metric=MSE,
        optimizer=Adam,
        device=device,
        saving_path=None,
        model_saving_strategy=None,
        verbose=False,
    )
    model._train_model(dataloader, dataloader)
    model.model.load_state_dict(model.best_model_dict)

    # 构造推理数据
    X = torch.tensor(data_filled, dtype=torch.float32).to(model.device)
    missing_mask = torch.tensor(missing_mask, dtype=torch.float32).to(model.device)

    # 推理填补
    model.model.eval()
    with torch.no_grad():
        output = model.model({
            'X': X,
            'missing_mask': missing_mask,
        })
        imputed = output['imputation']

    # 替换缺失位置
    X_ori_tensor = torch.tensor(X_ori_no_nan, dtype=torch.float32).to(model.device)
    result = X_ori_tensor.clone()
    result[missing_mask == 0] = imputed[missing_mask == 0]

    return result.cpu().numpy().squeeze()
def timesnet_impu(mx):
    import numpy as np
    from pypots.imputation.timesnet import TimesNet  # 根据实际项目结构调整

    # 复制原始数据
    mx = mx.copy()
    n_steps, n_features = mx.shape

    # 记录全空列
    all_nan_cols = np.all(np.isnan(mx), axis=0)

    # 计算全局均值用于填补全空列
    non_nan_values = mx[~np.isnan(mx)]
    global_mean = np.mean(non_nan_values) if non_nan_values.size > 0 else 0.0

    # 用全局均值填补全空列（完全 NaN 的列）
    for i in range(n_features):
        if all_nan_cols[i]:
            mx[:, i] = global_mean

    # 构造缺失掩码（注意此时已经没有全空列）
    mask = ~np.isnan(mx)
    mx_filled = np.nan_to_num(mx, nan=0.0)  # 其余 NaN 填 0，用作模型输入

    # 初始化 TimesNet 模型
    model = TimesNet(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        top_k=1,
        d_model=2,
        d_ffn=2,
        n_kernels=2,
        dropout=0.1,
        batch_size=1,
        epochs=5,
        patience=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    # 构造输入数据
    data = {
        "X": mx_filled[None, ...],            # (1, T, N)
        "missing_mask": mask[None, ...],      # (1, T, N)
        "X_ori": mx[None, ...],               # 原始带缺失值
        "indicating_mask": mask[None, ...],   # 与 missing_mask 相同
    }

    # 拟合模型
    model.fit(data)

    # 使用模型进行填补
    imputed = model.predict({"X": mx_filled[None, ...], "missing_mask": mask[None, ...]})
    return imputed["imputation"][0]  # 返回填补后的 (T, N) 矩阵

def tsde_impu(mx, n_samples: int = 40, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    from tsde import impute_missing_data
    mx = mx.copy()
    mx = impute_missing_data(
            mx, 
            n_samples=n_samples, 
            device=device
        )
    return mx

def grin_impu(mx):
    """GRIN填补方法 - 低内存版本"""
    from grin import grin_impute_low_memory
    try:
        mx = mx.copy()
        seq_len, n_features = mx.shape
        
        print(f"原始缺失值: {np.isnan(mx).sum()}")
        
        # ✅ 放宽限制条件，但保持低内存
        if seq_len < 10:
            print("⚠️ 序列太短，使用均值填补")
            return mean_impu(mx)
        
        # 根据数据大小调整参数
        total_size = seq_len * n_features
        
        if total_size > 50000:  # 大数据集
            window_size = min(10, seq_len // 10)
            hidden_dim = min(8, max(4, n_features // 10))
            epochs = 80
            print(f"🔧 大数据集模式: window={window_size}, hidden={hidden_dim}")
        elif total_size > 10000:  # 中等数据集
            window_size = min(15, seq_len // 8) 
            hidden_dim = min(16, max(8, n_features // 8))
            epochs = 100
            print(f"🔧 中等数据集模式: window={window_size}, hidden={hidden_dim}")
        else:  # 小数据集
            window_size = min(20, seq_len // 4)
            hidden_dim = min(32, max(16, n_features // 4))
            epochs = 120
            print(f"🔧 小数据集模式: window={window_size}, hidden={hidden_dim}")
        
        # 调用低内存版GRIN
        from grin import grin_impute_low_memory
        result = grin_impute_low_memory(
            mx, 
            window_size=window_size,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=0.01
        )
        
        # 验证填补结果
        if np.isnan(result).any():
            print("🔄 GRIN部分填补，补充均值填补")
            # 只对剩余缺失值用均值填补
            remaining_nan = np.isnan(result)
            col_means = np.nanmean(mx, axis=0)
            for j in range(n_features):
                if remaining_nan[:, j].any():
                    if not np.isnan(col_means[j]):
                        result[remaining_nan[:, j], j] = col_means[j]
                    else:
                        result[remaining_nan[:, j], j] = 0
        
        return result
        
    except Exception as e:
        return mean_impu(mx)