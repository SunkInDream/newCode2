import numpy as np
import pandas as pd
from scipy import stats
import os
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
    mx = mx.copy()
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


def knn_impu(mx, k=5):
    mx = mx.copy()
    all_nan_cols = np.all(np.isnan(mx), axis=0)

    # 计算全局均值（不为 NaN）
    global_mean = np.nanmean(mx)

    # 全空列先填全局均值，避免 KNNImputer 报错
    mx[:, all_nan_cols] = global_mean

    imputer = KNNImputer(n_neighbors=k)
    return imputer.fit_transform(mx)

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

def miracle_impu(mx):
    try:
        global_mean = np.nanmean(mx)
        from miracle import MIRACLE
        mx = mx.copy()
        
        # 检查是否有缺失值
        if not np.isnan(mx).any():
            print("数据中没有缺失值，直接返回原数据")
            return mx
        
        # 检查是否所有值都是NaN
        if np.isnan(mx).all():
            print("所有值都是NaN，使用0填充")
            return np.zeros_like(mx)
        
        # 检查是否有整列都是NaN
        all_nan_cols = np.all(np.isnan(mx), axis=0)
        if all_nan_cols.any():
            print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用填充")
            mx[:, all_nan_cols] = global_mean
        
        # 重新检查剩余的缺失值
        missing_idxs = np.where(np.any(np.isnan(mx), axis=0))[0]
        
        # 如果没有剩余的缺失值，直接返回
        if len(missing_idxs) == 0:
            print("处理完全NaN列后，没有剩余缺失值")
            return mx
        
        # 对剩余缺失值使用均值填充作为种子
        mx_imputed = mean_impu(mx)
        
        # 使用MIRACLE进行填补
        miracle = MIRACLE(
            num_inputs=mx.shape[1],
            reg_lambda=6,
            reg_beta=4,
            n_hidden=32,
            ckpt_file="tmp.ckpt",
            missing_list=missing_idxs,
            reg_m=0.1,
            lr=0.01,
            window=10,
            max_steps=200,  # 减少训练步数避免过拟合
        )
        
        miracle_imputed_data_x = miracle.fit(
            mx,
            X_seed=mx_imputed,
        )
        
        # ✅ 检查MIRACLE输出结果
        if miracle_imputed_data_x is None:
            print("MIRACLE返回None，使用0填充")
            return np.zeros_like(mx)
        
        # 检查是否所有值都是NaN
        if np.isnan(miracle_imputed_data_x).all():
            print("MIRACLE输出全为NaN，使用0填充")
            return np.zeros_like(mx)
        
        # 检查是否还有NaN值
        if np.isnan(miracle_imputed_data_x).any():
            print("MIRACLE输出包含NaN，将剩余NaN替换为0")
            miracle_imputed_data_x = np.where(np.isnan(miracle_imputed_data_x), 0.0, miracle_imputed_data_x)
        
        # 检查是否有异常大值
        if np.any(np.abs(miracle_imputed_data_x) > 1e6):
            print(f"MIRACLE输出包含异常大值 (max: {np.max(np.abs(miracle_imputed_data_x)):.2e})，使用均值填充")
            return mean_impu(mx)
        
        # 检查是否有无穷值
        if np.any(np.isinf(miracle_imputed_data_x)):
            print("MIRACLE输出包含无穷值，使用均值填充")
            return mean_impu(mx)
        
        print("MIRACLE填补成功")
        return miracle_imputed_data_x
        
    except Exception as e:
        print(f"MIRACLE填补失败: {e}")
        print("使用0填充作为fallback")
        mx = mx.copy()
        mx[np.isnan(mx)] = 0.0
        return mx

def saits_impu(mx, epochs=10, d_model=128, n_layers=2, n_heads=4, 
               d_k=32, d_v=32, d_ffn=64, dropout=0.4, device=None):
    from pypots.imputation import SAITS
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用填充")
        mx[:, all_nan_cols] = global_mean
    # print("_____________2______________________")
    mx = mx.copy()
    n_steps, n_features = mx.shape
    data_3d = mx[np.newaxis, :, :]  # shape: (1, n_steps, n_features)
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ffn=d_ffn,
        dropout=dropout,
        epochs=epochs,
        device=device,
    )
    
    train_set = {"X": data_3d}
    saits.fit(train_set)
    test_set = {"X": data_3d}
    imputed_data_3d = saits.impute(test_set)
    imputed_data_2d = imputed_data_3d[0]  # shape: (n_steps, n_features)
    return imputed_data_2d

def timemixerpp_impu(mx):
    """TimeMixer++ 填补函数的修复版本"""
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用填充")
        mx[:, all_nan_cols] = global_mean
    try:
        from pypots.imputation import TimeMixerpp
        
        # 检查输入维度
        if mx.shape[1] < 5:
            print(f"TimeMixer++ 需要至少5个特征，当前只有 {mx.shape[1]}，使用均值填补")
            return mean_impu(mx)
        
        # 确保输入格式正确
        if len(mx.shape) == 2:
            # 添加batch维度
            train_data = mx[np.newaxis, ...]
        else:
            train_data = mx
        
        # 创建模型时指定正确的参数
        timemixer = TimeMixerpp(
            n_steps=mx.shape[0],
            n_features=mx.shape[1],
            n_layers=1,       
            d_model=8,      
            d_ffn=8,          
            n_heads=1,      
            n_kernels=1,      
            top_k=1,          
            dropout=0.5,      
            channel_mixing=False,        
            channel_independence=False,   
            downsampling_layers=0,       
            apply_nonstationary_norm=False, 
            epochs=5,         
            patience=0,       
            batch_size=128,   
            verbose=False,   
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        
        # 训练和填补
        timemixer.fit(train_data)
        imputed_data = timemixer.predict(train_data)
        
        # 确保输出格式正确
        if len(imputed_data.shape) == 3:
            return imputed_data[0]  # 移除batch维度
        else:
            return imputed_data
            
    except Exception as e:
        print(f"TimeMixer++ 执行失败: {e}")
        return mean_impu(mx)




def tefn_impu(mx, epoch=10, device=None):
    from pypots.imputation import TEFN
    from pypots.optim.adam import Adam
    from pypots.nn.modules.loss import MAE, MSE
    print("_____________4______________________")
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
        top_k=8,
        d_model=4,
        d_ffn=8,
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

def tsde_impu(mx, n_samples: int = 20, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
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
            epochs = 3
            print(f"🔧 大数据集模式: window={window_size}, hidden={hidden_dim}")
        elif total_size > 10000:  # 中等数据集
            window_size = min(15, seq_len // 8) 
            hidden_dim = min(16, max(8, n_features // 8))
            epochs = 5
            print(f"🔧 中等数据集模式: window={window_size}, hidden={hidden_dim}")
        else:  # 小数据集
            window_size = min(20, seq_len // 4)
            hidden_dim = min(32, max(16, n_features // 4))
            epochs = 10
            print(f"🔧 小数据集模式: window={window_size}, hidden={hidden_dim}")
        
        # 调用低内存版GRIN
        from grin import grin_impute_low_memory
        result = grin_impute_low_memory(
            mx, 
            window_size=window_size,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=0.005
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