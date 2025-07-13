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
    col_means = np.nanmean(mx, axis=0)
    inds = np.where(np.isnan(mx))
    mx[inds] = np.take(col_means, inds[1])
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    return mx

def median_impu(mx):
    mx = mx.copy()
    col_medians = np.nanmedian(mx, axis=0)
    inds = np.where(np.isnan(mx))
    mx[inds] = np.take(col_medians, inds[1])
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    return mx

def mode_impu(mx):
    # df = pd.DataFrame(mx)
    # for column in df.columns:
    #     col_data = df[column]
    #     if col_data.isna().all():
    #         df[column] = -1
    #     else:
    #         non_nan_data = col_data.dropna()
    #         mode_value = non_nan_data.mode().iloc[0]  # 更直接取众数
    #         df[column] = col_data.fillna(mode_value)
    # return df.values
    mx = mx.copy()
    col_modes = stats.mode(mx, axis=0, nan_policy='omit').mode[0]
    inds = np.where(np.isnan(mx))
    mx[inds] = np.take(col_modes, inds[1])
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    return mx

def random_impu(mx):
    # df = pd.DataFrame(mx)
    # for column in df.columns:
    #     col_data = df[column]
    #     if col_data.isna().all():
    #         df[column] = -1
    #     else:
    #         non_nan_data = col_data.dropna()
    #         if not non_nan_data.empty:
    #             random_value = np.random.choice(non_nan_data)
    #             df[column] = col_data.fillna(random_value)
    # return df.values
    mx = mx.copy()
    for col in range(mx.shape[1]):
        nan_mask = np.isnan(mx[:, col])
        non_nan = mx[~nan_mask, col]
        if non_nan.size > 0:
            mx[nan_mask, col] = np.random.choice(non_nan, size=nan_mask.sum())
        else:
            mx[nan_mask, col] = -1
    return mx

def knn_impu(mx, k=5):
    mx = mx.copy()
    imputer = KNNImputer(n_neighbors=k)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    mx[:, all_nan_cols] = -1
    return imputer.fit_transform(mx)

def ffill_impu(mx):
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.ffill(axis=0)  # 沿着时间维度（行）前向填充
    df = df.fillna(-1)     # 若第一行是 NaN 会残留未填，补-1
    return df.values

def bfill_impu(mx):
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.bfill(axis=0)  # 沿着时间维度（行）后向填充
    df = df.fillna(-1)     # 若最后一行是 NaN 会残留未填，补-1
    return df.values

def miracle_impu(mx):
    try:
        from miracle import MIRACLE
        print("_____________1______________________")
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
            print(f"发现 {all_nan_cols.sum()} 列全为NaN，这些列将用0填充")
            mx[:, all_nan_cols] = 0.0
        
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
            n_layers=16,  # 减少层数
            d_model=32,  # 减少模型维度
            n_heads=4,   # 减少注意力头数
            epochs=50,   # 减少训练轮数
            batch_size=32,
            patience=1,
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