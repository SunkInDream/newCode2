import numpy as np
import pandas as pd
from scipy import stats
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn.impute import KNNImputer
from miracle import *
from pypots.imputation import SAITS,TimeMixerPP,TimeLLM,MOMENT,TEFN
from typing import Optional
from pypots.optim.adam import Adam
from pypots.nn.modules.loss import MAE, MSE
import torch
from torch.utils.data import Dataset, DataLoader

def zero_impu(mx):
   return np.nan_to_num(mx, nan=0)

def mean_impu(mx):
    mx = mx.copy()
    # original_shape = mx.shape  # 保存原始维度
    
    # # 按列计算均值
    # col_mean = np.nanmean(mx, axis=0)
    # all_nan_cols = np.isnan(col_mean)
    # col_mean[all_nan_cols] = 0
    
    # # 明确处理每一列
    # for col in range(mx.shape[1]):
    #     nan_mask = np.isnan(mx[:, col])
    #     if np.any(nan_mask):
    #         mx[nan_mask, col] = col_mean[col]
    
    # # 确保所有NaN都已处理
    # if np.isnan(mx).any():
    #     mx = np.nan_to_num(mx, nan=0)
    
    # # 确保维度没变
    # assert mx.shape == original_shape, "填充后维度变化!"
    col_means = np.nanmean(mx, axis=0)
    inds = np.where(np.isnan(mx))
    mx[inds] = np.take(col_means, inds[1])
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    return mx

def median_impu(mx):
    mx = mx.copy()
    # original_shape = mx.shape  # 保存原始维度
    
    # # 按列计算均值
    # col_median = np.nanmedian(mx, axis=0)
    # all_nan_cols = np.isnan(col_median)
    # col_median[all_nan_cols] = 0
    
    # # 明确处理每一列
    # for col in range(mx.shape[1]):
    #     nan_mask = np.isnan(mx[:, col])
    #     if np.any(nan_mask):
    #         mx[nan_mask, col] = col_median[col]
    
    # # 确保所有NaN都已处理
    # if np.isnan(mx).any():
    #     mx = np.nan_to_num(mx, nan=0)
    
    # # 确保维度没变
    # assert mx.shape == original_shape, "填充后维度变化!"
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
    # X = mx.copy()
    # col_mean = np.nanmean(X, axis=0)
    
    # # 全空列填充-1
    # col_mean = np.where(np.isnan(col_mean), -1, col_mean)
    
    # inds = np.where(np.isnan(X))
    # X[inds] = np.take(col_mean, inds[1])
    # imputed_data_x = X

    # missing_idxs = np.where(np.any(np.isnan(mx), axis=0))[0]
    # miracle = MIRACLE(
    #     num_inputs=mx.shape[1],
    #     reg_lambda=6,
    #     reg_beta=4,
    #     n_hidden=32,
    #     ckpt_file="tmp.ckpt",
    #     missing_list=missing_idxs,
    #     reg_m=0.1,
    #     lr=0.01,
    #     window=10,
    #     max_steps=800,
    # )
    # miracle_imputed_data_x = miracle.fit(
    #     mx,
    #     X_seed=imputed_data_x,
    # )
    # return miracle_imputed_data_x
    mx = mx.copy()
    missing_idxs = np.where(np.any(np.isnan(mx), axis=0))[0]
    mx_imputed = mean_impu(mx)
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
        max_steps=800,
    )
    miracle_imputed_data_x = miracle.fit(
        mx,
        X_seed=mx_imputed,
    )
    return miracle_imputed_data_x

def saits_impu(mx, epochs=100, d_model=256, n_layers=2, n_heads=4, 
               d_k=32, d_v=32, d_ffn=64, dropout=0.2):
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
        epochs=epochs
    )
    
    train_set = {"X": data_3d}
    saits.fit(train_set)
    test_set = {"X": data_3d}
    imputed_data_3d = saits.impute(test_set)
    imputed_data_2d = imputed_data_3d[0]  # shape: (n_steps, n_features)
    return imputed_data_2d

def timemixerpp_impu(mx, n_layers=3, d_model=16, d_ffn=32, top_k=3,
                     n_heads=4, n_kernels=4, dropout=0.1, epochs=300, batch_size=32,
                    patience=10, device=None, verbose=True, random_seed=42):
    mx = mx.copy()
    np.random.seed(random_seed)
    n_steps, n_features = mx.shape
    data_3d = mx[np.newaxis, :, :]  # 形状变为 (1, n_timesteps, n_features)
    
    if verbose:
        print(f"原始数据形状: {mx.shape}")
        print(f"转换后数据形状: {data_3d.shape}")
        missing_rate = np.isnan(data_3d).sum() / data_3d.size
        print(f"缺失率: {missing_rate:.2%}")

    train_data = {
        'X': data_3d.copy()
    }
    
    timemixer = TimeMixerPP(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=n_layers,
        d_model=d_model,
        d_ffn=d_ffn,
        top_k=top_k,
        n_heads=n_heads,
        n_kernels=n_kernels,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        verbose=verbose
    )
        
    if verbose:
        print("开始训练TimeMixerPP模型...")
    timemixer.fit(train_data)
    if verbose:
        print("训练完成，开始填补缺失值...")
    test_data = {'X': data_3d.copy()}
    imputed_data_3d = timemixer.predict(test_data)['imputation']
    imputed_data = imputed_data_3d[0, :, :]  # 取出第一个样本，形状变为 (n_timesteps, n_features) 
    if verbose:
        print("缺失值填补完成！")
        print(f"输出数据形状: {imputed_data.shape}")
    return imputed_data

def tefn_impu(mx, epoch=10):
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
        device="cuda" if torch.cuda.is_available() else "cpu",
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