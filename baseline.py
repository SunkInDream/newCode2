import numpy as np
import pandas as pd
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
    original_shape = mx.shape  # 保存原始维度
    
    # 按列计算均值
    col_mean = np.nanmean(mx, axis=0)
    all_nan_cols = np.isnan(col_mean)
    col_mean[all_nan_cols] = 0
    
    # 明确处理每一列
    for col in range(mx.shape[1]):
        nan_mask = np.isnan(mx[:, col])
        if np.any(nan_mask):
            mx[nan_mask, col] = col_mean[col]
    
    # 确保所有NaN都已处理
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    
    # 确保维度没变
    assert mx.shape == original_shape, "填充后维度变化!"
    
    return mx

def median_impu(mx):
    mx = mx.copy()
    original_shape = mx.shape  # 保存原始维度
    
    # 按列计算均值
    col_median = np.nanmedian(mx, axis=0)
    all_nan_cols = np.isnan(col_median)
    col_median[all_nan_cols] = 0
    
    # 明确处理每一列
    for col in range(mx.shape[1]):
        nan_mask = np.isnan(mx[:, col])
        if np.any(nan_mask):
            mx[nan_mask, col] = col_median[col]
    
    # 确保所有NaN都已处理
    if np.isnan(mx).any():
        mx = np.nan_to_num(mx, nan=0)
    
    # 确保维度没变
    assert mx.shape == original_shape, "填充后维度变化!"
    
    return mx
def mode_impu(mx):
    df = pd.DataFrame(mx)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
            df[column] = -1
        else:
            non_nan_data = col_data.dropna()
            mode_value = non_nan_data.mode().iloc[0]  # 更直接取众数
            df[column] = col_data.fillna(mode_value)
    return df.values

def random_impu(mx):
    df = pd.DataFrame(mx)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
            df[column] = -1
        else:
            non_nan_data = col_data.dropna()
            if not non_nan_data.empty:
                random_value = np.random.choice(non_nan_data)
                df[column] = col_data.fillna(random_value)
    return df.values
def knn_impu(mx, k=5):
    
    imputer = KNNImputer(n_neighbors=k)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    mx[:, all_nan_cols] = -1
    return imputer.fit_transform(mx)
def ffill_impu(mx):
    df = pd.DataFrame(mx)
    df = df.ffill(axis=0)  # 沿着时间维度（行）前向填充
    df = df.fillna(-1)     # 若第一行是 NaN 会残留未填，补-1
    return df.values
def bfill_impu(mx):
    df = pd.DataFrame(mx)
    df = df.bfill(axis=0)  # 沿着时间维度（行）后向填充
    df = df.fillna(-1)     # 若最后一行是 NaN 会残留未填，补-1
    return df.values
def miracle_impu(mx):
    X = mx.copy()
    col_mean = np.nanmean(X, axis=0)
    
    # 全空列填充-1
    col_mean = np.where(np.isnan(col_mean), -1, col_mean)
    
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    imputed_data_x = X

    missing_idxs = np.where(np.any(np.isnan(mx), axis=0))[0]
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
        X_seed=imputed_data_x,
    )
    return miracle_imputed_data_x
import numpy as np
from pypots.imputation import SAITS

def saits_impu(data_with_missing, 
                        epochs=100, 
                        d_model=256, 
                        n_layers=2, 
                        n_heads=4, 
                        d_k=32, 
                        d_v=32, 
                        d_ffn=64, 
                        dropout=0.2,
                        model_path=None):
    """
    使用SAITS算法填补2D时间序列数据中的缺失值
    
    参数:
        data_with_missing: numpy array, shape (n_steps, n_features)
                          带有缺失值的时间序列数据 (缺失值用np.nan表示)
        epochs: int, 训练轮数 (默认5)
        d_model: int, 模型维度 (默认256)
        n_layers: int, 网络层数 (默认2)
        n_heads: int, 注意力头数 (默认4)
        d_k: int, key维度 (默认64)
        d_v: int, value维度 (默认64)
        d_ffn: int, 前馈网络维度 (默认128)
        dropout: float, dropout率 (默认0.1)
        model_path: str, 预训练模型路径 (可选，如果提供则加载预训练模型)
    
    返回:
        numpy array, shape (n_steps, n_features)
        填补完整的时间序列数据
    """
    
    # 确保输入是2D数组
    if len(data_with_missing.shape) != 2:
        raise ValueError(f"输入数据必须是2D数组 (n_steps, n_features)，但得到了形状: {data_with_missing.shape}")
    
    n_steps, n_features = data_with_missing.shape
    
    # 将2D数据扩展为3D (添加batch维度)
    data_3d = data_with_missing[np.newaxis, :, :]  # shape: (1, n_steps, n_features)
    
    # 初始化SAITS模型
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
    
    # 如果提供了预训练模型路径，则加载模型
    if model_path is not None:
        try:
            saits.load(model_path)
            print(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("将重新训练模型...")
            
            # 准备训练数据
            train_set = {"X": data_3d}
            
            # 训练模型
            saits.fit(train_set)
    else:
        # 准备训练数据
        train_set = {"X": data_3d}
        
        # 训练模型
        saits.fit(train_set)
    
    # 进行填补
    test_set = {"X": data_3d}
    imputed_data_3d = saits.impute(test_set)
    
    # 将3D结果转回2D
    imputed_data_2d = imputed_data_3d[0]  # shape: (n_steps, n_features)
    
    return imputed_data_2d

def timemixerpp_impu(
    data_matrix: np.ndarray,
    n_layers: int = 3,
    d_model: int = 16,
    d_ffn: int = 32,
    top_k: int = 3,
    n_heads: int = 4,
    n_kernels: int = 4,
    dropout: float = 0.1,
    epochs: int = 300,
    batch_size: int = 32,
    patience: Optional[int] = 10,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
    verbose: bool = True,
    random_seed: int = 42
) -> np.ndarray:
    """
    使用TimeMixerPP模型填补时间序列数据中的缺失值
    """
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 输入验证
    if not isinstance(data_matrix, np.ndarray):
        raise ValueError("输入数据必须是numpy数组")
    
    if len(data_matrix.shape) != 2:
        raise ValueError("输入数据必须是2维数组 (n_timesteps, n_features)")
    
    # 将2D数据转换为3D格式 (添加batch维度)
    n_steps, n_features = data_matrix.shape
    data_3d = data_matrix[np.newaxis, :, :]  # 形状变为 (1, n_timesteps, n_features)
    
    if verbose:
        print(f"原始数据形状: {data_matrix.shape}")
        print(f"转换后数据形状: {data_3d.shape}")
        missing_rate = np.isnan(data_3d).sum() / data_3d.size
        print(f"缺失率: {missing_rate:.2%}")
    
    # 准备训练数据 - 单个样本，不分割验证集
    train_data = {
        'X': data_3d.copy()
    }
    
    try:
        # 初始化TimeMixerPP模型
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
        
        # 训练模型 - 不使用验证集
        timemixer.fit(train_data)
        
        if verbose:
            print("训练完成，开始填补缺失值...")
        
        # 使用训练好的模型进行填补
        test_data = {'X': data_3d.copy()}
        imputed_data_3d = timemixer.predict(test_data)['imputation']
        
        # 将3D结果转换回2D格式
        imputed_data = imputed_data_3d[0, :, :]  # 取出第一个样本，形状变为 (n_timesteps, n_features)
        
        if verbose:
            print("缺失值填补完成！")
            print(f"输出数据形状: {imputed_data.shape}")
        
        return imputed_data
        
    except Exception as e:
        print(f"模型训练或预测过程中出现错误: {e}")
        raise

def tefn_impu(data_np: np.ndarray) -> np.ndarray:
    """
    用 TEFN 模型对一个带缺失的二维 NumPy 数组进行填补。

    参数:
        data_np (np.ndarray): shape 为 (time_steps, features)，其中包含 np.nan 表示缺失值。

    返回:
        np.ndarray: 同样 shape 的已填补数据
    """
    assert data_np.ndim == 2, "输入必须是二维矩阵 (时间步, 特征数)"
    n_steps, n_features = data_np.shape

    # 构造 batch 数据：添加 batch 维度
    data = data_np[None, :, :]  # shape: (1, T, F)
    missing_mask = (~np.isnan(data)).astype(np.float32)
    indicating_mask = 1 - missing_mask
    data_filled = np.nan_to_num(data, nan=0.0).astype(np.float32)
    X_ori_no_nan = np.nan_to_num(data, nan=0.0).astype(np.float32)

    # 封装 Dataset
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

    # 初始化 TEFN 模型
    model = TEFN(
        n_steps=n_steps,
        n_features=n_features,
        n_fod=2,
        apply_nonstationary_norm=True,
        ORT_weight=1.0,
        MIT_weight=1.0,
        batch_size=1,
        epochs=10,
        patience=5,
        training_loss=MAE,
        validation_metric=MSE,
        optimizer=Adam,
        device="cuda" if torch.cuda.is_available() else "cpu",
        saving_path=None,
        model_saving_strategy=None,
        verbose=False,
    )

    # 使用自身数据训练模型
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