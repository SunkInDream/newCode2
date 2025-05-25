import numpy as np
import pandas as pd
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
    from sklearn.impute import KNNImputer
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
