import numpy as np
import pandas as pd
def zero_impu(mx):
   return np.nan_to_num(mx, nan=0)
def meam_impu(mx):
   return np.nan_to_num(mx, nan=np.nanmean(mx, axis=0, keepdims=True))
def median_impu(mx):
   return np.nan_to_num(mx, nan=np.nanmedian(mx, axis=0, keepdims=True))
def mode_impu(mx):
    df = pd.DataFrame(mx)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
            df[column] = -1
        else:
            non_nan_data = col_data.dropna()
            value_counts = non_nan_data.value_counts()
            mode_value = value_counts.index[0]
            mode_count = value_counts.iloc[0]
            if mode_count >= 0.8 * len(non_nan_data):
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
