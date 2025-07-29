import numpy as np
import pandas as pd
from scipy import stats
import os
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn.impute import KNNImputer
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from models_impute import *

def zero_impu(mx):
   return np.nan_to_num(mx, nan=0)

def mean_impu(mx):
    mean = np.nanmean(mx)
    return np.where(np.isnan(mx), mean, mx)

def median_impu(mx):
    mx = mx.copy()
    median = np.nanmedian(mx)
    return np.where(np.isnan(mx), median, mx)

def mode_impu(mx):
    mx = mx.copy()
    flat_values = mx[~np.isnan(mx)] 
    global_mode = stats.mode(flat_values, keepdims=False).mode
    if np.isnan(global_mode):
        global_mode = 0 
    inds = np.where(np.isnan(mx))
    mx[inds] = global_mode
    return mx

def random_impu(mx):
    mx = mx.copy()
    non_nan_values = mx[~np.isnan(mx)] 
    if non_nan_values.size == 0:
        mx[:] = -1
        return mx
    inds = np.where(np.isnan(mx))  
    mx[inds] = np.random.choice(non_nan_values, size=len(inds[0]), replace=True)
    return mx

def knn_impu(mx, k=3):
    mx = mx.copy()
    from sklearn.impute import KNNImputer
    
    # ✅ 记录原始形状
    original_shape = mx.shape
    
    # ✅ 确保k不超过有效样本数
    non_nan_rows = np.sum(~np.isnan(mx).any(axis=1))
    if non_nan_rows == 0:
        # 如果所有行都有缺失，用均值填补
        return zero_impu(mx)
    
    k = min(k, max(1, non_nan_rows - 1))
    
    try:
        imputer = KNNImputer(n_neighbors=k)
        result = imputer.fit_transform(mx)
        
        # ✅ 确保输出形状与输入一致
        if result.shape != original_shape:
            result = result[:original_shape[0], :original_shape[1]]
            
        return result
        
    except Exception as e:
        print(f"KNN imputation failed: {e}, falling back to mean imputation")
        return mean_impu(mx)
        

def mice_impu(mx, max_iter=5):
    mx = mx.copy()
    n_rows, n_cols = mx.shape
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        global_mean = np.nanmean(mx)
        mx[:, all_nan_cols] = global_mean
    imp = SimpleImputer(strategy='mean')
    matrix_filled = imp.fit_transform(mx)
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
    global_mean = np.nanmean(mx)
    df = df.fillna(global_mean)

    return df.values

def bfill_impu(mx):
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.bfill(axis=0)
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

def saits_impu(mx, epochs=None, d_model=None, n_layers=None, device=None):
    from pypots.imputation import SAITS
    
    mx = mx.copy()
    seq_len, n_features = mx.shape
    total_size = seq_len * n_features
    
    global_mean = np.nanmean(mx)
    if np.isnan(global_mean):
        global_mean = 0.0
    
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    
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
        print(f"SAITS fails: {e}")
        return mean_impu(mx)


def timemixerpp_impu(mx):
    import numpy as np
    import torch
    from pypots.imputation import TimeMixerPP
    from sklearn.impute import SimpleImputer

    mx = mx.astype(np.float32)
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    T, N = mx.shape
    data = mx[None, ...]  

    missing_mask = np.isnan(data).astype(np.float32)
    indicating_mask = (~np.isnan(data)).astype(np.float32)
    imp = SimpleImputer(strategy='mean', keep_empty_features=True)
    X_filled = imp.fit_transform(mx).astype(np.float32)
    X_filled = X_filled[None, ...]

    dataset = {
        "X": X_filled,
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask,
        "X_ori": data
    }

    model = TimeMixerPP(
            n_steps=T,
            n_features=N,
            n_layers=1,
            d_model=64,  
            d_ffn=128,  
            top_k=T//2,  
            n_heads=2,   
            n_kernels=6, 
            dropout=0.1,
            channel_mixing=True,  
            channel_independence=False,  
            downsampling_layers=1,    
            downsampling_window=2,   
            apply_nonstationary_norm=False,
            batch_size=1,
            epochs=10,
            patience=3,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    model.fit(train_set=dataset)

    result = model.predict(dataset)
    if isinstance(result, dict):
        imputed = result.get('imputation', list(result.values())[0])
    else:
        imputed = result

    if len(imputed.shape) == 3:
        imputed = imputed[0]

    return imputed


def tefn_impu(mx, epoch=100, device=None):
    from pypots.imputation import TEFN
    from pypots.optim.adam import Adam
    from pypots.nn.modules.loss import MAE, MSE
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    mx = mx.copy()
    n_steps, n_features = mx.shape

    data = mx[np.newaxis, :, :]
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

    X = torch.tensor(data_filled, dtype=torch.float32).to(model.device)
    missing_mask = torch.tensor(missing_mask, dtype=torch.float32).to(model.device)

    model.model.eval()
    with torch.no_grad():
        output = model.model({
            'X': X,
            'missing_mask': missing_mask,
        })
        imputed = output['imputation']
    X_ori_tensor = torch.tensor(X_ori_no_nan, dtype=torch.float32).to(model.device)
    result = X_ori_tensor.clone()
    result[missing_mask == 0] = imputed[missing_mask == 0]

    return result.cpu().numpy().squeeze()
def timesnet_impu(mx):
    import numpy as np
    from pypots.imputation.timesnet import TimesNet  

    mx = mx.copy()
    n_steps, n_features = mx.shape

    all_nan_cols = np.all(np.isnan(mx), axis=0)

    non_nan_values = mx[~np.isnan(mx)]
    global_mean = np.mean(non_nan_values) if non_nan_values.size > 0 else 0.0

    for i in range(n_features):
        if all_nan_cols[i]:
            mx[:, i] = global_mean

    mask = ~np.isnan(mx)
    mx_filled = np.nan_to_num(mx, nan=0.0) 

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

    data = {
        "X": mx_filled[None, ...],          
        "missing_mask": mask[None, ...],    
        "X_ori": mx[None, ...],             
        "indicating_mask": mask[None, ...],  
    }

    model.fit(data)

    imputed = model.predict({"X": mx_filled[None, ...], "missing_mask": mask[None, ...]})
    return imputed["imputation"][0]  

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
    from grin import grin_impute_low_memory
    try:
        mx = mx.copy()
        seq_len, n_features = mx.shape
        
        if seq_len < 10:
            return mean_impu(mx)

        total_size = seq_len * n_features
        
        if total_size > 50000:  
            window_size = min(10, seq_len // 10)
            hidden_dim = min(8, max(4, n_features // 10))
            epochs = 80
        elif total_size > 10000: 
            window_size = min(15, seq_len // 8) 
            hidden_dim = min(16, max(8, n_features // 8))
            epochs = 100
        else:  
            window_size = min(20, seq_len // 4)
            hidden_dim = min(32, max(16, n_features // 4))
            epochs = 120
        
        from grin import grin_impute_low_memory
        result = grin_impute_low_memory(
            mx, 
            window_size=window_size,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=0.01
        )
        
        if np.isnan(result).any():
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