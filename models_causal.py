import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.impute import KNNImputer
import random
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from models_TCDF import findpredictions
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.fftpack import fft, ifft
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from contextlib import redirect_stdout
from models_runTCDF import run_tcdf_analysis
from sklearn.preprocessing import StandardScaler
import time
import glob
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from io import StringIO
import io
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn as nn
import concurrent.futures
warnings.filterwarnings("ignore")  # 忽略警告信息
def FirstProcess(file, csv_dir):
    file_path = os.path.join(csv_dir, file)
    df = pd.read_csv(file_path)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
         # 对于全空的列，填充为-1
            df[column] = -1
        else:
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts()
                if not value_counts.empty:
                    mode_value = value_counts.index[0]
                    mode_count = value_counts.iloc[0]
                    # 使用有效数据数量判断是否超过阈值
                    if mode_count >= 0.8 * len(non_null_data):
                        df[column] = col_data.fillna(mode_value)
    return df
def SecondProcess(file, perturbation_prob=0.1, perturbation_scale=0.1):
    df_copy = file
    for column in df_copy.columns:
        series = df_copy[column]
        missing_mask = series.isna()

        if not missing_mask.any():
            continue  # 如果没有缺失值，跳过该列

        missing_segments = []
        start_idx = None

        # 查找缺失值的连续段
        for i, is_missing in enumerate(missing_mask):
            if is_missing and start_idx is None:
                start_idx = i
            elif not is_missing and start_idx is not None:
                missing_segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            missing_segments.append((start_idx, len(missing_mask) - 1))

        # 对每个缺失段进行填补
        for start, end in missing_segments:
            left_value, right_value = None, None
            left_idx, right_idx = start - 1, end + 1

            # 找到前后最近的非缺失值
            while left_idx >= 0 and np.isnan(series.iloc[left_idx]):
                left_idx -= 1
            if left_idx >= 0:
                left_value = series.iloc[left_idx]

            while right_idx < len(series) and np.isnan(series.iloc[right_idx]):
                right_idx += 1
            if right_idx < len(series):
                right_value = series.iloc[right_idx]

            # 如果前后都没有非缺失值，使用均值填充
            if left_value is None and right_value is None:
                fill_value = series.dropna().mean()
                df_copy.loc[missing_mask, column] = fill_value
                continue

            # 如果只有一个方向有非缺失值，使用另一个方向的值填充
            if left_value is None:
                left_value = right_value
            elif right_value is None:
                right_value = left_value

            # 使用等差数列填补缺失值
            segment_length = end - start + 1
            step = (right_value - left_value) / (segment_length + 1)
            values = [left_value + step * (i + 1) for i in range(segment_length)]

            # 添加扰动
            value_range = np.abs(right_value - left_value) or (np.abs(left_value) * 0.1 if left_value != 0 else 1.0)
            for i in range(len(values)):
                if random.random() < perturbation_prob:
                    perturbation = random.uniform(-1, 1) * perturbation_scale * value_range
                    values[i] += perturbation

            # 将填补后的值赋回数据框
            for i, value in enumerate(values):
                df_copy.iloc[start + i, df_copy.columns.get_loc(column)] = value

    return df_copy
def create_dataloader(csv_dir, batch_size=32, n_clusters=10, shuffle=True, num_workers=4):
    dataset = CustomDataset(csv_dir, n_clusters)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
def run_tcdf_for_data_gpu(data, epoch=100):
    try:
        # 定义设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果传入的是torch.Tensor，直接使用
        if isinstance(data, torch.Tensor):
            # 直接使用张量API，不需要转成DataFrame再保存为CSV
            cm = run_tcdf_analysis(
                data_tensor_list=[data],  # 传入张量列表
                epochs=epoch,
                use_cuda=True if device.type == 'cuda' else False,
                log_interval=1000,
                device=device.type  # 明确指定设备
            )
            
            if cm is not None and isinstance(cm, pd.DataFrame) and not cm.empty:
                print("使用TCDF生成的因果矩阵")
                return cm
        else:
            # 对于DataFrame，保持原有流程
            data_df = data
            
            # 确保我们有一个临时文件供TCDF使用
            temp_file_path = "temp_data_for_tcdf.csv"
            data_df.to_csv(temp_file_path, index=False)
            
            print(f"运行TCDF分析...")
            
            # 使用文件路径调用TCDF分析
            output_buffer = StringIO()
            with redirect_stdout(output_buffer):
                cm = run_tcdf_analysis(
                    data_files_list=[temp_file_path],
                    epochs=epoch,
                    log_interval=1000,
                    use_cuda=True if device.type == 'cuda' else False
                )
            
            # 删除临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            if cm is not None and isinstance(cm, pd.DataFrame) and not cm.empty:
                print("使用TCDF生成的因果矩阵")
                return cm
        
        print("未能生成有效的因果矩阵")
        return None
            
    except Exception as e:
        print(f"错误: 运行TCDF分析时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_filled_results(eval_dataset, first_sample_results):
    """
    保存第一个样本的各种填充方法结果为CSV文件，便于可视化比较
    
    Args:
        eval_dataset: 评估数据集
        first_sample_results: 包含各种填充方法结果的字典
    """
    try:
        # 创建结果文件夹
        result_dir = "imputation_results"
        os.makedirs(result_dir, exist_ok=True)
        
        # 获取原始数据的列名
        if len(eval_dataset.data) > 0 and 'original' in eval_dataset.data[0]:
            if hasattr(eval_dataset.data[0]['original'], 'columns'):
                columns = eval_dataset.data[0]['original'].columns
            else:
                # 如果没有columns属性，创建通用列名
                columns = [f"feature_{i}" for i in range(first_sample_results['causal_tcn'].shape[1])]
        else:
            # 创建通用列名
            columns = [f"feature_{i}" for i in range(first_sample_results['causal_tcn'].shape[1])]
        
        # 获取缺失数据的掩码（如果可用）
        mask = None
        if len(eval_dataset.data) > 0 and 'mask' in eval_dataset.data[0]:
            mask = eval_dataset.data[0]['mask']
        
        # 获取原始数据以与填充结果进行比较
        original_data = None
        if len(eval_dataset.data) > 0 and 'original' in eval_dataset.data[0]:
            if isinstance(eval_dataset.data[0]['original'], pd.DataFrame):
                original_data = eval_dataset.data[0]['original'].values
            else:
                original_data = eval_dataset.data[0]['original']
        
        # 保存所有填充结果到一个CSV文件，便于比较
        comparison_file = os.path.join(result_dir, "imputation_comparison.csv")
        
        # 创建一个大的DataFrame包含所有方法结果
        all_results = {}
        
        # 首先添加原始数据（如果有）
        if original_data is not None:
            for i, col in enumerate(columns):
                all_results[f"original_{col}"] = original_data[:, i]
        
        # 添加每种方法的结果
        for method, data in first_sample_results.items():
            for i, col in enumerate(columns):
                all_results[f"{method}_{col}"] = data[:, i]
                
        # 转换为DataFrame并保存
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"已保存填充结果比较到: {comparison_file}")
        
        # 为每种方法单独保存一个CSV文件
        for method, data in first_sample_results.items():
            method_file = os.path.join(result_dir, f"{method}_filled.csv")
            method_df = pd.DataFrame(data, columns=columns)
            method_df.to_csv(method_file, index=False)
            print(f"已保存 {method} 填充结果到: {method_file}")
            
        # 保存原始数据（如果有）
        if original_data is not None:
            original_file = os.path.join(result_dir, "original_data.csv")
            original_df = pd.DataFrame(original_data, columns=columns)
            original_df.to_csv(original_file, index=False)
            print(f"已保存原始数据到: {original_file}")
            
        # 保存掩码数据（如果有）
        if mask is not None:
            mask_file = os.path.join(result_dir, "missing_mask.csv")
            if isinstance(mask, np.ndarray):
                mask_df = pd.DataFrame(mask, columns=columns)
            else:
                mask_df = pd.DataFrame(mask)
            mask_df.to_csv(mask_file, index=False)
            print(f"已保存缺失值掩码到: {mask_file}")
        
    except Exception as e:
        print(f"保存填充结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def evaluate_imputation_methods(eval_dataset, original_dataset, cuda=False, epochs=10):
    # 设置设备
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    
    # 存储所有方法的MSE结果
    all_mse = {
        "causal_tcn": [],     # 因果TCN方法
        "preprocessed": [],  # 预处理数据
        "zero": [],           # 0填充
        "median": [],         # 中位数填充
        "mean": [],           # 均值填充
        "ffill": [],          # 前最近邻填充
        "bfill": [],          # 后最近邻填充
        "knn": [],            # KNN填充
        "mice": []            # MICE填充
    }
    
    # 确保eval_dataset有必要的属性
    if not hasattr(eval_dataset, 'causal_matrices'):
        eval_dataset.causal_matrices = original_dataset.causal_matrices
    if not hasattr(eval_dataset, 'total_causal_matrix'):
        eval_dataset.total_causal_matrix = original_dataset.total_causal_matrix

    # 遍历评估数据集中的每个样本
    print(f"开始评估 {len(eval_dataset.data)} 个样本...")
    for i in tqdm(range(len(eval_dataset.data))):
        try:
            # 获取带有人工缺失的数据
            missing_data = eval_dataset.data[i]['preprocessed']  # 修改为使用preprocessed
            # 获取真实值数据（无缺失）
            ground_truth = eval_dataset.data[i]['original']  # 使用original作为真值
            # 第一层预处理
            for column in missing_data.columns:
                col_data = missing_data[column]
                if col_data.isna().all():
                    missing_data[column] = -1
                else:
                    non_null_data = col_data.dropna()
                    if len(non_null_data) > 0:
                        value_counts = non_null_data.value_counts()
                        if not value_counts.empty:
                            mode_value = value_counts.index[0]
                            mode_count = value_counts.iloc[0]
                            if mode_count >= 0.8 * len(non_null_data):
                                missing_data[column] = col_data.fillna(mode_value)
            # 创建掩码 (True=缺失, False=已知)
            bool_mask = ~missing_data.isna()
            # 第二层预处理
            missing_data = SecondProcess(missing_data, perturbation_prob=0.4, perturbation_scale=0.6)
            # 获取原始数据集中对应的完整数据（真值）
            if i < len(original_dataset.data) and 'filled' in original_dataset.data[i]:
                if isinstance(original_dataset.data[i]['filled'], torch.Tensor):
                    ground_truth = original_dataset.data[i]['filled'].cpu().numpy()
                else:
                    ground_truth = original_dataset.data[i]['filled'].values
                if isinstance(ground_truth, np.ndarray):
                    ground_truth = pd.DataFrame(ground_truth, columns=missing_data.columns)
            else:
                print(f"警告: 样本 {i} 在原始数据集中无对应的filled数据，跳过")
                continue
            
            # 1. 使用因果TCN方法填充
            print(f"\n[{i+1}/{len(eval_dataset.data)}] 使用因果TCN方法填充...")
            causal_filled = missing_data.copy()
            
            # 创建临时文件用于TCN预测
            temp_file = f"temp_causal_eval_{i}.csv"
            causal_filled.to_csv(temp_file, index=False)
            
            # 移到GPU
            if cuda:
                causal_filled_tensor = torch.tensor(causal_filled.values, dtype=torch.float32).to(device)
            
            # 遍历每个特征
            for j in range(causal_filled.shape[1]):
                feature_name = causal_filled.columns[j]
                
                # 获取因果相关特征列表
                feature_list = []
                if eval_dataset.total_causal_matrix is not None and j < eval_dataset.total_causal_matrix.shape[1]:
                    causal_rows = np.where(eval_dataset.total_causal_matrix[:, j] == 1)[0]
                    feature_list = [causal_filled.columns[idx] for idx in causal_rows 
                                    if idx < len(causal_filled.columns)]
                    if feature_name not in feature_list:
                        feature_list.append(feature_name)
                
                try:
                    # 使用TCN预测
                    pred = findpredictions(
                        target=feature_name,
                        cuda=cuda,  # 传递GPU参数
                        epochs=epochs,
                        kernel_size=6,
                        layers=3,
                        log_interval=1,
                        lr=0.02,
                        optimizername='Adam',
                        seed=1111,
                        dilation_c=4,
                        significance=0.8,
                        file=temp_file,
                        feature_list=feature_list
                    )
                    
                    # 处理预测结果
                    if pred is not None and len(pred) > 0:
                        for idx in range(len(causal_filled)):
                            if ~bool_mask.iloc[idx, j] and idx < len(pred):  # 如果是缺失值且有预测结果
                                causal_filled.iloc[idx, j] = round(float(pred[idx][0]), 1)  # 四舍五入到1位小数
                            else:
                                causal_filled.iloc[idx, j] = missing_data.iloc[idx, j]
                except Exception as e:
                    print(f"  TCN预测 {feature_name} 失败: {str(e)}")
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)            
            # 2-8. 应用基线填充方法
            print("应用基线填充方法...")
                        
            # 2. 零填充
            zero_filled = missing_data.copy()
            mask_array = bool_mask.values
            for col_idx, col in enumerate(zero_filled.columns):
                col_data = zero_filled[col]
                for row_idx in range(len(col_data)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        zero_filled.iloc[row_idx, col_idx] = 0
            
            # 3. 中位数填充
            median_filled = missing_data.copy()
            mask_array = bool_mask.values
            for col_idx, col in enumerate(median_filled.columns):
                col_data = median_filled[col]
                for row_idx in range(len(col_data)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        median_filled.iloc[row_idx, col_idx] = col_data.median()
            
            # 4. 均值填充
            mean_filled = missing_data.copy()
            mask_array = bool_mask.values
            for col_idx, col in enumerate(mean_filled.columns):
                col_data = mean_filled[col]
                for row_idx in range(len(col_data)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        mean_filled.iloc[row_idx, col_idx] = col_data.mean()
            
            # 5. 前向填充
            ffill_filled = ground_truth.copy()  # 从真值复制，而不是从missing_data复制
            # 在缺失位置设置NaN
            for col_idx, col in enumerate(ffill_filled.columns):
                for row_idx in range(len(ffill_filled)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        ffill_filled.iloc[row_idx, col_idx] = np.nan

            # 应用前向填充
            ffill_filled = ffill_filled.fillna(method='ffill')
            # 处理第一行缺失
            for col in ffill_filled.columns:
                if ffill_filled[col].isna().any():
                    ffill_filled[col] = ffill_filled[col].fillna(method='bfill')

            # 6. 后向填充
            bfill_filled = ground_truth.copy()  # 从真值复制
            # 在缺失位置设置NaN
            for col_idx, col in enumerate(bfill_filled.columns):
                for row_idx in range(len(bfill_filled)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        bfill_filled.iloc[row_idx, col_idx] = np.nan

            # 应用后向填充
            bfill_filled = bfill_filled.fillna(method='bfill')
            # 处理最后一行缺失
            for col in bfill_filled.columns:
                if bfill_filled[col].isna().any():
                    bfill_filled[col] = bfill_filled[col].fillna(method='ffill')

            # 7. KNN填充
            knn_filled = ground_truth.copy()  # 从真值复制
            # 在缺失位置设置NaN
            for col_idx, col in enumerate(knn_filled.columns):
                for row_idx in range(len(knn_filled)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        knn_filled.iloc[row_idx, col_idx] = np.nan

            try:
                # 提取数值列，避免非数值列引起的错误
                numeric_cols = knn_filled.select_dtypes(include=['float', 'int']).columns
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    
                    # 准备数据
                    data_for_knn = knn_filled[numeric_cols].values
                    if cuda:
                        # 对于大矩阵，使用GPU加速距离计算
                        data_tensor = torch.tensor(data_for_knn, dtype=torch.float32).to(device)
                        # 此处可添加KNN的GPU实现
                        knn_values = imputer.fit_transform(knn_filled[numeric_cols])
                    else:
                        knn_values = imputer.fit_transform(knn_filled[numeric_cols])
                        
                    knn_filled[numeric_cols] = knn_values
                
                # 处理可能剩下的非数值列
                for col in knn_filled.columns:
                    if knn_filled[col].isna().any():
                        knn_filled[col] = knn_filled[col].fillna(method='ffill').fillna(method='bfill')
            except Exception as e:
                print(f"  KNN填充失败: {str(e)}，使用均值填充代替")
                for col in knn_filled.columns:
                    if knn_filled[col].isna().any():
                        knn_filled[col] = knn_filled[col].fillna(knn_filled[col].mean())

            # 8. MICE填充
            mice_filled = ground_truth.copy()  # 从真值复制
            # 在缺失位置设置NaN
            for col_idx, col in enumerate(mice_filled.columns):
                for row_idx in range(len(mice_filled)):
                    if not mask_array[row_idx, col_idx]:  # 如果是缺失位置
                        mice_filled.iloc[row_idx, col_idx] = np.nan

            try:
                # 提取数值列
                numeric_cols = mice_filled.select_dtypes(include=['float', 'int']).columns
                if len(numeric_cols) > 0:
                    # 1. 保存原始数据的统计信息用于后处理
                    col_means = mice_filled[numeric_cols].mean()
                    col_stds = mice_filled[numeric_cols].std().replace(0, 1)  # 防止除零错误
                    
                    # 2. 标准化数据
                    scaled_data = (mice_filled[numeric_cols] - col_means) / col_stds
                    
                    # 3. 使用更稳健的MICE设置
                    mice_imputer = IterativeImputer(
                        max_iter=20,          # 增加迭代次数
                        random_state=0,       # 固定随机种子
                        initial_strategy='median',  # 使用中位数初始化
                        min_value=-10,        # 限制极端值
                        max_value=10          # 限制极端值
                    )
                    
                    # 4. 对标准化数据进行填充
                    mice_values = mice_imputer.fit_transform(scaled_data)
                    
                    # 5. 将结果转回原始尺度并检查极端值
                    mice_values = mice_values * col_stds.values + col_means.values
                    
                    # 6. 限制极端填充值 (使用3σ规则)
                    for i, col in enumerate(numeric_cols):
                        q1 = np.nanpercentile(mice_filled[col].values, 25)
                        q3 = np.nanpercentile(mice_filled[col].values, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        # 只对填充的缺失值应用约束
                        for row_idx in range(len(mice_filled)):
                            if not mask_array[row_idx, col_idx] and mice_values[row_idx, i] > upper_bound:
                                mice_values[row_idx, i] = upper_bound
                            elif not mask_array[row_idx, col_idx] and mice_values[row_idx, i] < lower_bound:
                                mice_values[row_idx, i] = lower_bound
                    
                    # 7. 赋值回DataFrame
                    mice_filled[numeric_cols] = mice_values
                    
                    # 8. 打印调试信息
                    print(f"  MICE填充后值范围: [{mice_filled[numeric_cols].min().min():.2f}, {mice_filled[numeric_cols].max().max():.2f}]")
                    
                # 处理可能剩下的非数值列
                for col in mice_filled.columns:
                    if mice_filled[col].isna().any():
                        mice_filled[col] = mice_filled[col].fillna(method='ffill').fillna(method='bfill')
            except Exception as e:
                print(f"  MICE填充失败: {str(e)}，使用均值填充代替")
                for col in mice_filled.columns:
                    if mice_filled[col].isna().any():
                        mice_filled[col] = mice_filled[col].fillna(mice_filled[col].mean())
            
            # 计算各方法的MSE (仅在缺失位置)
            try:
                # 转换为numpy数组
                causal_np = causal_filled.values
                preprocessed_np = missing_data.values
                zero_np = zero_filled.values
                median_np = median_filled.values
                mean_np = mean_filled.values
                ffill_np = ffill_filled.values 
                bfill_np = bfill_filled.values
                knn_np = knn_filled.values
                mice_np = mice_filled.values

                # 计算MSE (仅评估缺失位置)
                def compute_mse_with_mask(filled, truth, mask):
                    """Calculate MSE only at missing positions (where mask is False)"""
                    # Convert to numpy if needed
                    if isinstance(filled, pd.DataFrame):
                        filled_np = filled.values
                    else:
                        filled_np = filled
                        
                    if isinstance(truth, pd.DataFrame):
                        truth_np = truth.values
                    else:
                        truth_np = truth
                    
                    # Calculate squared errors only at missing positions
                    missing_mask = ~mask  # True where values were missing
                    squared_errors = ((filled_np - truth_np) ** 2)[missing_mask]
                    
                    return np.mean(squared_errors) if len(squared_errors) > 0 else float('nan')

                # 对每种方法使用同样的缺失位置计算MSE
                # 转换为numpy数组
                causal_np = causal_filled.values
                preprocessed_np = missing_data.values
                zero_np = zero_filled.values
                median_np = median_filled.values
                mean_np = mean_filled.values
                ffill_np = ffill_filled.values 
                bfill_np = bfill_filled.values
                knn_np = knn_filled.values
                mice_np = mice_filled.values
                bool_mask_np = bool_mask.values
                ground_truth_np = ground_truth.values if isinstance(ground_truth, pd.DataFrame) else ground_truth

                # 对每种方法使用同样的缺失位置计算MSE
                mse_causal = compute_mse_with_mask(causal_np, ground_truth_np, bool_mask_np)
                mse_preprocessed = compute_mse_with_mask(preprocessed_np, ground_truth_np, bool_mask_np)
                mse_zero = compute_mse_with_mask(zero_np, ground_truth_np, bool_mask_np)
                mse_median = compute_mse_with_mask(median_np, ground_truth_np, bool_mask_np)
                mse_mean = compute_mse_with_mask(mean_np, ground_truth_np, bool_mask_np)
                mse_ffill = compute_mse_with_mask(ffill_np, ground_truth_np, bool_mask_np)
                mse_bfill = compute_mse_with_mask(bfill_np, ground_truth_np, bool_mask_np)
                mse_knn = compute_mse_with_mask(knn_np, ground_truth_np, bool_mask_np)
                mse_mice = compute_mse_with_mask(mice_np, ground_truth_np, bool_mask_np)

                
                # 添加到结果列表
                all_mse["causal_tcn"].append(mse_causal)
                all_mse["preprocessed"].append(mse_preprocessed)
                all_mse["zero"].append(mse_zero)
                all_mse["median"].append(mse_median)
                all_mse["mean"].append(mse_mean)
                all_mse["ffill"].append(mse_ffill)
                all_mse["bfill"].append(mse_bfill)
                all_mse["knn"].append(mse_knn)
                all_mse["mice"].append(mse_mice)
                
                print(f"\n样本 {i} 的MSE评估结果:")
                print(f"  因果TCN: {mse_causal:.4f}")
                print(f"  预处理数据: {mse_preprocessed:.4f}")
                print(f"  零填充: {mse_zero:.4f}")
                print(f"  中位数填充: {mse_median:.4f}")
                print(f"  均值填充: {mse_mean:.4f}")
                print(f"  前向填充: {mse_ffill:.4f}")
                print(f"  后向填充: {mse_bfill:.4f}")
                print(f"  KNN填充: {mse_knn:.4f}")
                print(f"  MICE填充: {mse_mice:.4f}")
                
                # 保存第一个样本的所有填充结果
                if i == 0:
                    first_sample_results = {
                        'causal_tcn': causal_np,
                        'preprocessed': preprocessed_np,
                        'zero': zero_np,
                        'median': median_np,
                        'mean': mean_np,
                        'ffill': ffill_np,
                        'bfill': bfill_np,
                        'knn': knn_np,
                        'mice': mice_np
                    }
                    save_filled_results(eval_dataset, first_sample_results)
                
            except Exception as e:
                print(f"计算样本 {i} 的MSE时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # 清理GPU内存
            if cuda:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 计算平均MSE
    avg_results = {}
    for method, values in all_mse.items():
        if values:
            avg_results[method] = np.mean(values)
        else:
            avg_results[method] = float('nan')
    
    # 输出结果汇总
    print("\n==== 填充方法评估结果 (平均MSE) ====")
    print(f"因果TCN填充: {avg_results['causal_tcn']:.4f}")
    print(f"预处理数据: {avg_results['preprocessed']:.4f}")
    print(f"零填充: {avg_results['zero']:.4f}")
    print(f"中位数填充: {avg_results['median']:.4f}")
    print(f"均值填充: {avg_results['mean']:.4f}")
    print(f"前向填充: {avg_results['ffill']:.4f}")
    print(f"后向填充: {avg_results['bfill']:.4f}")
    print(f"KNN填充: {avg_results['knn']:.4f}")
    print(f"MICE填充: {avg_results['mice']:.4f}")
    
    # 找出最佳方法
    methods = list(avg_results.keys())
    mse_values = [avg_results[m] for m in methods]
    best_idx = np.argmin(mse_values)
    best_method = methods[best_idx]
    best_mse = mse_values[best_idx]
    
    print(f"\n最佳填充方法: {best_method}，MSE = {best_mse:.4f}")
    
    # 计算相对于我们方法的性能比较
    causal_mse = avg_results['causal_tcn']
    print("\n各方法与因果TCN方法的性能比较:")
    for method in methods:
        if method != 'causal_tcn':
            ratio = avg_results[method] / causal_mse
            diff = ((avg_results[method] - causal_mse) / avg_results[method]) * 100
            if diff > 0:
                print(f"  {method}: MSE比因果TCN方法高 {ratio:.2f}倍 (因果TCN改进了 {diff:.2f}%)")
            else:
                print(f"  {method}: MSE比因果TCN方法低 {1/ratio:.2f}倍 (因果TCN性能差 {-diff:.2f}%)")
    
    return avg_results

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LSTMModel, self).__init__()
        
        # LSTM层，输入的特征维度是 input_size, 隐藏层的大小是 hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 全连接层，将LSTM的输出映射到一个单一的输出
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 移除sigmoid激活函数，因为我们使用BCEWithLogitsLoss
    
    def forward(self, x):
        # x的形状是 [batch_size, sequence_length, input_size]
        
        # 通过LSTM层处理数据
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 使用LSTM的最后一个时间步的隐藏状态作为特征进行预测
        # lstm_out的形状是 [batch_size, sequence_length, hidden_size]
        # 选择最后一个时间步的输出
        last_hidden_state = lstm_out[:, -1, :]
        
        # 通过全连接层，不再使用Sigmoid激活
        output = self.fc(last_hidden_state)
        
        return output

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, n_clusters=10, tag_file='./static_tag.csv'):
        self.file_name = None
        self.csv_dir = csv_dir
        self.tag_file = tag_file
        self.n_clusters = n_clusters
        self.files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        self.data = []  # 存储数据
        self.labels = []  # 存储聚类标签
        self.geometric_centers = []  # 存储几何中心
        self.causal_matrices = []  # 存储每个几何中心的因果矩阵
        self.total_causal_matrix = None  # 存储最终处理后的因果矩阵
        self.filled = None  # 存储填补后的数据
        # 加载并处理所有文件
        self.tag = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._process_files()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        original = torch.tensor(sample['original'].values, dtype=torch.float32)  # 转换为 tensor
        preprocessed = torch.tensor(sample['preprocessed'].values, dtype=torch.float32)  # 转换为 tensor
        mask = torch.tensor(sample['mask'], dtype=torch.float32)  # 转换为 tensor
        
        # 获取当前样本的聚类标签
        label = sample['cluster_labels'][0]  # 假设每个样本的所有行都属于同一聚类
        
        # 使用聚类标签(索引需要减1因为聚类标签从1开始)来获取对应的因果矩阵
        causal_matrix_idx = label - 1  # 转换为0-索引
        causal_matrix = None
        if 0 <= causal_matrix_idx < len(self.causal_matrices):
            causal_matrix = self.causal_matrices[causal_matrix_idx]
            # 检查'filled'键是否存在
        if 'filled' in sample:
            filled = torch.tensor(sample['filled'].values, dtype=torch.float32)
        else:
            # 如果不存在，创建一个与original相同形状的零张量或NaN张量
            filled = torch.full_like(original, float('nan'))
        sample_tag = sample.get('tag', None)
    
        return original, preprocessed, mask, label, causal_matrix, self.total_causal_matrix, filled, sample_tag
    def _process_files(self):
        all_samples = []  # 用于存储所有预处理数据，后续计算几何中心
        # 读取标签文件 - 使用try/except以增加健壮性
        try:
            tag_df = pd.read_csv(self.tag_file)
            print(f"成功读取标签文件，包含 {len(tag_df)} 行")
        except Exception as e:
            print(f"读取标签文件出错: {str(e)}")
            print("创建空DataFrame作为替代")
            tag_df = pd.DataFrame(columns=['ICUSTAY_ID', 'DIEINHOSPITAL'])
        for file in self.files:
            filtered_rows = tag_df[tag_df['ICUSTAY_ID'].astype(str) + '.csv' == file]
            die_value = filtered_rows['DIEINHOSPITAL'].values[0]
            # 第一层预处理
            df, df_original = FirstProcess(file, self.csv_dir)
            # 创建掩码（假设缺失值为 NaN）
            mask = df.isna().astype(int).values
            # 第二层预处理 - 传入DataFrame而不是文件名
            preprocessed = SecondProcess(df) 

            # 聚类：计算时间序列的距离矩阵
            distance_matrix = pairwise_distances(preprocessed, metric='euclidean')  # 可以根据需求换成其他距离度量
            # 使用层次聚类方法 (Hierarchical Clustering)
            Z = linkage(distance_matrix, method='ward')  # 使用Ward方法进行聚类
            cluster_labels = fcluster(Z, self.n_clusters, criterion='maxclust')  # 聚成 n_clusters 类

            # 存储数据
            self.data.append({
                'original': df_original,
                'mask': mask,
                'preprocessed': preprocessed,
                'cluster_labels': cluster_labels,  # 存储每个样本的聚类标签
                'file_name': file,  # 存储文件名
                'tag': die_value  # 存储死亡标签
            })

            # 将聚类标签添加到 labels 列表中
            self.labels.extend(cluster_labels)
            all_samples.append(preprocessed)

        # 计算几何中心
        self.geometric_centers = self._calculate_geometric_centers(all_samples)

        # 对每个几何中心运行TCDF分析并保存因果矩阵
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个可用GPU设备")

        if num_gpus <= 1:
            # 单GPU或CPU模式，顺序处理
            for center in self.geometric_centers:
                # 检查center是什么类型，并相应地处理
                if isinstance(center, pd.DataFrame):
                    # 如果是DataFrame，使用.values转换为numpy数组
                    center_tensor = torch.tensor(center.values, dtype=torch.float32, device=self.device)
                elif isinstance(center, np.ndarray):
                    # 如果已经是numpy数组
                    center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
                else:
                    print(f"警告：未知的中心类型 {type(center)}，跳过因果分析")
                    self.causal_matrices.append(None)
                    continue
            
                # 使用转换后的张量运行TCDF分析
                cm = run_tcdf_for_data_gpu(center_tensor)  # 每个几何中心当作时间序列处理
                if cm is not None:
                    self.causal_matrices.append(cm)
                else:
                    self.causal_matrices.append(None)
        else:
            # 多GPU模式，并行处理
            # 将几何中心分成num_gpus组
            center_groups = [[] for _ in range(num_gpus)]
            for i, center in enumerate(self.geometric_centers):
                group_idx = i % num_gpus
                center_groups[group_idx].append((i, center))
            
            # 定义每个GPU上的处理函数
            def process_centers_on_gpu(gpu_id, centers):
                results = []
                torch.cuda.set_device(gpu_id)  # 设置当前GPU设备
                for idx, center in centers:
                    try:
                        # 检查center类型
                        if isinstance(center, pd.DataFrame):
                            center_tensor = torch.tensor(center.values, dtype=torch.float32, device=f'cuda:{gpu_id}')
                        elif isinstance(center, np.ndarray):
                            center_tensor = torch.tensor(center, dtype=torch.float32, device=f'cuda:{gpu_id}')
                        else:
                            print(f"警告：未知的中心类型 {type(center)}，跳过因果分析")
                            results.append((idx, None))
                            continue
                        
                        # 使用转换后的张量运行TCDF分析
                        cm = run_tcdf_for_data_gpu(center_tensor)
                        results.append((idx, cm))
                    except Exception as e:
                        print(f"GPU {gpu_id} 处理中心 {idx} 时出错: {str(e)}")
                        results.append((idx, None))
                return results
            
            # 并行执行每个GPU上的任务
            self.causal_matrices = [None] * len(self.geometric_centers)  # 预分配空间
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                # 提交任务
                futures = []
                for gpu_id in range(num_gpus):
                    future = executor.submit(process_centers_on_gpu, gpu_id, center_groups[gpu_id])
                    futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    for idx, cm in future.result():
                        self.causal_matrices[idx] = cm
            
            print(f"已完成 {len(self.geometric_centers)} 个几何中心的并行因果分析")
        
    def process_causal_matrices(self, causal_matrices):
        """将每个几何中心的因果矩阵累加并二值化（Top‑3 或全1）"""
        # 1. 转成 numpy 数组列表
        mats = []
        for cm in causal_matrices:
            if isinstance(cm, pd.DataFrame):
                mats.append(cm.values)
            elif isinstance(cm, np.ndarray):
                mats.append(cm)
        if not mats:
            return None

        # 2. 累加
        total = np.zeros_like(mats[0], dtype=float)
        for m in mats:
            total += m

        # 3. 每列二值化：非零元素 <3 则全置1，否则取前三大置1
        rows, cols = total.shape
        for c in range(cols):
            idxs = np.nonzero(total[:, c])[0]
            if len(idxs) < 3:
                total[:, c] = 1
            else:
                top3 = np.argsort(total[:, c])[-3:]
                col_mask = np.zeros(rows, dtype=float)
                col_mask[top3] = 1
                total[:, c] = col_mask

        return total

    def _calculate_geometric_centers(self, all_samples):
        geometric_centers = []
        
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_samples = []
            cluster_indices = []  # 存储样本索引和文件索引
            
            # 处理每个文件
            for file_idx, file_data in enumerate(self.data):
                preprocessed_data = file_data['preprocessed']
                cluster_labels = file_data['cluster_labels']
                
                # 找到当前文件中属于此类别的所有行
                for row_idx, label in enumerate(cluster_labels):
                    if label == cluster_id:
                        # 添加此行到集群样本中
                        if hasattr(preprocessed_data, 'iloc'):  # DataFrame
                            cluster_samples.append(preprocessed_data.iloc[row_idx].values)
                        else:  # numpy array
                            cluster_samples.append(preprocessed_data[row_idx])
                        cluster_indices.append((file_idx, row_idx))
            
            if cluster_samples:
                cluster_samples_array = np.array(cluster_samples)
                
                # 计算均值中心
                mean_center = np.mean(cluster_samples_array, axis=0)
                
                # 找到距离中心最近的真实样本
                distances = [np.linalg.norm(sample - mean_center) for sample in cluster_samples_array]
                closest_idx = np.argmin(distances)
                
                # 获取距离中心最近的原始二维数据
                file_idx, row_idx = cluster_indices[closest_idx]
                representative_sample = self.data[file_idx]['preprocessed']
                
                # 添加代表性样本作为中心
                geometric_centers.append(representative_sample)
            else:
                print(f"警告：聚类 {cluster_id} 没有样本")
                if len(self.data) > 0:
                    # 使用第一个文件的空结构作为占位符
                    dummy = pd.DataFrame(0, index=range(self.data[0]['preprocessed'].shape[0]), 
                                        columns=self.data[0]['preprocessed'].columns)
                    geometric_centers.append(dummy)
                else:
                    geometric_centers.append(None)
        
        return geometric_centers

    def fill(self, cuda=False, epoch=100):
        """填充数据中的缺失值，使用因果发现和TCN预测，无需临时文件"""
        # 检测可用的GPU数量
        num_gpus = torch.cuda.device_count() if cuda else 1
        print(f"检测到 {num_gpus} 个可用GPU设备用于填充处理")
        
        if num_gpus <= 1:
            # 单GPU或CPU模式，顺序处理
            for i in range(len(self.data)):
                original, preprocessed, mask, label, causal_matrix, total_matrix, filled, tag = self[i]
                
                # 获取原始数据的列名
                original_df = self.data[i]['original']
                cols = original_df.columns
                
                # 直接将预处理数据转为DataFrame供findpredictions使用
                temp_df = pd.DataFrame(preprocessed.cpu().numpy(), columns=cols)
                
                # 遍历每个特征（使用索引）
                for j in range(preprocessed.shape[1]):
                    feature_name = cols[j]  # 使用列名而不是张量值
                    print(f'处理特征: {feature_name}')
                    
                    feature_list = []
                    if total_matrix is not None and j < total_matrix.shape[1]:
                        # 找到total_matrix中对应列中值为1的行索引
                        causal_rows = np.where(total_matrix[:, j] == 1)[0]
                        # 将这些行索引转换为相应的特征名称
                        feature_list = [cols[idx] for idx in causal_rows if idx < len(cols)]
                        # 确保目标特征也在列表中
                        if feature_name not in feature_list:
                            feature_list.append(feature_name)
                        
                        print(f'  使用因果相关特征: {feature_list}')
                    else:
                        print(f'  未找到因果相关特征，使用所有特征')
                    
                    mask_np = mask.cpu().numpy()
                    mask_feature = mask_np[:, j] == 1 if j < mask_np.shape[1] else []
                    
                    try:
                        # 直接传递DataFrame给findpredictions，不创建临时文件
                        pred = findpredictions(
                            target=feature_name,
                            cuda=cuda,
                            epochs=epoch,
                            kernel_size=6,
                            layers=3,
                            log_interval=1,
                            lr=0.02,
                            optimizername='Adam',
                            seed=1111,
                            dilation_c=4,
                            significance=0.8,
                            file=None,  # 不使用文件
                            feature_list=feature_list,
                            df=temp_df  # 直接传递DataFrame
                        )
                        
                        if pred is not None and len(pred) > 0:
                            for idx in range(len(temp_df)):
                                if not mask_feature[idx] and idx < len(pred):
                                    # 填充预测值 - 四舍五入到1位小数
                                    filled[idx, j] = round(float(pred[idx][0]), 1)
                                else:
                                    filled[idx, j] = preprocessed[idx, j]
                    except Exception as e:
                        print(f'处理特征 {feature_name} 时出错: {str(e)}')
                        continue
                
                self.data[i]['filled'] = filled
                print(f"完成样本 {i} 的填充处理")
        else:
            # 多GPU模式，并行处理
            # 将数据分成num_gpus组
            data_groups = [[] for _ in range(num_gpus)]
            for i in range(len(self.data)):
                group_idx = i % num_gpus
                data_groups[group_idx].append(i)
            
            # 定义每个GPU上的处理函数
            def process_data_on_gpu(gpu_id, data_indices):
                torch.cuda.set_device(gpu_id)  # 设置当前GPU设备
                results = {}
                
                for i in data_indices:
                    try:
                        original, preprocessed, mask, label, causal_matrix, total_matrix, filled, tag = self[i]
                        
                        # 获取原始数据的列名
                        original_df = self.data[i]['original']
                        cols = original_df.columns
                        
                        # 直接将预处理数据转为DataFrame
                        temp_df = pd.DataFrame(preprocessed.cpu().numpy(), columns=cols)
                        
                        # 遍历每个特征（使用索引）
                        for j in range(preprocessed.shape[1]):
                            feature_name = cols[j]
                            print(f'GPU {gpu_id} 处理样本 {i}, 特征: {feature_name}')
                            
                            feature_list = []
                            if total_matrix is not None and j < total_matrix.shape[1]:
                                causal_rows = np.where(total_matrix[:, j] == 1)[0]
                                feature_list = [cols[idx] for idx in causal_rows if idx < len(cols)]
                                if feature_name not in feature_list:
                                    feature_list.append(feature_name)
                            
                            mask_np = mask.cpu().numpy()
                            mask_feature = mask_np[:, j] == 1 if j < mask_np.shape[1] else []
                            
                            try:
                                # 直接传递DataFrame，不创建临时文件
                                pred = findpredictions(
                                    target=feature_name,
                                    cuda=True,  # 在GPU上运行
                                    epochs=epoch,
                                    kernel_size=6,
                                    layers=3,
                                    log_interval=1,
                                    lr=0.02,
                                    optimizername='Adam',
                                    seed=1111,
                                    dilation_c=4,
                                    significance=0.8,
                                    file=None,  # 不使用文件
                                    feature_list=feature_list,
                                    df=temp_df  # 直接传递DataFrame
                                )
                                
                                if pred is not None and len(pred) > 0:
                                    for idx in range(len(temp_df)):
                                        if not mask_feature[idx] and idx < len(pred):
                                            filled[idx, j] = round(float(pred[idx][0]), 1)
                                        else:
                                            filled[idx, j] = preprocessed[idx, j]
                            except Exception as e:
                                print(f'GPU {gpu_id} 处理特征 {feature_name} 时出错: {str(e)}')
                                continue
                        
                        # 保存结果
                        results[i] = filled.clone()
                        print(f"GPU {gpu_id} 完成样本 {i} 的填充处理")
                            
                    except Exception as e:
                        print(f"GPU {gpu_id} 处理样本 {i} 时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                return results
            
            # 并行执行每个GPU上的任务
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                # 提交任务
                futures = []
                for gpu_id in range(num_gpus):
                    future = executor.submit(process_data_on_gpu, gpu_id, data_groups[gpu_id])
                    futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    for i, filled_data in results.items():
                        self.data[i]['filled'] = filled_data
            
            print(f"已完成 {len(self.data)} 个样本的并行填充处理")
    
    def evaluate_preparedata(self, point_missing_rate=0.1, block_missing_rate=0.6, 
                        min_block_height=1, min_block_width=1,
                        max_block_height=10, max_block_width=5):
        """
        准备带有人工缺失的数据进行评估
        
        Args:
            point_missing_rate: 随机点缺失比例
            block_missing_rate: 随机块缺失比例
            min_block_height: 随机缺失块的最小高度
            min_block_width: 随机缺失块的最小宽度
            max_block_height: 随机缺失块的最大高度
            max_block_width: 随机缺失块的最大宽度
        """
        # 创建一个新的 CustomDataset
        new_dataset = CustomDataset(self.csv_dir, self.n_clusters)
        new_dataset.data = []  # 清空数据，准备添加新的处理过的数据
        
        # 确保最小值不超过最大值
        min_block_height = max(1, min(min_block_height, max_block_height))
        min_block_width = max(1, min(min_block_width, max_block_width))
        
        # 遍历原始数据集中的每个样本
        for i in range(len(self.data)):
            try:
                # 获取填充后的数据或预处理数据
                if 'filled' in self.data[i] and self.data[i]['filled'] is not None:
                    if isinstance(self.data[i]['filled'], torch.Tensor):
                        # 张量转NumPy再转DataFrame
                        filled_values = self.data[i]['filled'].cpu().numpy()
                        filled_df = pd.DataFrame(filled_values, columns=self.data[i]['original'].columns)
                    elif isinstance(self.data[i]['filled'], pd.DataFrame):
                        filled_df = self.data[i]['filled'].copy()
                    else:
                        # 如果类型未知，使用原始数据
                        filled_df = self.data[i]['original'].copy()
                else:
                    # 如果没有填充数据，使用预处理数据
                    filled_df = self.data[i]['preprocessed'].copy()
                    
                # 创建掩码矩阵（布尔类型的True表示不缺失）
                rows, cols = filled_df.shape
                bool_mask = np.ones((rows, cols), dtype=bool)
                
                # 1. 应用随机点缺失
                point_missing = np.random.rand(rows, cols) < point_missing_rate
                bool_mask[point_missing] = False  # 设置为False表示缺失
                
                # 2. 应用随机块缺失
                actual_min_block_height = min(min_block_height, rows)
                actual_min_block_width = min(min_block_width, cols)
                
                num_blocks = max(1, int(rows * cols * block_missing_rate / (max_block_height * max_block_width / 4)))
                
                for _ in range(num_blocks):
                    # 限制起始位置，确保有空间容纳最小块
                    max_start_row = rows - actual_min_block_height + 1
                    max_start_col = cols - actual_min_block_width + 1
                    
                    if max_start_row <= 0 or max_start_col <= 0:
                        print(f"  警告: 矩阵维度({rows}x{cols})不足以容纳最小块尺寸({actual_min_block_height}x{actual_min_block_width})")
                        continue  # 跳过此块
                        
                    # 随机选择块的起始位置(确保有足够空间)
                    start_row = np.random.randint(0, max_start_row)
                    start_col = np.random.randint(0, max_start_col)
                    
                    # 随机选择块的高度和宽度
                    height_space = rows - start_row
                    width_space = cols - start_col
                    
                    # 确保高度和宽度范围有效
                    max_possible_height = min(max_block_height, height_space)
                    max_possible_width = min(max_block_width, width_space)
                    
                    if max_possible_height <= actual_min_block_height:
                        block_height = max_possible_height
                    else:
                        block_height = np.random.randint(actual_min_block_height, max_possible_height + 1)
                        
                    if max_possible_width <= actual_min_block_width:
                        block_width = max_possible_width
                    else:
                        block_width = np.random.randint(actual_min_block_width, max_possible_width + 1)
                    
                    # 将块内的掩码设为False（表示缺失）
                    end_row = min(start_row + block_height, rows)
                    end_col = min(start_col + block_width, cols)
                    bool_mask[start_row:end_row, start_col:end_col] = False
                
                # 创建带有缺失值的数据
                missing_df = filled_df.copy()
                missing_df = missing_df.mask(~bool_mask)  # 使用布尔掩码快速设置缺失值
                
                # 构建新的数据条目 - 简化版，只保留原始和缺失数据
                new_item = {
                    'original': filled_df,  # 保存原始完整数据（无缺失）用于评估
                    'preprocessed': missing_df,  # 保存带有人工缺失的数据用于填充测试
                    'mask': (~bool_mask).astype(int),  # 保存正确的掩码
                    'cluster_labels': self.data[i]['cluster_labels'],
                    'file_name': self.data[i]['file_name']  # 添加文件名
                }
                
                # 添加到新数据集
                new_dataset.data.append(new_item)
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue  # 跳过此样本
        
        # 复制必要的属性
        new_dataset.labels = self.labels.copy() if hasattr(self, 'labels') else []
        new_dataset.geometric_centers = self.geometric_centers.copy() if hasattr(self, 'geometric_centers') else []
        new_dataset.causal_matrices = self.causal_matrices.copy() if hasattr(self, 'causal_matrices') else []
        new_dataset.total_causal_matrix = self.total_causal_matrix
        
        print(f"已创建带有人工缺失的评估数据集，包含 {len(new_dataset.data)} 个样本")
        print(f"点缺失率: {point_missing_rate:.1%}, 块缺失率: {block_missing_rate:.1%}")
        print(f"块尺寸范围: {min_block_height}-{max_block_height} x {min_block_width}-{max_block_width}")
        
        return new_dataset
    def evaluate_imputation_downstream(self, cuda=False, epochs=10):
        x_tcn = []
        x_zero = []
        x_median = []
        x_mean = []
        x_ffill = []
        x_bfill = []
        x_knn = []
        x_mice = []
        y_true = []
        for i in range(len(self.data)):
            filled_tcn = self.data[i]['filled']
            filled_zero = self.data[i]['original'].fillna(0)
            filled_median = self.data[i]['original'].fillna(self.data[i]['original'].median())
            filled_mean = self.data[i]['original'].fillna(self.data[i]['original'].mean())
            filled_ffill = self.data[i]['original'].fillna(method='ffill').fillna(method='bfill')  # 先前向后后向
            filled_bfill = self.data[i]['original'].fillna(method='bfill').fillna(method='ffill') 

            # 替换原有的KNN填充代码
            preprocessed_data = self.data[i]['original']

            # 1. 检查每列，对全空列填充-1
            empty_cols = preprocessed_data.columns[preprocessed_data.isna().all()]
            non_empty_cols = preprocessed_data.columns[~preprocessed_data.isna().all()]

            # 2. 创建结果DataFrame
            filled_knn = preprocessed_data.copy()

            # 3. 对全空列填充-1
            if len(empty_cols) > 0:
                filled_knn[empty_cols] = -1

            # 4. 对非空列使用KNN填充
            if len(non_empty_cols) > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                filled_knn_values = knn_imputer.fit_transform(preprocessed_data[non_empty_cols])
                filled_knn[non_empty_cols] = filled_knn_values

            # 替换原有的MICE填充代码
            preprocessed_data2 = self.data[i]['original']

            # 1. 检查每列，对全空列填充-1
            empty_cols2 = preprocessed_data2.columns[preprocessed_data2.isna().all()]
            non_empty_cols2 = preprocessed_data2.columns[~preprocessed_data2.isna().all()]

            # 2. 创建结果DataFrame
            filled_mice = preprocessed_data2.copy()

            # 3. 对全空列填充-1
            if len(empty_cols2) > 0:
                filled_mice[empty_cols2] = -1

            # 4. 对非空列使用MICE填充
            if len(non_empty_cols2) > 0:
                # 创建MICE填充器
                mice_imputer = IterativeImputer(max_iter=10, random_state=0)
                # 只对非空列进行填充
                filled_mice_values = mice_imputer.fit_transform(preprocessed_data2[non_empty_cols2])
                # 将结果赋值回对应列
                filled_mice[non_empty_cols2] = filled_mice_values
            
            x_tcn.append(filled_tcn)
            x_zero.append(filled_zero)
            x_median.append(filled_median)
            x_mean.append(filled_mean)
            x_ffill.append(filled_ffill)
            x_bfill.append(filled_bfill)
            x_knn.append(filled_knn)
            x_mice.append(filled_mice)
            y_true.append(self.data[i]['tag'])
        # 分别评估每种填充方法
        fill_methods = {
            'TCN': x_tcn,
            'Zero': x_zero,
            'Median': x_median,
            'Mean': x_mean,
            'FFill': x_ffill,
            'BFill': x_bfill,
            'KNN': x_knn,
            'MICE': x_mice
        }

            # 存储所有方法的评估结果
        all_results = {}
        
        for method_name, method_data in fill_methods.items():
            print(f"\n===== 评估 {method_name} 填充方法 =====")
            # 转换为NumPy数组
            method_data_np = np.array([data.values if isinstance(data, pd.DataFrame) 
                                    else data.detach().cpu().numpy() if isinstance(data, torch.Tensor) 
                                    else data for data in method_data])
            
            metrics = cross_validate_lstm(
                X=method_data_np,
                y=np.array(y_true),
                model_class=LSTMModel,
                input_size=self.data[0]['original'].shape[1],  # 使用第一个样本确保尺寸一致
                hidden_size=64,  # LSTM隐藏层大小
                num_epochs=epochs
            )
            
            # 计算平均指标并存储
            if metrics:
                method_avg_metrics = {}
                # 修改此处，添加'f1'和'auprc'
                for metric in ['accuracy', 'recall', 'precision', 'fpr', 'auroc', 'f1', 'auprc']:
                    avg = np.mean([fold[metric] for fold in metrics])
                    method_avg_metrics[metric] = avg
                    print(f"{metric}: {avg:.4f}")
                all_results[method_name] = method_avg_metrics
        
        # 打印汇总表格
        print("\n\n====================================================================")
        print("                      下游任务预测性能对比表                           ")
        print("====================================================================")
        
        # 表头
        headers = ["方法", "准确率", "召回率", "精确率", "F1分数", "假正例率", "AUROC", "AUPRC"]
        format_row = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}"
        
        print(format_row.format(*headers))
        print("-" * 85)
        
        # 表格内容
        for method, metrics in all_results.items():
            row = [
                method, 
                f"{metrics.get('accuracy', 0):.4f}", 
                f"{metrics.get('recall', 0):.4f}", 
                f"{metrics.get('precision', 0):.4f}", 
                f"{metrics.get('f1', 0):.4f}",           # 添加F1-score
                f"{metrics.get('fpr', 0):.4f}", 
                f"{metrics.get('auroc', 0):.4f}",
                f"{metrics.get('auprc', 0):.4f}"         # 添加AUPRC
            ]
            print(format_row.format(*row))
        
        print("====================================================================")
        
        # 找出每个指标的最佳方法
        best_methods = {}
        for metric in ['accuracy', 'recall', 'precision', 'f1', 'auroc', 'auprc']:  # 添加f1和auprc
            best_method = max(all_results.items(), key=lambda x: x[1].get(metric, 0))[0]
            best_methods[metric] = best_method
        
        # 找出假正例率最低的方法
        best_methods['fpr'] = min(all_results.items(), key=lambda x: x[1].get('fpr', 1))[0]
        
        print("\n各指标最佳方法:")
        print(f"准确率最高: {best_methods['accuracy']}")
        print(f"召回率最高: {best_methods['recall']}")
        print(f"精确率最高: {best_methods['precision']}")
        print(f"F1分数最高: {best_methods['f1']}")        # 添加F1-score
        print(f"假正例率最低: {best_methods['fpr']}")
        print(f"AUROC最高: {best_methods['auroc']}")
        print(f"AUPRC最高: {best_methods['auprc']}")      # 添加AUPRC

def cross_validate_lstm(X, y, model_class, input_size, hidden_size, num_epochs=10, batch_size=32, k=5, learning_rate=0.001):
    """
    使用K折交叉验证评估LSTM模型在填充数据上的性能。
    
    Args:
        X: 特征数据 (样本数, 序列长度, 特征数)
        y: 标签数据 (样本数,)
        model_class: 模型类，例如LSTMModel
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层大小
        num_epochs: 训练轮数
        batch_size: 批量大小
        k: 交叉验证折数
        learning_rate: 学习率
        
    Returns:
        包含每折评估指标的列表
    """
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            pt = torch.exp(-bce_loss)  # pt = p if y=1, pt = 1-p if y=0
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss)
            return focal_loss.mean()
    # 确保输入数据格式正确
    if len(X) == 0:
        print("错误: 输入数据为空")
        return []
    
    if len(X) != len(y):
        print(f"错误: 特征数据({len(X)})和标签数据({len(y)})长度不匹配")
        return []
    
    # 检查是否有足够的样本进行k折交叉验证
    if len(X) < k:
        print(f"警告: 样本数({len(X)})小于折数({k})，调整为留一法")
        k = min(len(X), 3)  # 最少使用3折或样本数
        
    # 初始化K折交叉验证器
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_all_folds = []
    
    try:
        # 对每一折进行训练和评估
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold+1}/{k} ===")
            
            # 划分训练集和测试集
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 检查训练数据和测试数据的形状
            print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
            
            try:
                # 转换为PyTorch张量
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
                
                # 创建数据加载器
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)
                
                                # 实例化模型
                model = model_class(input_size=input_size, hidden_size=hidden_size, output_size=1)
                
                # 计算类别权重来处理不平衡问题
                pos_count = np.sum(y_train == 1)
                neg_count = np.sum(y_train == 0)
                total = pos_count + neg_count
                # 确保没有零除错误
                if pos_count == 0:
                    pos_weight = torch.tensor([1.0])
                else:
                    # 根据反比例设置权重，少数类权重更高
                    pos_weight = torch.tensor([neg_count / max(1, pos_count)])
                    print(f"类别权重 - 正例: {pos_weight.item():.2f}")
                
                # 使用带权重的BCE损失
                #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                # 调整Focal Loss参数，限制最大不平衡率并增加gamma值
                alpha = min(0.75, neg_count / (pos_count + 1))  # 限制最大不平衡率，防止极端值
                gamma = 2.5  # 增加gamma值，更强调难分类样本
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                # 训练模型
                model.train()
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # 每隔几轮输出训练进度
                    if (epoch+1) % max(1, num_epochs//5) == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
                
                # 验证模型
                model.eval()
                with torch.no_grad():
                    # 获取输出logits，确保保持维度
                    logits = model(X_test_tensor)
                    
                    # 确保logits是二维的 [samples, 1]
                    if len(logits.shape) == 1:
                        logits = logits.unsqueeze(1)
                    
                    # 应用sigmoid获取概率
                    y_pred_probs = torch.sigmoid(logits).cpu().numpy()
                    
                    # 确保y_pred_probs是数组而不是标量
                    if y_pred_probs.ndim == 0:
                        y_pred_probs = np.array([y_pred_probs])
                    
                    # 确保形状正确
                    y_pred_probs = y_pred_probs.reshape(-1)
                    
                    # 大于0.5的判断为正类，否则为负类
                                        # 降低决策阈值提高对少数类的敏感性(0.2-0.3效果较好)
                    decision_threshold = 0.25
                    print(f"使用决策阈值: {decision_threshold}")
                    y_pred_labels = (y_pred_probs > decision_threshold).astype(int)
                    
                    # 确保y_true也是扁平数组
                    y_true = y_test_tensor.cpu().numpy().reshape(-1)
                    
                    # 打印调试信息
                    print(f"预测标签形状: {y_pred_labels.shape}, 真实标签形状: {y_true.shape}")
                    print(f"预测标签: {y_pred_labels}")
                    print(f"真实标签: {y_true}")
                    
                    # 计算评估指标
                    acc = accuracy_score(y_true, y_pred_labels)
                    
                    # 处理边缘情况
                    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred_labels)) <= 1:
                        print("警告: 数据或预测中只有一个类别，某些指标可能不准确")
                        recall = 1.0 if np.sum(y_true) == np.sum(y_pred_labels) else 0.0
                        precision = 1.0 if np.sum(y_true) == np.sum(y_pred_labels) else 0.0
                        fpr = 0.0
                        auroc = 0.5
                        f1 = 0.0
                        auprc = 0.5
                    else:
                        # 计算其他指标
                        recall = recall_score(y_true, y_pred_labels)
                        precision = precision_score(y_true, y_pred_labels)
                        
                        # 计算假正例率
                        cm = confusion_matrix(y_true, y_pred_labels)
                        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) and (cm[0, 0] + cm[0, 1]) > 0 else 0.0
                        
                        # 计算ROC曲线下面积
                        auroc = roc_auc_score(y_true, y_pred_probs)

                        # 添加F1-score计算
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        # 添加AUPRC (Area Under Precision-Recall Curve)计算
                        from sklearn.metrics import precision_recall_curve, auc
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
                        auprc = auc(recall_curve, precision_curve)
                        
                    metrics_all_folds.append({
                        'accuracy': acc,
                        'recall': recall,
                        'precision': precision,
                        'fpr': fpr,
                        'auroc': auroc,
                        'f1': f1,           # 添加F1-score
                        'auprc': auprc 
                    })
                    
                    print(f"测试集性能 - Accuracy: {acc:.4f}, Recall: {recall:.4f}, "
                          f"Precision: {precision:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}, "
                          f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
                        
            except Exception as e:
                print(f"处理第 {fold+1} 折时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"交叉验证过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 如果所有折都失败，返回空列表
    if not metrics_all_folds:
        print("警告: 所有折都处理失败")
        return []
    
    return metrics_all_folds