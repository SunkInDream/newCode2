import os
import random
import numpy as np
import pandas as pd
from models_TCDF import *

def FirstProcess(matrix, threshold=0.8):
    df = pd.DataFrame(matrix)
    for column in df.columns:
        col_data = df[column]
        if col_data.isna().all():
            df[column] = -1
        else:
            non_nan_data = col_data.dropna()
            value_counts = non_nan_data.value_counts()
            mode_value = value_counts.index[0]
            mode_count = value_counts.iloc[0]
            if mode_count >= threshold * len(non_nan_data):
                df[column] = col_data.fillna(mode_value)
    return df


def SecondProcess(df, perturbation_prob=0.1, perturbation_scale=0.1):
    df_copy = df.copy()
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

def initial_process(matrix, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    df = pd.DataFrame(matrix)
    df = FirstProcess(df, threshold)
    df = SecondProcess(df, perturbation_prob, perturbation_scale)
    return df.values.astype(np.float32)

