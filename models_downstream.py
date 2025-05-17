import torch
import torch.nn as nn
import numpy as np
import os
import torch
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from models_dataset import MyDataset
import warnings
warnings.filterwarnings("ignore")

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 取最后时刻的隐藏状态
        out = self.fc(last_hidden)
        return self.sigmoid(out)


def preprocess_input(x_np):
    """
    输入 x_np 是 shape=[seq_len, num_features] 的 numpy 数组。
    返回 torch.Tensor，shape=[1, seq_len, num_features]（添加 batch 维）
    并做标准化。
    """
    # 标准化（对每列做 z-score）
    mean = np.mean(x_np, axis=0)
    std = np.std(x_np, axis=0)
    std[std == 0] = 1  # 避免除0
    x_norm = (x_np - mean) / std

    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, num_features]
    return x_tensor

def fill_with_method(data, mask, method):
    """使用特定方法填充缺失值"""
    filled = data.copy()
    
    if method == 'zero':
        filled[mask == 0] = 0
    elif method == 'mean':
        for j in range(data.shape[1]):
            mean_val = np.nanmean(np.where(mask[:, j] == 1, data[:, j], np.nan))
            if np.isnan(mean_val): mean_val = 0
            filled[mask[:, j] == 0, j] = mean_val
    elif method == 'median':
        for j in range(data.shape[1]):
            median_val = np.nanmedian(np.where(mask[:, j] == 1, data[:, j], np.nan))
            if np.isnan(median_val): median_val = 0
            filled[mask[:, j] == 0, j] = median_val
    elif method in ['bfill', 'ffill']:
        df = pd.DataFrame(data)
        df_mask = pd.DataFrame(mask)
        df[df_mask == 0] = np.nan
        if method == 'bfill':
            filled = df.bfill().ffill().values
        else:
            filled = df.ffill().bfill().values
    elif method == 'knn':
        filled_temp = data.copy()
        filled_temp[mask == 0] = np.nan
        imputer = KNNImputer(n_neighbors=5)
        filled = imputer.fit_transform(filled_temp)
    elif method == 'mice':
        # 修改的MICE填充方法
        filled_temp = data.copy()
        # 先用均值填充NaN，这样IterativeImputer就不会遇到NaN
        simple_imputer = SimpleImputer(strategy='mean')
        initial_fill = simple_imputer.fit_transform(np.where(mask == 1, data, np.nan))
        # 然后用MICE细化填充结果
        mice_imputer = IterativeImputer(max_iter=10, random_state=0, 
                                      skip_complete=True)
        filled = mice_imputer.fit_transform(initial_fill)
    elif method == 'model':
        process_single_matrix(args)
    
    return filled

def evaluate_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim=64, epochs=30, lr=0.001, device='cuda'):
    """训练并评估LSTM模型"""
    # 初始化模型、优化器和损失函数
    model = SimpleLSTMClassifier(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    # 训练模型
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]
            
            x_tensor = preprocess_input(x).to(device)
            y_tensor = torch.tensor([[y_true]], dtype=torch.float32).to(device)
            
            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train):.4f}')
    
    # 评估模型
    model.eval()
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for x in X_test:
            x_tensor = preprocess_input(x).to(device)
            output = model(x_tensor)
            y_prob.append(output.item())
            y_pred.append(1 if output.item() >= 0.5 else 0)
    
    # 计算指标
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    
    return f1, auroc

def evaluate_filling_method(method_name, filled_data, labels, k_folds=4, device='cuda'):
    """使用K折交叉验证评估特定填充方法"""
    print(f"\n评估填充方法: {method_name} 在设备 {device} 上")
    
    # 过滤有效样本
    valid_indices = []
    for i, (data, label) in enumerate(zip(filled_data, labels)):
        if data is not None and label is not None:
            valid_indices.append(i)
    
    X = [filled_data[i] for i in valid_indices]
    y = [labels[i] for i in valid_indices]
    y = np.array(y, dtype=np.float32)
    
    # K折交叉验证
    kf = KFold(n_splits=min(k_folds, len(y)), shuffle=True, random_state=42)
    
    f1_scores = []
    auroc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(X)))):
        print(f"  执行第 {fold+1}/{min(k_folds, len(y))} 折...")
        X_train = [X[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = y[test_idx]
        
        input_dim = X[0].shape[1]  # 特征维度
        f1, auroc = evaluate_model(X_train, y_train, X_test, y_test, input_dim, device=device)
        f1_scores.append(f1)
        auroc_scores.append(auroc)
    
    avg_f1 = np.mean(f1_scores)
    avg_auroc = np.mean(auroc_scores)
    print(f"  {method_name} 评估完成: 平均F1={avg_f1:.4f}, 平均AUROC={avg_auroc:.4f}")
    
    return {'f1': avg_f1, 'auroc': avg_auroc}

def process_method(args):
    """单个进程内评估填充方法的包装函数"""
    method, data, mask, labels, device, k_folds, return_dict = args
    # 设置GPU环境
    if device != 'cpu':
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = 'cuda:0'  # 重置为0，因为此进程只能看到一个GPU
        
    if method == 'model':
        # 直接使用final_filled
        filled_data = data
    else:
        # 对每个样本应用填充方法
        filled_data = []
        for j in range(len(data)):
            filled = fill_with_method(data[j], mask[j], method)
            filled_data.append(filled)
        
    result = evaluate_filling_method(method, filled_data, labels, k_folds, device)
    return_dict[method] = result

def evaluate_downstream_methods(dataset, k_folds=4):
    """评估不同填充方法对下游分类任务的影响"""
    print("\n" + "="*50)
    print("开始下游任务评估: LSTM 二分类任务")
    
    # 快速获取数据集信息
    valid_labels = [l for l in dataset.labels if l is not None]
    print(f"数据集：{len(dataset)}样本，{len(valid_labels)}个有标签 "
          f"({sum(valid_labels)}正例/{len(valid_labels)-sum(valid_labels)}负例)")
    
    # 设置设备和方法
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
    print(f"检测到 {len(devices)} 个计算设备")
    methods = ['zero', 'mean', 'median', 'bfill', 'ffill', 'knn', 'mice', 'model']
    
    # 并行评估所有方法
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    
    # 创建并启动进程
    for i, method in enumerate(methods):
        device = devices[i % len(devices)]
        mask = dataset.mask_data
        if method == 'model':
            data = dataset.final_filled  # 对final/model方法用最终填充数据
        else:
            data = dataset.initial_filled  # 对基准方法用初始数据
        p = multiprocessing.Process(
            target=process_method, 
            args=((method, data, mask, dataset.labels, device, k_folds, return_dict),)
        )
        processes.append(p)
        p.start()
    
    # 等待完成并打印结果
    for p in processes:
        p.join()
    
    # 打印结果表格
    results = dict(return_dict)
    print("\n===== 填充方法性能比较 =====")
    print(f"{'方法':<10}{'F1分数':<15}{'AUROC分数':<15}")
    print("-" * 40)
    
    for method in methods:
        if method in results:
            print(f"{method:<10}{results[method]['f1']:.4f}{'':<10}{results[method]['auroc']:.4f}")
    
    # 添加保存结果到文件的代码
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建结果文件
    results_file = os.path.join(results_dir, 'downstream_comparison.txt')
    with open(results_file, 'w') as f:
        f.write("===== 填充方法性能比较 =====\n")
        f.write(f"{'方法':<10}{'F1分数':<15}{'AUROC分数':<15}\n")
        f.write("-" * 40 + "\n")
        
        for method in methods:
            if method in results:
                f.write(f"{method:<10}{results[method]['f1']:.4f}{'':<10}{results[method]['auroc']:.4f}\n")
    
    # 同时保存为CSV格式便于后续分析
    df_results = pd.DataFrame([
        {'method': method, 'f1': results[method]['f1'], 'auroc': results[method]['auroc']}
        for method in methods if method in results
    ])
    csv_file = os.path.join(results_dir, 'downstream_comparison.csv')
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n结果已保存到 {results_file} 和 {csv_file}")
    
    return results