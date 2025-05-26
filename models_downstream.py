import torch
import torch.nn as nn
import numpy as np
import os
import torch
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from models_dataset import *
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
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

class MatrixDataset(Dataset):
    def __init__(self, matrices, labels):
        self.matrices = matrices  # list of [seq_len, input_dim] tensors or arrays
        self.labels = labels      # list of 0/1

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        x = torch.tensor(self.matrices[idx], dtype=torch.float32)  # [seq_len, input_dim]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)    # scalar
        return x, y

def prepare_data(data_dir, label_file, id_name, label_name):
    data_arr = []
    label_arr = []
    label_df = pd.read_csv(label_file)  
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        this_np = pd.read_csv(file_path).to_numpy()
        data_arr.append(this_np)
        file_id = file_name[:-4]  
        label_df[id_name] = [str(i) for i in label_df[id_name]]
        matched_row = label_df[label_df[id_name] == file_id]
        label = matched_row[label_name].values[0]
        label_arr.append(label)
    return data_arr, label_arr
        
        
def train_and_evaluate(data_arr, label_arr, k=5, epochs=100, lr=0.01):
    dataset = MatrixDataset(data_arr, label_arr)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold + 1}/{k} ===")
        
        # 用 Subset 拆出 train 和 val 数据集
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)
        
        # 初始化模型
        model = SimpleLSTMClassifier(input_dim=data_arr[0].shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 🔁 训练 5 个 epoch
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                y = y.unsqueeze(1)
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ✅ 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                y = y.unsqueeze(1)
                preds = (model(x) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Fold {fold+1} Accuracy: {correct / total:.2%}")
        




















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
    elif method == 'timemixerpp':
        from pypots.imputation import TimeMixerPP, TimeLLM, MOMENT
        data_for_pypots = filled.copy()
        data_for_pypots[mask == 0] = np.nan
        data_for_pypots = data_for_pypots[np.newaxis, ...]
        train_set = {"X": data_for_pypots}
        model = TimeMixerPP(
            n_steps=data_for_pypots.shape[1], 
            n_features=data_for_pypots.shape[2], 
            n_layers=2, 
            d_model=4,
            n_heads=4,  
            top_k=4,       
            d_ffn=16,  
            n_kernels=6,
            dropout=0.1,
            epochs=100,
        )
                    
        model.fit(train_set)
        imputed_data = model.impute(train_set)
        filled = imputed_data.squeeze(0)
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
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    return f1, auroc, accuracy, precision, recall


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
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(X)))):
        print(f"  执行第 {fold+1}/{min(k_folds, len(y))} 折...")
        X_train = [X[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = y[test_idx]
        
        input_dim = X[0].shape[1]  # 特征维度
        f1, auroc, accuracy, precision, recall = evaluate_model(X_train, y_train, X_test, y_test, input_dim, device=device)
        f1_scores.append(f1)
        auroc_scores.append(auroc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    avg_f1 = np.mean(f1_scores)
    avg_auroc = np.mean(auroc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    
    print(f"  {method_name} 评估完成: 平均F1={avg_f1:.4f}, 平均AUROC={avg_auroc:.4f}, 准确率={avg_accuracy:.4f}, 精确率={avg_precision:.4f}, 召回率={avg_recall:.4f}")
    
    return {
        'f1': avg_f1, 
        'auroc': avg_auroc,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall
    }

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
    methods = ['zero', 'mean', 'median', 'bfill', 'ffill', 'knn', 'mice', 'DLine', 'model']
    
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
    print(f"{'方法':<10}{'accuracy':<15}{'precision':<15}{'recall':<15}{'F1分数':<15}{'AUROC分数':<15}")
    print("-" * 75)  # 增加分隔线长度以覆盖所有列

    for method in methods:
        if method in results:
            print(f"{method:<10}{results[method]['accuracy']:.4f}{'':<10}{results[method]['precision']:.4f}{'':<10}{results[method]['recall']:.4f}{'':<10}{results[method]['f1']:.4f}{'':<10}{results[method]['auroc']:.4f}")

    # 添加保存结果到文件的代码
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)

    # 创建结果文件
    results_file = os.path.join(results_dir, 'downstream_comparison.txt')
    with open(results_file, 'w') as f:
        f.write("===== 填充方法性能比较 =====\n")
        f.write(f"{'方法':<10}{'accuracy':<15}{'precision':<15}{'recall':<15}{'F1分数':<15}{'AUROC分数':<15}\n")
        f.write("-" * 75 + "\n")  # 增加分隔线长度
        
        for method in methods:
            if method in results:
                # 添加换行符确保每个方法在单独一行
                f.write(f"{method:<10}{results[method]['accuracy']:.4f}{'':<10}{results[method]['precision']:.4f}{'':<10}{results[method]['recall']:.4f}{'':<10}{results[method]['f1']:.4f}{'':<10}{results[method]['auroc']:.4f}\n")

    # 同时保存为CSV格式便于后续分析
    df_results = pd.DataFrame([
        {
            'method': method,
            'accuracy': results[method]['accuracy'], 
            'precision': results[method]['precision'], 
            'recall': results[method]['recall'], 
            'f1': results[method]['f1'],
            'auroc': results[method]['auroc']
        }
        for method in methods if method in results
    ])
    csv_file = os.path.join(results_dir, 'downstream_comparison.csv')
    df_results.to_csv(csv_file, index=False)

    print(f"\n结果已保存到 {results_file} 和 {csv_file}")

    return results