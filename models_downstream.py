import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from baseline import *
import warnings
warnings.filterwarnings("ignore")

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, 
                 use_batch_norm=True, use_layer_norm=False):
        super(SimpleLSTMClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # 输入标准化层 - 这里是关键创新点
        if use_batch_norm:
            # BatchNorm1d 适用于特征标准化，在batch维度上计算统计量
            # 这对于处理不同患者间的基线差异很有效
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif use_layer_norm:
            # LayerNorm 在特征维度上标准化，对每个样本独立处理
            # 这更适合处理时间序列内部的特征差异
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = None
        
        # 使用双向LSTM，增加模型表达能力
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM能同时利用前后信息
        )
        
        # LSTM输出标准化 - 稳定后续层的训练
        if use_batch_norm:
            self.lstm_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.lstm_norm = None
        
        # 注意力机制，让模型关注重要的时间步
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM输出维度翻倍
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力输出标准化
        if use_batch_norm:
            self.attention_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.attention_norm = None
        
        # 分类头，使用更深的网络
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32) if use_batch_norm else nn.LayerNorm(32) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape
        
        # 步骤1: 输入特征标准化
        # 这一步非常关键 - 确保所有特征在相同的数值范围内
        if self.input_norm is not None:
            if self.use_batch_norm:
                # BatchNorm需要将时间维度和批次维度合并
                # 重塑: [batch_size, seq_len, features] -> [batch_size*seq_len, features]
                x_reshaped = x.view(-1, num_features)
                x_normalized = self.input_norm(x_reshaped)
                # 恢复形状: [batch_size*seq_len, features] -> [batch_size, seq_len, features]
                x = x_normalized.view(batch_size, seq_len, num_features)
            else:  # LayerNorm情况
                x = self.input_norm(x)
        
        # 步骤2: LSTM处理 - 提取时间序列特征
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 步骤3: LSTM输出标准化（可选）
        if self.lstm_norm is not None:
            if self.use_batch_norm:
                # 对LSTM输出进行标准化，同样需要重塑
                lstm_reshaped = lstm_out.reshape(-1, lstm_out.size(-1))
                lstm_normalized = self.lstm_norm(lstm_reshaped) 
                lstm_out = lstm_normalized.view(batch_size, seq_len, -1)
            else:
                lstm_out = self.lstm_norm(lstm_out)
        
        # 步骤4: 自注意力机制 - 识别重要的时间点
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 步骤5: 注意力输出标准化（可选）
        if self.attention_norm is not None:
            if self.use_batch_norm:
                attn_reshaped = attn_out.reshape(-1, attn_out.size(-1))
                attn_normalized = self.attention_norm(attn_reshaped)
                attn_out = attn_normalized.view(batch_size, seq_len, -1)
            else:
                attn_out = self.attention_norm(attn_out)
        
        # 步骤6: 时间维度聚合 - 全局平均池化而不是只取最后时间步
        # 这样能更好地利用整个时间序列的信息
        pooled = torch.mean(attn_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 步骤7: 分类 - 最终预测
        out = self.classifier(pooled)
        
        return out
class SimpleGRUClassifier(nn.Module): 
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3,  
                 use_batch_norm=True, use_layer_norm=False):
        super(SimpleGRUClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # 输入标准化层
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = None
        
        # 使用双向GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # GRU输出标准化
        if use_batch_norm:
            self.lstm_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.lstm_norm = None
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力输出标准化
        if use_batch_norm:
            self.attention_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.attention_norm = None
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32) if use_batch_norm else nn.LayerNorm(32) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape
        
        # 1. 输入标准化
        if self.input_norm is not None:
            if self.use_batch_norm:
                x_reshaped = x.view(-1, num_features)
                x_normalized = self.input_norm(x_reshaped)
                x = x_normalized.view(batch_size, seq_len, num_features)
            else:
                x = self.input_norm(x)
        
        # 2. GRU处理
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 3. GRU输出标准化
        if self.lstm_norm is not None:
            if self.use_batch_norm:
                lstm_reshaped = gru_out.reshape(-1, gru_out.size(-1))
                lstm_normalized = self.lstm_norm(lstm_reshaped)
                gru_out = lstm_normalized.view(batch_size, seq_len, -1)
            else:
                gru_out = self.lstm_norm(gru_out)
        
        # 4. 自注意力机制
        attn_out, attention_weights = self.attention(gru_out, gru_out, gru_out)
        
        # 5. 注意力输出标准化
        if self.attention_norm is not None:
            if self.use_batch_norm:
                attn_reshaped = attn_out.reshape(-1, attn_out.size(-1))
                attn_normalized = self.attention_norm(attn_reshaped)
                attn_out = attn_normalized.view(batch_size, seq_len, -1)
            else:
                attn_out = self.attention_norm(attn_out)
        
        # 6. 时间维度聚合
        pooled = torch.mean(attn_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 7. 分类
        out = self.classifier(pooled)
        
        return out

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

def Prepare_data(data_dir, label_file=None, id_name=None, label_name=None):
    file_list = os.listdir(data_dir)

    if label_file is None or id_name is None or label_name is None:
        data_arr = []
        for file_name in tqdm(file_list, desc="读取数据文件"):  # ✅ 加进度条
            file_path = os.path.join(data_dir, file_name)
            this_np = pd.read_csv(file_path).to_numpy()
            data_arr.append(this_np)
        return data_arr
    else:
        data_arr = []
        label_arr = []
        label_df = pd.read_csv(label_file)
        label_df[id_name] = [str(i) for i in label_df[id_name]]

        for file_name in tqdm(file_list, desc="读取数据并匹配标签"):  # ✅ 加进度条
            file_path = os.path.join(data_dir, file_name)
            this_np = pd.read_csv(file_path).to_numpy()
            data_arr.append(this_np)

            file_id = file_name[:-4]
            matched_row = label_df[label_df[id_name] == file_id]
            label = matched_row[label_name].values[0]
            label_arr.append(label)

        return data_arr, label_arr
        
def train_fold(fold_args): 
    import os
    import torch
    fold, train_idx, val_idx, data_arr, label_arr, epochs, lr, gpu_uuid = fold_args

    # 限制只可见一个GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    torch.cuda.set_device(0)  # 对每个进程来说，只可见一个 GPU，其编号是 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = MatrixDataset(data_arr, label_arr)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

    model = SimpleGRUClassifier(input_dim=data_arr[0].shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, zero_division=0),
        recall_score(all_labels, all_preds, zero_division=0),
        f1_score(all_labels, all_preds, zero_division=0),
        roc_auc_score(all_labels, all_scores)
    )



def train_and_evaluate(data_arr, label_arr, k=5, epochs=200, lr=0.02):
    from multiprocessing import get_context  # ✅ 正确方式
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    num_gpus = torch.cuda.device_count()
    tasks = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_arr)):
        physical_gpu_id = fold % num_gpus
        tasks.append((fold, train_idx, val_idx, data_arr, label_arr, epochs, lr, physical_gpu_id))

    with get_context("spawn").Pool(processes=min(k, num_gpus)) as pool:  # ✅ 使用 spawn 上下文，不会重复设置
        results = pool.map(train_fold, tasks)

    accs, precs, recs, f1s, aurocs = zip(*results)
    return {
        'Accuracy': (np.mean(accs), np.std(accs)),
        'Precision': (np.mean(precs), np.std(precs)),
        'Recall': (np.mean(recs), np.std(recs)),
        'F1': (np.mean(f1s), np.std(f1s)),
        'AUROC': (np.mean(aurocs), np.std(aurocs)),
    }


def evaluate_downstream(data_arr, label_arr, k=4, epochs=100, lr=0.02):
    """
    评估多种插补方法的性能
    """
    results = {}
    
    # ✅ 定义要评估的方法列表
    methods = [
        ('Scit-Impute', lambda: Prepare_data('./data_imputed/my_model/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('Zero-Impute', lambda: ([zero_impu(matrix) for matrix in data_arr], label_arr)),
        # ('MICE-Impute', lambda: Prepare_data('./data_imputed/mice/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('Random-Impute', lambda: ([random_impu(matrix) for matrix in data_arr], label_arr)),
        # ('KNN-Impute', lambda:  Prepare_data('./data_imputed/knn/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('Mean-Impute', lambda: ([mean_impu(matrix) for matrix in data_arr], label_arr)),
        # ('BFill-Impute', lambda: ([bfill_impu(matrix) for matrix in data_arr], label_arr)),
        # ('Miracle-Impute', lambda:  Prepare_data('./data_imputed/miracle/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        ('SAITS-Impute', lambda: Prepare_data('./data_imputed/saits/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('TimeMixerPP-Impute', lambda:  Prepare_data('./data_imputed/timemixerpp/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('TEFN-Impute', lambda: Prepare_data('./data_imputed/tefn/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        # ('TSDE-Impute', lambda: Prepare_data('./data_imputed/tsde/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        ('GRIN-Impute', lambda:  Prepare_data('./data_imputed/grin/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'FirstICU24_AKI_ALL')),
        ('TimesNet-Impute', lambda:  Prepare_data('./data_imputed/timesnet/III', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')),
    ]
    
    # ✅ 添加进度条
    for method_name, data_func in tqdm(methods, desc="评估插补方法"):
        print(f"\n🔄 正在评估 {method_name}...")
        try:
            data_arr_method, label_arr_method = data_func()
            accs = train_and_evaluate(data_arr_method, label_arr_method, k=k, epochs=epochs, lr=lr)
            results[method_name] = accs
            print(f"✅ {method_name} 完成，结果：{accs}")
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            continue
    
    # 生成结果表格...（保持原代码不变）
    table = []
    for method, metrics in results.items():
        row = {
            'Method': method,
            'Accuracy (mean ± std)': f"{metrics['Accuracy'][0]:.2%} ± {metrics['Accuracy'][1]:.2%}",
            'Precision (mean ± std)': f"{metrics['Precision'][0]:.2%} ± {metrics['Precision'][1]:.2%}",
            'Recall (mean ± std)': f"{metrics['Recall'][0]:.2%} ± {metrics['Recall'][1]:.2%}",
            'F1 Score (mean ± std)': f"{metrics['F1'][0]:.2%} ± {metrics['F1'][1]:.2%}",
            'AUROC (mean ± std)': f"{metrics['AUROC'][0]:.4f} ± {metrics['AUROC'][1]:.4f}",
        }
        table.append(row)

    df_results = pd.DataFrame(table)
    print(df_results)
    df_results.to_csv('imputation_comparison_results.csv', index=False)
    return results
















