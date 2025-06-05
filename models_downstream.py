import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
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
        
def train_and_evaluate(data_arr, label_arr, k=5, epochs=200, lr=0.02):
    dataset = MatrixDataset(data_arr, label_arr)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accs, precs, recs, f1s, aurocs = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=16)

        model = SimpleLSTMClassifier(input_dim=data_arr[0].shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in tqdm(range(epochs), desc=f"Training Fold {fold+1}"):
            model.train()
            for x, y in train_loader:
                y = y.unsqueeze(1).float()
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        all_labels, all_preds, all_scores = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                y = y.unsqueeze(1).float()
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        acc   = accuracy_score(all_labels, all_preds)
        prec  = precision_score(all_labels, all_preds, zero_division=0)
        rec   = recall_score(all_labels, all_preds, zero_division=0)
        f1    = f1_score(all_labels, all_preds, zero_division=0)
        auroc = roc_auc_score(all_labels, all_scores)

        print(f"Fold {fold+1} — Accuracy: {acc:.2%}, Precision: {prec:.2%}, "
              f"Recall: {rec:.2%}, F1: {f1:.2%}, AUROC: {auroc:.4f}")

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        aurocs.append(auroc)

    print("\n=== Average over folds ===")
    print(f"Accuracy: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"Precision: {np.mean(precs):.2%} ± {np.std(precs):.2%}")
    print(f"Recall: {np.mean(recs):.2%} ± {np.std(recs):.2%}")
    print(f"F1: {np.mean(f1s):.2%} ± {np.std(f1s):.2%}")
    print(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    return {
        'Accuracy': (np.mean(accs), np.std(accs)),
        'Precision': (np.mean(precs), np.std(precs)),
        'Recall': (np.mean(recs), np.std(recs)),
        'F1': (np.mean(f1s), np.std(f1s)),
        'AUROC': (np.mean(aurocs), np.std(aurocs)),
    }


















