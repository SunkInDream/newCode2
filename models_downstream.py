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
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

def prepare_data(data_dir, label_file=None, id_name=None, label_name=None):
    if label_file is None or id_name is None or label_name is None:
        data_arr = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            this_np = pd.read_csv(file_path).to_numpy()
            data_arr.append(this_np)
        return data_arr
    else:
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
        
def train_and_evaluate(data_arr, label_arr, k=5, epochs=100, lr=0.02): 
    dataset = MatrixDataset(data_arr, label_arr) 
    kf = KFold(n_splits=k, shuffle=True, random_state=42) 
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)): 
        print(f"\n=== Fold {fold + 1}/{k} ===") 
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=16)
        
        model = SimpleLSTMClassifier(input_dim=data_arr[0].shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # —— 训练
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                y = y.unsqueeze(1).float()
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # —— 验证
        model.eval()
        all_labels = []
        all_preds  = []
        all_scores = []  # 存放 sigmoid(logits) 作为 AUROC 分数
        with torch.no_grad():
            for x, y in val_loader:
                y = y.unsqueeze(1).float()
                logits = model(x)
                probs = torch.sigmoid(logits)

                preds = (probs > 0.5).float()

                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        # 统一计算各项指标
        acc   = accuracy_score(all_labels, all_preds)
        prec  = precision_score(all_labels, all_preds, zero_division=0)
        rec   = recall_score(all_labels, all_preds, zero_division=0)
        f1    = f1_score(all_labels, all_preds, zero_division=0)
        auroc = roc_auc_score(all_labels, all_scores)

        print(f"Fold {fold+1} — Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, "
              f"F1: {f1:.2%}, AUROC: {auroc:.4f}")
        


















