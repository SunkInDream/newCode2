import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

class MIRACLE(nn.Module):
    def __init__(self, num_inputs, missing_list, n_hidden=32, lr=0.008, max_steps=50, window=5, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ✅ 修复：正确计算总输入维度
        self.original_features = num_inputs
        self.missing_list = missing_list  # ✅ 保存missing_list
        self.missing_features = len(missing_list)
        self.total_input_dim = num_inputs + len(missing_list)  # 原始特征 + 缺失指示器
        
        self.max_steps = max_steps
        self.window = window

        # ✅ 修复：使用正确的输入维度创建网络
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_input_dim, n_hidden),  # 使用总输入维度
                nn.ELU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, 1)
            ) for _ in range(self.original_features)  # 只输出原始特征数量
        ])
        
        self.to(self.device)
        self.opt = optim.Adam(self.parameters(), lr=lr)
    def forward(self, X):
        # X的维度应该是 (seq_len, total_input_dim)
        outputs = []
        for net in self.networks:
            output = net(X).squeeze(-1)  # (seq_len,)
            outputs.append(output)
        return torch.stack(outputs, dim=1)  # (seq_len, original_features)

    def fit(self, X_miss: np.ndarray) -> np.ndarray:
        X = X_miss.copy()
        X_mask = (~np.isnan(X)).astype(np.float32)
        
        # ✅ 修复：确保维度一致性
        seq_len, n_features = X.shape
        assert n_features == self.original_features, f"输入特征数({n_features})与模型期望({self.original_features})不匹配"
        
        # 用列均值填补NaN
        col_mean = np.nanmean(X, axis=0)
        for j in range(n_features):
            col_mask = np.isnan(X[:, j])
            if col_mask.any():
                if not np.isnan(col_mean[j]):
                    X[col_mask, j] = col_mean[j]
                else:
                    X[col_mask, j] = 0.0

        # 构造缺失指示器 - 只对有缺失的列
        missing_cols = np.any(np.isnan(X_miss), axis=0)
        # 构造缺失指示器 - 使用missing_list
        if len(self.missing_list) > 0:
            indicators = np.zeros((seq_len, len(self.missing_list)), dtype=np.float32)
            for i, col_idx in enumerate(self.missing_list):
                if col_idx < n_features:
                    indicators[:, i] = (1 - X_mask[:, col_idx]).astype(np.float32)
        else:
            indicators = np.zeros((seq_len, 0), dtype=np.float32)
        
        # ✅ 修复：确保拼接维度正确
        X_all = np.concatenate([X, indicators], axis=1)  # (seq_len, original_features + missing_features)
        X_mask_all = np.concatenate([X_mask, np.ones_like(indicators)], axis=1)
        
        # ✅ 验证维度
        assert X_all.shape[1] == self.total_input_dim, f"拼接后维度({X_all.shape[1]})与期望({self.total_input_dim})不匹配"

        avg_preds = []

        for step in range(self.max_steps):
            if step >= self.window and len(avg_preds) >= self.window:
                X_pred = np.mean(np.stack(avg_preds), axis=0)
                # ✅ 只更新原始特征部分
                X = X * X_mask + X_pred * (1 - X_mask)
                X_all = np.concatenate([X, indicators], axis=1)

            self.train()
            xt = torch.tensor(X_all, dtype=torch.float32, device=self.device)
            mt = torch.tensor(X_mask_all, dtype=torch.float32, device=self.device)
            
            pred = self(xt)  # (seq_len, original_features)
            
            # ✅ 修复：只对原始特征计算损失
            # ✅ 修复：将indicators转换为torch张量
            indicators_tensor = torch.tensor(indicators, dtype=torch.float32, device=self.device)
            pred_all = torch.cat([pred, indicators_tensor], dim=1) if indicators.shape[1] > 0 else pred
            loss = ((pred_all - xt) ** 2 * mt).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.eval()
                avg_preds.append(self(xt).cpu().numpy())
                if len(avg_preds) > self.window:
                    avg_preds.pop(0)

        final_pred = np.mean(np.stack(avg_preds), axis=0)
        return X * X_mask + final_pred * (1 - X_mask)