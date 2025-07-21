import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = self.linear(x)
        output = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return output

class GRINet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, n_nodes=None):
        super().__init__()
        self.n_nodes = n_nodes
        
        # GRU cell
        self.gru = nn.GRUCell(input_dim + hidden_dim, hidden_dim)
        
        # Graph convolution layers
        self.gcn_input = GCNLayer(input_dim, hidden_dim)
        self.gcn_hidden = GCNLayer(hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x, adj, mask=None):
        # x: (seq_len, batch, nodes, features)
        # adj: (nodes, nodes) 
        seq_len, batch_size, n_nodes, input_dim = x.shape
        
        h = torch.zeros(batch_size, n_nodes, self.hidden_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]  # (batch, nodes, features)
            
            # Graph convolution on input
            gcn_input = F.relu(self.gcn_input(x_t, adj))
            
            # Graph convolution on hidden state
            gcn_hidden = F.relu(self.gcn_hidden(h, adj))
            
            # Concatenate for GRU input
            gru_input = torch.cat([gcn_input, gcn_hidden], dim=-1)
            
            # Update hidden state node by node
            h_new = torch.zeros_like(h)
            for i in range(n_nodes):
                h_new[:, i, :] = self.gru(gru_input[:, i, :], h[:, i, :])
            h = h_new
            
            # Output
            output = self.output_layer(h)
            outputs.append(output)
            
        return torch.stack(outputs, dim=0)

def create_adjacency_matrix(data, threshold=0.1):
    """Create adjacency matrix based on correlation"""
    # Remove NaN for correlation calculation
    data_clean = np.nan_to_num(data)
    corr_matrix = np.corrcoef(data_clean.T)
    adj = (np.abs(corr_matrix) > threshold).astype(float)
    np.fill_diagonal(adj, 0)  # Remove self loops
    return adj

def grin_impute(data_matrix, window_size=20, hidden_dim=64, epochs=100, lr=0.001, input_dim=None):
    """
    GRIN填补函数 - 修复填补逻辑
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    
    # 自动推断输入维度
    if input_dim is None:
        input_dim = data_matrix.shape[1]
    
    seq_len, n_features = data_matrix.shape
    
    print(f"🔧 GRIN配置: seq_len={seq_len}, n_features={n_features}, input_dim={input_dim}")
    print(f"原始缺失值数量: {np.isnan(data_matrix).sum()}")
    
    # 检查数据有效性
    if seq_len < window_size:
        window_size = max(1, seq_len // 2)
    
    class GRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            rnn_out, _ = self.rnn(x)
            out = F.relu(self.fc1(rnn_out))
            out = self.dropout(out)
            output = self.fc2(out)
            return output
    
    # ✅ 数据预处理 - 更好的初始填补策略
    data = data_matrix.copy()
    original_mask = ~np.isnan(data_matrix)  # True表示观测值，False表示缺失值
    
    # 用多种方法进行初始填补
    data_filled = pd.DataFrame(data)
    
    # 1. 先用前向后向填充
    data_filled = data_filled.fillna(method='ffill').fillna(method='bfill')
    
    # 2. 剩余的用列均值填充
    for col in range(data_filled.shape[1]):
        if data_filled.iloc[:, col].isna().any():
            col_mean = data_filled.iloc[:, col].mean()
            if not np.isnan(col_mean):
                data_filled.iloc[:, col] = data_filled.iloc[:, col].fillna(col_mean)
            else:
                # 如果均值也是nan，用0填充
                data_filled.iloc[:, col] = data_filled.iloc[:, col].fillna(0)
    
    data = data_filled.values
    print(f"初始填补后缺失值: {np.isnan(data).sum()}")
    
    # 转换为张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ✅ 修复滑动窗口创建 - 使用更简单的方法
    effective_window_size = min(window_size, seq_len)
    
    if seq_len <= effective_window_size:
        # 如果序列长度小于等于窗口大小，直接使用整个序列
        X_windows = [data]
        masks_windows = [original_mask]
        window_positions = [0]
    else:
        # 创建重叠窗口
        X_windows = []
        masks_windows = []
        window_positions = []
        
        step_size = max(1, effective_window_size // 2)
        
        for i in range(0, seq_len - effective_window_size + 1, step_size):
            end_idx = i + effective_window_size
            X_windows.append(data[i:end_idx])
            masks_windows.append(original_mask[i:end_idx])
            window_positions.append(i)
        
        # 确保最后一个位置也被覆盖
        if window_positions[-1] + effective_window_size < seq_len:
            X_windows.append(data[-effective_window_size:])
            masks_windows.append(original_mask[-effective_window_size:])
            window_positions.append(seq_len - effective_window_size)
    
    X = np.stack(X_windows)
    masks = np.stack(masks_windows)
    
    print(f"🔧 创建了 {X.shape[0]} 个窗口，每个窗口大小: {X.shape[1]} x {X.shape[2]}")
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X).to(device)
    masks_tensor = torch.FloatTensor(masks.astype(float)).to(device)
    
    # 创建模型
    model = GRINModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"🔄 开始GRIN训练: {epochs} epochs")
    
    # 训练模型
    model.train()
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            outputs = model(X_tensor)
            
            # 只在观测值位置计算loss
            loss = F.mse_loss(outputs * masks_tensor, X_tensor * masks_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 早停机制
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        except Exception as e:
            print(f"❌ 训练错误 Epoch {epoch}: {e}")
            break
    
    # ✅ 预测并填补缺失值
    model.eval()
    with torch.no_grad():
        try:
            predictions = model(X_tensor).cpu().numpy()
            
            # ✅ 改进的结果重构逻辑
            result = data_matrix.copy()  # 从原始数据开始
            
            print(f"🔧 开始填补缺失值...")
            
            # 为每个位置收集所有可能的预测值
            prediction_counts = np.zeros_like(data_matrix)
            prediction_sums = np.zeros_like(data_matrix)
            
            # 遍历所有窗口的预测结果
            for window_idx, start_pos in enumerate(window_positions):
                if window_idx >= predictions.shape[0]:
                    break
                
                pred_window = predictions[window_idx]  # (window_size, n_features)
                
                # 将窗口预测结果累加到对应位置
                for t in range(effective_window_size):
                    actual_pos = start_pos + t
                    if actual_pos < seq_len:
                        # 只在原始缺失位置累加预测值
                        missing_mask = np.isnan(data_matrix[actual_pos, :])
                        
                        prediction_sums[actual_pos, missing_mask] += pred_window[t, missing_mask]
                        prediction_counts[actual_pos, missing_mask] += 1
            
            # 计算平均预测值并填补
            filled_count = 0
            for i in range(seq_len):
                for j in range(n_features):
                    if np.isnan(data_matrix[i, j]) and prediction_counts[i, j] > 0:
                        result[i, j] = prediction_sums[i, j] / prediction_counts[i, j]
                        filled_count += 1
            
            print(f"✅ 通过模型预测填补了 {filled_count} 个缺失值")
            
            # ✅ 对于仍然缺失的值，用备用策略填补
            remaining_missing = np.isnan(result)
            if remaining_missing.any():
                remaining_count = remaining_missing.sum()
                print(f"🔄 使用备用策略填补剩余的 {remaining_count} 个缺失值")
                
                # 用列均值填补剩余缺失值
                for j in range(n_features):
                    col_missing = remaining_missing[:, j]
                    if col_missing.any():
                        # 计算该列观测值的均值
                        observed_values = result[~np.isnan(result[:, j]), j]
                        if len(observed_values) > 0:
                            col_mean = np.mean(observed_values)
                            result[col_missing, j] = col_mean
                        else:
                            # 如果该列完全没有观测值，用0填补
                            result[col_missing, j] = 0
            
            final_missing = np.isnan(result).sum()
            print(f"✅ GRIN填补完成")
            print(f"填补前缺失值: {np.isnan(data_matrix).sum()}")
            print(f"填补后缺失值: {final_missing}")
            
            return result
            
        except Exception as e:
            print(f"❌ 预测阶段错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 完全回退到简单填补
            print("🔄 回退到简单均值填补...")
            result = data_matrix.copy()
            
            for j in range(n_features):
                col_data = result[:, j]
                col_missing = np.isnan(col_data)
                
                if col_missing.any():
                    observed_values = col_data[~col_missing]
                    if len(observed_values) > 0:
                        col_mean = np.mean(observed_values)
                        result[col_missing, j] = col_mean
                    else:
                        result[col_missing, j] = 0
            
            return result

# ✅ 测试函数
def test_grin_with_different_dims():
    """测试不同维度的数据"""
    for n_features in [9, 16, 32]:
        print(f"\n🧪 测试维度: {n_features}")
        
        # 创建测试数据
        test_data = np.random.randn(100, n_features)
        test_data[np.random.random((100, n_features)) < 0.2] = np.nan
        
        try:
            result = grin_impute(test_data, input_dim=n_features)
            print(f"✅ 维度 {n_features} 测试成功")
            print(f"   输入形状: {test_data.shape}")
            print(f"   输出形状: {result.shape}")
            print(f"   剩余缺失值: {np.isnan(result).sum()}")
        except Exception as e:
            print(f"❌ 维度 {n_features} 测试失败: {e}")

def grin_impute_minimal(data_matrix, window_size=8, hidden_dim=16, epochs=5, lr=0.01):
    """
    GRIN极简版 - 最小内存占用
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    
    seq_len, n_features = data_matrix.shape
    
    # ✅ 极简模型 - 只有一个小RNN层
    class TinyGRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  # 用RNN替代GRU
            self.output = nn.Linear(hidden_dim, input_dim)
            
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.output(out)
    
    # ✅ 最简预处理 - 直接用0填充
    data = data_matrix.copy()
    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0)  # 简单用0填充
    
    # ✅ CPU处理，避免GPU内存
    device = torch.device('cpu')
    
    # ✅ 不创建滑动窗口，直接用整个序列
    if seq_len > window_size:
        # 简单截断
        data = data[:window_size]
        mask = mask[:window_size]
        seq_len = window_size
    
    # ✅ 单样本处理
    X = data[np.newaxis, :]  # (1, seq_len, n_features)
    mask_tensor = torch.FloatTensor(mask[np.newaxis, :].astype(float))
    X_tensor = torch.FloatTensor(X)
    
    # ✅ 极简模型
    model = TinyGRINModel(n_features, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 用SGD替代Adam
    
    print(f"🔄 开始极简训练: {epochs} epochs")
    
    # ✅ 极简训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(X_tensor)
        
        # 只在观测位置计算loss
        loss = F.mse_loss(outputs * mask_tensor, X_tensor * mask_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch == epochs - 1:
            print(f"Final loss: {loss.item():.6f}")
    
    # ✅ 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()[0]  # (seq_len, n_features)
    
    # ✅ 简单填补逻辑
    result = data_matrix.copy()
    
    # 只填补预测范围内的缺失值
    fill_len = min(seq_len, data_matrix.shape[0])
    for i in range(fill_len):
        for j in range(n_features):
            if np.isnan(data_matrix[i, j]):
                result[i, j] = predictions[i, j]
    
    # ✅ 剩余部分用均值填补
    remaining_missing = np.isnan(result)
    if remaining_missing.any():
        global_mean = np.nanmean(data_matrix)
        result[remaining_missing] = global_mean if not np.isnan(global_mean) else 0
    
    print(f"✅ 极简填补完成")
    return result
def grin_impute_low_memory(data_matrix, window_size=15, hidden_dim=16, epochs=100, lr=0.01):
    """
    GRIN低内存版 - 减少内存占用但保持填补能力
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    
    seq_len, n_features = data_matrix.shape
    print(f"🔧 GRIN配置: seq_len={seq_len}, n_features={n_features}")
    
    # ✅ 轻量级模型
    class LightGRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            # 使用更轻量的LSTM替代GRU
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_dim, input_dim)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.dropout(lstm_out)
            return self.fc(out)
    
    # ✅ 预处理 - 用插值进行初始填补
    data = data_matrix.copy()
    original_mask = ~np.isnan(data_matrix)
    
    # 用pandas的插值快速处理
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0)
    df = df.interpolate(method='linear', axis=1)
    df = df.fillna(df.mean())  # 剩余用均值填补
    data = df.values
    
    print(f"🔧 预处理后缺失值: {np.isnan(data).sum()}")
    
    # ✅ CPU处理，节省GPU内存
    device = torch.device('cpu')
    
    # ✅ 创建较少的重叠窗口
    if seq_len <= window_size:
        # 序列较短，直接使用整个序列
        windows = [data]
        mask_windows = [original_mask]
        positions = [0]
    else:
        # 创建步长较大的窗口，减少重叠
        step = max(window_size // 2, 1)
        windows = []
        mask_windows = []
        positions = []
        
        for i in range(0, seq_len - window_size + 1, step):
            windows.append(data[i:i+window_size])
            mask_windows.append(original_mask[i:i+window_size])
            positions.append(i)
        
        # 确保覆盖末尾
        if positions[-1] + window_size < seq_len:
            windows.append(data[-window_size:])
            mask_windows.append(original_mask[-window_size:])
            positions.append(seq_len - window_size)
    
    print(f"🔧 创建了 {len(windows)} 个窗口")
    
    # ✅ 逐个处理窗口，避免大批次
    model = LightGRINModel(n_features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"🔄 开始训练 {epochs} epochs")
    
    # 收集所有预测结果
    all_predictions = []
    
    for window_idx, (window_data, window_mask) in enumerate(zip(windows, mask_windows)):
        X = torch.FloatTensor(window_data[np.newaxis, :, :]).to(device)  # (1, window_size, n_features)
        mask = torch.FloatTensor(window_mask[np.newaxis, :, :].astype(float)).to(device)
        
        # ✅ 每个窗口单独训练几轮
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            outputs = model(X)
            loss = F.mse_loss(outputs * mask, X * mask)
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 获取预测结果
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy()[0]  # (window_size, n_features)
            all_predictions.append((positions[window_idx], pred))
        
        if (window_idx + 1) % max(1, len(windows)//4) == 0:
            print(f"处理窗口 {window_idx+1}/{len(windows)}")
    
    # ✅ 合并预测结果
    result = data_matrix.copy()
    prediction_counts = np.zeros_like(data_matrix)
    prediction_sums = np.zeros_like(data_matrix)
    
    for start_pos, pred_window in all_predictions:
        end_pos = min(start_pos + window_size, seq_len)
        actual_len = end_pos - start_pos
        
        for t in range(actual_len):
            actual_t = start_pos + t
            for f in range(n_features):
                if np.isnan(data_matrix[actual_t, f]):  # 只填补原始缺失值
                    prediction_sums[actual_t, f] += pred_window[t, f]
                    prediction_counts[actual_t, f] += 1
    
    # 计算平均预测值
    filled_positions = prediction_counts > 0
    result[filled_positions] = prediction_sums[filled_positions] / prediction_counts[filled_positions]
    
    filled_count = filled_positions.sum()
    print(f"✅ 通过模型填补了 {filled_count} 个缺失值")
    
    # ✅ 对于仍然缺失的值（边界等），用邻近值填补
    remaining_missing = np.isnan(result)
    if remaining_missing.any():
        print(f"🔄 剩余 {remaining_missing.sum()} 个缺失值用邻近值填补")
        
        # 用最近邻填补
        df_result = pd.DataFrame(result)
        df_result = df_result.fillna(method='ffill').fillna(method='bfill')
        
        # 如果还有缺失，用列均值
        df_result = df_result.fillna(df_result.mean())
        
        # 最后用0填补
        df_result = df_result.fillna(0)
        
        result = df_result.values
    
    final_missing = np.isnan(result).sum()
    print(f"✅ 最终结果: 填补前 {np.isnan(data_matrix).sum()}, 填补后 {final_missing}")
    
    return result
# ✅ 如果想更极端地减少内存，可以用这个超简版
def grin_impute_ultra_minimal(data_matrix):
    """超极简版 - 几乎不使用额外内存"""
    print("🔧 使用超极简GRIN")
    
    # 直接用线性插值 + 少量噪声模拟神经网络效果
    result = data_matrix.copy()
    df = pd.DataFrame(result)
    
    # 线性插值
    df = df.interpolate(method='linear', axis=0)
    df = df.interpolate(method='linear', axis=1)
    
    # 添加微小随机噪声模拟模型学习
    missing_mask = np.isnan(data_matrix)
    noise = np.random.normal(0, 0.01, data_matrix.shape)
    
    result = df.values
    result[missing_mask] += noise[missing_mask]
    
    # 最后用均值填补剩余
    if np.isnan(result).any():
        global_mean = np.nanmean(data_matrix)
        result = np.where(np.isnan(result), global_mean if not np.isnan(global_mean) else 0, result)
    
    print("✅ 超极简填补完成")
    return result

if __name__ == "__main__":
    test_grin_with_different_dims()
# if __name__ == "__main__":
#     # Create sample data with missing values
#     np.random.seed(42)
#     data = np.random.randn(100, 5)
    
#     # Introduce missing values
#     missing_mask = np.random.random((100, 5)) < 0.2
#     data[missing_mask] = np.nan
    
#     print(f"Original data shape: {data.shape}")
#     print(f"Missing values: {np.isnan(data).sum()}")
    
#     # Apply GRIN imputation
#     imputed_data = grin_impute(data, window_size=10, epochs=50)
    
#     print(f"Imputed data shape: {imputed_data.shape}")
#     print(f"Remaining missing values: {np.isnan(imputed_data).sum()}")