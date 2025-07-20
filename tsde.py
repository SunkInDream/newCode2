import numpy as np
import torch
import torch.nn as nn
import yaml
import json
import os
from typing import Optional

# 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ✅ 修复SimpleDiffusionModel，确保buffer在正确设备上
class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_steps=50):
        super().__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        
        # 简化的去噪网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # ✅ 扩散调度 - 这些buffer会自动跟随模型设备
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def forward(self, x, t):
        # ✅ 确保时间嵌入在正确设备上
        device = next(self.parameters()).device
        if t.device != device:
            t = t.to(device)
        
        # 时间嵌入
        t_emb = t.float().unsqueeze(-1) / self.num_steps
        x_t = torch.cat([x, t_emb], dim=-1)
        return self.net(x_t)

# ✅ 修复SimpleTSDE类，确保所有组件都在正确设备上
class SimpleTSDE(nn.Module):
    def __init__(self, feature_dim, seq_len, device="cuda"):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 扩散模型
        self.diffusion = SimpleDiffusionModel(feature_dim)
        
        # ✅ 时间和特征嵌入 - 确保在正确设备上
        self.time_emb = nn.Embedding(seq_len, 64)
        self.feature_emb = nn.Linear(feature_dim, 64)
        
        # 融合层
        self.fusion = nn.Linear(64 * 2, feature_dim)
        
        # ✅ 立即移动到指定设备
        self.to(self.device)
        
    def get_embeddings(self, x, mask, timepoints):
        # ✅ 确保timepoints在正确设备上
        if timepoints.device != self.device:
            timepoints = timepoints.to(self.device)
        
        # 时间嵌入
        time_emb = self.time_emb(timepoints.long())  # (B, T, 64)
        
        # 特征嵌入
        feat_emb = self.feature_emb(x)  # (B, T, 64)
        
        # 融合
        combined = torch.cat([time_emb, feat_emb], dim=-1)  # (B, T, 128)
        return self.fusion(combined)  # (B, T, F)
    
    def impute(self, observed_data, mask, n_samples=50):
        B, T, F = observed_data.shape
        device = self.device  # ✅ 使用模型的设备
        
        # ✅ 确保输入数据在正确设备上
        if observed_data.device != device:
            observed_data = observed_data.to(device)
        if mask.device != device:
            mask = mask.to(device)
        
        # ✅ 在模型设备上创建时间点
        timepoints = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
        
        # 获取条件嵌入
        cond_emb = self.get_embeddings(observed_data, mask, timepoints)
        
        samples = []
        
        # 显示进度
        try:
            from tqdm import tqdm
            sample_iterator = tqdm(range(n_samples), desc="生成样本", leave=False)
        except ImportError:
            sample_iterator = range(n_samples)
        
        for sample_idx in sample_iterator:
            # ✅ 在正确设备上从噪声开始
            x_t = torch.randn_like(observed_data, device=device)
            
            # 反向扩散过程
            for t in reversed(range(self.diffusion.num_steps)):
                # ✅ 确保所有张量都在同一设备上
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                
                # 批量处理所有时间步，避免循环
                x_reshaped = x_t.view(-1, F)  # (B*T, F)
                cond_reshaped = cond_emb.view(-1, F)  # (B*T, F)
                t_expanded = t_tensor.unsqueeze(1).repeat(1, T).view(-1)  # (B*T,)
                
                # 批量预测噪声
                x_input = x_reshaped + 0.1 * cond_reshaped
                noise_pred = self.diffusion(x_input, t_expanded)
                
                # 重塑回原来的形状
                noise_pred = noise_pred.view(B, T, F)
                
                # 去噪步骤
                if t > 0:
                    alpha = self.diffusion.alphas[t]
                    alpha_cumprod = self.diffusion.alphas_cumprod[t]
                    
                    # 简化的DDPM更新
                    x_t = (x_t - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha)
                    
                    # 添加噪声
                    if t > 1:
                        noise = torch.randn_like(x_t, device=device)
                        x_t = x_t + torch.sqrt(self.diffusion.betas[t]) * noise
                else:
                    x_t = (x_t - noise_pred) / torch.sqrt(self.diffusion.alphas_cumprod[0])
            
            # 结合观测数据
            x_t = mask * observed_data + (1 - mask) * x_t
            samples.append(x_t)
            
            # 定期清理GPU缓存
            if device.type == 'cuda' and sample_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 返回所有样本
        return torch.stack(samples, dim=1)  # (B, n_samples, T, F)

def load_model_from_checkpoint(model_path, config_path, data_shape, device="cpu"):
    """从检查点加载模型"""
    seq_len, feature_dim = data_shape
    
    # ✅ 创建模型时传入设备信息
    model = SimpleTSDE(feature_dim, seq_len, device)
    
    if os.path.exists(model_path):
        try:
            # 尝试加载预训练权重（可能不完全兼容）
            checkpoint = torch.load(model_path, map_location=device)
            # 只加载兼容的权重
            model_dict = model.state_dict()
            compatible_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
            print(f"✅ 加载了 {len(compatible_dict)} 个兼容的权重参数")
        except Exception as e:
            print(f"⚠️ 无法加载预训练权重: {e}")
            print("使用随机初始化")
    
    return model.to(device)

def impute_missing_data(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None,
    model_folder: Optional[str] = None,
    n_samples: int = 50,
    device: str = "cuda"  # ✅ 默认使用GPU
) -> np.ndarray:
    """
    使用TSDE填补缺失数据的主函数（GPU优化版）
    
    Args:
        data: 二维numpy数组 (时间步, 特征数)，包含NaN表示缺失值
        missing_mask: 可选，缺失掩码 (1=观测值, 0=缺失值)
        model_folder: 可选，预训练模型文件夹路径
        n_samples: 生成样本数，用于取中位数
        device: 计算设备 ("cuda", "cpu", 或 "cuda:0", "cuda:1" 等)
        
    Returns:
        填补后的数据，形状与输入相同
    """
    
    # 输入验证
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        raise ValueError("输入数据必须是二维numpy数组")
    
    seq_len, feature_dim = data.shape
    
    # ✅ 智能设备选择
    if device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device)
        print(f"🚀 使用GPU: {device}")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用CPU (GPU不可用)")
    
    # 处理缺失掩码
    if missing_mask is None:
        missing_mask = ~np.isnan(data)
        data_clean = np.nan_to_num(data, nan=0.0)
    else:
        data_clean = data.copy()
        data_clean[~missing_mask] = 0.0
    
    # ✅ 优化张量转换和内存管理
    data_tensor = torch.FloatTensor(data_clean).unsqueeze(0).to(device, non_blocking=True)  # (1, T, F)
    mask_tensor = torch.FloatTensor(missing_mask.astype(float)).unsqueeze(0).to(device, non_blocking=True)
    
    # 加载或创建模型
    if model_folder and os.path.exists(model_folder):
        model_path = os.path.join(model_folder, "model.pth")
        config_path = os.path.join(model_folder, "config.json")
        model = load_model_from_checkpoint(model_path, config_path, (seq_len, feature_dim), device)
        print(f"🔄 尝试加载预训练模型: {model_folder}")
    else:
        model = SimpleTSDE(feature_dim, seq_len, device)
        print("🔄 使用随机初始化模型")
    
    # ✅ 确保模型在正确设备上
    model = model.to(device)
    model.eval()
    set_seed(42)
    
    # ✅ GPU优化的填补过程
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # 清理GPU缓存
        
        print(f"🔄 开始生成 {n_samples} 个样本...")
        
        # 生成多个样本
        samples = model.impute(data_tensor, mask_tensor, n_samples)
        
        # 取中位数作为最终结果
        result = torch.median(samples, dim=1)[0]  # (1, T, F)
        
        # ✅ 异步传输回CPU
        result_np = result.cpu().numpy().squeeze(0)  # (T, F)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # 清理GPU缓存
    
    print("✅ 填补完成!")
    return result_np

# ✅ 添加设备检查函数
def check_device_consistency(model, *tensors):
    """检查模型和张量是否在同一设备上"""
    model_device = next(model.parameters()).device
    
    for i, tensor in enumerate(tensors):
        if tensor.device != model_device:
            print(f"⚠️ 设备不匹配: 模型在 {model_device}, 张量 {i} 在 {tensor.device}")
            return False
    
    print(f"✅ 所有组件都在设备 {model_device} 上")
    return True

# ✅ 多GPU并行版本
def impute_missing_data_parallel(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None,
    model_folder: Optional[str] = None,
    n_samples: int = 50,
    gpu_ids: list = [0]  # 指定使用的GPU ID列表
) -> np.ndarray:
    """使用多GPU并行的TSDE填补"""
    
    if len(gpu_ids) == 1:
        # 单GPU情况
        device = f"cuda:{gpu_ids[0]}"
        return impute_missing_data(data, missing_mask, model_folder, n_samples, device)
    
    # 多GPU并行处理
    print(f"🚀 使用多GPU并行: {gpu_ids}")
    
    seq_len, feature_dim = data.shape
    samples_per_gpu = n_samples // len(gpu_ids)
    
    import concurrent.futures
    
    def worker(gpu_id, n_samples_worker):
        device = f"cuda:{gpu_id}"
        return impute_missing_data(data, missing_mask, model_folder, n_samples_worker, device)
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for gpu_id in gpu_ids:
            future = executor.submit(worker, gpu_id, samples_per_gpu)
            futures.append(future)
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # 合并结果（取平均）
    final_result = np.mean(results, axis=0)
    return final_result

# 使用示例和测试
if __name__ == "__main__":
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"🚀 可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ 未检测到CUDA GPU")
    
    # 创建测试数据
    np.random.seed(42)
    seq_len, n_features = 200, 20  # ✅ 增大测试数据
    t = np.linspace(0, 4*np.pi, seq_len)
    data = np.zeros((seq_len, n_features))
    
    for i in range(n_features):
        data[:, i] = np.sin(t + i) + 0.1 * np.random.randn(seq_len)
    
    # 随机添加30%的缺失值
    missing_ratio = 0.3
    missing_indices = np.random.choice(data.size, size=int(missing_ratio * data.size), replace=False)
    data_with_missing = data.copy()
    data_with_missing.flat[missing_indices] = np.nan
    
    print(f"原始数据形状: {data.shape}")
    print(f"缺失值数量: {np.isnan(data_with_missing).sum()}")
    print(f"缺失比例: {np.isnan(data_with_missing).sum() / data.size:.2%}")
    
    # ✅ GPU加速测试
    try:
        import time
        start_time = time.time()
        
        if torch.cuda.is_available():
            # 使用GPU
            imputed_data = impute_missing_data(
                data_with_missing, 
                n_samples=30, 
                device="cuda:0"  # 指定使用GPU 0
            )
        else:
            # 回退到CPU
            imputed_data = impute_missing_data(
                data_with_missing, 
                n_samples=30, 
                device="cpu"
            )
        
        elapsed_time = time.time() - start_time
        
        print(f"填补后形状: {imputed_data.shape}")
        print(f"填补后缺失值: {np.isnan(imputed_data).sum()}")
        print(f"处理时间: {elapsed_time:.2f}秒")
        
        # 计算填补精度
        mse = np.mean((data - imputed_data) ** 2)
        print(f"填补MSE: {mse:.4f}")
        print("✅ GPU加速填补成功!")
        
    except Exception as e:
        print(f"❌ 填补失败: {e}")
        import traceback
        traceback.print_exc()