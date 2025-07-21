import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000.0) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class DenoisingGRUModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.cond_proj = nn.Linear(feature_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        x_proj = self.input_proj(x)
        cond_proj = self.cond_proj(cond)
        blend = x_proj + t_emb + cond_proj
        gru_out, _ = self.gru(blend)
        return self.output_proj(gru_out)

class TSDE_Imputer(nn.Module):
    def __init__(self, feature_dim, seq_len, device="cuda", hidden_dim=64, num_steps=50):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps
        self.model = DenoisingGRUModel(feature_dim, hidden_dim).to(self.device)

        betas = torch.linspace(0.0001, 0.02, num_steps)
        alphas = 1. - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        a_t = self.alphas_cumprod[t].to(x0.device).view(-1, 1, 1)
        xt = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        return xt, noise

    def reverse_step(self, xt, t, cond):
        B = xt.size(0)
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
        pred_noise = self.model(xt, t_tensor, cond)
        a = self.alphas[t].to(xt.device)
        a_bar = self.alphas_cumprod[t].to(xt.device)
        term1 = (xt - (1 - a).sqrt() / (1 - a_bar).sqrt() * pred_noise) / a.sqrt()
        if t > 0:
            term1 += torch.randn_like(xt) * self.betas[t].to(xt.device).sqrt()
        return term1


    def train_on_instance(self, x, mask, epochs=200, lr=1e-3):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            t = torch.randint(0, self.num_steps, (x.size(0),), device=self.device)
            xt, noise = self.forward_diffusion(x, t)
            pred_noise = self.model(xt, t, x)
            loss = F.mse_loss(pred_noise * mask, noise * mask)
            loss.backward()
            opt.step()
            opt.zero_grad()

    @torch.no_grad()
    def impute(self, x_obs, mask, n_samples=50):
        self.eval()
        samples = []
        for _ in range(n_samples):
            xt = torch.randn_like(x_obs)
            for t in reversed(range(self.num_steps)):
                xt = self.reverse_step(xt, t, x_obs)
                xt = mask * x_obs + (1 - mask) * xt
            samples.append(xt)
        samples = torch.stack(samples, dim=1)
        result = torch.median(samples, dim=1)[0]
        return mask * x_obs + (1 - mask) * result  # 强制还原观测值

def impute_missing_data(data: np.ndarray, n_samples=50, device="cuda", epochs=200) -> np.ndarray:
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("输入数据必须是二维 numpy 数组")

    B, F = data.shape
    mask = ~np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    col_means[np.isnan(col_means)] = 0
    filled = np.where(mask, data, col_means)

    # 标准化
    mean, std = np.nanmean(filled), np.nanstd(filled) + 1e-6
    normed = (filled - mean) / std

    # 构造张量
    x = torch.FloatTensor(normed).unsqueeze(0).to(device)
    m = torch.FloatTensor(mask.astype(float)).unsqueeze(0).to(device)

    model = TSDE_Imputer(F, B, device=device)
    model.train_on_instance(x, m, epochs=epochs)

    set_seed(42)
    imputed = model.impute(x, m, n_samples=n_samples)
    imputed = imputed.squeeze(0).cpu().numpy() * std + mean
    final_result = data.copy()
    final_result[np.isnan(data)] = imputed[np.isnan(data)]
    return final_result
