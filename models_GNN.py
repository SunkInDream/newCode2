import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def normalize_adjacency(A: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    """
    Row-normalize adjacency with self-loops: A_hat = D^{-1} (A + I).
    A: (M, M), 0/1 or weighted
    """
    M = A.size(0)
    A_hat = A + th.eye(M, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1, keepdim=True) + eps
    return A_hat / deg


class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.Ws = nn.Linear(in_dim, out_dim, bias=bias)
        self.Wn = nn.Linear(in_dim, out_dim, bias=False)
        # self.act = nn.PReLU(out_dim)            # ❌ 原来这行
        self.act = nn.GELU()                       # ✅ 任意逐元素激活都行

    def forward(self, X: th.Tensor, A_hat: th.Tensor) -> th.Tensor:
        Xs = self.Ws(X)                            # (B, T, M, Dout)
        B, T, M, Din = X.shape
        X2 = X.reshape(B * T, M, Din)
        Xn = self.Wn(X2)                           # (B*T, M, Dout)
        Xn = th.matmul(A_hat, Xn)                  # (B*T, M, Dout)
        Xn = Xn.view(B, T, M, -1)
        out = self.act(Xs + Xn)                    # 不再依赖“通道维”
        return out


class GNNKernelPerTarget(nn.Module):
    """
    一个小型的 GNN “核”，对一个目标列 i 所需的子图做逐时刻图卷积，输出 (B, T, 1)。
    - 输入: x_sub (B, T, M) 只包含父集合+自身的列
    - A_sub: (M, M) 该子图的邻接（来自 causal_matrix 的子矩阵）
    - num_layers / hidden_dim 可调
    """
    def __init__(self, A_sub: np.ndarray, num_layers: int = 2, hidden_dim: int = 32, cuda: bool = False):
        super().__init__()
        A = th.tensor(A_sub, dtype=th.float32)
        self.register_buffer('A_hat', normalize_adjacency(A))  # (M, M)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 输入是标量时间序列 -> 提升到 hidden，再堆叠图卷积层，最后投影回标量
        layers = []
        # 第一层从 1 -> hidden_dim
        layers.append(SimpleGraphConv(1, hidden_dim))
        # 中间层 hidden -> hidden
        for _ in range(num_layers - 1):
            layers.append(SimpleGraphConv(hidden_dim, hidden_dim))
        self.gnn = nn.ModuleList(layers)

        # 最后把每个节点 hidden 映射到标量，并取“目标节点”的输出
        self.readout = nn.Linear(hidden_dim, 1)

        if cuda:
            self.cuda()

    def forward(self, x_sub: th.Tensor, target_local_idx: int) -> th.Tensor:
        """
        x_sub: (B, T, M)  —— 只包含该目标的父集合+自身的序列
        target_local_idx: 该目标在子图中的局部索引（input_idx 列表里它的位置）
        return: (B, T, 1)
        """
        B, T, M = x_sub.shape
        # 扩一维作为节点特征维度 Din=1
        h = x_sub.unsqueeze(-1)  # (B, T, M, 1)
        for layer in self.gnn:
            h = layer(h, self.A_hat)  # (B, T, M, hidden)

        # per-node线性投影 -> 标量
        y = self.readout(h)  # (B, T, M, 1)
        # 取目标节点
        y_tgt = y[:, :, target_local_idx, :]  # (B, T, 1)
        return y_tgt


class ParallelFeatureGNN(nn.Module):
    """
    用 GNN 内核替换原 ParallelFeatureADDSTCN：
    - 输入:  (B, T, N)
    - 输出:  (B, T, N)
    - 仍然为每个目标特征 i，挑选其父集合 (causal_matrix[:, i]==1) + 自身，构成子图做逐时刻图卷积。
    - 时间维不在 GNN 中混合（逐时刻独立图卷积），因此形状与用途与原模型完全兼容。
    """
    def __init__(self, causal_matrix: np.ndarray, model_params: dict = None, cuda: bool = False):
        super().__init__()
        C = np.array(causal_matrix).astype(bool)
        assert C.shape[0] == C.shape[1], "causal_matrix must be square (N x N)"
        N = C.shape[0]
        self.N = N
        self.cuda_flag = cuda

        if model_params is None:
            model_params = {}
        num_layers = int(model_params.get("num_layers", 2))
        hidden_dim = int(model_params.get("hidden_dim", 32))

        self.models = nn.ModuleList()
        self.input_idx_list = []

        for i in range(N):
            # 取第 i 列：谁指向 i（父集合），再并上自身 i
            parents = np.where(C[:, i])[0].tolist()
            if i not in parents:
                parents = parents + [i]  # ensure self loop
            input_idx = sorted(list(set(parents)))
            self.input_idx_list.append(input_idx)

            # 子图邻接
            A_sub = C[np.ix_(input_idx, input_idx)].astype(np.float32)

            self.models.append(
                GNNKernelPerTarget(
                    A_sub=A_sub,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    cuda=cuda
                )
            )

        if cuda:
            self.cuda()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, T, N)
        return: (B, T, N)
        """
        B, T, N = x.shape
        outs = []
        for i, gnn_core in enumerate(self.models):
            input_idx = self.input_idx_list[i]   # 父集合+自身的列索引
            # 构建子图输入 (B, T, M)
            x_sub = x[:, :, input_idx]
            # 目标在子图里的局部索引
            target_local_idx = input_idx.index(i)
            # (B, T, 1)
            y_i = gnn_core(x_sub, target_local_idx)
            outs.append(y_i)
        # 拼回 (B, T, N)
        return th.cat(outs, dim=-1)