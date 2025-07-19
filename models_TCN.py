import torch as th
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, 
    so it is implemented (with some inefficiency) by simply using a standard 
    convolution with zero padding on both sides, and chopping off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()
        
        self.target = target
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)      
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        
    def forward(self, x):
        out = self.net(x)
        return self.relu(out)    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        

    def forward(self, x):
        out = self.net(x)
        return self.relu(out+x) #residual connection

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x):
        out = self.net(x)
        return self.linear(out.transpose(1,2)+x.transpose(1,2)).transpose(1,2) #residual connection

class DepthwiseNet(nn.Module):
    def __init__(self, target, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                layers += [FirstBlock(target, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            elif l==num_levels-1:
                layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class ADDSTCN(nn.Module):
    #每个特征单独训练网络
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()

        self.target=target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = th.ones(input_size,1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = th.nn.Parameter(self._attention.data)
        
        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()
                  
    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1) 
        return y1.transpose(1,2)

class MultiADDSTCN(nn.Module):
    #所有特征一起训练一整个网络
    """
    One-shot imputer for all features.
    causal_mask : ndarray/torch.BoolTensor (N, N) – parent graph with diag = 1
    """
    def __init__(self, causal_mask, num_levels, kernel_size=2, dilation_c=2, cuda=False):
        super().__init__()
        N = causal_mask.shape[0]
        self.register_buffer('C', th.tensor(causal_mask, dtype=th.float32))  # (N,N)

        # 可学习注意力, 但梯度只能流经 C==1 的位置
        self.A = nn.Parameter(th.ones_like(self.C))

        self.dwn = DepthwiseNet(
            target=None,            # 不再需要 target 索引
            num_inputs=N,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dilation_c=dilation_c,
        )
        # 逐列输出，保持深度分离
        self.pointwise = nn.Conv1d(N, N, kernel_size=1, groups=N)

        if cuda:
            self.cuda()

    def forward(self, x):
        """
        x: (B, T, N) — 全表数值，缺失已填补
        模型内部处理为 (B, N, T)
        """
        # 转置到 (B, N, T)
        x = x.transpose(1, 2)

        B, N, T = x.shape

        gate_logits = self.A * self.C - 1e6 * (1 - self.C)
        gate = th.softmax(gate_logits, dim=0)  # (N, N)

        gate_b = gate.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        x_b = x.unsqueeze(2)                      # (B, N, 1, T)

        x_gated = (x_b * gate_b).sum(dim=1)       # (B, N, T)

        h = self.dwn(x_gated)                     # (B, N, T)
        out = self.pointwise(h)                   # (B, N, T)

        return out.transpose(1, 2)                # (B, T, N) ← 与输入保持一致

class ParallelFeatureADDSTCN(nn.Module):
    #每个特征单独训练网络但是每个数据表的所有特征作为卷积的一个通道并行
    """
    并行训练每列特征的 ADDSTCN 模型集合
    input: (B, T, N) -- 多个特征
    output: (B, T, N) -- 多个预测结果
    """
    def __init__(self, causal_matrix, model_params, cuda=False):  # ✅ 添加 cuda 参数
        super().__init__()
        N = causal_matrix.shape[0]
        self.models = nn.ModuleList()

        for i in range(N):
            causal_mask = causal_matrix[:, i]  # 现在提取的是第 i 列
            input_idx = np.where(causal_mask)[0].tolist() + [i]  # 找到所有为 1 的行 + 自己
            self.models.append(
                ADDSTCN(
                    target=i,
                    input_size=len(input_idx),
                    cuda=cuda,  
                    **model_params
                )
            )
            setattr(self, f'input_idx_{i}', input_idx)  # 保存索引用于前向传播

    def forward(self, x):
        """
        x: (B, T, N)
        returns: (B, T, N)
        """
        B, T, N = x.shape
        outputs = []
        for i, model in enumerate(self.models):
            input_idx = getattr(self, f'input_idx_{i}')
            x_input = x[:, :, input_idx]  # (B, T, len(input_idx))
            
            # ✅ 转置输入以匹配 ADDSTCN 的期望格式 (B, N, T)
            x_input = x_input.transpose(1,2)  # (B, len(input_idx), T)
            
            out = model(x_input)  # (B, T, 1) - ADDSTCN 内部会转置回来
            outputs.append(out)  # append (B, T, 1)

        return th.cat(outputs, dim=-1)  # (B, T, N)