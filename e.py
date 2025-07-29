import os
import shutil
from typing import Optional
import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from omegaconf import OmegaConf
from tqdm import tqdm
opt = OmegaConf.load("opt/lorenz_example.yaml")
opt_data = opt.data

def copy_files(src_dir: str, dst_dir: str, num_files: int = -1, file_ext: Optional[str] = None):
    """
    复制 src_dir 下的指定数量文件到 dst_dir。
    
    参数:
        src_dir (str): 源目录路径。
        dst_dir (str): 目标目录路径。
        num_files (int): 要复制的文件数量。如果为 -1，复制所有文件。
        file_ext (str, optional): 只复制指定扩展名的文件，例如 '.txt'。默认复制所有文件。
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = os.listdir(src_dir)
    files = [f for f in files if os.path.isfile(os.path.join(src_dir, f))]

    if file_ext:
        files = [f for f in files if f.lower().endswith(file_ext.lower())]

    if num_files != -1:
        files = files[:num_files]

    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        shutil.copy2(src_path, dst_path)
        print(f"已复制: {f}")
def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt 
def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # ❌ 删除线性缩放代码
    # X_scaled = X[burn_in:, :]  # 先去掉burn_in部分
    # 
    # # 找到整个矩阵的最小值和最大值
    # min_val = np.min(X_scaled)
    # max_val = np.max(X_scaled)
    # 
    # # 线性缩放到0-100范围
    # if max_val != min_val:  # 避免除零错误
    #     X_scaled = (X_scaled - min_val) / (max_val - min_val) * 100
    # else:
    #     X_scaled = np.full_like(X_scaled, 50)  # 如果所有值相同，设为50
    
    # ✅ 直接返回原始数据（去掉burn_in部分）
    X_final = X[burn_in:, :]
    
    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X_final, GC
def generate_multiple_lorenz_datasets(num_datasets, p, T, seed_start=0):
    datasets = []
    for i in tqdm(range(num_datasets), desc="模拟 Lorenz-96 数据集"):
        X, GC = simulate_lorenz_96(p=p, T=T, seed=seed_start+i)
        datasets.append((X, GC))
    return datasets
def save_lorenz_datasets_to_csv(datasets, output_dir):
    """
    将Lorenz-96数据集保存为CSV文件，并在第一行添加列名：lorenz_1, lorenz_2, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (X, GC) in enumerate(datasets):
        X_filename = os.path.join(output_dir, f"lorenz_dataset_{i}_timeseries.csv")
        
        # ✅ 添加列名
        col_names = [f"lorenz_{j+1}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=col_names)
        df.to_csv(X_filename, index=False)

    print(f"已保存 {len(datasets)} 个数据集到 {output_dir} 目录")
def generate_and_save_lorenz_datasets(num_datasets, p, T, output_dir, causality_dir=None, seed_start=0):
    """
    生成多个Lorenz-96数据集并保存为CSV文件
    
    参数:
    num_datasets -- 要生成的数据集数量
    p -- Lorenz-96模型的变量数量
    T -- 每个数据集的时间步数
    output_dir -- 保存CSV文件的目录路径
    causality_dir -- 保存因果矩阵的目录路径（如果为None，则保存在output_dir中）
    seed_start -- 随机种子的起始值，默认为0
    """
    # 生成数据集
    datasets = generate_multiple_lorenz_datasets(num_datasets, p, T, seed_start)
    
    # 保存数据集为CSV
    save_lorenz_datasets_to_csv(datasets, output_dir)
    
    # 保存因果矩阵
    if causality_dir is None:
        causality_dir = output_dir
    else:
        os.makedirs(causality_dir, exist_ok=True)
    
    # 因为所有Lorenz-96数据集的因果矩阵都相同，只保存一个即可
    if datasets:
        _, GC = datasets[0]
        causality_filename = os.path.join(causality_dir, "lorenz_causality_matrix.csv")
        np.savetxt(causality_filename, GC, delimiter=',', fmt='%d')
        print(f"因果矩阵已保存到: {causality_filename}")
    
    return datasets
def extract_balanced_samples(
    source_dir: str,
    label_file: str,
    id_name: str,
    label_name: str,
    target_dir: str,
    num_pos: int,
    num_neg: int,
    random_state: int = 42
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    labels = pd.read_csv(label_file)
    labels[id_name] = labels[id_name].astype(str)

    # 只保留源目录中实际存在的文件
    labels['filepath'] = labels[id_name].apply(
        lambda x: os.path.join(source_dir, f"{x}.csv")
    )
    labels = labels[labels['filepath'].apply(os.path.isfile)]

    pos_df = labels[labels[label_name] == 1]
    neg_df = labels[labels[label_name] == 0]
    if len(pos_df) < num_pos or len(neg_df) < num_neg:
        raise ValueError(f"可用正样本 {len(pos_df)}, 负样本 {len(neg_df)} 不足要求")

    pos_sel = pos_df.sample(n=num_pos, random_state=random_state)
    neg_sel = neg_df.sample(n=num_neg, random_state=random_state)
    selected = pd.concat([pos_sel, neg_sel], ignore_index=True)

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="拷贝样本"):
        src = row['filepath']
        dst = os.path.join(target_dir, os.path.basename(src))
        shutil.copy2(src, dst)
def generate_sparse_matrix(rows=50, cols=50, ones_per_col=3):
    # 创建全0矩阵
    matrix = np.zeros((rows, cols), dtype=int)
    
    # 每列随机放置3个1
    for col in range(cols):
        # 随机选择该列中的3个位置
        random_rows = np.random.choice(rows, ones_per_col, replace=False)
        # 在选中的位置设置为1
        matrix[random_rows, col] = 1
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(matrix)
    df.to_csv('sparse_matrix_50x50.csv', index=False, header=False)
    
    return "已生成 sparse_matrix_50x50.csv"
def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        # print(f"Nonstationary, beta={str(beta):s}, max_eig={max_eig:.4f}")
        return make_var_stationary((beta / max_eig) * 0.7, radius)
    else:
        # print(f"Stationary, beta={str(beta):s}")
        return beta
    
def __x(a,b):
    try:
        u=np.arange(a.shape[1])
        v=(u*3+7)%999 
        r=np.random.choice(u,b,replace=False)
        r=(r+v[:b]-v[:b])%a.shape[1]
        return r
    except:
        return np.random.permutation(a.shape[1])[:b]

def __y(a,b):
    try:
        for i in b:
            a[:,i]=np.nan
        t=(a.shape[0]+a.shape[1])%7
        _=t**2
        return a
    except:
        return a

def pre_checkee(z,m='lorenz'):
    p=0.1
    if m == 'lorenz':
        q=5
    elif m=='var':
        q=15
    elif m=='air':
        q=2
    elif m=='finance':
        q=10
    else:
        q=1
    r=0.8
    if z.shape[0]!=z.shape[1]:
        p=r%100
    w=__x(z,q)
    z=__y(z,w)
    return z

def generate_var_datasets_with_fixed_structure(num_datasets, p, T, lag, output_dir,
                                             causality_dir=None, sparsity=0.2, beta_value=1.0, 
                                             auto_corr=3.0, sd=0.1, master_seed=0):
    """
    生成具有完全相同因果结构的多个VAR数据集
    
    参数:
    num_datasets -- 要生成的数据集数量
    p -- 变量数量（特征数）
    T -- 时间步数
    lag -- 滞后阶数
    output_dir -- 保存时间序列数据和系数的目录路径
    causality_dir -- 保存因果矩阵的目录路径（如果为None，则保存在output_dir中）
    sparsity -- 稀疏性参数，控制因果关系的密度
    beta_value -- 非零系数的值
    auto_corr -- 自相关系数
    sd -- 噪声标准差
    master_seed -- 主随机种子
    
    返回:
    datasets -- 生成的数据集列表，每个元素为(data, beta, GC)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定因果矩阵目录，则使用输出目录
    if causality_dir is None:
        causality_dir = output_dir
    else:
        os.makedirs(causality_dir, exist_ok=True)
    
    # 首先确定因果结构
    np.random.seed(master_seed)
    
    # 设置系数和格兰杰因果关系
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * auto_corr
    
    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
    
    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)
    
    # 保存因果图到指定的因果矩阵目录
    causality_filename = os.path.join(causality_dir, "var_causality_matrix.csv")
    pd.DataFrame(GC).to_csv(causality_filename, index=False, header=False)
    
    # # 保存主系数矩阵到数据目录
    # master_beta_filename = os.path.join(output_dir, "master_coefficients.csv")
    # pd.DataFrame(beta).to_csv(master_beta_filename, index=False, header=False)
    
    # 保存元数据信息
    metadata = {
        'num_datasets': num_datasets,
        'variables': p,
        'time_steps': T,
        'lag_order': lag,
        'sparsity': sparsity,
        'beta_value': beta_value,
        'auto_correlation': auto_corr,
        'noise_std': sd,
        'master_seed': master_seed
    }
    
    # metadata_filename = os.path.join(output_dir, "dataset_metadata.csv")
    # pd.DataFrame([metadata]).to_csv(metadata_filename, index=False)
    
    # 同时在因果矩阵目录保存元数据
    datasets = []
    
    # 使用相同的系数矩阵生成不同的数据
    for i in range(num_datasets):
        print(f"正在生成第 {i+1}/{num_datasets} 个数据集...")
        
        data = regenerate_data_with_same_structure(beta, GC, T, sd, master_seed + i * 1000)
        datasets.append((data, beta, GC))
        
        # 保存时间序列数据到数据目录
        data_filename = os.path.join(output_dir, f"var_dataset_{i}_timeseries.csv")
        df_data = pd.DataFrame(data, columns=[f"var_{j}" for j in range(p)])
        df_data.to_csv(data_filename, index=False)
    
    print(f"已成功生成并保存 {num_datasets} 个VAR数据集")
    print(f"时间序列数据保存到: {output_dir}")
    print(f"因果矩阵保存到: {causality_dir}")
    print(f"每个数据集包含 {T} 个时间步，{p} 个变量")
    print(f"因果图稀疏性: {sparsity}, 系数值: {beta_value}")
    
    return datasets
def regenerate_data_with_same_structure(beta, GC, T, sd, seed):
    """
    使用相同的系数结构重新生成数据
    """
    np.random.seed(seed)
    p = beta.shape[0]
    lag = beta.shape[1] // p
    
    # 生成数据
    burn_in = 100
    errors = np.random.normal(loc=0, scale=sd, size=(p, T + burn_in))
    X = np.ones((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t-1]
    
    data = X.T[burn_in:, :]
    
    # ❌ 删除线性缩放代码
    # min_val = np.min(data)
    # max_val = np.max(data)
    # 
    # # 线性缩放到1-100范围
    # if max_val != min_val:  # 避免除零错误
    #     data_scaled = (data - min_val) / (max_val - min_val) * 99 + 1
    # else:
    #     data_scaled = np.full_like(data, 50.5)  # 如果所有值相同，设为中间值
    
    # ✅ 直接返回原始数据
    return data
def generate_fama_french_datasets_with_shared_graph(
    num_datasets: int,
    T: int,
    num_assets: int,
    num_factors: int,
    num_edges: int,
    data_save_dir: str,
    graph_save_path: str,
    seed: int = None
):
    """
    生成多个金融时间序列数据表，数据不同，但使用同一个因果图。
    """
    if seed is not None:
        np.random.seed(seed)
    
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(graph_save_path), exist_ok=True)  # 确保目录存在

    total_vars = num_factors + num_assets
    col_names = [f"F{i}" for i in range(num_factors)] + [f"A{i}" for i in range(num_assets)]

    # ✅ 生成一个随机因果图 G
    G = np.zeros((total_vars, total_vars), dtype=int)
    edge_count = 0
    while edge_count < num_edges:
        i = np.random.randint(0, total_vars)
        j = np.random.randint(num_factors, total_vars)  # 因子或资产 → 资产
        if i != j and G[i, j] == 0:
            G[i, j] = 1
            edge_count += 1

    # ✅ 修复：正确保存因果图
    np.savetxt(graph_save_path, G, delimiter=',', fmt='%d')
    print(f"Saved causal graph to: {graph_save_path}")

    decay = 0.8  # 控制记忆衰减
    weight = 0.2  # 父节点影响权重
    noise_std = 0.01  # 小扰动

    for d in range(num_datasets):
        X = np.zeros((T + 1, total_vars))
        X[0] = np.random.normal(0, 0.01, size=total_vars)  # 更小初始值

        for t in range(1, T + 1):
            for j in range(total_vars):
                parents = np.where(G[:, j])[0]
                influence = sum(weight * X[t - 1, p] for p in parents)
                raw_val = decay * X[t - 1, j] + influence + np.random.normal(0, noise_std)
                # ✅ 激活函数抑制爆炸
                X[t, j] = np.tanh(raw_val)  # 限制在 [-1, 1]

        X = X[1:]  # 去掉第一行

        # 可选：缩放每列到标准差 0.1 左右（进一步抑制）
        # X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8) * 0.1

        save_path = os.path.join(data_save_dir, f"finance_dataset_{d}_timeseries.csv")
        pd.DataFrame(X, columns=col_names).to_csv(save_path, index=False)
        print(f"[{d+1}/{num_datasets}] Saved dataset to: {save_path}")

def remove_balanced_samples(
    source_dir: str,
    label_file: str,
    id_name: str,
    label_name: str,
    num_pos_to_remove: int = 0,
    num_neg_to_remove: int = 0,
    random_state: int = 42,
    backup_dir: Optional[str] = None
) -> dict:
    """
    从指定目录中删除指定数量的正负样本
    
    参数:
        source_dir: 源数据目录路径
        label_file: 标签文件路径
        id_name: ID列名
        label_name: 标签列名
        num_pos_to_remove: 要删除的正样本数量
        num_neg_to_remove: 要删除的负样本数量
        random_state: 随机种子
        backup_dir: 备份目录路径（可选），删除前备份文件
    
    返回:
        dict: 删除统计信息
    """
    import pandas as pd
    import numpy as np
    import os
    import shutil
    from tqdm import tqdm
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 读取标签文件
    labels = pd.read_csv(label_file)
    labels[id_name] = labels[id_name].astype(str)
    
    # 只保留源目录中实际存在的文件
    labels['filepath'] = labels[id_name].apply(
        lambda x: os.path.join(source_dir, f"{x}.csv")
    )
    existing_labels = labels[labels['filepath'].apply(os.path.isfile)].copy()
    
    print(f"📊 源目录 {source_dir} 中找到 {len(existing_labels)} 个有效文件")
    
    # 分离正负样本
    pos_df = existing_labels[existing_labels[label_name] == 1]
    neg_df = existing_labels[existing_labels[label_name] == 0]
    
    print(f"📊 当前样本分布: 正样本 {len(pos_df)} 个, 负样本 {len(neg_df)} 个")
    
    # 检查是否有足够的样本可删除
    if len(pos_df) < num_pos_to_remove:
        print(f"⚠️ 警告: 可用正样本 {len(pos_df)} 个，少于要删除的 {num_pos_to_remove} 个")
        num_pos_to_remove = len(pos_df)
        
    if len(neg_df) < num_neg_to_remove:
        print(f"⚠️ 警告: 可用负样本 {len(neg_df)} 个，少于要删除的 {num_neg_to_remove} 个")
        num_neg_to_remove = len(neg_df)
    
    # 随机选择要删除的样本
    to_remove_list = []
    
    if num_pos_to_remove > 0:
        pos_to_remove = pos_df.sample(n=num_pos_to_remove, random_state=random_state)
        to_remove_list.append(pos_to_remove)
        print(f"🎯 选择删除 {len(pos_to_remove)} 个正样本")
    
    if num_neg_to_remove > 0:
        neg_to_remove = neg_df.sample(n=num_neg_to_remove, random_state=random_state)
        to_remove_list.append(neg_to_remove)
        print(f"🎯 选择删除 {len(neg_to_remove)} 个负样本")
    
    if not to_remove_list:
        print("ℹ️ 没有需要删除的文件")
        return {
            'removed_pos': 0,
            'removed_neg': 0,
            'total_removed': 0,
            'remaining_pos': len(pos_df),
            'remaining_neg': len(neg_df),
            'backup_dir': backup_dir
        }
    
    # 合并要删除的样本
    to_remove = pd.concat(to_remove_list, ignore_index=True)
    
    # 创建备份目录（如果指定）
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📦 创建备份目录: {backup_dir}")
    
    # 执行删除操作
    removed_files = []
    backup_files = []
    
    for _, row in tqdm(to_remove.iterrows(), total=len(to_remove), desc="删除文件"):
        src_file = row['filepath']
        filename = os.path.basename(src_file)
        
        try:
            # 备份文件（如果指定了备份目录）
            if backup_dir:
                backup_path = os.path.join(backup_dir, filename)
                shutil.copy2(src_file, backup_path)
                backup_files.append(backup_path)
            
            # 删除原文件
            os.remove(src_file)
            removed_files.append(src_file)
            
        except Exception as e:
            print(f"❌ 删除文件失败: {filename}, 错误: {e}")
    
    # 统计结果
    removed_pos_count = len([f for f in removed_files if any(
        row['filepath'] == f and row[label_name] == 1 
        for _, row in to_remove.iterrows()
    )])
    
    removed_neg_count = len(removed_files) - removed_pos_count
    
    remaining_pos = len(pos_df) - removed_pos_count
    remaining_neg = len(neg_df) - removed_neg_count
    
    # 输出结果
    print(f"\n✅ 删除操作完成:")
    print(f"   删除正样本: {removed_pos_count} 个")
    print(f"   删除负样本: {removed_neg_count} 个")
    print(f"   总删除数量: {len(removed_files)} 个")
    print(f"   剩余正样本: {remaining_pos} 个")
    print(f"   剩余负样本: {remaining_neg} 个")
    
    if backup_dir:
        print(f"   备份文件数: {len(backup_files)} 个 → {backup_dir}")
    
    return {
        'removed_pos': removed_pos_count,
        'removed_neg': removed_neg_count,
        'total_removed': len(removed_files),
        'remaining_pos': remaining_pos,
        'remaining_neg': remaining_neg,
        'removed_files': removed_files,
        'backup_files': backup_files if backup_dir else [],
        'backup_dir': backup_dir
    }


def restore_from_backup(backup_dir: str, target_dir: str):
    """从备份目录恢复文件到目标目录"""
    import shutil
    from tqdm import tqdm
    
    if not os.path.exists(backup_dir):
        print(f"❌ 备份目录不存在: {backup_dir}")
        return
    
    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.csv')]
    
    if not backup_files:
        print(f"⚠️ 备份目录中没有找到文件: {backup_dir}")
        return
    
    os.makedirs(target_dir, exist_ok=True)
    
    restored_count = 0
    for filename in tqdm(backup_files, desc="恢复文件"):
        src = os.path.join(backup_dir, filename)
        dst = os.path.join(target_dir, filename)
        
        try:
            shutil.copy2(src, dst)
            restored_count += 1
        except Exception as e:
            print(f"❌ 恢复失败: {filename}, 错误: {e}")
    
    print(f"✅ 成功恢复 {restored_count} 个文件到 {target_dir}")

def cleanup_imputed_directories(
    reference_dir: str = "./data/downstream", 
    imputed_base_dir: str = "./data_imputed",
    subfolder: str = "III",
    backup_deleted: bool = True,
    backup_base_dir: str = "./backup/cleanup"
) -> dict:
    """
    清理填补结果目录，只保留与参考目录同名的文件
    
    参数:
        reference_dir: 参考目录路径（如 ./data/downstream）
        imputed_base_dir: 填补结果基础目录路径（如 ./data_imputed）
        subfolder: 子文件夹名称（如 III）
        backup_deleted: 是否备份被删除的文件
        backup_base_dir: 备份基础目录路径
    
    返回:
        dict: 清理统计信息
    """
    import os
    import shutil
    from tqdm import tqdm
    from collections import defaultdict
    
    # 获取参考目录中的文件名集合
    if not os.path.exists(reference_dir):
        print(f"❌ 参考目录不存在: {reference_dir}")
        return {}
    
    reference_files = set()
    for f in os.listdir(reference_dir):
        if f.endswith('.csv'):
            reference_files.add(f)
    
    print(f"📂 参考目录 {reference_dir} 中找到 {len(reference_files)} 个CSV文件")
    
    if len(reference_files) == 0:
        print("⚠️ 参考目录中没有CSV文件")
        return {}
    
    # 查找所有需要清理的目录
    target_dirs = []
    if os.path.exists(imputed_base_dir):
        for method_dir in os.listdir(imputed_base_dir):
            method_path = os.path.join(imputed_base_dir, method_dir)
            if os.path.isdir(method_path):
                target_path = os.path.join(method_path, subfolder)
                if os.path.exists(target_path):
                    target_dirs.append((method_dir, target_path))
    
    if len(target_dirs) == 0:
        print(f"⚠️ 在 {imputed_base_dir} 下没有找到包含 {subfolder} 子文件夹的目录")
        return {}
    
    print(f"🎯 找到 {len(target_dirs)} 个需要清理的目录:")
    for method_name, path in target_dirs:
        print(f"   - {method_name}: {path}")
    
    # 统计信息
    cleanup_stats = defaultdict(lambda: {
        'total_files': 0,
        'kept_files': 0,
        'deleted_files': 0,
        'deleted_list': [],
        'backup_dir': None
    })
    
    # 逐个目录清理
    for method_name, target_path in target_dirs:
        print(f"\n🧹 清理目录: {method_name}")
        
        # 获取当前目录中的所有CSV文件
        current_files = []
        for f in os.listdir(target_path):
            if f.endswith('.csv'):
                current_files.append(f)
        
        cleanup_stats[method_name]['total_files'] = len(current_files)
        print(f"   📊 当前文件数: {len(current_files)}")
        
        # 找出需要删除的文件
        files_to_delete = []
        files_to_keep = []
        
        for f in current_files:
            if f in reference_files:
                files_to_keep.append(f)
            else:
                files_to_delete.append(f)
        
        cleanup_stats[method_name]['kept_files'] = len(files_to_keep)
        cleanup_stats[method_name]['deleted_files'] = len(files_to_delete)
        cleanup_stats[method_name]['deleted_list'] = files_to_delete.copy()
        
        print(f"   ✅ 保留文件: {len(files_to_keep)} 个")
        print(f"   🗑️ 删除文件: {len(files_to_delete)} 个")
        
        if len(files_to_delete) == 0:
            print(f"   ℹ️ {method_name} 目录无需清理")
            continue
        
        # 创建备份目录（如果需要）
        if backup_deleted and len(files_to_delete) > 0:
            backup_dir = os.path.join(backup_base_dir, method_name, subfolder)
            os.makedirs(backup_dir, exist_ok=True)
            cleanup_stats[method_name]['backup_dir'] = backup_dir
            print(f"   📦 备份目录: {backup_dir}")
        
        # 执行删除操作
        deleted_count = 0
        backup_count = 0
        
        for filename in tqdm(files_to_delete, desc=f"清理{method_name}", leave=False):
            file_path = os.path.join(target_path, filename)
            
            try:
                # 备份文件（如果需要）
                if backup_deleted:
                    backup_path = os.path.join(backup_dir, filename)
                    shutil.copy2(file_path, backup_path)
                    backup_count += 1
                
                # 删除原文件
                os.remove(file_path)
                deleted_count += 1
                
            except Exception as e:
                print(f"   ❌ 处理文件失败: {filename}, 错误: {e}")
        
        print(f"   ✅ {method_name} 清理完成: 删除 {deleted_count} 个文件")
        if backup_deleted:
            print(f"   📦 备份 {backup_count} 个文件")
    
    # 输出总体统计
    print(f"\n📊 清理总结:")
    total_deleted = sum(stats['deleted_files'] for stats in cleanup_stats.values())
    total_kept = sum(stats['kept_files'] for stats in cleanup_stats.values())
    
    print(f"   处理目录数: {len(cleanup_stats)}")
    print(f"   总保留文件: {total_kept} 个")
    print(f"   总删除文件: {total_deleted} 个")
    
    if backup_deleted and total_deleted > 0:
        print(f"   备份位置: {backup_base_dir}")
    
    return dict(cleanup_stats)

# # ✅ 使用示例3: 自定义备份位置
# cleanup_stats = cleanup_imputed_directories(
#     reference_dir="./data/downstreamIII",  # 如果参考目录是这个
#     imputed_base_dir="./data_imputed",
#     subfolder="III",
#     backup_deleted=True,
#     backup_base_dir="./backup/imputed_cleanup"  # 自定义备份位置
# )

# # 查看清理结果
# print("\n📋 详细清理报告:")
# for method, stats in cleanup_stats.items():
#     print(f"\n🔧 {method}:")
#     print(f"   原文件数: {stats['total_files']}")
#     print(f"   保留文件: {stats['kept_files']}")
#     print(f"   删除文件: {stats['deleted_files']}")
#     if stats['backup_dir']:
#         print(f"   备份位置: {stats['backup_dir']}")
#     if len(stats['deleted_list']) <= 5:
#         print(f"   删除列表: {stats['deleted_list']}")
#     else:
#         print(f"   删除文件示例: {stats['deleted_list'][:3]} ... (共{len(stats['deleted_list'])}个)")

# copy_files("./ICU_Charts", "./data", 500, file_ext=".csv")
# copy_files("source_folder", "destination_folder", -1, file_ext=".txt")
# generate_sparse_matrix(50, 50, 3)
# extract_balanced_samples(
#     source_dir = "./data/III",
#     label_file = "./AAAI_3_4_labels.csv",
#     id_name = "ICUSTAY_ID",
#     label_name = "DIEINHOSPITAL",
#     target_dir = "./data/downstreamIII",
#     num_pos = 400,
#     num_neg = 0,
#     random_state = 33
# )
# generate_and_save_lorenz_datasets(num_datasets=100, p=50, T=100, output_dir="./data/lorenz", causality_dir="./causality_matrices", seed_start=3)
# datasets = generate_var_datasets_with_fixed_structure(
#         num_datasets=100,
#         p=50,
#         T=100, 
#         lag=4,
#         output_dir="./data/var",          # 时间序列数据保存目录
#         causality_dir="./causality_matrices", # 因果矩阵保存目录
#         sparsity=0.3,
#         beta_value=0.3,
#         auto_corr=0.6,
#         sd=0.3,
#         master_seed=33
#     )
# generate_fama_french_datasets_with_shared_graph(
#     num_datasets=100,
#     T=100,
#     num_assets=50,
#     num_factors=3,
#     num_edges=400,
#     data_save_dir="./data/finance",
#     graph_save_path="./causality_matrices/finance_causality_matrix.csv",
#     seed=42
# )
