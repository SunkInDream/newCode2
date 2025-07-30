# import os
# import pandas as pd
# import numpy as np
# import torch
# from multiprocessing import Process, Queue, current_process

# # ========= 兜底均值填补 =========
# def mean_impu(mx: np.ndarray) -> np.ndarray:
#     mx = mx.copy().astype(np.float32)
#     col_means = np.nanmean(mx, axis=0)
#     col_means = np.where(np.isnan(col_means), 0.0, col_means)
#     inds = np.where(np.isnan(mx))
#     if inds[0].size > 0:
#         mx[inds] = np.take(col_means, inds[1])
#     return mx

# # ========= KNN 填补（按你提供的实现）=========
# def knn_impu(mx, k=5):
#     from sklearn.impute import KNNImputer

#     mx = mx.copy()
#     original_shape = mx.shape

#     # 确保 k 不超过“无缺失行”的数量
#     non_nan_rows = np.sum(~np.isnan(mx).any(axis=1))
#     if non_nan_rows == 0:
#         return mean_impu(mx)

#     k = min(k, max(1, non_nan_rows - 1))

#     try:
#         imputer = KNNImputer(n_neighbors=k)
#         result = imputer.fit_transform(mx)

#         # 确保输出形状一致
#         if result.shape != original_shape:
#             result = result[:original_shape[0], :original_shape[1]]
#         return result.astype(np.float32)

#     except Exception as e:
#         print(f"KNN imputation failed: {e}, falling back to mean imputation")
#         return mean_impu(mx).astype(np.float32)

# # ========= 子进程：处理队列 =========
# def gpu_worker(file_queue, input_dir, output_dir, gpu_id, use_gpu=True):
#     # MICE/KNN 在 CPU 上运行，这里仅保留 GPU 隔离框架以统一管理
#     os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if use_gpu else ""

#     pid = current_process().pid

#     while not file_queue.empty():
#         try:
#             fname = file_queue.get_nowait()
#         except Exception:
#             break

#         out_path = os.path.join(output_dir, fname)
#         # 二次保护：已存在则跳过
#         if os.path.exists(out_path):
#             print(f"[PID {pid} GPU {gpu_id if use_gpu else 'CPU'}] ⏩ 跳过已存在文件: {fname}")
#             continue

#         try:
#             fpath = os.path.join(input_dir, fname)
#             mx = pd.read_csv(fpath).values.astype(np.float32)

#             filled = knn_impu(mx, k=5)

#             pd.DataFrame(filled).to_csv(out_path, index=False)
#             print(f"[PID {pid} GPU {gpu_id if use_gpu else 'CPU'}] ✅ 处理完成: {fname}")
#         except Exception as e:
#             print(f"[PID {pid} GPU {gpu_id if use_gpu else 'CPU'}] ❌ 处理失败: {fname}，错误：{e}")

# # ========= 主调度：多进程并行 =========
# def parallel_knn_impute(input_dir, output_dir, n_processes_per_gpu=2):
#     os.makedirs(output_dir, exist_ok=True)

#     # 仅把“目标中不存在”的文件入队
#     file_list = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
#     file_list = [f for f in file_list if not os.path.exists(os.path.join(output_dir, f))]

#     file_queue = Queue()
#     for fname in file_list:
#         file_queue.put(fname)

#     num_gpus = torch.cuda.device_count()
#     use_gpu = num_gpus > 0
#     processes = []

#     if use_gpu:
#         total_procs = num_gpus * n_processes_per_gpu
#         print(f"🚀 检测到 {num_gpus} 张 GPU；每卡 {n_processes_per_gpu} 进程，共 {total_procs} 进程处理 {len(file_list)} 个文件")
#         for gpu_id in range(num_gpus):
#             for _ in range(n_processes_per_gpu):
#                 p = Process(target=gpu_worker, args=(file_queue, input_dir, output_dir, gpu_id, True))
#                 p.start()
#                 processes.append(p)
#     else:
#         print(f"⚠️ 未检测到 GPU，改为 CPU 单进程处理 {len(file_list)} 个文件")
#         p = Process(target=gpu_worker, args=(file_queue, input_dir, output_dir, 0, False))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     print("✅ 所有文件处理完成！")

# # ========= 入口 =========
# if __name__ == "__main__":
#     import multiprocessing as mp
#     mp.set_start_method("spawn", force=True)

#     input_dir = "./data/downstreamIII"
#     output_dir = "./data_imputed/knn/III"
#     parallel_knn_impute(input_dir, output_dir, n_processes_per_gpu=2)
# from pathlib import Path

# root = Path("./data_imputed/my_model/III")  # 改成你的目标文件夹路径
# for p in root.iterdir():
#     if p.is_file() and "_imputed" in p.name:
#         new_path = p.with_name(p.name.replace("_imputed", ""))
#         if new_path.exists():
#             print(f"[冲突] 跳过：{p.name} -> {new_path.name}（目标已存在）")
#             continue
#         p.rename(new_path)
#         print(f"重命名：{p.name} -> {new_path.name}")
#!/usr/bin/env python3
# from pathlib import Path

# TARGET_ROWS = 193  # 193 行：第 1 行是列名 + 192 行数据

# def pad_csv_to_193_lines(fp: Path):
#     # 读取全部行（通常 CSV 行数不大；若特别大可改为只读取末尾）
#     try:
#         text = fp.read_text(encoding="utf-8", errors="ignore")
#     except Exception:
#         # 若编码不是 utf-8，可改成 fp.read_bytes() 再手动处理
#         print(f"[跳过] 无法读取：{fp}")
#         return

#     lines = text.splitlines(keepends=True)  # 保留换行符，便于原样追加
#     n = len(lines)

#     if n == 0:
#         print(f"[跳过] 空文件：{fp.name}")
#         return

#     if n >= TARGET_ROWS:
#         # 已达标或超过，不处理
#         # print(f"[OK] {fp.name} 已有 {n} 行")
#         return

#     if n == 1:
#         # 只有表头，没有数据行，无法复制“最后一行数据”
#         print(f"[警告] 仅有表头，无数据可复制：{fp.name}（当前 1 行）")
#         return

#     # 最后一行数据（保留末尾换行；若没有则补一个）
#     last_line = lines[-1]
#     if not last_line.endswith(("\n", "\r")):
#         last_line += "\n"

#     need = TARGET_ROWS - n
#     # 若原文件最后一行没有换行，先补一个换行，再开始追加
#     need_prefix_newline = not lines[-1].endswith(("\n", "\r"))
#     to_append = (("\n" if need_prefix_newline else "") + last_line * need)

#     # 追加写回（更安全可写入临时文件再替换）
#     with fp.open("a", encoding="utf-8", newline="") as f:
#         f.write(to_append)

#     print(f"[补全] {fp.name}: {n} -> {TARGET_ROWS} 行，复制最后一行 {need} 次")

# def main():
#     folder = Path("./data_imputed/grin/III")  # ← 改成你的目标文件夹
#     for fp in folder.glob("*.csv"):        # 若要所有文件改为：for fp in folder.iterdir():
#         if fp.is_file():
#             pad_csv_to_193_lines(fp)

# if __name__ == "__main__":
#     main()
# import os
# import numpy as np
# import pandas as pd
# from multiprocessing import Pool
# from sklearn.impute import KNNImputer

# # ====== 简单填补函数 ======
# def zero_impu(mx):
#     return np.where(np.isnan(mx), 0, mx)

# def mean_impu(mx):
#     df = pd.DataFrame(mx)
#     return df.fillna(df.mean()).to_numpy()

# # ====== KNN 填补核心 ======
# def knn_impu(mx, k=3):
#     mx = mx.copy()
#     original_shape = mx.shape

#     # 1. 先找出整列全 NaN 的列
#     all_nan_cols = np.all(np.isnan(mx), axis=0)
#     if all_nan_cols.any():
#         global_mean = np.nanmean(mx)
#         if np.isnan(global_mean):
#             global_mean = 0
#         # 这些列先用全局均值填
#         mx[:, all_nan_cols] = global_mean

#     # 2. 计算无缺失行数
#     non_nan_rows = np.sum(~np.isnan(mx).any(axis=1))
#     if non_nan_rows == 0:
#         # 所有行都有缺失，只能退回均值填补
#         return mean_impu(mx)

#     k = min(k, max(1, non_nan_rows - 1))
#     try:
#         imputer = KNNImputer(n_neighbors=k)
#         result = imputer.fit_transform(mx)

#         # 3. 确保输出形状一致
#         if result.shape != original_shape:
#             result = result[:original_shape[0], :original_shape[1]]
#         return result
#     except Exception as e:
#         print(f"KNN imputation failed: {e}, falling back to mean imputation")
#         return mean_impu(mx)

# # ====== 单文件处理 ======
# def process_file(file_info):
#     input_path, output_dir = file_info
#     try:
#         df = pd.read_csv(input_path)
#         mx = df.to_numpy(dtype=float)

#         # 执行 KNN 填补
#         result = knn_impu(mx, k=3)

#         # 保存
#         os.makedirs(output_dir, exist_ok=True)
#         out_file = os.path.join(output_dir, os.path.basename(input_path))
#         pd.DataFrame(result, columns=df.columns).to_csv(out_file, index=False)
#         return f"✅ Done: {os.path.basename(input_path)}"
#     except Exception as e:
#         return f"❌ Failed: {os.path.basename(input_path)} -> {e}"

# # ====== 并行处理函数 ======
# def parallel_knn_impute(input_dir, output_dir, num_workers=8):
#     files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
#              if f.endswith(".csv")]
#     tasks = [(f, output_dir) for f in files]

#     with Pool(num_workers) as pool:
#         for res in pool.imap_unordered(process_file, tasks):
#             print(res)

# # ====== 主程序入口 ======
# if __name__ == "__main__":
#     input_dir = "./data/downstreamIII"
#     output_dir = "./data_imputed/knn/III"
#     parallel_knn_impute(input_dir, output_dir, num_workers=8)
#!/usr/bin/env python3
# from pathlib import Path

# TARGET_ROWS = 193  # 193 行：第 1 行是列名 + 192 行数据

# def pad_csv_to_193_lines(fp: Path):
#     # 读取全部行（通常 CSV 行数不大；若特别大可改为只读取末尾）
#     try:
#         text = fp.read_text(encoding="utf-8", errors="ignore")
#     except Exception:
#         # 若编码不是 utf-8，可改成 fp.read_bytes() 再手动处理
#         print(f"[跳过] 无法读取：{fp}")
#         return

#     lines = text.splitlines(keepends=True)  # 保留换行符，便于原样追加
#     n = len(lines)

#     if n == 0:
#         print(f"[跳过] 空文件：{fp.name}")
#         return

#     if n >= TARGET_ROWS:
#         # 已达标或超过，不处理
#         # print(f"[OK] {fp.name} 已有 {n} 行")
#         return

#     if n == 1:
#         # 只有表头，没有数据行，无法复制“最后一行数据”
#         print(f"[警告] 仅有表头，无数据可复制：{fp.name}（当前 1 行）")
#         return

#     # 最后一行数据（保留末尾换行；若没有则补一个）
#     last_line = lines[-1]
#     if not last_line.endswith(("\n", "\r")):
#         last_line += "\n"

#     need = TARGET_ROWS - n
#     # 若原文件最后一行没有换行，先补一个换行，再开始追加
#     need_prefix_newline = not lines[-1].endswith(("\n", "\r"))
#     to_append = (("\n" if need_prefix_newline else "") + last_line * need)

#     # 追加写回（更安全可写入临时文件再替换）
#     with fp.open("a", encoding="utf-8", newline="") as f:
#         f.write(to_append)

#     print(f"[补全] {fp.name}: {n} -> {TARGET_ROWS} 行，复制最后一行 {need} 次")

# def main():
#     folder = Path("./data/downstreamIII")  # ← 改成你的目标文件夹
#     for fp in folder.glob("*.csv"):        # 若要所有文件改为：for fp in folder.iterdir():
#         if fp.is_file():
#             pad_csv_to_193_lines(fp)

# if __name__ == "__main__":
#     main()
# import numpy as np
# import torch
# from sklearn.impute import KNNImputer  # 备用
# import os

# def mean_impu(mx):
#     import pandas as pd
#     return pd.DataFrame(mx).fillna(pd.DataFrame(mx).mean()).to_numpy()

# def saits_impu(mx, epochs=None, d_model=None, n_layers=None, device=None):
#     from pypots.imputation import SAITS

#     mx = mx.copy()
#     seq_len, n_features = mx.shape
#     total_size = seq_len * n_features

#     # 全局均值
#     global_mean = np.nanmean(mx)
#     if np.isnan(global_mean):
#         global_mean = 0.0

#     # 全列NaN先填充
#     all_nan_cols = np.all(np.isnan(mx), axis=0)
#     if all_nan_cols.any():
#         mx[:, all_nan_cols] = global_mean

#     # 自动配置参数（比之前更轻）
#     if epochs is None:
#         if total_size > 50000:
#             epochs = 10
#             d_model = 16
#             n_layers = 1
#         elif total_size > 10000:
#             epochs = 10
#             d_model = 32
#             n_layers = 1
#         else:
#             epochs = 20
#             d_model = 32
#             n_layers = 1

#     if d_model is None:
#         d_model = min(64, max(16, n_features * 2))

#     if n_layers is None:
#         n_layers = 1

#     try:
#         data_3d = mx[np.newaxis, :, :]

#         saits = SAITS(
#             n_steps=seq_len,
#             n_features=n_features,
#             n_layers=n_layers,
#             d_model=d_model,
#             n_heads=min(2, max(1, d_model // 32)),
#             d_k=max(4, d_model // 8),
#             d_v=max(4, d_model // 8),
#             d_ffn=d_model,
#             dropout=0.1,
#             epochs=epochs,
#             patience=5,
#             batch_size=16,  # 降低 batch size
#             device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),
#         )

#         train_set = {"X": data_3d}
#         saits.fit(train_set)
#         imputed_data_3d = saits.impute(train_set)
#         return imputed_data_3d[0]

#     except Exception as e:
#         print(f"SAITS fails: {e}")
#         return mean_impu(mx)
# from multiprocessing import Pool
# import pandas as pd

# def process_file_saits(task):
#     input_path, output_dir, gpu_id = task
#     try:
#         # 固定该进程只用一个 GPU
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#         df = pd.read_csv(input_path)
#         mx = df.to_numpy(dtype=float)

#         result = saits_impu(mx)  # 轻量版

#         os.makedirs(output_dir, exist_ok=True)
#         out_file = os.path.join(output_dir, os.path.basename(input_path))
#         pd.DataFrame(result, columns=df.columns).to_csv(out_file, index=False)
#         return f"[GPU {gpu_id}] Done: {os.path.basename(input_path)}"
#     except Exception as e:
#         return f"[GPU {gpu_id}] Failed: {os.path.basename(input_path)} -> {e}"

# def parallel_saits_impute(input_dir, output_dir, num_gpus=2, workers_per_gpu=1):
#     files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
#     tasks = []
#     gpu_list = list(range(num_gpus))
#     for i, f in enumerate(files):
#         gpu_id = gpu_list[i % num_gpus]
#         tasks.append((f, output_dir, gpu_id))

#     with Pool(num_gpus * workers_per_gpu) as pool:
#         for res in pool.imap_unordered(process_file_saits, tasks):
#             print(res)
# if __name__ == "__main__":
#     input_dir = "./data/downstreamIII"
#     output_dir = "./data_imputed/saits/III"
#     parallel_saits_impute(input_dir, output_dir, num_gpus=2, workers_per_gpu=1)
# ===============================
# 选项 2: 每个文件单独计算 std
# ===============================
import os
import numpy as np
import pandas as pd

# 文件夹路径
folder = "./data/air"


all_values = []

for fname in os.listdir(folder):
    if fname.endswith(".csv"):
        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)
        all_values.append(df.values.flatten())

all_values = np.concatenate(all_values)
global_mean = np.mean(all_values)
print(f"全局 mean = {global_mean}")