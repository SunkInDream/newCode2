import os
import gc
import pandas as pd
import numpy as np
import torch
from multiprocessing import Process, Queue, current_process
from tqdm import tqdm

# ====== TSDE 填补函数 ======
def tsde_impu(mx, n_samples: int = 40, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    from tsde import impute_missing_data
    mx = mx.copy()
    mx = impute_missing_data(
        mx,
        n_samples=n_samples,
        device=device
    )
    return mx

# ====== GPU 工作进程函数 ======
def gpu_worker(file_queue, input_dir, output_dir, gpu_id):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    pid = current_process().pid

    while not file_queue.empty():
        try:
            fname = file_queue.get_nowait()
        except:
            break

        try:
            fpath = os.path.join(input_dir, fname)
            mx = pd.read_csv(fpath).values.astype(np.float32)
            filled = tsde_impu(mx, device=device)

            out_path = os.path.join(output_dir, fname)
            pd.DataFrame(filled).to_csv(out_path, index=False)
            print(f"[PID {pid} GPU {gpu_id}] ✅ 处理完成: {fname}")
        except Exception as e:
            print(f"[PID {pid} GPU {gpu_id}] ❌ 处理失败: {fname}，错误：{e}")

# ====== 主调度函数 ======
def parallel_tsde_impute(input_dir, output_dir, n_processes_per_gpu=1):
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    file_queue = Queue()
    for fname in file_list:
        file_queue.put(fname)

    num_gpus = torch.cuda.device_count()
    total_procs = num_gpus * n_processes_per_gpu
    processes = []

    print(f"🚀 启动 {num_gpus} 个 GPU，每 GPU {n_processes_per_gpu} 个进程，共 {total_procs} 个进程处理 {len(file_list)} 个文件")

    for gpu_id in range(num_gpus):
        for _ in range(n_processes_per_gpu):
            p = Process(target=gpu_worker, args=(file_queue, input_dir, output_dir, gpu_id))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("✅ 所有文件处理完成！")

# ====== 主程序入口 ======
if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    input_dir = "./data/downstreamIII"
    output_dir = "./data_imputed/tsde/III"
    parallel_tsde_impute(input_dir, output_dir, n_processes_per_gpu=2)  # 每GPU并行2个进程
