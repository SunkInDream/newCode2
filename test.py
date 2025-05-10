import os
from concurrent.futures import ProcessPoolExecutor
from models_TCDF import *
import torch
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
params = {
    'epochs': 30,
    'kernel_size': 3,
    'layers': 3,
    'dilation_c': 2,
    'lr': 0.01,
    'optimizername': 'Adam',
    'significance': 0.5
}

def task(args):
    file, params, gpu = args
    matrix, columns = compute_causal_matrix(file, params, gpu)
    print(f"\nResult for {os.path.basename(file)}:")
    print(np.array(matrix))
    return matrix

if __name__ == "__main__":
    csv_dir = "./data"
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    gpus = list(range(torch.cuda.device_count())) or ['cpu']
    tasks = [(f, params, gpus[i % len(gpus)]) for i, f in enumerate(files)]

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(task, tasks))

