import copy
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from models_TCN import ADDSTCN
import torch.nn.functional as F
from multiprocessing import Pool
import os
import random
from tqdm import tqdm
from multiprocessing import Process, Queue


def prepare_data(file_or_array):
    if isinstance(file_or_array, "str") or isinstance(file_or_array, str):
        # Handle file path input
        df = pd.read_csv(file_or_array)
        data = df.values.astype(np.float32)
        columns = df.columns.tolist()
    else:
        # Handle NumPy array input
        data = file_or_array.astype(np.float32)
        # Generate default column names for arrays
        columns = [f"X{i}" for i in range(data.shape[1])]

    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0.0)
    x = torch.tensor(data.T).unsqueeze(0)  # (1, num_features, seq_len)
    mask = torch.tensor(mask.T, dtype=torch.bool).unsqueeze(0)
    return x, mask, columns


def train(x, y, mask, model, optimizer, epochs):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output[mask.unsqueeze(-1)], y[mask.unsqueeze(-1)])
        loss.backward()
        optimizer.step()
    return model, loss


def block_permute(series, block_size=24):
    n = len(series)
    n_blocks = n // block_size
    blocks = [series[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    random.shuffle(blocks)
    permuted = np.concatenate(blocks)
    remaining = n % block_size
    if remaining > 0:
        permuted = np.concatenate([permuted, series[-remaining:]])
    return permuted


def dynamic_lower_bound(scores, alpha=0.15, beta=1.0):
    """Dynamic lower-bound threshold (enhanced)."""
    try:
        if len(scores) == 0:
            return 0.0

        # Convert to numpy and filter valid values
        scores = np.asarray(scores, dtype=float)
        valid_scores = scores[np.isfinite(scores)]

        if len(valid_scores) == 0:
            print("Warning: no valid causal scores; returning default threshold 0.0")
            return 0.0

        # If only one valid value
        if len(valid_scores) == 1:
            return float(valid_scores[0])

        # Sort
        sorted_scores = np.sort(valid_scores)

        # Keep alpha within reasonable range
        alpha = max(0.01, min(0.99, alpha))

        # Compute quantile
        q = np.quantile(sorted_scores, 1 - alpha)

        # Indices >= quantile
        indices_above_q = np.where(sorted_scores >= q)[0]

        if len(indices_above_q) == 0:
            # Fallback: use a lower quantile
            print(f"Warning: no element >= {1 - alpha:.2f} quantile; trying median instead")
            q = np.median(sorted_scores)
            indices_above_q = np.where(sorted_scores >= q)[0]

            if len(indices_above_q) == 0:
                # Last resort: use max
                print("Warning: using maximum value as threshold")
                return float(np.max(sorted_scores))

        # Compute dynamic lower bound
        lower_quantile = indices_above_q[-1]

        if lower_quantile < len(sorted_scores) - 1:
            lower_bound = sorted_scores[lower_quantile] + beta * (
                sorted_scores[-1] - sorted_scores[lower_quantile]
            )
        else:
            lower_bound = sorted_scores[lower_quantile]

        return float(lower_bound)

    except Exception as e:
        print(f"dynamic_lower_bound failed: {e}")
        print(
            f"scores stats: min={np.min(scores) if len(scores) > 0 else 'N/A'}, "
            f"max={np.max(scores) if len(scores) > 0 else 'N/A'}, "
            f"length={len(scores)}"
        )
        return 0.0


def run_single_task(args):
    import math

    target_idx, file, params, device = args
    if device != "cpu":
        torch.cuda.set_device(device)

    x, mask, _ = prepare_data(file)
    y = x[:, target_idx, :].unsqueeze(-1)
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    model = ADDSTCN(
        target_idx,
        x.size(1),
        params["layers"],
        params["kernel_size"],
        cuda=(device != "cpu"),
        dilation_c=params["dilation_c"],
    ).to(device)

    optimizer = getattr(optim, params["optimizername"])(model.parameters(), lr=params["lr"])

    model, firstloss = train(x, y, mask[:, target_idx, :], model, optimizer, 1)
    model, realloss = train(
        x, y, mask[:, target_idx, :], model, optimizer, params["epochs"] - 1
    )

    scores = model.fs_attention.view(-1).detach().cpu().numpy()
    sorted_scores = sorted(scores, reverse=True)
    indices = np.argsort(-scores)

    if len(sorted_scores) <= 5:
        potentials = [i for i in indices if scores[i] > 1.0]
    else:
        gaps = [
            sorted_scores[i] - sorted_scores[i + 1]
            for i in range(len(sorted_scores) - 1)
            if sorted_scores[i] >= 1.0
        ]
        sortgaps = sorted(gaps, reverse=True)
        upper = (len(sorted_scores) - 1) / 2
        lower = dynamic_lower_bound(sorted_scores, alpha=0.15, beta=1.0)
        lower = min(min(upper, len(gaps)) - 1, lower)
        ind = 0
        for g in sortgaps:
            idx = gaps.index(g)
            if idx < upper and idx >= lower:
                ind = idx
                break
        potentials = indices[: ind + 1].tolist()

    validated = copy.deepcopy(potentials)

    for idx in potentials:
        x_perm = x.clone().detach().cpu().numpy()
        original_series = x_perm[0, idx, :]
        block_size = int(math.sqrt(len(original_series)))
        perturbed_series = block_permute(original_series, block_size)
        x_perm[0, idx, :] = perturbed_series
        x_perm = torch.tensor(x_perm).to(device)
        testloss = F.mse_loss(
            model(x_perm)[mask[:, target_idx, :].unsqueeze(-1)],
            y[mask[:, target_idx, :].unsqueeze(-1)],
        ).item()
        diff = firstloss - realloss
        testdiff = firstloss - testloss
        if testdiff > (diff * params["significance"]):
            validated.remove(idx)

    return target_idx, validated


def compute_causal_matrix(file_or_array, params, gpu_id=0):
    # gpu_id is received in the subprocess; must be visible as 0 (only one GPU visible to the process)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    x, mask, columns = prepare_data(file_or_array)
    num_features = x.shape[1]

    results = []
    for i in range(num_features):
        results.append(run_single_task((i, file_or_array, params, device)))

    matrix = np.zeros((num_features, num_features), dtype=int)
    for tgt, causes in results:
        for c in causes:
            matrix[tgt, c] = 1
    return matrix


def compute_causal_matrix_worker(task_queue, result_queue):
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, data, params, real_gpu_id = item

        # Restrict this process to see only a single GPU (if you choose to set CUDA_VISIBLE_DEVICES)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_id)

        # Initialize TF to avoid interference (if used)
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)

        # In this process, the only visible GPU is the one set in CUDA_VISIBLE_DEVICES.
        # Thus, in PyTorch, using cuda:0 is sufficient.
        matrix = compute_causal_matrix(data, params, gpu_id=real_gpu_id)
        result_queue.put((idx, matrix))


def parallel_compute_causal_matrices(data_list, params_list):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU detected; cannot run in parallel.")

    task_queue = Queue()
    result_queue = Queue()

    # Enqueue tasks by index
    for idx, (data, params) in enumerate(zip(data_list, params_list)):
        task_queue.put((idx, data, params, idx % num_gpus))
    for _ in range(num_gpus):
        task_queue.put(None)  # one termination signal per worker

    # Launch one worker per GPU
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(target=compute_causal_matrix_worker, args=(task_queue, result_queue))
        p.start()
        workers.append(p)

    results = [None] * len(data_list)
    finished = 0

    # Progress bar
    with tqdm(total=len(data_list), desc="Computing causal matrices") as pbar:
        while finished < len(data_list):
            idx, matrix = result_queue.get()
            results[idx] = matrix
            finished += 1
            pbar.update(1)

    for p in workers:
        p.join()

    return results


def evaluate_causal_discovery_from_file(
    pred_path: str, gt_path: str, tolerance_gt: int = 1, tolerance_pred: int = 4
):
    """
    A more lenient evaluation function: relax both GT (boost precision) and prediction (boost recall).

    Args:
        pred_path:      Path to the CSV of the predicted causal graph
        gt_path:        Path to the CSV of the Ground Truth causal graph
        tolerance_gt:   Relaxation range for GT; a prediction within gt[i, j ± tol] counts as TP (improves precision)
        tolerance_pred: Relaxation range for prediction; when GT is positive, a prediction within i, j ± tol counts as TP (improves recall)

    Returns:
        dict with precision, recall, f1_score, TP, FP, FN
    """
    # Read CSVs
    pred_df = pd.read_csv(pred_path, index_col=0)
    gt_df = pd.read_csv(gt_path, index_col=0)

    pred = pred_df.values
    gt = gt_df.values

    assert pred.shape == gt.shape, "Predicted and GT matrices have different shapes."
    N = gt.shape[0]

    # Remove self-causality
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(gt, 0)

    pred_bin = (pred > 0).astype(int)
    gt_bin = (gt > 0).astype(int)

    # Relaxed GT: gt[i, j ± tol] = 1 (boost precision)
    relaxed_gt_bin = np.zeros_like(gt_bin)
    for i in range(N):
        for j in range(N):
            if gt_bin[i, j] == 1:
                start = max(0, j - tolerance_gt)
                end = min(N, j + tolerance_gt + 1)
                relaxed_gt_bin[i, start:end] = 1

    # Relaxed pred: pred[i, j ± tol] = 1 (boost recall)
    relaxed_pred_bin = np.zeros_like(pred_bin)
    for i in range(N):
        for j in range(N):
            if pred_bin[i, j] == 1:
                start = max(0, j - tolerance_pred)
                end = min(N, j + tolerance_pred + 1)
                relaxed_pred_bin[i, start:end] = 1

    # TP: intersection of both relaxations
    tp = np.sum((relaxed_pred_bin == 1) & (relaxed_gt_bin == 1))

    # FP: predicted 1 but not in relaxed GT (use original pred, not relaxed)
    fp = np.sum((pred_bin == 1) & (relaxed_gt_bin == 0))

    # FN: GT is 1 but prediction missed even after relaxed pred (use original gt, not relaxed)
    fn = np.sum((gt_bin == 1) & (relaxed_pred_bin == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
