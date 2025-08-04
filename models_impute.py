import os
import torch
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from models_TCDF import *
import torch.nn.functional as F
from pygrinder import (
    mcar,
    mar_logistic,
    mnar_x,
)

from sklearn.cluster import KMeans
from baseline import *
from e import pre_checkee
from sklearn.preprocessing import StandardScaler
from multiprocessing import set_start_method
from scipy.stats import wasserstein_distance
from models_downstream import *
from multiprocessing import Process, Queue
from models_TCN import MultiADDSTCN, ParallelFeatureADDSTCN, ADDSTCN
import subprocess
import time
from multiprocessing import Pool, get_context
from functools import partial


def set_seed_all(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    import os

    # Python random
    random.seed(seed)

    # Numpy random (pygrinder depends on this)
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"🎲 Set global random seed: {seed}")


def wait_for_gpu_free(threshold_mb=500, sleep_time=10):
    """
    Wait until the memory usage of all GPUs is below the threshold (MiB), then return.
    By default, wait until all GPUs use less than 500 MiB.
    """
    print(f"⏳ Waiting for GPUs to be free (memory used < {threshold_mb} MiB)...")
    while True:
        try:
            output = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", shell=True
            )
            used_memory = [int(x) for x in output.decode().strip().split("\n")]
            if all(mem < threshold_mb for mem in used_memory):
                print("✅ All GPUs are free; starting miracle_impu.")
                break
            else:
                print(f"🚧 Current VRAM usage: {used_memory} MiB, not low enough; waiting {sleep_time}s...")
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Failed to check GPU memory: {e}")
            time.sleep(sleep_time)


def FirstProcess(matrix, threshold=0.8):
    matrix = np.array(matrix, dtype=np.float32)

    # Stage 1: handle empty columns and high-duplication columns
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]

        if np.isnan(col_data).all():
            matrix[:, col_idx] = -1
            continue

        valid_mask = ~np.isnan(col_data)
        if not valid_mask.any():
            continue

        valid_data = col_data[valid_mask]
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        max_count_idx = np.argmax(counts)
        mode_value = unique_vals[max_count_idx]
        mode_count = counts[max_count_idx]

        if mode_count >= threshold * len(valid_data):
            matrix[np.isnan(col_data), col_idx] = mode_value
    return matrix


def SecondProcess(matrix, perturbation_prob=0.3, perturbation_scale=0.3):
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]
        missing_mask = np.isnan(col_data)

        if not missing_mask.any():
            continue

        series = pd.Series(col_data)
        interpolated = series.interpolate(method="linear", limit_direction="both").values

        if np.isnan(interpolated).any():
            interpolated[np.isnan(interpolated)] = np.nanmean(col_data)

        # Add perturbation
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0 and perturbation_prob > 0:
            n_perturb = int(len(missing_indices) * perturbation_prob)
            if n_perturb > 0:
                perturb_indices = np.random.choice(missing_indices, n_perturb, replace=False)
                value_range = np.ptp(col_data[~missing_mask]) or 1.0
                perturbations = np.random.uniform(-1, 1, n_perturb) * perturbation_scale * value_range
                interpolated[perturb_indices] += perturbations

        matrix[:, col_idx] = interpolated

    return matrix.astype(np.float32)  # ✅ Fix: move this outside the loop


def initial_process(matrix, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    matrix = FirstProcess(matrix, threshold)
    matrix = SecondProcess(matrix, perturbation_prob, perturbation_scale)
    return matrix


def impute(
    original,
    causal_matrix,
    model_params,
    epochs=100,
    lr=0.02,
    gpu_id=None,
    ifGt=False,
    gt=None,
    ablation=0,
    seed=42,
):
    """Add seed parameter."""
    # ✅ Set seed to ensure reproducible training
    set_seed_all(seed)

    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
    print("missing_count", np.isnan(original).sum())

    # Preprocessing
    if ablation == 3:
        first = FirstProcess(original.copy())
        mask = (~np.isnan(first)).astype(int)
        initial_filled = zero_impu(first)
        initial_filled_copy = initial_filled.copy()
    else:
        first = FirstProcess(original.copy())
        mask = (~np.isnan(first)).astype(int)
        initial_filled = SecondProcess(first)
        initial_filled_copy = initial_filled.copy()

    # Standardization
    scaler = StandardScaler()
    initial_filled_scaled = scaler.fit_transform(initial_filled)

    # Create tensors with standardized data
    x = torch.tensor(initial_filled_scaled[None, ...], dtype=torch.float32, device=device)
    y = torch.tensor(initial_filled_scaled[None, ...], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, ...], dtype=torch.float32, device=device)

    # ✅ Set seed again before creating model
    set_seed_all(seed)
    if ablation == 1:
        ablation_causal = causal_matrix.copy()
        ablation_causal = ablation_causal[...] == 1
        model = ParallelFeatureADDSTCN(causal_matrix=ablation_causal, model_params=model_params).to(device)
    elif ablation == 0 or ablation == 3:
        model = ParallelFeatureADDSTCN(causal_matrix=causal_matrix, model_params=model_params).to(device)
    elif ablation == 2:
        model = MultiADDSTCN(causal_mask=causal_matrix, num_levels=4, cuda=True).to(device)

    # torch.compile acceleration
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass

    # ✅ Set seed before optimizer init
    set_seed_all(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    grad_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and torch.cuda.is_available() else None

    # Early stopping
    best_loss = float("inf")
    best_imputed = None
    patience = 15
    no_improve_count = 0

    # Precompute stats
    y_mean = y.mean(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)
    quantiles = [0.25, 0.5, 0.75]
    y_quantiles = [torch.quantile(y.float(), q, dim=1, keepdim=True) for q in quantiles]

    for epoch in range(epochs):
        opt.zero_grad()

        if grad_scaler:  # ✅ Use grad_scaler
            with torch.cuda.amp.autocast():
                pred = model(x)

                # Loss1: prediction error at observed entries
                loss_1 = F.mse_loss(pred * m, y * m)

                # ✅ Ensure pred is float32 for statistical computations
                pred_float = pred.float()

                pred_mean = pred_float.mean(dim=1, keepdim=True)
                pred_std = pred_float.std(dim=1, keepdim=True)

                mean_loss = F.mse_loss(pred_mean, y_mean)
                std_loss = F.mse_loss(pred_std, y_std)

                # ✅ Use float32 for quantiles
                quantile_losses = []
                for i, q in enumerate(quantiles):
                    pred_q = torch.quantile(pred_float, q, dim=1, keepdim=True)
                    quantile_losses.append(F.mse_loss(pred_q, y_quantiles[i]))

                loss_3 = (mean_loss + std_loss + sum(quantile_losses)) / (2 + len(quantiles))
                total_loss = 0.6 * loss_1 + 0.4 * loss_3

            grad_scaler.scale(total_loss).backward()  # ✅ Use grad_scaler
            grad_scaler.unscale_(opt)  # ✅ Use grad_scaler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(opt)  # ✅ Use grad_scaler
            grad_scaler.update()  # ✅ Use grad_scaler
        else:
            # Standard training
            pred = model(x)

            loss_1 = F.mse_loss(pred * m, y * m)

            # ✅ Ensure pred is float32
            pred_float = pred.float()

            pred_mean = pred_float.mean(dim=1, keepdim=True)
            pred_std = pred_float.std(dim=1, keepdim=True)

            mean_loss = F.mse_loss(pred_mean, y_mean)
            std_loss = F.mse_loss(pred_std, y_std)

            quantile_losses = []
            for i, q in enumerate(quantiles):
                pred_q = torch.quantile(pred_float, q, dim=1, keepdim=True)  # ✅ use float
                quantile_losses.append(F.mse_loss(pred_q, y_quantiles[i]))

            loss_3 = (mean_loss + std_loss + sum(quantile_losses)) / (2 + len(quantiles))
            total_loss = 0.6 * loss_1 + 0.4 * loss_3

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        scheduler.step()

        # Early stopping check
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve_count = 0
            with torch.no_grad():
                best_imputed = model(x).float().cpu().squeeze(0).numpy()  # ✅ ensure float32 conversion
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Reduce print frequency and clean VRAM
        if epoch % 2 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {current_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    # Fill missing with best result and inverse-scale
    res = initial_filled.copy()
    if best_imputed is not None:
        best_imputed_rescaled = scaler.inverse_transform(best_imputed)  # ✅ inverse transform
        res[mask == 0] = best_imputed_rescaled[mask == 0]

    pd.DataFrame(res).to_csv("result_1.csv", index=False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return res, mask, initial_filled_copy


def impute_wrapper(args):
    import torch
    import os

    # ✅ Unpack the new skip_existing parameter
    if len(args) == 11:  # New version has 10 parameters after idx
        idx, mx, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr, skip_existing = args
    else:  # Backward compatibility (9 parameters after idx)
        idx, mx, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr = args
        skip_existing = False

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    try:
        # ✅ Build output file path
        filename = os.path.basename(file_path).replace(".csv", "_imputed.csv")
        save_path = os.path.join(output_dir, filename)

        # ✅ Double-check skip logic at worker level
        if skip_existing and os.path.exists(save_path):
            print(f"⏩ Skipping existing file at worker level: {filename}")
            return idx, save_path

        # Run imputation
        imputed_result, mask, initial_processed = impute(
            mx, causal_matrix, model_params=model_params, epochs=epochs, lr=lr, gpu_id=gpu_id
        )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        # Save result
        pd.DataFrame(imputed_result).to_csv(save_path, index=False)

        print(f"✅ Imputation complete: {os.path.basename(file_path)} → {filename}")
        return idx, save_path

    except Exception as e:
        print(f"❌ Imputation failed: {os.path.basename(file_path)}, error: {e}")
        return idx, f"Error: {e}"


def parallel_impute(
    file_paths,  # str directory path (e.g., ./data/mimic-iii)
    causal_matrix,  # causal graph cg
    model_params,  # parameters for the imputation model
    epochs=100,
    lr=0.02,
    simultaneous_per_gpu=2,
    output_dir="imputed_results",
    skip_existing=False,  # ✅ new: whether to skip files that already exist
):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[INFO] No GPU available; using CPU and processing sequentially")
        num_gpus = 1

    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[INFO] Using {num_gpus} GPU(s), up to {simultaneous_per_gpu} task(s) per GPU, "
        f"total processes: {num_gpus * simultaneous_per_gpu}"
    )

    # ✅ Get file list
    file_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith(".csv")]
    print(f"[INFO] Found {len(file_list)} file(s) to process")

    # ✅ Skip logic
    if skip_existing:
        print(f"🔍 Skip-existing mode ON; checking output directory: {output_dir}")
        existing_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

        # Filter files to process
        filtered_file_list = []
        skipped_count = 0

        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename in existing_files:
                skipped_count += 1
                print(f"⏩ Skip existing file: {filename}")
            else:
                filtered_file_list.append(file_path)

        file_list = filtered_file_list
        print(f"📊 Skip stats: {skipped_count} skipped, {len(file_list)} to process")

        if len(file_list) == 0:
            print("✅ All files already exist; nothing to do")
            return {}

    args_list = []
    for idx, file_path in enumerate(file_list):
        df = pd.read_csv(file_path)
        data = df.values.astype(np.float32)
        gpu_id = idx % num_gpus
        # ✅ Pass skip_existing to worker
        args_list.append(
            (idx, data, file_path, causal_matrix, gpu_id, output_dir, model_params, epochs, lr, skip_existing)
        )

    with mp.Pool(processes=num_gpus * simultaneous_per_gpu) as pool:
        results = list(tqdm(pool.imap(impute_wrapper, args_list), total=len(args_list), desc="Filling"))

    results.sort(key=lambda x: x[0])
    output_paths = {file_list[idx]: result for idx, result in results}
    return output_paths


def agregate(initial_filled_array, n_cluster):
    # Step 1: compute per-sample column-wise means to form clustering input
    data = np.array([np.nanmean(x, axis=0) for x in initial_filled_array])

    # Step 2: KMeans clustering
    km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
    labels = km.fit_predict(data)

    # Step 3: select a representative sample per cluster (with progress bar)
    idx_arr = []
    for k in tqdm(range(n_cluster), desc="Selecting representative sample per cluster"):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        cluster_data = data[idxs]
        dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
        best_idx = idxs[np.argmin(dists)]
        idx_arr.append(int(best_idx))

    return idx_arr


def causal_discovery(
    original_matrix_arr,
    n_cluster=5,
    isStandard=False,
    standard_cg=None,
    met="lorenz",
    params={
        "layers": 6,
        "kernel_size": 6,
        "dilation_c": 4,
        "optimizername": "Adam",
        "lr": 0.02,
        "epochs": 100,
        "significance": 1.2,
    },
):
    if isStandard:
        if standard_cg is None:
            raise ValueError("standard_cg must be provided when isStandard is True")
        return pd.read_csv(standard_cg, header=None).values

    # Step 1: batch preprocessing
    initial_matrix_arr = []
    batch_size = 100

    for i in tqdm(range(0, len(original_matrix_arr), batch_size), desc="Batch preprocessing"):
        batch = original_matrix_arr[i : i + batch_size]
        batch_results = [initial_process(matrix) for matrix in batch]
        initial_matrix_arr.extend(batch_results)

        if i % (batch_size * 5) == 0:
            gc.collect()

    # Step 2: clustering & representative extraction
    idx_arr = agregate(initial_matrix_arr, n_cluster)
    data_list = [initial_matrix_arr[idx] for idx in idx_arr]
    params_list = [params] * len(data_list)

    # Step 3: multi-GPU parallel causal discovery
    results = parallel_compute_causal_matrices(data_list, params_list)

    # Step 4: aggregate results
    cg_total = None
    for matrix in results:
        if matrix is None:
            continue
        if cg_total is None:
            cg_total = matrix.copy()
        else:
            cg_total += matrix

    if cg_total is None:
        raise RuntimeError("All tasks failed; no valid causal matrix was obtained")

    # Step 5: pick top-4 per column to construct final causal graph
    np.fill_diagonal(cg_total, 0)
    new_matrix = np.zeros_like(cg_total)
    for col in range(cg_total.shape[1]):
        col_values = cg_total[:, col]
        if np.count_nonzero(col_values) < 4:
            new_matrix[:, col] = 1
        else:
            top5 = np.argsort(col_values)[-4:]
            new_matrix[top5, col] = 1
    pd.DataFrame(new_matrix).to_csv(f"./causality_matrices/{met}_causality_matrix.csv", index=False, header=False)
    return new_matrix


# ================================
# 1. Single-file evaluation
# ================================
def mse_evaluate_single_file(mx, causal_matrix, gpu_id=0, device=None, met="lorenz", missing="mar", seed=42, ablation=0):
    """Add seed parameter to control randomness."""
    # ✅ Reset seed for each call to make the missingness process reproducible
    set_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ground truth
    gt = mx.copy()
    gt2 = gt.copy()
    pd.DataFrame(gt).to_csv("gt_matrix.csv", index=False)  # rename to avoid conflicts

    # ✅ Missingness generation — run immediately after seeding
    try:
        print(f"🔍 Starting missingness generation (seed={seed})...")
        # Seed again to ensure mar_logistic determinism
        set_seed_all(seed)
        if missing == "mar":
            X = mar_logistic(mx, obs_rate=0.1, missing_rate=0.6)

        # Keep determinism for subsequent steps
        if missing == "mnar":
            X = mx.copy()
            X = X[np.newaxis, ...]
            X = mnar_x(X, offset=0.6)
            X = X.squeeze(0)

        if missing == "mcar":
            X = mx.copy()
            X = X[np.newaxis, ...]
            X = mcar(X, p=0.5)
            X = X.squeeze(0)
        pre_checkee(X, met)
        print(f"✅ Missingness generated, missing rate: {np.isnan(X).sum() / X.size:.2%}")

    except (ValueError, RuntimeError) as e:
        print(f"⚠️ mar_logistic failed, skipping this file: {e}")
        return None

    pd.DataFrame(X).to_csv("missing_matrix.csv", index=False)

    # Mask: observed 1, missing 0
    Mask = (~np.isnan(X)).astype(int)

    # Masked MSE evaluated only at missing positions
    def mse(a, b, mask):
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        b = torch.as_tensor(b, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        mask = 1 - mask  # invert mask: 1 indicates missing

        # Save debug artifacts
        pd.DataFrame((a * mask).cpu().numpy()).to_csv("pred_missing.csv", index=False)
        pd.DataFrame((b * mask).cpu().numpy()).to_csv("gt_missing.csv", index=False)
        pd.DataFrame(mask.cpu().numpy()).to_csv("missing_mask.csv", index=False)

        # Compute masked MSE
        masked_error = F.mse_loss(a * mask, b * mask).item()
        return masked_error

    res = {}

    # ✅ Our model evaluation — pass seed
    print("Starting my_model...")
    set_seed_all(seed)  # Ensure model training is deterministic as well
    imputed_result, mask, initial_processed = impute(
        X, causal_matrix,
        model_params={'num_levels':10, 'kernel_size': 8, 'dilation_c': 2},
        epochs=100, lr=0.02, gpu_id=gpu_id, ifGt=True, gt=gt, seed=seed, ablation=ablation
    )
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    # if ablation==1:
    #     res['ablation1'] = mse(imputed_result, gt2, mask)
    # elif ablation==2:
    #     res['ablation2'] = mse(imputed_result, gt2, mask)
    # elif ablation==3:
    #     res['ablation3'] = mse(imputed_result, gt2, mask)

    def is_reasonable_mse(mse_value, threshold=1_000_000.0):
        return (not np.isnan(mse_value)) and (not np.isinf(mse_value)) and (0 <= mse_value <= threshold)

    # ✅ Baselines — set seed before each method
    baseline = [
        # ("initial_process", initial_process),
        ("zero_impu", zero_impu),
        ("mean_impu", mean_impu),
        ("knn_impu", knn_impu),
        ("mice_impu", mice_impu),
        ("ffill_impu", ffill_impu),
        ("bfill_impu", bfill_impu),
        ("miracle_impu", miracle_impu),
        ("saits_impu", saits_impu),
        ("timemixerpp_impu", timemixerpp_impu),
        ("tefn_impu", tefn_impu),
        ("timesnet_impu", timesnet_impu),
        ("tsde_impu", tsde_impu),
        ("grin_impu", grin_impu),
    ]
    if not ablation:
        res['my_model'] = mse(imputed_result, gt2, mask)

    for name, fn in baseline:
        print(f"Running {name}...")

        # ✅ Set the same seed before each baseline
        set_seed_all(seed)

        try:
            result = fn(X)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if np.any(np.abs(result) > 1e6):
                print(f"❌ {name}: imputed result contains extremely large values (max: {np.max(np.abs(result)):.2e})")
                res[name] = float("nan")
            else:
                mse_value = mse(result, gt, Mask)
                if is_reasonable_mse(mse_value):
                    res[name] = mse_value
                    print(f"✅ {name}: {mse_value:.6f}")
                else:
                    print(f"❌ {name}: abnormal MSE ({mse_value:.2e})")
                    res[name] = float("nan")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            res[name] = float("nan")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    print(f"All results: {res}")
    return res


# ================================
# 2. Wrapper for Pool (per-task)
# ================================
def worker_wrapper(args):
    import torch
    import os

    idx, mx, causal_matrix, gpu_id, met, missing, seed, ablation = args  # ✅ include seed

    # ✅ Set a deterministic seed per worker/sample
    set_seed_all(seed + idx)  # each sample uses a different but deterministic seed

    print(f"[Worker PID {os.getpid()}] Assigned GPU: {gpu_id}, Seed: {seed + idx}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Worker PID {os.getpid()}] Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"[Worker PID {os.getpid()}] Warning: No GPU detected, using CPU")

    # ✅ Pass seed into evaluation function
    res = mse_evaluate_single_file(
        mx, causal_matrix, gpu_id=gpu_id, device=device, met=met, missing=missing, seed=seed + idx, ablation=ablation
    )
    return idx, res


# ================================
# 3. Parallel scheduler (process pool)
# ================================
def parallel_mse_evaluate(res_list, causal_matrix, met, simultaneous_per_gpu=3, missing="mar", seed=42, ablation=0):
    """Add seed parameter."""
    # ✅ Set seed in the main process
    set_seed_all(seed)

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("[INFO] No GPU available; evaluating sequentially on CPU")
        all_res = []
        for i, x in enumerate(tqdm(res_list, desc="CPU")):
            set_seed_all(seed + i)  # deterministic per sample
            result = mse_evaluate_single_file(x, causal_matrix, seed=seed + i)
            all_res.append(result)

        # Compute averages...
        return {k: float(np.nanmean([d[k] for d in all_res if d is not None])) for k in all_res[0] if all_res[0] is not None}

    max_workers = num_gpus * simultaneous_per_gpu
    print(
        f"[INFO] Using {num_gpus} GPU(s), up to {simultaneous_per_gpu} task(s) per GPU, total processes: {max_workers}"
    )
    print(f"🎲 Base seed: {seed}")

    # ✅ Assign GPU and seed for each task
    gpu_ids = [i % num_gpus for i in range(len(res_list))]
    args_list = [
        (i, res_list[i], causal_matrix, gpu_ids[i], met, missing, seed, ablation) for i in range(len(res_list))
    ]

    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(worker_wrapper, args_list), total=len(args_list), desc="All-tasks"))

    results.sort(key=lambda x: x[0])  # restore order
    only_result_dicts = [res for _, res in results if res is not None]

    avg = {}
    for k in only_result_dicts[0]:
        values = [r[k] for r in only_result_dicts]
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) > 0:
            avg[k] = float(np.nanmean(values))
            print(f"📊 {k}: {len(valid_values)}/{len(values)} valid values, mean: {avg[k]:.6f}")
        else:
            avg[k] = float("nan")
            print(f"❌ {k}: all values are NaN")

    # ✅ Include seed info when saving
    pd.DataFrame([{"Method": k, "Average_MSE": v, "Seed": seed} for k, v in avg.items()]).to_csv(
        f"mse_evaluation_results_seed{seed}.csv", index=False
    )

    return avg
