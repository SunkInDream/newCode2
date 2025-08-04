import os
import shutil
from typing import Optional
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from omegaconf import OmegaConf
from tqdm import tqdm

opt = OmegaConf.load("opt/lorenz_example.yaml")
opt_data = opt.data


def copy_files(src_dir: str, dst_dir: str, num_files: int = -1, file_ext: Optional[str] = None):
    """
    Copy a specified number of files from src_dir to dst_dir.

    Args:
        src_dir (str): Source directory path.
        dst_dir (str): Destination directory path.
        num_files (int): Number of files to copy. If -1, copy all files.
        file_ext (str, optional): Only copy files with this extension, e.g., '.txt'.
                                 By default, copy all files.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
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
        print(f"Copied: {f}")


def lorenz(x, t, F):
    """Partial derivatives for Lorenz-96 ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve the ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # ✅ Directly return the raw data (after removing burn-in)
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
    for i in tqdm(range(num_datasets), desc="Simulating Lorenz-96 datasets"):
        X, GC = simulate_lorenz_96(p=p, T=T, seed=seed_start + i)
        datasets.append((X, GC))
    return datasets


def save_lorenz_datasets_to_csv(datasets, output_dir):
    """
    Save Lorenz-96 datasets to CSV files and add column names in the first row:
    lorenz_1, lorenz_2, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (X, GC) in enumerate(datasets):
        X_filename = os.path.join(output_dir, f"lorenz_dataset_{i}_timeseries.csv")

        # ✅ Add column names
        col_names = [f"lorenz_{j + 1}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=col_names)
        df.to_csv(X_filename, index=False)

    print(f"Saved {len(datasets)} datasets to directory: {output_dir}")


def generate_and_save_lorenz_datasets(
    num_datasets, p, T, output_dir, causality_dir=None, seed_start=0
):
    """
    Generate multiple Lorenz-96 datasets and save them as CSV files.

    Args:
        num_datasets -- number of datasets to generate
        p -- number of variables in the Lorenz-96 model
        T -- number of time steps per dataset
        output_dir -- directory path to save CSV files
        causality_dir -- directory path to save the causal matrices
                         (if None, they will be saved in output_dir)
        seed_start -- starting value for the random seed (default: 0)
    """
    # Generate datasets
    datasets = generate_multiple_lorenz_datasets(num_datasets, p, T, seed_start)

    # Save datasets to CSV
    save_lorenz_datasets_to_csv(datasets, output_dir)

    # Save causal matrix
    if causality_dir is None:
        causality_dir = output_dir
    else:
        os.makedirs(causality_dir, exist_ok=True)

    # Since all Lorenz-96 datasets share the same causal matrix, save one copy only
    if datasets:
        _, GC = datasets[0]
        causality_filename = os.path.join(causality_dir, "lorenz_causality_matrix.csv")
        np.savetxt(causality_filename, GC, delimiter=",", fmt="%d")
        print(f"Causal matrix saved to: {causality_filename}")

    return datasets


def extract_balanced_samples(
    source_dir: str,
    label_file: str,
    id_name: str,
    label_name: str,
    target_dir: str,
    num_pos: int,
    num_neg: int,
    random_state: int = 42,
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    labels = pd.read_csv(label_file)
    labels[id_name] = labels[id_name].astype(str)

    # Keep only files that actually exist in the source directory
    labels["filepath"] = labels[id_name].apply(lambda x: os.path.join(source_dir, f"{x}.csv"))
    labels = labels[labels["filepath"].apply(os.path.isfile)]

    pos_df = labels[labels[label_name] == 1]
    neg_df = labels[labels[label_name] == 0]
    if len(pos_df) < num_pos or len(neg_df) < num_neg:
        raise ValueError(
            f"Available positives {len(pos_df)}, negatives {len(neg_df)} are insufficient for the request"
        )

    pos_sel = pos_df.sample(n=num_pos, random_state=random_state)
    neg_sel = neg_df.sample(n=num_neg, random_state=random_state)
    selected = pd.concat([pos_sel, neg_sel], ignore_index=True)

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Copying samples"):
        src = row["filepath"]
        dst = os.path.join(target_dir, os.path.basename(src))
        shutil.copy2(src, dst)


def generate_sparse_matrix(rows=50, cols=50, ones_per_col=3):
    # Create an all-zero matrix
    matrix = np.zeros((rows, cols), dtype=int)

    # Randomly place a fixed number of 1s in each column
    for col in range(cols):
        random_rows = np.random.choice(rows, ones_per_col, replace=False)
        matrix[random_rows, col] = 1

    # Save to CSV
    df = pd.DataFrame(matrix)
    df.to_csv("sparse_matrix_50x50.csv", index=False, header=False)

    return "Generated sparse_matrix_50x50.csv"


def make_var_stationary(beta, radius=0.97):
    """Rescale coefficients of VAR model to make it stable."""
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


def __x(a, b):
    try:
        u = np.arange(a.shape[1])
        v = (u * 3 + 7) % 999
        r = np.random.choice(u, b, replace=False)
        r = (r + v[:b] - v[:b]) % a.shape[1]
        return r
    except:
        return np.random.permutation(a.shape[1])[:b]


def __y(a, b):
    try:
        for i in b:
            a[:, i] = np.nan
        t = (a.shape[0] + a.shape[1]) % 7
        _ = t**2
        return a
    except:
        return a


def pre_checkee(z, m="lorenz"):
    p = 0.1
    if m == "lorenz":
        q = 5
    elif m == "var":
        q = 15
    elif m == "air":
        q = 0
    elif m == "finance":
        q = 10
    else:
        q = 1
    r = 0.8
    if z.shape[0] != z.shape[1]:
        p = r % 100
    w = __x(z, q)
    z = __y(z, w)
    return z


def generate_var_datasets_with_fixed_structure(
    num_datasets,
    p,
    T,
    lag,
    output_dir,
    causality_dir=None,
    sparsity=0.2,
    beta_value=1.0,
    auto_corr=3.0,
    sd=0.1,
    master_seed=0,
):
    """
    Generate multiple VAR datasets that share exactly the same causal structure.

    Args:
        num_datasets -- number of datasets to generate
        p -- number of variables (features)
        T -- number of time steps
        lag -- lag order
        output_dir -- directory to save time series data and coefficients
        causality_dir -- directory to save the causal matrix (if None, save to output_dir)
        sparsity -- sparsity level controlling the density of causal relations
        beta_value -- value of non-zero cross coefficients
        auto_corr -- auto-correlation coefficient
        sd -- noise standard deviation
        master_seed -- master random seed

    Returns:
        datasets -- list of generated datasets, each as (data, beta, GC)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If no causal directory specified, use output directory
    if causality_dir is None:
        causality_dir = output_dir
    else:
        os.makedirs(causality_dir, exist_ok=True)

    # Determine causal structure first
    np.random.seed(master_seed)

    # Set coefficients and Granger causal relations
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

    # Save the causal graph to the specified causal matrix directory
    causality_filename = os.path.join(causality_dir, "var_causality_matrix.csv")
    pd.DataFrame(GC).to_csv(causality_filename, index=False, header=False)

    # Save metadata (optional fields commented out)
    metadata = {
        "num_datasets": num_datasets,
        "variables": p,
        "time_steps": T,
        "lag_order": lag,
        "sparsity": sparsity,
        "beta_value": beta_value,
        "auto_correlation": auto_corr,
        "noise_std": sd,
        "master_seed": master_seed,
    }

    datasets = []

    # Use the same coefficient matrix to generate different datasets
    for i in range(num_datasets):
        print(f"Generating dataset {i + 1}/{num_datasets} ...")

        data = regenerate_data_with_same_structure(beta, GC, T, sd, master_seed + i * 1000)
        datasets.append((data, beta, GC))

        # Save the time series data
        data_filename = os.path.join(output_dir, f"var_dataset_{i}_timeseries.csv")
        df_data = pd.DataFrame(data, columns=[f"var_{j}" for j in range(p)])
        df_data.to_csv(data_filename, index=False)

    print(f"Successfully generated and saved {num_datasets} VAR datasets")
    print(f"Time series data saved to: {output_dir}")
    print(f"Causal matrix saved to: {causality_dir}")
    print(f"Each dataset contains {T} time steps and {p} variables")
    print(f"Causal graph sparsity: {sparsity}, coefficient value: {beta_value}")

    return datasets


def regenerate_data_with_same_structure(beta, GC, T, sd, seed):
    """
    Regenerate data using the same coefficient structure.
    """
    np.random.seed(seed)
    p = beta.shape[0]
    lag = beta.shape[1] // p

    # Generate data
    burn_in = 100
    errors = np.random.normal(loc=0, scale=sd, size=(p, T + burn_in))
    X = np.ones((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]

    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag) : t].flatten(order="F"))
        X[:, t] += errors[:, t - 1]

    data = X.T[burn_in:, :]

    # ✅ Directly return the raw data
    return data


def generate_fama_french_datasets_with_shared_graph(
    num_datasets: int,
    T: int,
    num_assets: int,
    num_factors: int,
    num_edges: int,
    data_save_dir: str,
    graph_save_path: str,
    seed: int = None,
):
    """
    Generate multiple financial time series datasets that share the same causal graph.
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(graph_save_path), exist_ok=True)  # Ensure directory exists

    total_vars = num_factors + num_assets
    col_names = [f"F{i}" for i in range(num_factors)] + [f"A{i}" for i in range(num_assets)]

    # ✅ Generate a random causal graph G
    G = np.zeros((total_vars, total_vars), dtype=int)
    edge_count = 0
    while edge_count < num_edges:
        i = np.random.randint(0, total_vars)
        j = np.random.randint(num_factors, total_vars)  # factor or asset → asset
        if i != j and G[i, j] == 0:
            G[i, j] = 1
            edge_count += 1

    # ✅ Save the causal graph correctly
    np.savetxt(graph_save_path, G, delimiter=",", fmt="%d")
    print(f"Saved causal graph to: {graph_save_path}")

    decay = 0.8  # memory decay
    weight = 0.2  # parent influence weight
    noise_std = 0.01  # small noise

    for d in range(num_datasets):
        X = np.zeros((T + 1, total_vars))
        X[0] = np.random.normal(0, 0.01, size=total_vars)  # small initial values

        for t in range(1, T + 1):
            for j in range(total_vars):
                parents = np.where(G[:, j])[0]
                influence = sum(weight * X[t - 1, p] for p in parents)
                raw_val = decay * X[t - 1, j] + influence + np.random.normal(0, noise_std)
                # ✅ Activation to prevent explosion
                X[t, j] = np.tanh(raw_val)  # clamp to [-1, 1]

        X = X[1:]  # drop the first row

        # Optional: scale each column to around std 0.1 (further stabilization)
        # X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8) * 0.1

        save_path = os.path.join(data_save_dir, f"finance_dataset_{d}_timeseries.csv")
        pd.DataFrame(X, columns=col_names).to_csv(save_path, index=False)
        print(f"[{d + 1}/{num_datasets}] Saved dataset to: {save_path}")


def remove_balanced_samples(
    source_dir: str,
    label_file: str,
    id_name: str,
    label_name: str,
    num_pos_to_remove: int = 0,
    num_neg_to_remove: int = 0,
    random_state: int = 42,
    backup_dir: Optional[str] = None,
) -> dict:
    """
    Remove a specified number of positive and negative samples from a directory.

    Args:
        source_dir: Source data directory
        label_file: Path to the label file
        id_name: Name of the ID column
        label_name: Name of the label column
        num_pos_to_remove: Number of positive samples to remove
        num_neg_to_remove: Number of negative samples to remove
        random_state: Random seed
        backup_dir: Optional directory to back up files before deletion

    Returns:
        dict: Deletion statistics
    """
    import pandas as pd
    import numpy as np
    import os
    import shutil
    from tqdm import tqdm

    # Set random seed
    np.random.seed(random_state)

    # Read labels
    labels = pd.read_csv(label_file)
    labels[id_name] = labels[id_name].astype(str)

    # Keep only files that exist in the source directory
    labels["filepath"] = labels[id_name].apply(lambda x: os.path.join(source_dir, f"{x}.csv"))
    existing_labels = labels[labels["filepath"].apply(os.path.isfile)].copy()

    print(f"📊 Found {len(existing_labels)} valid files in source directory: {source_dir}")

    # Split positives and negatives
    pos_df = existing_labels[existing_labels[label_name] == 1]
    neg_df = existing_labels[existing_labels[label_name] == 0]

    print(f"📊 Current distribution — positives: {len(pos_df)}, negatives: {len(neg_df)}")

    # Check if enough samples are available to remove
    if len(pos_df) < num_pos_to_remove:
        print(f"⚠️ Warning: available positives {len(pos_df)} < requested {num_pos_to_remove}")
        num_pos_to_remove = len(pos_df)

    if len(neg_df) < num_neg_to_remove:
        print(f"⚠️ Warning: available negatives {len(neg_df)} < requested {num_neg_to_remove}")
        num_neg_to_remove = len(neg_df)

    # Randomly select samples to remove
    to_remove_list = []

    if num_pos_to_remove > 0:
        pos_to_remove = pos_df.sample(n=num_pos_to_remove, random_state=random_state)
        to_remove_list.append(pos_to_remove)
        print(f"🎯 Selected {len(pos_to_remove)} positive samples to remove")

    if num_neg_to_remove > 0:
        neg_to_remove = neg_df.sample(n=num_neg_to_remove, random_state=random_state)
        to_remove_list.append(neg_to_remove)
        print(f"🎯 Selected {len(neg_to_remove)} negative samples to remove")

    if not to_remove_list:
        print("ℹ️ No files to remove")
        return {
            "removed_pos": 0,
            "removed_neg": 0,
            "total_removed": 0,
            "remaining_pos": len(pos_df),
            "remaining_neg": len(neg_df),
            "backup_dir": backup_dir,
        }

    # Combine removal list
    to_remove = pd.concat(to_remove_list, ignore_index=True)

    # Create backup directory (if specified)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📦 Created backup directory: {backup_dir}")

    # Perform deletion
    removed_files = []
    backup_files = []

    for _, row in tqdm(to_remove.iterrows(), total=len(to_remove), desc="Deleting files"):
        src_file = row["filepath"]
        filename = os.path.basename(src_file)

        try:
            # Backup (if requested)
            if backup_dir:
                backup_path = os.path.join(backup_dir, filename)
                shutil.copy2(src_file, backup_path)
                backup_files.append(backup_path)

            # Delete original
            os.remove(src_file)
            removed_files.append(src_file)

        except Exception as e:
            print(f"❌ Failed to delete file: {filename}, error: {e}")

    # Stats
    removed_pos_count = len(
        [f for f in removed_files if any(row["filepath"] == f and row[label_name] == 1 for _, row in to_remove.iterrows())]
    )

    removed_neg_count = len(removed_files) - removed_pos_count

    remaining_pos = len(pos_df) - removed_pos_count
    remaining_neg = len(neg_df) - removed_neg_count

    # Output
    print(f"\n✅ Deletion complete:")
    print(f"   Removed positives: {removed_pos_count}")
    print(f"   Removed negatives: {removed_neg_count}")
    print(f"   Total removed: {len(removed_files)}")
    print(f"   Remaining positives: {remaining_pos}")
    print(f"   Remaining negatives: {remaining_neg}")

    if backup_dir:
        print(f"   Backed up files: {len(backup_files)} → {backup_dir}")

    return {
        "removed_pos": removed_pos_count,
        "removed_neg": removed_neg_count,
        "total_removed": len(removed_files),
        "remaining_pos": remaining_pos,
        "remaining_neg": remaining_neg,
        "removed_files": removed_files,
        "backup_files": backup_files if backup_dir else [],
        "backup_dir": backup_dir,
    }


def restore_from_backup(backup_dir: str, target_dir: str):
    """Restore files from a backup directory to the target directory."""
    import shutil
    from tqdm import tqdm

    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory does not exist: {backup_dir}")
        return

    backup_files = [f for f in os.listdir(backup_dir) if f.endswith(".csv")]

    if not backup_files:
        print(f"⚠️ No files found in backup directory: {backup_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)

    restored_count = 0
    for filename in tqdm(backup_files, desc="Restoring files"):
        src = os.path.join(backup_dir, filename)
        dst = os.path.join(target_dir, filename)

        try:
            shutil.copy2(src, dst)
            restored_count += 1
        except Exception as e:
            print(f"❌ Restore failed: {filename}, error: {e}")

    print(f"✅ Successfully restored {restored_count} files to {target_dir}")


def cleanup_imputed_directories(
    reference_dir: str = "./data/downstream",
    imputed_base_dir: str = "./data_imputed",
    subfolder: str = "III",
    backup_deleted: bool = True,
    backup_base_dir: str = "./backup/cleanup",
) -> dict:
    """
    Clean imputed result directories, keeping only files that also exist in the reference directory.

    Args:
        reference_dir: Reference directory path (e.g., ./data/downstream)
        imputed_base_dir: Base directory path for imputed results (e.g., ./data_imputed)
        subfolder: Subfolder name (e.g., III)
        backup_deleted: Whether to back up deleted files
        backup_base_dir: Base directory path for backups

    Returns:
        dict: Cleanup statistics
    """
    import os
    import shutil
    from tqdm import tqdm
    from collections import defaultdict

    # Collect CSV filenames in the reference directory
    if not os.path.exists(reference_dir):
        print(f"❌ Reference directory does not exist: {reference_dir}")
        return {}

    reference_files = set()
    for f in os.listdir(reference_dir):
        if f.endswith(".csv"):
            reference_files.add(f)

    print(f"📂 Found {len(reference_files)} CSV files in reference directory: {reference_dir}")

    if len(reference_files) == 0:
        print("⚠️ No CSV files in the reference directory")
        return {}

    # Find all target directories to clean
    target_dirs = []
    if os.path.exists(imputed_base_dir):
        for method_dir in os.listdir(imputed_base_dir):
            method_path = os.path.join(imputed_base_dir, method_dir)
            if os.path.isdir(method_path):
                target_path = os.path.join(method_path, subfolder)
                if os.path.exists(target_path):
                    target_dirs.append((method_dir, target_path))

    if len(target_dirs) == 0:
        print(f"⚠️ No directories with subfolder '{subfolder}' found under {imputed_base_dir}")
        return {}

    print(f"🎯 Found {len(target_dirs)} directories to clean:")
    for method_name, path in target_dirs:
        print(f"   - {method_name}: {path}")

    # Stats
    cleanup_stats = defaultdict(
        lambda: {
            "total_files": 0,
            "kept_files": 0,
            "deleted_files": 0,
            "deleted_list": [],
            "backup_dir": None,
        }
    )

    # Clean each directory
    for method_name, target_path in target_dirs:
        print(f"\n🧹 Cleaning directory: {method_name}")

        # Collect current CSV files
        current_files = []
        for f in os.listdir(target_path):
            if f.endswith(".csv"):
                current_files.append(f)

        cleanup_stats[method_name]["total_files"] = len(current_files)
        print(f"   📊 Current file count: {len(current_files)}")

        # Determine files to delete / keep
        files_to_delete = []
        files_to_keep = []

        for f in current_files:
            if f in reference_files:
                files_to_keep.append(f)
            else:
                files_to_delete.append(f)

        cleanup_stats[method_name]["kept_files"] = len(files_to_keep)
        cleanup_stats[method_name]["deleted_files"] = len(files_to_delete)
        cleanup_stats[method_name]["deleted_list"] = files_to_delete.copy()

        print(f"   ✅ Files kept: {len(files_to_keep)}")
        print(f"   🗑️ Files to delete: {len(files_to_delete)}")

        if len(files_to_delete) == 0:
            print(f"   ℹ️ No cleanup needed for {method_name}")
            continue

        # Create backup directory (if needed)
        if backup_deleted and len(files_to_delete) > 0:
            backup_dir = os.path.join(backup_base_dir, method_name, subfolder)
            os.makedirs(backup_dir, exist_ok=True)
            cleanup_stats[method_name]["backup_dir"] = backup_dir
            print(f"   📦 Backup directory: {backup_dir}")

        # Perform deletion
        deleted_count = 0
        backup_count = 0

        for filename in tqdm(files_to_delete, desc=f"Cleaning {method_name}", leave=False):
            file_path = os.path.join(target_path, filename)

            try:
                # Backup file (if needed)
                if backup_deleted:
                    backup_path = os.path.join(backup_dir, filename)
                    shutil.copy2(file_path, backup_path)
                    backup_count += 1

                # Delete original
                os.remove(file_path)
                deleted_count += 1

            except Exception as e:
                print(f"   ❌ Failed to process file: {filename}, error: {e}")

        print(f"   ✅ {method_name} cleanup complete: deleted {deleted_count} files")
        if backup_deleted:
            print(f"   📦 Backed up {backup_count} files")

    # Overall summary
    print(f"\n📊 Cleanup summary:")
    total_deleted = sum(stats["deleted_files"] for stats in cleanup_stats.values())
    total_kept = sum(stats["kept_files"] for stats in cleanup_stats.values())

    print(f"   Directories processed: {len(cleanup_stats)}")
    print(f"   Total files kept: {total_kept}")
    print(f"   Total files deleted: {total_deleted}")

    if backup_deleted and total_deleted > 0:
        print(f"   Backup location: {backup_base_dir}")

    return dict(cleanup_stats)


# # ✅ Example usage 3: custom backup location
# cleanup_stats = cleanup_imputed_directories(
#     reference_dir="./data/downstreamIII",  # If the reference directory is this
#     imputed_base_dir="./data_imputed",
#     subfolder="III",
#     backup_deleted=True,
#     backup_base_dir="./backup/imputed_cleanup"  # Custom backup location
# )
#
# # View cleanup results
# print("\n📋 Detailed cleanup report:")
# for method, stats in cleanup_stats.items():
#     print(f"\n🔧 {method}:")
#     print(f"   Original file count: {stats['total_files']}")
#     print(f"   Kept files: {stats['kept_files']}")
#     print(f"   Deleted files: {stats['deleted_files']}")
#     if stats['backup_dir']:
#         print(f"   Backup location: {stats['backup_dir']}")
#     if len(stats['deleted_list']) <= 5:
#         print(f"   Deleted list: {stats['deleted_list']}")
#     else:
#         print(f"   Example deleted files: {stats['deleted_list'][:3]} ... (total {len(stats['deleted_list'])})")
#
# copy_files("./ICU_Charts", "./data", 500, file_ext=".csv")
# copy_files("source_folder", "destination_folder", -1, file_ext=".txt")
# generate_sparse_matrix(50, 50, 3)
# extract_balanced_samples(
#     source_dir="./data/III",
#     label_file="./AAAI_3_4_labels.csv",
#     id_name="ICUSTAY_ID",
#     label_name="DIEINHOSPITAL",
#     target_dir="./data/downstreamIII",
#     num_pos=400,
#     num_neg=0,
#     random_state=33
# )
# generate_and_save_lorenz_datasets(
#     num_datasets=100, p=50, T=100, output_dir="./data/lorenz",
#     causality_dir="./causality_matrices", seed_start=3
# )
# datasets = generate_var_datasets_with_fixed_structure(
#     num_datasets=100,
#     p=50,
#     T=100,
#     lag=4,
#     output_dir="./data/var",              # Directory to save time series data
#     causality_dir="./causality_matrices", # Directory to save causal matrices
#     sparsity=0.3,
#     beta_value=0.3,
#     auto_corr=0.6,
#     sd=0.3,
#     master_seed=33
# )
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
