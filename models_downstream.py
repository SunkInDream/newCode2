import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from baseline import *
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🎲 Set random seed: {seed}")


class SimpleLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_batch_norm=True,
        use_layer_norm=False,
    ):
        super(SimpleLSTMClassifier, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        if use_batch_norm:
            self.lstm_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.lstm_norm = None

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        if use_batch_norm:
            self.attention_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.attention_norm = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
            if use_batch_norm
            else nn.LayerNorm(hidden_dim)
            if use_layer_norm
            else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32)
            if use_batch_norm
            else nn.LayerNorm(32)
            if use_layer_norm
            else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        if self.input_norm is not None:
            if self.use_batch_norm:
                x_reshaped = x.view(-1, num_features)
                x_normalized = self.input_norm(x_reshaped)
                x = x_normalized.view(batch_size, seq_len, num_features)
            else:
                x = self.input_norm(x)

        lstm_out, _ = self.lstm(x)

        if self.lstm_norm is not None:
            if self.use_batch_norm:
                lstm_reshaped = lstm_out.reshape(-1, lstm_out.size(-1))
                lstm_normalized = self.lstm_norm(lstm_reshaped)
                lstm_out = lstm_normalized.view(batch_size, seq_len, -1)
            else:
                lstm_out = self.lstm_norm(lstm_out)

        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        if self.attention_norm is not None:
            if self.use_batch_norm:
                attn_reshaped = attn_out.reshape(-1, attn_out.size(-1))
                attn_normalized = self.attention_norm(attn_reshaped)
                attn_out = attn_normalized.view(batch_size, seq_len, -1)
            else:
                attn_out = self.attention_norm(attn_out)

        pooled = torch.mean(attn_out, dim=1)
        out = self.classifier(pooled)
        return out


class SimpleGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_batch_norm=True,
        use_layer_norm=False,
    ):
        super(SimpleGRUClassifier, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = None

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        if use_batch_norm:
            self.lstm_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.lstm_norm = None

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        if use_batch_norm:
            self.attention_norm = nn.BatchNorm1d(hidden_dim * 2)
        elif use_layer_norm:
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.attention_norm = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
            if use_batch_norm
            else nn.LayerNorm(hidden_dim)
            if use_layer_norm
            else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32)
            if use_batch_norm
            else nn.LayerNorm(32)
            if use_layer_norm
            else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        if self.input_norm is not None:
            if self.use_batch_norm:
                x_reshaped = x.view(-1, num_features)
                x_normalized = self.input_norm(x_reshaped)
                x = x_normalized.view(batch_size, seq_len, num_features)
            else:
                x = self.input_norm(x)

        gru_out, _ = self.gru(x)

        if self.lstm_norm is not None:
            if self.use_batch_norm:
                lstm_reshaped = gru_out.reshape(-1, gru_out.size(-1))
                lstm_normalized = self.lstm_norm(lstm_reshaped)
                gru_out = lstm_normalized.view(batch_size, seq_len, -1)
            else:
                gru_out = self.lstm_norm(gru_out)

        attn_out, attention_weights = self.attention(gru_out, gru_out, gru_out)

        if self.attention_norm is not None:
            if self.use_batch_norm:
                attn_reshaped = attn_out.reshape(-1, attn_out.size(-1))
                attn_normalized = self.attention_norm(attn_reshaped)
                attn_out = attn_normalized.view(batch_size, seq_len, -1)
            else:
                attn_out = self.attention_norm(attn_out)

        pooled = torch.mean(attn_out, dim=1)
        out = self.classifier(pooled)
        return out


class MatrixDataset(Dataset):
    def __init__(self, matrices, labels):
        self.matrices = matrices
        self.labels = labels

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        x = torch.tensor(self.matrices[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def Prepare_data(data_dir, label_file=None, id_name=None, label_name=None):
    file_list = os.listdir(data_dir)

    if label_file is None or id_name is None or label_name is None:
        data_arr = []
        for file_name in tqdm(file_list, desc="Reading data files"):
            file_path = os.path.join(data_dir, file_name)
            this_np = pd.read_csv(file_path).to_numpy()
            data_arr.append(this_np)
        return data_arr
    else:
        data_arr = []
        label_arr = []
        label_df = pd.read_csv(label_file)
        label_df[id_name] = [str(i) for i in label_df[id_name]]

        for file_name in tqdm(file_list, desc="Reading data and matching labels"):
            file_path = os.path.join(data_dir, file_name)
            this_np = pd.read_csv(file_path).to_numpy()
            data_arr.append(this_np)

            file_id = file_name[:-4]
            matched_row = label_df[label_df[id_name] == file_id]
            label = matched_row[label_name].values[0]
            label_arr.append(label)

        label_counts = np.bincount(np.array(label_arr, dtype=int))
        print(f"Label counts: #0 = {label_counts[0]}, #1 = {label_counts[1]}")

        return data_arr, label_arr


def train_fold(fold_args):
    import torch

    fold, train_idx, val_idx, test_idx, data_arr, label_arr, epochs, lr, gpu_uuid, seed = fold_args

    set_seed(seed + fold)
    torch.cuda.set_device(gpu_uuid)
    device = torch.device(f"cuda:{gpu_uuid}" if torch.cuda.is_available() else "cpu")

    dataset = MatrixDataset(data_arr, label_arr)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=16)

    def worker_init_fn(worker_id):
        np.random.seed(seed + fold + worker_id)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=16,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed + fold),
        drop_last=True,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

    set_seed(seed + fold)
    model = SimpleLSTMClassifier(input_dim=data_arr[0].shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    with torch.no_grad():
        dummy_preds, dummy_scores = [], []
        for x, _ in test_loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dummy_scores.extend(probs.cpu().numpy())
            dummy_preds.extend(preds.cpu().numpy())
        _ = (dummy_preds, dummy_scores)
    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, zero_division=0),
        recall_score(all_labels, all_preds, zero_division=0),
        f1_score(all_labels, all_preds, zero_division=0),
        roc_auc_score(all_labels, all_scores),
    )


def train_and_evaluate(data_arr, label_arr, k=5, epochs=200, lr=0.02, seed=42):
    from multiprocessing import get_context

    set_seed(seed)
    n = len(data_arr)
    all_indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices)
    test_size = max(1, int(0.1 * n))
    test_idx = all_indices[:test_size]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    num_gpus = torch.cuda.device_count()
    tasks = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_arr)):
        physical_gpu_id = fold % num_gpus
        tasks.append(
            (
                fold,
                train_idx,
                val_idx,
                test_idx,
                data_arr,
                label_arr,
                epochs,
                lr,
                physical_gpu_id,
                seed,
            )
        )

    with get_context("spawn").Pool(processes=min(k, num_gpus)) as pool:
        results = pool.map(train_fold, tasks)

    accs, precs, recs, f1s, aurocs = zip(*results)
    return {
        "Accuracy": (np.mean(accs), np.std(accs)),
        "Precision": (np.mean(precs), np.std(precs)),
        "Recall": (np.mean(recs), np.std(recs)),
        "F1": (np.mean(f1s), np.std(f1s)),
        "AUROC": (np.mean(aurocs), np.std(aurocs)),
    }


def evaluate_downstream(
    data_arr, label_arr, k=4, epochs=100, lr=0.02, seed=42, tag="DIEINHOSPITAL"
):
    """Evaluate downstream task."""
    set_seed(seed)
    results = {}
    tag = tag
    methods = [
        (
            "Scit-Impute",
            lambda: Prepare_data(
                "./data_imputed/my_model/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "GRIN-Impute",
            lambda: Prepare_data(
                "./data_imputed/grin/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        ("Zero-Impute", lambda: ([zero_impu(matrix) for matrix in data_arr], label_arr)),
        ("Mean-Impute", lambda: ([mean_impu(matrix) for matrix in data_arr], label_arr)),
        ("BFill-Impute", lambda: ([bfill_impu(matrix) for matrix in data_arr], label_arr)),
        (
            "KNN-Impute",
            lambda: Prepare_data(
                "./data_imputed/knn/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "MICE-Impute",
            lambda: Prepare_data(
                "./data_imputed/mice/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "TimeMixerPP-Impute",
            lambda: Prepare_data(
                "./data_imputed/timemixerpp/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "SAITS-Impute",
            lambda: Prepare_data(
                "./data_imputed/saits/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "TEFN-Impute",
            lambda: Prepare_data(
                "./data_imputed/tefn/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "TimesNet-Impute",
            lambda: Prepare_data(
                "./data_imputed/timesnet/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "TSDE-Impute",
            lambda: Prepare_data(
                "./data_imputed/tsde/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
        (
            "Miracle-Impute",
            lambda: Prepare_data(
                "./data_imputed/miracle/III",
                "./AAAI_3_4_labels.csv",
                "ICUSTAY_ID",
                tag,
            ),
        ),
    ]

    for method_name, data_func in tqdm(methods, desc="Evaluating imputation methods"):
        print(f"\n🔄 Evaluating {method_name} ...")
        try:
            set_seed(seed)
            data_arr_method, label_arr_method = data_func()
            accs = train_and_evaluate(
                data_arr_method, label_arr_method, k=k, epochs=epochs, lr=lr, seed=seed
            )
            results[method_name] = accs
            print(f"✅ {method_name} completed. Results: {accs}")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            continue

    table = []
    for method, metrics in results.items():
        row = {
            "Method": method,
            "Seed": seed,
            "Accuracy (mean ± std)": f"{metrics['Accuracy'][0]:.2%} ± {metrics['Accuracy'][1]:.2%}",
            "Precision (mean ± std)": f"{metrics['Precision'][0]:.2%} ± {metrics['Precision'][1]:.2%}",
            "Recall (mean ± std)": f"{metrics['Recall'][0]:.2%} ± {metrics['Recall'][1]:.2%}",
            "F1 Score (mean ± std)": f"{metrics['F1'][0]:.2%} ± {metrics['F1'][1]:.2%}",
            "AUROC (mean ± std)": f"{metrics['AUROC'][0]:.4f} ± {metrics['AUROC'][1]:.4f}",
        }
        table.append(row)

    df_results = pd.DataFrame(table)
    print(df_results)
    df_results.to_csv(f"imputation_comparison_results_seed{seed}.csv", index=False)
    return results
