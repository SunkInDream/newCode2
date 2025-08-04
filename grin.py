import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = self.linear(x)
        output = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return output


class GRINet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, n_nodes=None):
        super().__init__()
        self.n_nodes = n_nodes

        # GRU cell
        self.gru = nn.GRUCell(input_dim + hidden_dim, hidden_dim)

        # Graph convolution layers
        self.gcn_input = GCNLayer(input_dim, hidden_dim)
        self.gcn_hidden = GCNLayer(hidden_dim, hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x, adj, mask=None):
        # x: (seq_len, batch, nodes, features)
        # adj: (nodes, nodes)
        seq_len, batch_size, n_nodes, input_dim = x.shape

        h = torch.zeros(batch_size, n_nodes, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[t]  # (batch, nodes, features)

            # Graph convolution on input
            gcn_input = F.relu(self.gcn_input(x_t, adj))

            # Graph convolution on hidden state
            gcn_hidden = F.relu(self.gcn_hidden(h, adj))

            # Concatenate for GRU input
            gru_input = torch.cat([gcn_input, gcn_hidden], dim=-1)

            # Update hidden state node by node
            h_new = torch.zeros_like(h)
            for i in range(n_nodes):
                h_new[:, i, :] = self.gru(gru_input[:, i, :], h[:, i, :])
            h = h_new

            # Output
            output = self.output_layer(h)
            outputs.append(output)

        return torch.stack(outputs, dim=0)


def create_adjacency_matrix(data, threshold=0.1):
    """Create adjacency matrix based on correlation."""
    # Remove NaN for correlation calculation
    data_clean = np.nan_to_num(data)
    corr_matrix = np.corrcoef(data_clean.T)
    adj = (np.abs(corr_matrix) > threshold).astype(float)
    np.fill_diagonal(adj, 0)  # Remove self loops
    return adj


def grin_impute(data_matrix, window_size=20, hidden_dim=64, epochs=100, lr=0.001, input_dim=None):
    """
    GRIN imputation function — fixed/improved imputation logic.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

    # Auto-detect input dimension
    if input_dim is None:
        input_dim = data_matrix.shape[1]

    seq_len, n_features = data_matrix.shape

    print(f"🔧 GRIN config: seq_len={seq_len}, n_features={n_features}, input_dim={input_dim}")
    print(f"Original missing values: {np.isnan(data_matrix).sum()}")

    # Basic validity checks
    if seq_len < window_size:
        window_size = max(1, seq_len // 2)

    class GRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            rnn_out, _ = self.rnn(x)
            out = F.relu(self.fc1(rnn_out))
            out = self.dropout(out)
            output = self.fc2(out)
            return output

    # ✅ Preprocessing — better initial fill strategy
    data = data_matrix.copy()
    original_mask = ~np.isnan(data_matrix)  # True: observed, False: missing

    # Initial fill with multiple strategies
    data_filled = pd.DataFrame(data)

    # 1) Forward-fill then backfill
    data_filled = data_filled.fillna(method="ffill").fillna(method="bfill")

    # 2) Remaining NaNs filled with per-column mean
    for col in range(data_filled.shape[1]):
        if data_filled.iloc[:, col].isna().any():
            col_mean = data_filled.iloc[:, col].mean()
            if not np.isnan(col_mean):
                data_filled.iloc[:, col] = data_filled.iloc[:, col].fillna(col_mean)
            else:
                # If the mean is also NaN, fill with 0
                data_filled.iloc[:, col] = data_filled.iloc[:, col].fillna(0)

    data = data_filled.values
    print(f"Missing after initial fill: {np.isnan(data).sum()}")

    # To tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Sliding windows — simpler creation
    effective_window_size = min(window_size, seq_len)

    if seq_len <= effective_window_size:
        # If sequence length <= window size, use whole sequence directly
        X_windows = [data]
        masks_windows = [original_mask]
        window_positions = [0]
    else:
        # Create overlapping windows
        X_windows = []
        masks_windows = []
        window_positions = []

        step_size = max(1, effective_window_size // 2)

        for i in range(0, seq_len - effective_window_size + 1, step_size):
            end_idx = i + effective_window_size
            X_windows.append(data[i:end_idx])
            masks_windows.append(original_mask[i:end_idx])
            window_positions.append(i)

        # Ensure coverage of the tail
        if window_positions[-1] + effective_window_size < seq_len:
            X_windows.append(data[-effective_window_size:])
            masks_windows.append(original_mask[-effective_window_size:])
            window_positions.append(seq_len - effective_window_size)

    X = np.stack(X_windows)
    masks = np.stack(masks_windows)

    print(f"🔧 Created {X.shape[0]} windows, each of size: {X.shape[1]} x {X.shape[2]}")

    # To tensors
    X_tensor = torch.FloatTensor(X).to(device)
    masks_tensor = torch.FloatTensor(masks.astype(float)).to(device)

    # Create model
    model = GRINModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"🔄 Start GRIN training: {epochs} epochs")

    # Train
    model.train()
    best_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        try:
            outputs = model(X_tensor)

            # Loss only on observed entries
            loss = F.mse_loss(outputs * masks_tensor, X_tensor * masks_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        except Exception as e:
            print(f"❌ Training error at epoch {epoch}: {e}")
            break

    # ✅ Predict and impute
    model.eval()
    with torch.no_grad():
        try:
            predictions = model(X_tensor).cpu().numpy()

            # ✅ Improved reconstruction logic
            result = data_matrix.copy()

            print("🔧 Start filling missing values...")

            # Collect predictions for each missing position
            prediction_counts = np.zeros_like(data_matrix)
            prediction_sums = np.zeros_like(data_matrix)

            # Iterate over windows
            for window_idx, start_pos in enumerate(window_positions):
                if window_idx >= predictions.shape[0]:
                    break

                pred_window = predictions[window_idx]  # (window_size, n_features)

                # Accumulate predictions for aligned positions
                for t in range(effective_window_size):
                    actual_pos = start_pos + t
                    if actual_pos < seq_len:
                        # Only accumulate where original values are missing
                        missing_mask = np.isnan(data_matrix[actual_pos, :])

                        prediction_sums[actual_pos, missing_mask] += pred_window[t, missing_mask]
                        prediction_counts[actual_pos, missing_mask] += 1

            # Average predictions to fill
            filled_count = 0
            for i in range(seq_len):
                for j in range(n_features):
                    if np.isnan(data_matrix[i, j]) and prediction_counts[i, j] > 0:
                        result[i, j] = prediction_sums[i, j] / prediction_counts[i, j]
                        filled_count += 1

            print(f"✅ Filled {filled_count} missing values via model predictions")

            # ✅ For any remaining NaNs, use a fallback strategy
            remaining_missing = np.isnan(result)
            if remaining_missing.any():
                remaining_count = remaining_missing.sum()
                print(f"🔄 Using fallback strategy for remaining {remaining_count} missing values")

                # Column means for remaining missing values
                for j in range(n_features):
                    col_missing = remaining_missing[:, j]
                    if col_missing.any():
                        observed_values = result[~np.isnan(result[:, j]), j]
                        if len(observed_values) > 0:
                            col_mean = np.mean(observed_values)
                            result[col_missing, j] = col_mean
                        else:
                            # If the entire column has no observed values, fill with 0
                            result[col_missing, j] = 0

            final_missing = np.isnan(result).sum()
            print("✅ GRIN imputation complete")
            print(f"Missing before: {np.isnan(data_matrix).sum()}")
            print(f"Missing after:  {final_missing}")

            return result

        except Exception as e:
            print(f"❌ Error during prediction stage: {e}")
            import traceback

            traceback.print_exc()

            # Full fallback to simple mean imputation
            print("🔄 Falling back to simple mean imputation...")
            result = data_matrix.copy()

            for j in range(n_features):
                col_data = result[:, j]
                col_missing = np.isnan(col_data)

                if col_missing.any():
                    observed_values = col_data[~col_missing]
                    if len(observed_values) > 0:
                        col_mean = np.mean(observed_values)
                        result[col_missing, j] = col_mean
                    else:
                        result[col_missing, j] = 0

            return result


# ✅ Test helper
def test_grin_with_different_dims():
    """Test data with different dimensionalities."""
    for n_features in [9, 16, 32]:
        print(f"\n🧪 Testing dimension: {n_features}")

        # Create test data
        test_data = np.random.randn(100, n_features)
        test_data[np.random.random((100, n_features)) < 0.2] = np.nan

        try:
            result = grin_impute(test_data, input_dim=n_features)
            print(f"✅ Dimension {n_features} test passed")
            print(f"   Input shape: {test_data.shape}")
            print(f"   Output shape: {result.shape}")
            print(f"   Remaining NaNs: {np.isnan(result).sum()}")
        except Exception as e:
            print(f"❌ Dimension {n_features} test failed: {e}")


def grin_impute_minimal(data_matrix, window_size=8, hidden_dim=16, epochs=5, lr=0.01):
    """
    Minimal GRIN — minimal memory usage.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

    seq_len, n_features = data_matrix.shape

    # ✅ Tiny model — a small RNN layer only
    class TinyGRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  # use RNN instead of GRU
            self.output = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.output(out)

    # ✅ Simplest preprocessing — fill with zeros
    data = data_matrix.copy()
    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0)

    # ✅ Force CPU to save memory
    device = torch.device("cpu")

    # ✅ No sliding windows — use the whole sequence
    if seq_len > window_size:
        # Simple truncation
        data = data[:window_size]
        mask = mask[:window_size]
        seq_len = window_size

    # ✅ Single-sample processing
    X = data[np.newaxis, :]  # (1, seq_len, n_features)
    mask_tensor = torch.FloatTensor(mask[np.newaxis, :].astype(float))
    X_tensor = torch.FloatTensor(X)

    # ✅ Minimal model
    model = TinyGRINModel(n_features, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)  # use SGD instead of Adam

    print(f"🔄 Start minimal training: {epochs} epochs")

    # ✅ Minimal training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_tensor)

        # Loss only on observed entries
        loss = F.mse_loss(outputs * mask_tensor, X_tensor * mask_tensor)

        loss.backward()
        optimizer.step()

        if epoch == epochs - 1:
            print(f"Final loss: {loss.item():.6f}")

    # ✅ Prediction
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()[0]  # (seq_len, n_features)

    # ✅ Simple imputation logic
    result = data_matrix.copy()

    # Fill missing values within the predicted range
    fill_len = min(seq_len, data_matrix.shape[0])
    for i in range(fill_len):
        for j in range(n_features):
            if np.isnan(data_matrix[i, j]):
                result[i, j] = predictions[i, j]

    # ✅ For the rest, fill with global mean
    remaining_missing = np.isnan(result)
    if remaining_missing.any():
        global_mean = np.nanmean(data_matrix)
        result[remaining_missing] = global_mean if not np.isnan(global_mean) else 0

    print("✅ Minimal GRIN imputation complete")
    return result


def grin_impute_low_memory(data_matrix, window_size=15, hidden_dim=16, epochs=100, lr=0.01):
    """
    GRIN low-memory version — reduce memory while maintaining imputation capability.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

    seq_len, n_features = data_matrix.shape
    print(f"🔧 GRIN config: seq_len={seq_len}, n_features={n_features}")

    # ✅ Lightweight model
    class LightGRINModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            # Use a small LSTM instead of a GRU
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_dim, input_dim)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.dropout(lstm_out)
            return self.fc(out)

    # ✅ Preprocessing — use interpolation for initial fill
    data = data_matrix.copy()
    original_mask = ~np.isnan(data_matrix)

    # Fast interpolation via pandas
    df = pd.DataFrame(data)
    df = df.interpolate(method="linear", axis=0)
    df = df.interpolate(method="linear", axis=1)
    df = df.fillna(df.mean())  # remaining with mean
    data = df.values

    print(f"🔧 Missing after preprocessing: {np.isnan(data).sum()}")

    # ✅ Force CPU to save GPU memory
    device = torch.device("cpu")

    # ✅ Create fewer overlapping windows
    if seq_len <= window_size:
        # Use whole sequence
        windows = [data]
        mask_windows = [original_mask]
        positions = [0]
    else:
        # Larger step to reduce overlap
        step = max(window_size // 2, 1)
        windows = []
        mask_windows = []
        positions = []

        for i in range(0, seq_len - window_size + 1, step):
            windows.append(data[i : i + window_size])
            mask_windows.append(original_mask[i : i + window_size])
            positions.append(i)

        # Ensure tail coverage
        if positions[-1] + window_size < seq_len:
            windows.append(data[-window_size:])
            mask_windows.append(original_mask[-window_size:])
            positions.append(seq_len - window_size)

    print(f"🔧 Created {len(windows)} windows")

    # ✅ Process window-by-window to avoid large batches
    model = LightGRINModel(n_features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"🔄 Start training {epochs} epochs")

    # Collect predictions
    all_predictions = []

    for window_idx, (window_data, window_mask) in enumerate(zip(windows, mask_windows)):
        X = torch.FloatTensor(window_data[np.newaxis, :, :]).to(device)  # (1, window_size, n_features)
        mask = torch.FloatTensor(window_mask[np.newaxis, :, :].astype(float)).to(device)

        # ✅ Train a few epochs per window
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = model(X)
            loss = F.mse_loss(outputs * mask, X * mask)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy()[0]  # (window_size, n_features)
            all_predictions.append((positions[window_idx], pred))

        if (window_idx + 1) % max(1, len(windows) // 4) == 0:
            print(f"Processed window {window_idx+1}/{len(windows)}")

    # ✅ Merge predictions
    result = data_matrix.copy()
    prediction_counts = np.zeros_like(data_matrix)
    prediction_sums = np.zeros_like(data_matrix)

    for start_pos, pred_window in all_predictions:
        end_pos = min(start_pos + window_size, seq_len)
        actual_len = end_pos - start_pos

        for t in range(actual_len):
            actual_t = start_pos + t
            for f in range(n_features):
                if np.isnan(data_matrix[actual_t, f]):  # only fill originally missing values
                    prediction_sums[actual_t, f] += pred_window[t, f]
                    prediction_counts[actual_t, f] += 1

    # Average where we have predictions
    filled_positions = prediction_counts > 0
    result[filled_positions] = prediction_sums[filled_positions] / prediction_counts[filled_positions]

    filled_count = filled_positions.sum()
    print(f"✅ Filled {filled_count} missing values via model")

    # ✅ For any remaining NaNs (e.g., borders), fill with neighbors
    remaining_missing = np.isnan(result)
    if remaining_missing.any():
        print(f"🔄 Remaining {remaining_missing.sum()} missing values will be filled with neighbors")

        df_result = pd.DataFrame(result)
        df_result = df_result.fillna(method="ffill").fillna(method="bfill")

        # If still missing, use column means
        df_result = df_result.fillna(df_result.mean())

        # Finally, fill any residual with 0
        df_result = df_result.fillna(0)

        result = df_result.values

    final_missing = np.isnan(result).sum()
    print(f"✅ Final result: missing before {np.isnan(data_matrix).sum()}, missing after {final_missing}")

    return result


# ✅ If you need to minimize memory even further, use this ultra-minimal version
def grin_impute_ultra_minimal(data_matrix):
    """Ultra-minimal version — uses virtually no extra memory."""
    print("🔧 Using ultra-minimal GRIN")

    # Linear interpolation + tiny noise as a proxy for model learning
    result = data_matrix.copy()
    df = pd.DataFrame(result)

    # Linear interpolation
    df = df.interpolate(method="linear", axis=0)
    df = df.interpolate(method="linear", axis=1)

    # Add small random noise for missing entries
    missing_mask = np.isnan(data_matrix)
    noise = np.random.normal(0, 0.01, data_matrix.shape)

    result = df.values
    result[missing_mask] += noise[missing_mask]

    # Finally, fill any remaining missing values
    if np.isnan(result).any():
        global_mean = np.nanmean(data_matrix)
        result = np.where(np.isnan(result), global_mean if not np.isnan(global_mean) else 0, result)

    print("✅ Ultra-minimal imputation complete")
    return result


if __name__ == "__main__":
    test_grin_with_different_dims()

# if __name__ == "__main__":
#     # Create sample data with missing values
#     np.random.seed(42)
#     data = np.random.randn(100, 5)
#
#     # Introduce missing values
#     missing_mask = np.random.random((100, 5)) < 0.2
#     data[missing_mask] = np.nan
#
#     print(f"Original data shape: {data.shape}")
#     print(f"Missing values: {np.isnan(data).sum()}")
#
#     # Apply GRIN imputation
#     imputed_data = grin_impute(data, window_size=10, epochs=50)
#
#     print(f"Imputed data shape: {imputed_data.shape}")
#     print(f"Remaining missing values: {np.isnan(imputed_data).sum()}")
