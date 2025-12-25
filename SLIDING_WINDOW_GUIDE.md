# Sliding Window Training Guide

## Overview

The aircraft delay prediction pipeline uses **time unit-based sliding windows** for temporal prediction. This approach enables:

- **Flexible temporal granularity**: Configure time units (e.g., 60 minutes = 1 hour) to match your prediction requirements
- **Learn-from-past, predict-future**: Separate learning window (historical context) from prediction window (target period)
- **Data leakage prevention**: Arrival delay features are automatically masked during the prediction window
- **Single graph with masks**: Build the graph once from all data, use train/val/test masks for efficiency
- **Chronological data split**: Data is split chronologically (train/val/test) before windowing to prevent temporal leakage

---

## Key Concepts

### Time Units

Instead of fixed hourly snapshots, the system uses configurable **time units**:
- Default: 60 minutes (1 hour)
- Can be adjusted for finer/coarser granularity (e.g., 30 minutes, 120 minutes)
- Data is binned into these units based on departure timestamps

### Learning Window vs Prediction Window

Each training window consists of two parts:

```
[-------- Learning Window --------][-- Prediction Window --]
       (learn from past)              (predict future)
       e.g., 10 hours                 e.g., 1 hour
```

- **Learning Window** (`learn_window`): Historical context the model learns from
- **Prediction Window** (`pred_window`): Future period the model must predict
- **Key difference**: ARR_DELAY features are masked (set to 0) for all flights in the prediction window to prevent data leakage

### Data Leakage Prevention

The system automatically prevents data leakage:
1. During training, ARR_DELAY features in the prediction window are temporarily masked (zeroed)
2. Model learns from learning window only
3. Model predicts delays for flights in prediction window
4. After each window/epoch, ARR_DELAY features are restored

This ensures the model cannot "cheat" by seeing future arrival delays during training.

---

## Configuration Parameters

### Core Sliding Window Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unit` | 60 | Time unit duration in minutes (e.g., 60 = 1 hour bins) |
| `learn_window` | 10 | Number of time units for learning window (historical context) |
| `pred_window` | 1 | Number of time units for prediction window (target period) |
| `window_stride` | 1 | Number of time units to slide between windows |
| `data_split` | [80, 10, 10] | Train/val/test split percentages (chronological) |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | "data/datasets/" | Path to CSV dataset file (or directory for auto-select) |
| `normalize` | true | Enable feature normalization (recommended) |
| `rows` | null | Optional row limit for quick testing |

---

## Usage Examples

### Basic Training with Sliding Windows

Train with default settings (10-hour learning window, 1-hour prediction window):

```bash
python main.py -m train -t rgcn -g hetero4 \
  --unit 60 \
  --learn_window 10 \
  --pred_window 1 \
  --window_stride 1
```

### Using a Config File

Create a YAML config file (e.g., `my_config.yaml`):

```yaml
mode: train
model_type: rgcn
graph_type: hetero4
prediction_type: regression

# Data
data_path: "data/datasets/2M_20251221_210514.csv"
data_split: [80, 10, 10]
normalize: true

# Sliding Window
unit: 60          # 1 hour per unit
learn_window: 10  # learn from 10 hours
pred_window: 1    # predict next 1 hour
window_stride: 1  # slide by 1 hour

# Training
epochs: 50
lr: 0.001
criterion: huber
batch_size: 150000
```

Run with:

```bash
python main.py -c my_config.yaml
```

### Testing Mode

Test a trained model (uses val/test splits from chronological split):

```bash
python main.py -m test -t rgcn -g hetero4
```

---

## Data Flow Architecture

### 1. Data Loading & Chronological Split

```
Load CSV Data
  ↓
Filter Cancelled Flights
  ↓
Clip Negative ARR_DELAY to 0
  ↓
Assign Time Units (based on departure timestamp + unit size)
  ↓
Sort by Departure Timestamp
  ↓
Chronological Split: train (80%) | val (10%) | test (10%)
  ↓
Create Sliding Windows (only within train portion)
```

### 2. Graph Construction

```
Build Graph from FULL Dataset (train + val + test)
  ├─ Node Features: flights, airports, aircraft, airlines
  ├─ Edges: temporal connections, spatial relationships
  ├─ Masks: train_mask, val_mask, test_mask
  ↓
Single Graph Stored in Memory
```

**Key advantage**: Build graph once, reuse for all windows with different masks.

### 3. Training Loop

```
For Each Epoch:
  For Each Sliding Window:
    ├─ Mask ARR_DELAY features for prediction window flights
    ├─ Apply window-specific masks (learn_mask, pred_mask)
    ├─ Forward pass: learn from learning window
    ├─ Compute loss: predictions on prediction window
    ├─ Backward pass & optimizer step
    ├─ Restore ARR_DELAY features
    └─ Update progress bar
```

**Epoch definition**: One complete pass through all sliding windows.

### 4. Evaluation

After training, model is evaluated on:
- **Validation set**: Chronologically held-out data from middle split
- **Test set**: Chronologically held-out data from final split

Metrics are computed on denormalized values (original scale in minutes).

---

## Window Generation Example

With 24 time units, learn_window=10, pred_window=1, stride=1:

```
Time Units:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23

Window 0:   [L  L  L  L  L  L  L  L  L  L][P]
Window 1:      [L  L  L  L  L  L  L  L  L  L][P]
Window 2:         [L  L  L  L  L  L  L  L  L  L][P]
...
Window 13:                                            [L  L  L  L  L  L  L  L  L  L][P]

Legend:
  L = Learning window (learn from these flights)
  P = Prediction window (predict delays for these flights)
```

Total windows: 14 (for 24 units with learn_window=10, pred_window=1, stride=1)

---

## Normalization

**Per-dataset normalization** is applied:

1. Statistics (μ, σ) are computed on the **training set only**
2. Same normalization is applied to train/val/test sets
3. Prevents data leakage from val/test into training statistics
4. Metrics are automatically denormalized for interpretability

**Normalized features** (specified in `data/normalize.txt`):
- Numeric columns like `DISTANCE`, `CRS_ELAPSED_TIME`
- Target variable `ARR_DELAY` (for regression)

---

## Output & Logging

### Training Progress

```
Time unit: 60 minutes
Total time units: 8760
Dataset spans: 2016-01-01 00:05:00 to 2016-12-31 23:59:00

Generated 500 training windows
  First window: units [0, 11], learn=12450, pred=1250
  Last window: units [489, 500], learn=12380, pred=1245

PLACEHOLDER
```

### Predictions CSV

After testing, predictions are saved to `logs/{model_file}_preds.csv`:

| Column | Description |
|--------|-------------|
| `node_id` | Flight index in dataset |
| `true` | Actual arrival delay (minutes) |
| `pred` | Predicted arrival delay (minutes) |
| `error` | Absolute error (minutes) |

---

## Best Practices

1. **Start with default parameters**: 60-minute units, 10-unit learning window, 1-unit prediction window
2. **Use chronological split**: Ensures temporal validity (past → future)
3. **Enable normalization**: Improves training stability and convergence
4. **Monitor denormalized metrics**: RMSE/MAE in minutes are interpretable
5. **Save graphs for reuse**: Use `--save_graph` to avoid rebuilding
6. **Use Huber loss**: More robust to outliers than MSE
7. **Full-batch training**: More stable for heterogeneous graphs (use `neighbor_sampling: false`)

---


## Troubleshooting

### Issue: Out of Memory

**Solutions**:
- Reduce `batch_size`
- Increase `unit` size (fewer, larger time bins)
- Increase `window_stride` (fewer overlapping windows)
- Use `neighbor_sampling: true` for mini-batch training

### Issue: Poor Prediction Accuracy

**Solutions**:
- Increase `learn_window` (more historical context)
- Decrease `unit` size (finer temporal resolution)
- Tune learning rate (`lr`) and weight decay
- Verify ARR_DELAY masking is working (check training logs)

### Issue: Training Too Slow

**Solutions**:
- Increase `window_stride` (fewer windows)
- Increase `unit` size (fewer time bins)
- Enable `--compile` (PyTorch 2.x JIT compilation)
- Reduce `epochs`

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Data Split** | Chronological: train (80%) → val (10%) → test (10%) |
| **Time Units** | Configurable bins (default 60 minutes) |
| **Windows** | Learn window (context) + pred window (target) |
| **Data Leakage Prevention** | ARR_DELAY masked during prediction window |
| **Graph** | Built once from full dataset, reused with masks |
| **Epoch** | One pass through all sliding windows |
| **Metrics** | Automatically denormalized to original scale (minutes) |
| **Normalization** | Per-dataset, computed on training set only |

This architecture is **time-aware**, **efficient**, and **prevents data leakage**—ideal for temporal prediction tasks like flight delay forecasting.
