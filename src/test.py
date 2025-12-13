import time
from datetime import datetime
import torch
from src.utils import regression_metrics, classification_metrics

try:
    import psutil
except Exception:
    psutil = None


def test(
        model: torch.nn.Module,
        graph,
        args
        ) -> None:
    start_dt = datetime.now()
    start_ts = time.time()
    print(f"Test start: {start_dt.isoformat()}")

    model.eval()
    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)

        # Compute metrics and format like train.py
        if args.prediction_type == "regression":
            labels = graph["flight"].y.float().squeeze(-1)
            preds = out.squeeze(-1)
            metrics_results = regression_metrics(labels, preds)
            metrics_str = (
                f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
                f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
            )
        else:
            labels = graph["flight"].y.long()
            preds = torch.argmax(out, dim=1)
            metrics_results = classification_metrics(labels, preds)
            metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

        # Resource usage similar to train.py
        gpu_mem = None
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
            except Exception:
                gpu_mem = None

        cpu_info = None
        if psutil:
            p = psutil.Process()
            mem_mb = p.memory_info().rss / 1024**2
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_info = (mem_mb, cpu_pct)

        elapsed = time.time() - start_ts
        info_parts = [f"Test", f"{metrics_str}", f"time: {elapsed:.2f}s"]
        if gpu_mem is not None:
            info_parts.append(f"gpu_mem_peak: {gpu_mem:.1f} MB")
        if cpu_info is not None:
            info_parts.append(f"proc_mem: {cpu_info[0]:.1f} MB")
            info_parts.append(f"cpu%: {cpu_info[1]:.1f}")

        print(" - ".join(info_parts))

    end_ts = time.time()
    end_dt = datetime.now()
    print(f"Test end: {end_dt.isoformat()}")
    print(f"Elapsed: {end_ts - start_ts:.2f}s")

