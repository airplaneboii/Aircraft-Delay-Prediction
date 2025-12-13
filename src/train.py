import torch
import time
from datetime import datetime
from torch import nn, optim
from src.utils import regression_metrics, classification_metrics

try:
    import psutil
except Exception:
    psutil = None

def train(
        model: nn.Module,
        graph,
        args
    ) -> None:
    # Set model to training mode
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define loss function
    if args.prediction_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    overall_start = time.time()
    start_dt = datetime.now()
    print(f"Training start: {start_dt.isoformat()}")

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Always remember to zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(graph.x_dict, graph.edge_index_dict)
        if args.prediction_type == "regression":
            labels = graph["flight"].y.squeeze(-1).to(out.device)
            preds = out.squeeze(-1)
        else:
            labels = graph["flight"].y.long().to(out.device)
            preds = out

        # Compute loss
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start

        # Compute metrics for monitoring
        if args.prediction_type == "regression":
            metrics_results = regression_metrics(labels, preds)
            metrics_str = (
                f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
                f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
            )
        else:
            metrics_results = classification_metrics(labels, torch.argmax(out, dim=1))
            metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

        # Resource usage
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

        info_parts = [f"Epoch {epoch+1}/{args.epochs}", f"{metrics_str}", f"time: {epoch_time:.2f}s"]
        if gpu_mem is not None:
            info_parts.append(f"gpu_mem_peak: {gpu_mem:.1f} MB")
        if cpu_info is not None:
            info_parts.append(f"proc_mem: {cpu_info[0]:.1f} MB")
            info_parts.append(f"cpu%: {cpu_info[1]:.1f}")

        print(" - ".join(info_parts))

    overall_end = time.time()
    end_dt = datetime.now()
    total_time = overall_end - overall_start
    print(f"Training end: {end_dt.isoformat()}")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    if args.epochs > 0:
        print(f"Average epoch time: {total_time/args.epochs:.2f}s")