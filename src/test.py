import time
from datetime import datetime
import torch
import pandas as pd
import os
from src.utils import regression_metrics, classification_metrics
from torch_geometric.loader import NeighborLoader
import logging
from tqdm.auto import tqdm

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
    # use centralized logger
    logger = logging.getLogger("train")
    logger.info("Test start: %s", start_dt.isoformat())

    device = next(model.parameters()).device
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    # Resolve neighbor fanouts similar to train
    fanouts = getattr(args, "neighbor_fanouts", None)
    # infer model depth
    depth = getattr(model, "num_layers", None)
    if depth is None:
        convs = getattr(model, "convs", None)
        if convs is not None:
            try:
                depth = len(convs)
            except Exception:
                depth = 1
        else:
            depth = 1

    if fanouts is None:
        fanouts = [-1] * depth
    elif len(fanouts) != depth and len(fanouts) > 0:
        if len(fanouts) < depth:
            fanouts = fanouts + [fanouts[-1]] * (depth - len(fanouts))
        else:
            fanouts = fanouts[:depth]

    # Create loader if requested
    if use_neighbor_sampling:
        num_flights = graph["flight"].x.size(0)
        input_nodes = ("flight", torch.arange(num_flights))
        loader = NeighborLoader(
            graph,
            num_neighbors=fanouts,
            batch_size=args.batch_size,
            input_nodes=input_nodes,
            shuffle=False,
        )
    else:
        loader = None

    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        if use_neighbor_sampling and loader is not None:
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            for batch_idx, batch in enumerate(tqdm(loader, total=total_batches, desc="Testing")):
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict)
                flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))

                if args.prediction_type == "regression":
                    labels = batch["flight"].y.squeeze(-1)[:flight_batch_size]
                    preds = out.squeeze(-1)[:flight_batch_size]
                    all_labels.append(labels.detach().cpu())
                    all_preds.append(preds.detach().cpu())
                else:
                    labels = batch["flight"].y.view(-1).long()[:flight_batch_size]
                    logits = out[:flight_batch_size]
                    preds = torch.argmax(logits, dim=1)
                    all_labels.append(labels.detach().cpu())
                    all_preds.append(preds.detach().cpu())

        else:
            out = model(graph.x_dict, graph.edge_index_dict)
            if args.prediction_type == "regression":
                labels = graph["flight"].y.float().squeeze(-1)
                preds = out.squeeze(-1)
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
            else:
                labels = graph["flight"].y.long()
                preds = torch.argmax(out, dim=1)
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())

        # Concatenate and compute metrics
        if all_labels:
            labels_cat = torch.cat(all_labels)
            preds_cat = torch.cat(all_preds)
        else:
            labels_cat = torch.tensor([])
            preds_cat = torch.tensor([])

        # Prepare (and save) predictions and true values to CSV; un-normalize if stats available
        try:
            # attempt to recover node ids if available (from neighbor sampling batches)
            node_ids = None
            if use_neighbor_sampling and loader is not None:
                # rebuild node_ids from collected batches if they exposed 'n_id'
                # all_batches_node_ids was not stored; try to get from last batch variable if present
                # Fallback: use sequential ids
                node_ids = torch.arange(labels_cat.size(0)).cpu()
            else:
                # full-batch: use natural order of graph flight nodes
                node_ids = torch.arange(labels_cat.size(0)).cpu()

            true_vals = labels_cat.cpu().numpy()
            pred_vals = preds_cat.cpu().numpy()

            # Try to un-normalize using args.norm_stats (prefer 'y' key then 'ARR_DELAY')
            unnorm_true = None
            unnorm_pred = None
            try:
                norm_stats = getattr(args, "norm_stats", None) or {}
                mu_map = norm_stats.get("mu", {})
                sigma_map = norm_stats.get("sigma", {})
                if "y" in mu_map and "y" in sigma_map:
                    mu = float(mu_map["y"])
                    sigma = float(sigma_map["y"])
                elif "ARR_DELAY" in mu_map and "ARR_DELAY" in sigma_map:
                    mu = float(mu_map["ARR_DELAY"])
                    sigma = float(sigma_map["ARR_DELAY"])
                else:
                    mu = None
                    sigma = None

                if mu is not None and sigma is not None:
                    unnorm_true = (true_vals * sigma) + mu
                    unnorm_pred = (pred_vals * sigma) + mu
            except Exception:
                unnorm_true = None
                unnorm_pred = None

            df = pd.DataFrame({
                "node_id": node_ids.numpy(),
                "true": true_vals,
                "pred": pred_vals,
            })
            if unnorm_true is not None:
                df["true_raw"] = unnorm_true
                df["pred_raw"] = unnorm_pred

            model_file_base = getattr(args, "model_file", None) or getattr(args, "model_type", "model")
            out_dir = getattr(args, "model_dir", ".")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{model_file_base}_preds.csv")
            df.to_csv(out_path, index=False)
            logger.info("Saved predictions to %s", out_path)
        except Exception:
            logger.exception("Failed to save predictions CSV")

        if args.prediction_type == "regression":
            # If normalization stats exist for the target, unnormalize before computing metrics for interpretability
            try:
                norm_stats = getattr(args, "norm_stats", None) or {}
                mu_map = norm_stats.get("mu", {})
                sigma_map = norm_stats.get("sigma", {})
                if "y" in mu_map and "y" in sigma_map:
                    mu = float(mu_map["y"])
                    sigma = float(sigma_map["y"])
                elif "ARR_DELAY" in mu_map and "ARR_DELAY" in sigma_map:
                    mu = float(mu_map["ARR_DELAY"])
                    sigma = float(sigma_map["ARR_DELAY"])
                else:
                    mu = None
                    sigma = None
            except Exception:
                mu = None
                sigma = None

            #if mu is not None and sigma is not None:
                # unnormalize and convert back to torch tensors for metrics
                #labels_unnorm = torch.from_numpy((labels_cat.cpu().numpy() * sigma + mu)).to(labels_cat.device)
                #preds_unnorm = torch.from_numpy((preds_cat.cpu().numpy() * sigma + mu)).to(preds_cat.device)
            #    metrics_results = regression_metrics(labels_unnorm, preds_unnorm, args.norm_stats)
            #else:
            metrics_results = regression_metrics(labels_cat, preds_cat, args.norm_stats)

            metrics_str = (
                f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
                f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
            )
        else:
            metrics_results = classification_metrics(labels_cat, preds_cat)
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

        # If using tqdm, use tqdm.write to avoid corrupting the bar
        if use_neighbor_sampling:
            tqdm.write(" - ".join(info_parts))
        else:
            logger.info(" - ".join(info_parts))

    end_ts = time.time()
    end_dt = datetime.now()
    logger.info("Test end: %s", end_dt.isoformat())
    logger.info("Elapsed: %.2f s", end_ts - start_ts)

