import time
from datetime import datetime
import torch
import pandas as pd
import os
from src.utils import regression_metrics, classification_metrics, compute_epoch_stats, resolve_fanouts
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
    
    if args.mode == "val":
        eval_mask = graph["flight"].val_mask
    elif args.mode == "test":
        eval_mask = graph["flight"].test_mask
    else:
        raise ValueError(f"Invalid mode for testing: {args.mode}")

    eval_nodes = eval_mask.nonzero(as_tuple=False).view(-1)

    # Resolve neighbor fanouts similar to train
    fanouts = resolve_fanouts(model, getattr(args, "neighbor_fanouts", None))

    # Create loader if requested
    if use_neighbor_sampling:
        input_nodes = ("flight", eval_nodes)
        
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
                    labels = batch["flight"].y.view(-1).float()[:flight_batch_size]
                    logits = out[:flight_batch_size]
                    probs = torch.sigmoid(logits)
                    preds = (probs >= args.border).long()
                    all_labels.append(labels.detach().cpu())
                    all_preds.append(preds.detach().cpu())

        else:
            out = model(graph.x_dict, graph.edge_index_dict)
            
            if args.prediction_type == "regression":
                labels = graph["flight"].y.float().squeeze(-1)[eval_mask]
                preds = out.squeeze(-1)[eval_mask]
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
            else:
                labels = graph["flight"].y.float()[eval_mask]  # float for BCE
                logits = out[eval_mask]  # shape [num_nodes]

                # convert logits to 0/1 predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= args.border).long()

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
        if args.prediction_type == "regression":
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
                    norm_stats = getattr(graph, "norm_stats", None) or getattr(args, "norm_stats", None) or {}
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
        elif args.prediction_type == "classification":
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

                true_vals = labels_cat.view(-1).cpu().numpy()
                pred_vals = preds_cat.view(-1).cpu().numpy()

                df = pd.DataFrame({
                    "node_id": node_ids.numpy(),
                    "true": true_vals,
                    "pred": pred_vals,
                })

                model_file_base = getattr(args, "model_file", None) or getattr(args, "model_type", "model")
                out_dir = getattr(args, "model_dir", ".")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{model_file_base}_preds.csv")
                df.to_csv(out_path, index=False)
                logger.info("Saved predictions to %s", out_path)
            except Exception:
                logger.exception("Failed to save predictions CSV")
        else:
            raise ValueError(f"Unknown prediction type: {args.prediction_type}")

        if args.prediction_type == "regression":
            # If normalization stats exist for the target, unnormalize before computing metrics for interpretability
            try:
                norm_stats = getattr(graph, "norm_stats", None) or getattr(args, "norm_stats", None) or {}
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
        # Use the shared logging helper to print metrics and resource usage
        # Pass epoch=0 and a dummy epoch_losses list for compatibility
        compute_epoch_stats(0, args, graph, labels_cat, preds_cat, [0.0], start_ts, logger)

    end_ts = time.time()
    end_dt = datetime.now()
    logger.info("Test end: %s", end_dt.isoformat())
    logger.info("Elapsed: %.2f s", end_ts - start_ts)

