import time
import torch
import logging
from tqdm.auto import tqdm
from src.utils import resolve_fanouts, get_labels

try:
    import psutil
except Exception:
    psutil = None


def _evaluate_single(model, graph, args, eval_mask, use_neighbor_sampling, fanouts, device, logger, start_ts):
    """Evaluate on a single dataset split (no windows)."""
    all_labels = []
    all_preds = []
    
    eval_nodes = eval_mask.nonzero(as_tuple=False).view(-1)
    
    # Create loader if using neighbor sampling
    if use_neighbor_sampling:
        from torch_geometric.loader import NeighborLoader
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

    with torch.no_grad():
        if use_neighbor_sampling and loader is not None:
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            for batch_idx, batch in enumerate(tqdm(loader, total=total_batches, desc=f"Evaluating {args.mode}")):
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict)
                flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))

                if args.prediction_type == "regression":
                    labels = get_labels(batch, "regression")[:flight_batch_size]
                    preds = out.squeeze(-1)[:flight_batch_size]
                    all_labels.append(labels.detach().cpu())
                    all_preds.append(preds.detach().cpu())
                else:
                    labels = get_labels(batch, "classification")[:flight_batch_size]
                    logits = out[:flight_batch_size]
                    probs = torch.sigmoid(logits)
                    preds = (probs >= args.border).long()
                    all_labels.append(labels.detach().cpu())
                    all_preds.append(preds.detach().cpu())

        else:
            # Full-batch evaluation
            out = model(graph.x_dict, graph.edge_index_dict)
            
            if args.prediction_type == "regression":
                labels = get_labels(graph, "regression", eval_mask)
                preds = out.squeeze(-1)[eval_mask]
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
            else:
                labels = get_labels(graph, "classification", eval_mask)
                logits = out[eval_mask]
                probs = torch.sigmoid(logits)
                preds = (probs >= args.border).long()
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())

    # Concatenate and compute metrics using compute_epoch_stats
    labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([])
    preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])
    
    from src.utils import compute_epoch_stats
    compute_epoch_stats(0, args, graph, labels_cat, preds_cat, [0.0], start_ts, logger)


def _evaluate_windows(model, graph, args, window_defs, eval_mask, use_neighbor_sampling, fanouts, device, logger, start_ts):
    """Evaluate on sliding windows using induced subgraphs."""
    from src.subgraph_builder import build_window_subgraph
    
    print(f"Evaluating {len(window_defs)} windows using induced subgraphs")
    
    # Get ARR_DELAY feature index for masking
    feat_map = getattr(graph["flight"], "feat_index", None)
    if feat_map is not None and "arr_delay" in feat_map:
        arr_idx = feat_map["arr_delay"]
    else:
        arr_idx = -2  # fallback
    
    all_window_labels = []
    all_window_preds = []
    
    for window_info in tqdm(window_defs, desc=f"Evaluating {args.mode} windows", unit="window"):
        learn_indices = window_info['learn_indices']
        pred_indices = window_info['pred_indices']
        
        # Build induced subgraph for this window
        subgraph, local_pred_mask = build_window_subgraph(
            graph, learn_indices, pred_indices, device=device
        )
        
        # Mask ARR_DELAY in the subgraph for prediction window
        subgraph["flight"].x = subgraph["flight"].x.clone()
        subgraph["flight"].x[local_pred_mask, arr_idx] = 0.0
        
        with torch.no_grad():
            if use_neighbor_sampling:
                # For neighbor sampling with windows: more complex, skip for now
                logger.warning("Neighbor sampling with sliding windows not fully implemented for evaluation")
                continue
            else:
                # Full-batch evaluation on subgraph
                out = model(subgraph.x_dict, subgraph.edge_index_dict)
                
                if args.prediction_type == "regression":
                    # Evaluate on prediction window only (local_pred_mask)
                    labels = get_labels(subgraph, "regression", local_pred_mask)
                    preds = out.squeeze(-1)[local_pred_mask]
                    all_window_labels.append(labels.detach().cpu())
                    all_window_preds.append(preds.detach().cpu())
                else:
                    # Use prediction-window mask for labels so lengths match preds
                    labels = get_labels(subgraph, "classification", local_pred_mask)
                    logits = out[local_pred_mask]
                    probs = torch.sigmoid(logits)
                    preds = (probs >= args.border).long()
                    all_window_labels.append(labels.detach().cpu())
                    all_window_preds.append(preds.detach().cpu())
    
    # Concatenate all windows and compute metrics
    labels_cat = torch.cat(all_window_labels) if all_window_labels else torch.tensor([])
    preds_cat = torch.cat(all_window_preds) if all_window_preds else torch.tensor([])
    
    from src.utils import compute_epoch_stats
    # Sanity check: labels and preds should have same length
    if labels_cat.numel() != preds_cat.numel():
        logger.warning("Mismatch between labels and preds: %d vs %d; trimming to minimum for metrics", labels_cat.numel(), preds_cat.numel())
        mn = min(labels_cat.numel(), preds_cat.numel())
        if mn == 0:
            logger.warning("No labels or predictions to evaluate. Skipping metrics for this run.")
            return
        labels_cat = labels_cat[:mn]
        preds_cat = preds_cat[:mn]

    compute_epoch_stats(0, args, graph, labels_cat, preds_cat, [0.0], start_ts, logger)


def test(
        model: torch.nn.Module,
        graph,
        args,
        window_defs: list = None,  # New: sliding window definitions for evaluation
        ) -> None:
    """
    Evaluate the model on val/test data.
    If window_defs is provided, evaluate on sliding windows.
    Otherwise, evaluate on full split.
    """
    import time
    start_ts = time.time()
    logger = logging.getLogger("train")
    
    device = next(model.parameters()).device
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))
    
    if args.mode == "val":
        eval_mask = graph["flight"].val_mask
    elif args.mode == "test":
        eval_mask = graph["flight"].test_mask
    else:
        raise ValueError(f"Invalid mode for testing: {args.mode}")

    # Resolve neighbor fanouts similar to train
    fanouts = resolve_fanouts(model, getattr(args, "neighbor_fanouts", None))
    
    model.eval()
    
    if window_defs is None:
        # No sliding windows: evaluate on full split (legacy behavior)
        logger.info(f"Evaluating {args.mode} on full split (no windows)")
        _evaluate_single(model, graph, args, eval_mask, use_neighbor_sampling, fanouts, device, logger, start_ts)
    else:
        # Sliding window evaluation
        logger.info(f"Evaluating {args.mode} on {len(window_defs)} sliding windows")
        _evaluate_windows(model, graph, args, window_defs, eval_mask, use_neighbor_sampling, fanouts, device, logger, start_ts)

