import time
import torch
import logging
from tqdm.auto import tqdm
from src.utils import resolve_fanouts

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
            # Full-batch evaluation
            out = model(graph.x_dict, graph.edge_index_dict)
            
            if args.prediction_type == "regression":
                labels = graph["flight"].y.float().squeeze(-1)[eval_mask]
                preds = out.squeeze(-1)[eval_mask]
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
            else:
                labels = graph["flight"].y.float()[eval_mask]
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
    """Evaluate on sliding windows."""
    # Store original ARR_DELAY if masking is needed (for hetero4)
    if "ARR_DELAY" in [col for col in range(graph["flight"].x.size(1))]:
        arr_idx = getattr(graph["flight"], "feat_index", {}).get("arr_delay", -2)
        arr_delay_original = graph["flight"].x[:, arr_idx].clone()
    else:
        arr_delay_original = None
    
    all_window_labels = []
    all_window_preds = []
    
    for window_info in tqdm(window_defs, desc=f"Evaluating {args.mode} windows", unit="window"):
        # Restore original ARR_DELAY
        if arr_delay_original is not None:
            graph["flight"].x[:, arr_idx] = arr_delay_original
        
        # Mask ARR_DELAY for pred window (same as training)
        pred_indices = window_info['pred_indices']
        if arr_delay_original is not None:
            graph["flight"].x[pred_indices, arr_idx] = 0.0
        
        # Set masks for evaluation
        learn_mask_tensor = torch.tensor(window_info['learn_mask'], dtype=torch.bool, device=graph["flight"].x.device)
        pred_mask_tensor = torch.tensor(window_info['pred_mask'], dtype=torch.bool, device=graph["flight"].x.device)
        
        # Combine with eval mask (val or test split)
        eval_learn_mask = eval_mask & learn_mask_tensor
        eval_pred_mask = eval_mask & pred_mask_tensor
        
        with torch.no_grad():
            if use_neighbor_sampling:
                # For neighbor sampling with windows: more complex, skip for now
                logger.warning("Neighbor sampling with sliding windows not fully implemented for evaluation")
                continue
            else:
                # Full-batch evaluation
                out = model(graph.x_dict, graph.edge_index_dict)
                
                if args.prediction_type == "regression":
                    # Evaluate on prediction window only
                    labels = graph["flight"].y.float().squeeze(-1)[eval_pred_mask]
                    preds = out.squeeze(-1)[eval_pred_mask]
                    all_window_labels.append(labels.detach().cpu())
                    all_window_preds.append(preds.detach().cpu())
                else:
                    labels = graph["flight"].y.float()[eval_pred_mask]
                    logits = out[eval_pred_mask]
                    probs = torch.sigmoid(logits)
                    preds = (probs >= args.border).long()
                    all_window_labels.append(labels.detach().cpu())
                    all_window_preds.append(preds.detach().cpu())
    
    # Restore original ARR_DELAY
    if arr_delay_original is not None:
            graph["flight"].x[:, arr_idx] = arr_delay_original
    
    # Concatenate all windows and compute metrics
    labels_cat = torch.cat(all_window_labels) if all_window_labels else torch.tensor([])
    preds_cat = torch.cat(all_window_preds) if all_window_preds else torch.tensor([])
    
    from src.utils import compute_epoch_stats
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

