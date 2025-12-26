#from asyncio import graph
import torch
import time
from datetime import datetime
from torch import nn, optim
from src.utils import compute_epoch_stats, resolve_fanouts, get_labels, move_graph_to_device
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import logging
import os
import csv
from tqdm.auto import tqdm

try:
    import psutil
except Exception:
    psutil = None

OVERSAMPLE_FACTOR = 1.0


def _save_checkpoint(model, optimizer, scheduler, args, epoch, model_path, logger):
    """Save model checkpoint (helper).

    Keeps same dict layout as prior inline saves so resume works unchanged.
    """
    logging.debug(f"Saving checkpoint for epoch {epoch+1}/{getattr(args, 'epochs', '?')}")
    try:
        ckpt = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "norm_stats": getattr(args, "norm_stats", None),
        }
        torch.save(ckpt, model_path)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint for epoch {epoch+1}: {e}")


def _train_windowed(model, graph, args, window_defs, optimizer, scheduler, model_path, csv_writer, csv_file, logger, device, amp_enabled, scaler, start_epoch=0, criterion=None):
    """Train using sliding windows with induced subgraphs (hetero4)."""
    from src.utils import WindowSubgraphBuilder, build_window_subgraph
    
    print(f"\n=== Training with sliding windows: {len(window_defs)} windows per epoch ===")
    print(f"Using induced subgraphs to prevent message passing over non-window flights")

    overall_start = time.time()

    # Get ARR_DELAY feature index for masking
    arr_idx = getattr(graph["flight"], "feat_index", {}).get("arr_delay", -2)

    num_epochs = args.epochs

    # Reusable iterative builder for subgraphs across windows
    builder = WindowSubgraphBuilder(graph)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_losses = []
        all_labels = []
        all_preds = []

        for window_info in tqdm(window_defs, desc=f"Epoch {epoch+1}/{num_epochs}", unit="window"):
            learn_indices = window_info["learn_indices"]
            pred_indices = window_info["pred_indices"]
            
            # Build induced subgraph for this window (includes only window flights + connected nodes)
            # Iterative approach: reuse the builder across windows
            subgraph, local_pred_mask = builder.build_subgraph(
                learn_indices, pred_indices, device=device
            )
            
            # Mask ARR_DELAY in the subgraph for prediction window only
            # Clone to avoid mutating the original subgraph across iterations
            subgraph["flight"].x = subgraph["flight"].x.clone()
            subgraph["flight"].x[local_pred_mask, arr_idx] = 0.0
            
            # Create local train mask (all flights in subgraph belong to train split already)
            # Since we filter by window_info from train windows in main.py
            train_mask_local = torch.ones(subgraph["flight"].x.size(0), dtype=torch.bool, device=device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(subgraph.x_dict, subgraph.edge_index_dict)

            if args.prediction_type == "regression":
                labels = get_labels(subgraph, "regression", local_pred_mask).to(device)
                preds = out.squeeze(-1)[local_pred_mask]

                if criterion is None:
                    raise ValueError("A loss `criterion` must be provided to _train_windowed via train()")

                loss = criterion(preds, labels)

                # Backward and step
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
            else:
                labels = get_labels(subgraph, "classification", local_pred_mask).to(device)
                # Ensure logits is 1-D before masking to avoid 0-d scalar when selecting single item
                logits_all = out.squeeze(-1)
                logits = logits_all[local_pred_mask].to(device)

                if criterion is None:
                    raise ValueError("A loss `criterion` must be provided to _train_windowed via train()")

                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                preds_for_metrics = (probs > args.border).long().cpu()

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)

        labels_cat = torch.cat(all_labels, dim=0)
        preds_cat = torch.cat(all_preds, dim=0)

        stats = compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start_time, logger)
        if csv_writer:
            csv_writer.writerow({k: stats.get(k, "") for k in csv_writer.fieldnames})
            csv_file.flush()

        if scheduler:
            scheduler.step()

        _save_checkpoint(model, optimizer, scheduler, args, epoch, model_path, logger)

    if csv_file:
        csv_file.close()

    print(f"\nTraining completed in {time.time() - overall_start:.2f}s")
    return


def _train_legacy(model, graph, args, loader, use_neighbor_sampling, fanouts, optimizer, scheduler, model_path, csv_writer, csv_file, logger, device, amp_enabled, scaler, start_epoch=0, criterion=None):
    """Legacy training path (neighbor sampling or full-batch)."""
    overall_start = time.time()
    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_losses = []
        all_labels = []
        all_preds = []

        logger.debug("epoch %d: use_neighbor_sampling=%s, fanouts=%s, batch_size=%s", epoch+1, use_neighbor_sampling, fanouts, args.batch_size)

        if use_neighbor_sampling:
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            for batch_idx, batch in enumerate(tqdm(loader, total=total_batches, desc=f"Epoch {epoch+1}")):
                batch = batch.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    out = model(batch.x_dict, batch.edge_index_dict)
                flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))

                if args.prediction_type == "regression":
                    labels = get_labels(batch, "regression")[:flight_batch_size].to(device)
                    preds = out.squeeze(-1)[:flight_batch_size]
                    if criterion is None:
                        raise ValueError("A loss `criterion` must be provided to _train_legacy via train()")
                    loss = criterion(preds, labels)
                    preds_for_metrics = preds.detach().cpu()
                else:
                    labels_full = get_labels(batch, "classification")[:flight_batch_size].to(device)
                    # Make logits 1-D for consistent indexing
                    logits_full = out[:flight_batch_size].squeeze(-1)
                    labels = labels_full
                    pos_mask = labels_full == 1
                    neg_mask = labels_full == 0
                    pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
                    neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)

                    oversample_factor = int(OVERSAMPLE_FACTOR)
                    pos_indices_os = pos_indices.repeat(oversample_factor) if pos_indices.numel() > 0 else pos_indices
                    train_indices_os = torch.cat([neg_indices, pos_indices_os]) if neg_indices.numel() > 0 else pos_indices_os
                    if train_indices_os.numel() == 0:
                        train_indices_os = torch.arange(labels_full.size(0), device=labels_full.device)

                    labels_os = labels_full[train_indices_os]
                    logits_os = logits_full[train_indices_os]
                    if criterion is None:
                        raise ValueError("A loss `criterion` must be provided to _train_legacy via train()")
                    loss = criterion(logits_os, labels_os)
                    probs = torch.sigmoid(logits_full)
                    preds_for_metrics = (probs > args.border).long().cpu()

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)
        else:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(graph.x_dict, graph.edge_index_dict)

            if args.prediction_type == "regression":
                base_mask = graph["flight"].train_mask
                if hasattr(graph["flight"], "target_mask"):
                    mask = base_mask & graph["flight"].target_mask
                elif hasattr(graph["flight"], "window_mask"):
                    mask = base_mask & graph["flight"].window_mask
                else:
                    mask = base_mask

                labels = get_labels(graph, "regression", mask).to(device)
                preds = out.squeeze(-1)[mask]
                if criterion is None:
                    raise ValueError("A loss `criterion` must be provided to _train_legacy via train()")
                loss = criterion(preds, labels)
                preds_for_metrics = preds.detach().cpu()

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)
            else:
                mask = graph["flight"].train_mask
                labels = get_labels(graph, "classification", mask).to(device)
                # Ensure logits is 1-D before masking
                logits_all = out.squeeze(-1)
                logits = logits_all[mask].to(device)
                if criterion is None:
                    raise ValueError("A loss `criterion` must be provided to _train_legacy via train()")
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds_for_metrics = (probs > args.border).long().cpu()

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)

        labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([])
        preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])

        epoch_stats = compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start_time, logger)
        if csv_writer is not None and epoch_stats is not None:
            csv_row = {key: epoch_stats.get(key) for key in csv_writer.fieldnames}
            csv_writer.writerow(csv_row)
            csv_file.flush()

        _save_checkpoint(model, optimizer, scheduler, args, epoch, model_path, logger)

    if csv_file is not None:
        csv_file.close()
        logger.info(f"Training statistics saved to: {os.path.basename(csv_file.name)}")

    logger.info("Training completed. Final checkpoint saved to %s", model_path)
    overall_end = time.time()
    total_time = overall_end - overall_start
    logger.info("Total training time: %.2f s (%.2f minutes)", total_time, total_time/60)
    if args.epochs > 0:
        logger.info("Average epoch time: %.2f s", total_time/args.epochs)
    return

def train(
        model: nn.Module,
        graph,
        args,
        window_defs: list = None,  # New: list of window definitions for hetero4
    ) -> None:
    print(args)
    # Use args.epochs for number of epochs
    num_epochs = args.epochs
    
    # Set model to training mode
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Optional learning-rate scheduler (not configured by default)
    scheduler = None
    device = next(model.parameters()).device

    # y = graph["flight"].y.squeeze(-1).cpu()
    # print("y mean/std:", y.mean().item(), y.std().item(), "min/max:", y.min().item(), y.max().item())

    # Define loss function
    if args.prediction_type == "regression":
        if args.criterion == "mse":
            criterion = nn.MSELoss()
        elif args.criterion == "huber":
            criterion = nn.SmoothL1Loss()
        elif args.criterion == "l1":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown regression criterion: {args.criterion}")
    else:
        y = get_labels(graph, "classification", graph["flight"].train_mask).view(-1)
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        print(num_pos, num_neg)
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
        #pos_weight = torch.tensor([num_neg / num_neg], device=device) # 1.0 no weighting
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    # Resolve neighbor fanouts (None => full neighbors)
    fanouts = resolve_fanouts(model, getattr(args, "neighbor_fanouts", None))

    if use_neighbor_sampling:
        train_nodes = graph["flight"].train_mask.nonzero(as_tuple=False).view(-1)
        input_nodes = ("flight", train_nodes)
        
        loader = NeighborLoader(
            graph,
            num_neighbors=fanouts,
            batch_size=args.batch_size,
            input_nodes=input_nodes,
            shuffle=True,
        )
    else:
        loader = None

    # Use the global logger configured by main; get local logger
    logger = logging.getLogger("train")

    overall_start = time.time()
    start_dt = datetime.now()
    logger.info("Training start: %s", start_dt.isoformat())

    model_path = os.path.join(args.model_dir, f"{args.model_file}.pt")
    csv_path = os.path.join(args.log_dir, f"{args.model_file}_training_stats.csv")

    # Initialize CSV file for epoch statistics
    csv_file = None
    csv_writer = None
    if args.prediction_type == "regression":
        csv_headers = ["epoch", "loss", "MSE", "MAE", "RMSE", "R2", "time_s", "gpu_mem_mb", "proc_mem_mb", "cpu_pct"]
    else:
        csv_headers = ["epoch", "loss", "Accuracy", "Precision", "Recall", "F1_Score", "time_s", "gpu_mem_mb", "proc_mem_mb", "cpu_pct"]
    
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    csv_writer.writeheader()
    logger.info(f"Training statistics will be logged to: {csv_path}")

    # resume from checkpoint when enabled
    start_epoch = 0
    if getattr(args, "resume", False) and os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0))
                if "optimizer_state_dict" in ckpt and optimizer is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt and scheduler is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                args.norm_stats = ckpt.get("norm_stats", getattr(args, "norm_stats", None))
                print(f"Resuming training from epoch {start_epoch+1}, loaded checkpoint {model_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {model_path}: {e}")

    # AMP scaler (optional mixed precision)
    # Optional: AMP via torch.amp (CUDA-only)
    amp_enabled = bool(getattr(args, "amp", False)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # If AMP is enabled, patch PyG segment_matmul to compute in float32 when inputs are float16
    if amp_enabled:
        try:
            import pyg_lib.ops as pyg_ops
            _orig_segment_matmul = getattr(pyg_ops, 'segment_matmul', None)
            if _orig_segment_matmul is not None:
                def _safe_segment_matmul(inputs, ptr, other):
                    # If either input is float16, perform matmul in float32 and cast back
                    try:
                        if inputs.dtype == torch.float16 or other.dtype == torch.float16:
                            inputs32 = inputs.to(torch.float32)
                            other32 = other.to(torch.float32)
                            out = _orig_segment_matmul(inputs32, ptr, other32)
                            return out.to(inputs.dtype)
                        else:
                            return _orig_segment_matmul(inputs, ptr, other)
                    except Exception:
                        # On any failure, fallback to original op to raise native errors
                        return _orig_segment_matmul(inputs, ptr, other)
                pyg_ops.segment_matmul = _safe_segment_matmul
                logging.getLogger("train").info("Patched pyg_lib.ops.segment_matmul to use float32 compute when inputs are float16 (AMP compatibility).")
        except Exception as e:
            logging.getLogger("train").warning(f"Failed to patch pyg segment_matmul for AMP: {e}")

    # Optional: compile the model for speed (PyTorch 2.x)
    if bool(getattr(args, "compile", False)):
        backend = getattr(args, "compile_backend", "inductor")
        mode = getattr(args, "compile_mode", "default")
        compile_attempted = False

        # If using inductor backend, ensure Triton is available (it's required by inductor).
        # If Triton is missing, or MSVC isn't available on Windows, fall back to aot_eager.
        selected_backend = backend
        if backend == "inductor":
            try:
                # Some Triton packages (especially on Windows) may not expose the
                # symbols needed by torch-inductor. Probe for the expected API.
                from triton.compiler.compiler import triton_key  # type: ignore
                import triton  # type: ignore
                triton_ver = getattr(triton, "__version__", "unknown")
                logger.info(f"Found Triton (version={triton_ver}), proceeding with inductor backend")
            except Exception as e:
                logger.warning(
                    "Triton is present but incompatible (or missing required symbols): %s; "
                    "falling back from 'inductor' to 'aot_eager' backend. "
                    "If you want to use inductor, install a Triton build compatible with your PyTorch version (see: https://github.com/openai/triton)",
                    e,
                )
                selected_backend = "aot_eager"
            else:
                # On Windows, torch-inductor requires MSVC (cl.exe) to JIT-compile
                # CPU-side kernels; if cl is not on PATH, fallback to aot_eager.
                try:
                    import shutil, platform
                    if platform.system() == "Windows":
                        if shutil.which("cl") is None:
                            logger.warning(
                                "MSVC cl.exe not found on PATH; falling back from 'inductor' to 'aot_eager' backend. "
                                "Install Visual Studio Build Tools (with C++ build tools) or run from a Developer Command Prompt to enable inductor on Windows.")
                            selected_backend = "aot_eager"
                except Exception:
                    # Defensive: if anything goes wrong in the probe, continue and let
                    # torch.compile surface errors (we'll handle its exceptions below).
                    logger.debug("Failed to probe MSVC availability; proceeding and letting torch.compile handle missing compiler errors.")

        try:
            # Ensure scalar outputs (e.g., Tensor.item()) are captured when compiling
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.capture_scalar_outputs = True
                logger.info("Enabled torch._dynamo.config.capture_scalar_outputs=True to include scalar outputs in captured graphs")
            except Exception:
                logger.debug("torch._dynamo not available or config couldn't be set")

            model = torch.compile(model, backend=selected_backend, mode=mode)
            logger.info(f"Model compiled with backend={selected_backend}, mode={mode}")
            compile_attempted = True
        except Exception as e:
            logger.warning(f"torch.compile failed (backend={selected_backend}, mode={mode}): {e}")

        if not compile_attempted:
            logger.info("Continuing without torch.compile (performance will be unoptimized)")

    # Hetero4 sliding window training: delegate to helper
    if window_defs is not None:
        _train_windowed(model, graph, args, window_defs, optimizer, scheduler, model_path, csv_writer, csv_file, logger, device, amp_enabled, scaler, start_epoch=start_epoch, criterion=criterion)
        return
    
    # Legacy training: delegate to helper
    _train_legacy(model, graph, args, loader, use_neighbor_sampling, fanouts, optimizer, scheduler, model_path, csv_writer, csv_file, logger, device, amp_enabled, scaler, start_epoch=start_epoch, criterion=criterion)
    return