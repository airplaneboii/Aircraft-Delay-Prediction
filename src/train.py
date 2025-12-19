#from asyncio import graph
import torch
import time
from datetime import datetime
from torch import nn, optim
from src.utils import compute_epoch_stats, resolve_fanouts
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

def train(
        model: nn.Module,
        graph,
        args
    ) -> None:
    print(args)
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
        
        y = graph["flight"].y[graph["flight"].train_mask].view(-1)
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
    csv_path = os.path.join(args.model_dir, f"{args.model_file}_training_stats.csv")

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

    # Optional: compile the model for speed (PyTorch 2.x)
    if bool(getattr(args, "compile", False)):
        backend = getattr(args, "compile_backend", "inductor")
        mode = getattr(args, "compile_mode", "default")
        try:
            model = torch.compile(model, backend=backend, mode=mode)
            logger.info(f"Model compiled with backend={backend}, mode={mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed (backend={backend}, mode={mode}): {e}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_losses = []
        all_labels = []
        all_preds = []

        logger.debug("epoch %d: use_neighbor_sampling=%s, fanouts=%s, batch_size=%s", epoch+1, use_neighbor_sampling, fanouts, args.batch_size)

        if use_neighbor_sampling:
            # Wrap loader with tqdm for per-epoch progress
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            for batch_idx, batch in enumerate(tqdm(loader, total=total_batches, desc=f"Epoch {epoch+1}")):
                logger.debug("epoch %d batch %d: loaded, to(device) starting", epoch+1, batch_idx)
                batch = batch.to(device)
                logger.debug("epoch %d batch %d: to(device) done", epoch+1, batch_idx)
                optimizer.zero_grad()

                logger.debug("epoch %d batch %d: forward start", epoch+1, batch_idx)
                
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    out = model(batch.x_dict, batch.edge_index_dict)
                flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))
                
                logger.debug("epoch %d batch %d: forward done, out shape=%s", epoch+1, batch_idx, tuple(out.shape))
                if args.prediction_type == "regression":
                    labels = batch["flight"].y.squeeze(-1)[:flight_batch_size].to(device)
                    preds = out.squeeze(-1)[:flight_batch_size]
                    loss = criterion(preds, labels)
                    preds_for_metrics = preds.detach().cpu()
                else:
                    # Classification: use only input (seed) flight nodes in this batch
                    labels_full = batch["flight"].y.view(-1).float()[:flight_batch_size].to(device)
                    logits_full = out[:flight_batch_size]

                    # Keep a common variable name for metrics aggregation
                    labels = labels_full

                    # Oversample positives within the batch to address imbalance
                    pos_mask = labels_full == 1
                    neg_mask = labels_full == 0
                    pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
                    neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)

                    oversample_factor = int(OVERSAMPLE_FACTOR)
                    pos_indices_os = pos_indices.repeat(oversample_factor) if pos_indices.numel() > 0 else pos_indices
                    train_indices_os = torch.cat([neg_indices, pos_indices_os]) if neg_indices.numel() > 0 else pos_indices_os

                    # Fallback: if batch has only one class, avoid empty selection
                    if train_indices_os.numel() == 0:
                        train_indices_os = torch.arange(labels_full.size(0), device=labels_full.device)

                    labels_os = labels_full[train_indices_os]
                    logits_os = logits_full[train_indices_os]

                    loss = criterion(logits_os, labels_os)

                    # Metrics based on original batch seed nodes
                    probs = torch.sigmoid(logits_full)
                    preds_for_metrics = (probs > args.border).long().cpu()

                logger.debug("epoch %d batch %d: loss computed %.6f", epoch+1, batch_idx, loss.item())
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    logger.debug("epoch %d batch %d: backward done (AMP)", epoch+1, batch_idx)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    logger.debug("epoch %d batch %d: backward done", epoch+1, batch_idx)
                    optimizer.step()
                logger.debug("epoch %d batch %d: optimizer step done", epoch+1, batch_idx)

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)
        else:
            optimizer.zero_grad()
            logger.debug("epoch %d: full-batch forward start", epoch+1)
            
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(graph.x_dict, graph.edge_index_dict)
            
            logger.debug("epoch %d: full-batch forward done, out shape=%s", epoch+1, tuple(out.shape))

        if not use_neighbor_sampling:
            if args.prediction_type == "regression":
                # do prediction only on training nodes
                mask = graph["flight"].train_mask
                labels = graph["flight"].y.squeeze(-1)[mask].to(device)
                preds = out.squeeze(-1)[mask]
                loss = criterion(preds, labels)
                preds_for_metrics = preds.detach().cpu()
                
                logger.debug("epoch %d: loss computed %.6f", epoch+1, loss.item())
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    logger.debug("epoch %d: backward done (AMP)", epoch+1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    logger.debug("epoch %d: backward done", epoch+1)
                    optimizer.step()
                logger.debug("epoch %d: optimizer step done", epoch+1)
                
                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)
            else:
                mask = graph["flight"].train_mask
                labels = graph["flight"].y.view(-1).float()[mask].to(device)
                logits = out[mask].squeeze(-1)

                #print("logits:", logits[0:10].detach().cpu().numpy())
                #print("labels:", labels[0:10].detach().cpu().numpy())

                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                preds_for_metrics = (probs > args.border).long().cpu()

                logger.debug("epoch %d: loss computed %.6f", epoch+1, loss.item())
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    logger.debug("epoch %d: backward done (AMP)", epoch+1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    logger.debug("epoch %d: backward done", epoch+1)
                    optimizer.step()
                logger.debug("epoch %d: optimizer step done", epoch+1)

                epoch_losses.append(loss.item())
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_for_metrics)

        labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([])
        preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])

        # Compute metrics and log via shared helper
        epoch_stats = compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start, logger)

        # Write epoch stats to CSV using the returned dictionary
        if csv_writer is not None and epoch_stats is not None:
            csv_row = {key: epoch_stats.get(key) for key in csv_headers}
            csv_writer.writerow(csv_row)
            csv_file.flush()  # Ensure data is written immediately

        # Save checkpoint after each epoch
        logging.debug(f"Saving checkpoint for epoch {epoch+1}/{args.epochs}")
        try: 
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "norm_stats": getattr(args, "norm_stats", None),
            }
            torch.save(ckpt, model_path)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for epoch {epoch+1}: {e}")
    # Close CSV file
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Training statistics saved to: {csv_path}")


    logger.info("Training completed. Final checkpoint saved to %s", model_path)

    overall_end = time.time()
    end_dt = datetime.now()
    total_time = overall_end - overall_start
    logger.info("Training end: %s", end_dt.isoformat())
    logger.info("Total training time: %.2f s (%.2f minutes)", total_time, total_time/60)
    if args.epochs > 0:
        logger.info("Average epoch time: %.2f s", total_time/args.epochs)