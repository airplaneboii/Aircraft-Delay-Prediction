import torch
import time
from datetime import datetime
from torch import nn, optim
from src.utils import compute_epoch_stats
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import logging
import os
from tqdm.auto import tqdm

try:
    import psutil
except Exception:
    psutil = None

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
        criterion = nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    # Resolve neighbor fanouts (None => full neighbors)
    fanouts = getattr(args, "neighbor_fanouts", None)
    depth = getattr(model, "num_layers", None)
    if depth is None or depth <= 0:
        depth = len(fanouts) if fanouts else 1

    if fanouts is None:
        fanouts = [-1] * depth
    elif len(fanouts) != depth and len(fanouts) > 0:
        if len(fanouts) < depth:
            fanouts = fanouts + [fanouts[-1]] * (depth - len(fanouts))
        else:
            fanouts = fanouts[:depth]

    # Detect if graph is heterogeneous or homogeneous
    is_hetero = isinstance(graph, HeteroData)
    
    if use_neighbor_sampling:
        if is_hetero:
            train_nodes = graph["flight"].train_mask.nonzero(as_tuple=False).view(-1)
            input_nodes = ("flight", train_nodes)
        else:
            train_nodes = graph.train_mask.nonzero(as_tuple=False).view(-1)
            input_nodes = train_nodes
        
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

    # resume if checkpoint exists (backward-compatible: will skip if file is not a full checkpoint)
    start_epoch = 0
    if os.path.exists(model_path):
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

    # Training loop
    try:
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
                    
                    if is_hetero:
                        out = model(batch.x_dict, batch.edge_index_dict)
                        flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))
                    else:
                        out = model(batch.x, batch.edge_index)
                        flight_batch_size = batch.x.size(0)
                    
                    logger.debug("epoch %d batch %d: forward done, out shape=%s", epoch+1, batch_idx, tuple(out.shape))

                    if args.prediction_type == "regression":
                        if is_hetero:
                            labels = batch["flight"].y.squeeze(-1)[:flight_batch_size].to(device)
                        else:
                            labels = batch.y.squeeze(-1)[:flight_batch_size].to(device)
                        preds = out.squeeze(-1)[:flight_batch_size]
                        loss = criterion(preds, labels)
                        preds_for_metrics = preds.detach().cpu()
                    else:
                        if is_hetero:
                            labels = batch["flight"].y.view(-1).long()[:flight_batch_size].to(device)
                        else:
                            labels = batch.y.view(-1).long()[:flight_batch_size].to(device)
                        logits = out[:flight_batch_size]
                        loss = criterion(logits, labels)
                        preds_for_metrics = torch.argmax(logits.detach().cpu(), dim=1)

                    logger.debug("epoch %d batch %d: loss computed %.6f", epoch+1, batch_idx, loss.item())
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
                
                if is_hetero:
                    out = model(graph.x_dict, graph.edge_index_dict)
                else:
                    out = model(graph.x, graph.edge_index)
                
                logger.debug("epoch %d: full-batch forward done, out shape=%s", epoch+1, tuple(out.shape))

                if args.prediction_type == "regression":
                    if is_hetero:
                        mask = graph["flight"].train_mask
                        labels = graph["flight"].y.squeeze(-1)[mask].to(device)
                    else:
                        mask = graph.train_mask
                        labels = graph.y.squeeze(-1)[mask].to(device)
                    preds = out.squeeze(-1)[mask]
                    loss = criterion(preds, labels)
                    preds_for_metrics = preds.detach().cpu()
                else:
                    if is_hetero:
                        mask = graph["flight"].train_mask
                        labels = graph["flight"].y.view(-1).long()[mask].to(device)
                    else:
                        mask = graph.train_mask
                        labels = graph.y.view(-1).long()[mask].to(device)
                    logits = out[mask]
                    loss = criterion(logits, labels)
                    preds_for_metrics = torch.argmax(logits.detach().cpu(), dim=1)

                logger.debug("epoch %d: loss computed %.6f", epoch+1, loss.item())
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
            compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start, logger)
    except KeyboardInterrupt:
        # Save partial model state when training interrupted by user
        print("Interrupted — saving checkpoint...")
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "norm_stats": getattr(args, "norm_stats", None),
        }
        torch.save(ckpt, model_path)
        raise

    # Save final checkpoint after training completes
    try:
        final_ckpt = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "norm_stats": getattr(args, "norm_stats", None),
        }
        torch.save(final_ckpt, model_path)
        logger.info("Saved final checkpoint to %s", model_path)
    except Exception as e:
        logger.warning("Failed to save final checkpoint: %s", e)

    overall_end = time.time()
    end_dt = datetime.now()
    total_time = overall_end - overall_start
    logger.info("Training end: %s", end_dt.isoformat())
    logger.info("Total training time: %.2f s (%.2f minutes)", total_time, total_time/60)
    if args.epochs > 0:
        logger.info("Average epoch time: %.2f s", total_time/args.epochs)