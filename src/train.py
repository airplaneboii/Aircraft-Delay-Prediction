import torch
import time
from datetime import datetime
from torch import nn, optim
from src.utils import regression_metrics, classification_metrics
from torch_geometric.loader import NeighborLoader
import logging
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

    if use_neighbor_sampling:
        # num_flights = graph["flight"].x.size(0)
        # input_nodes = ("flight", torch.arange(num_flights))
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

    # Training loop
    for epoch in range(args.epochs):
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
                out = model(batch.x_dict, batch.edge_index_dict)
                logger.debug("epoch %d batch %d: forward done, out shape=%s", epoch+1, batch_idx, tuple(out.shape))
                flight_batch_size = getattr(batch["flight"], "batch_size", batch["flight"].x.size(0))

                if args.prediction_type == "regression":
                    labels = batch["flight"].y.squeeze(-1)[:flight_batch_size].to(device)
                    preds = out.squeeze(-1)[:flight_batch_size]
                    loss = criterion(preds, labels)
                    preds_for_metrics = preds.detach().cpu()
                else:
                    labels = batch["flight"].y.view(-1).long()[:flight_batch_size].to(device)
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
            out = model(graph.x_dict, graph.edge_index_dict)
            logger.debug("epoch %d: full-batch forward done, out shape=%s", epoch+1, tuple(out.shape))

            if args.prediction_type == "regression":
                # do prediction only on training nodes
                mask = graph["flight"].train_mask
                labels = graph["flight"].y.squeeze(-1)[mask].to(device)
                preds = out.squeeze(-1)[mask]
                loss = criterion(preds, labels)
                preds_for_metrics = preds.detach().cpu()
            else:
                mask = graph["flight"].train_mask
                labels = graph["flight"].y.view(-1).long()[mask].to(device)
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

        epoch_time = time.time() - epoch_start

        labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([])
        preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])

        # Compute metrics for monitoring
        if args.prediction_type == "regression":
            norm_stats = getattr(graph, "norm_stats", None) or getattr(args, "norm_stats", None)
            metrics_results = regression_metrics(labels_cat, preds_cat, norm_stats)
            metrics_str = (
                f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
                f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
            )
        else:
            metrics_results = classification_metrics(labels_cat, preds_cat)
            metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

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

        info_parts = [
            f"Epoch {epoch+1}/{args.epochs}",
            f"loss: {avg_loss:.4f}",
            f"{metrics_str}",
            f"time: {epoch_time:.2f}s",
        ]
        if gpu_mem is not None:
            info_parts.append(f"gpu_mem_peak: {gpu_mem:.1f} MB")
        if cpu_info is not None:
            info_parts.append(f"proc_mem: {cpu_info[0]:.1f} MB")
            info_parts.append(f"cpu%: {cpu_info[1]:.1f}")

        logger.info(" - ".join(info_parts))

    overall_end = time.time()
    end_dt = datetime.now()
    total_time = overall_end - overall_start
    logger.info("Training end: %s", end_dt.isoformat())
    logger.info("Total training time: %.2f s (%.2f minutes)", total_time, total_time/60)
    if args.epochs > 0:
        logger.info("Average epoch time: %.2f s", total_time/args.epochs)