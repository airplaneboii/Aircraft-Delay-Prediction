import os

import numpy as np
import torch

from src.config import get_args
from src.data_loader import (
    compute_split_and_normalize,
    compute_splits_from_graph,
    compute_windows_from_graph,
    load_data,
)
from src.graph.hetero3 import HeteroGraph3
from src.graph.hetero5 import HeteroGraph5
from src.models.dummy import DummyModel
from src.models.hgt import HGT
from src.models.rgcn import RGCN
from src.test import test
from src.train import train
from src.utils import (
    ensure_dir,
    move_graph_to_device,
    print_available_memory,
    print_graph_stats,
    setup_logging,
)

GRAPH_BUILDERS = {
    "hetero3": HeteroGraph3,
    "hetero5": HeteroGraph5,
}


def build_graph(args, dataframe, train_idx, val_idx, test_idx, norm_stats):
    builder_cls = GRAPH_BUILDERS.get(args.graph_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported graph type: {args.graph_type}")
    return builder_cls(
        dataframe, args, train_idx, val_idx, test_idx, norm_stats
    ).build()


def build_model(args, metadata, in_channels_dict):
    out_channels = 1
    mt = args.model_type
    # Hyperparameters from args (config overrides via YAML are applied in get_args())
    hc = getattr(args, "hidden_channels", 64)
    nl = getattr(args, "num_layers", 2)
    do = getattr(args, "dropout", 0.2)

    if mt == "none":
        return None
    if mt == "dummy":
        return DummyModel(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=hc,
            out_channels=out_channels,
        ).to(args.device)
    if mt == "rgcn":
        return RGCN(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=hc,
            out_channels=out_channels,
            num_layers=nl,
            dropout=do,
        ).to(args.device)
    if mt == "hgt":
        return HGT(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=hc,
            out_channels=out_channels,
            num_layers=nl,
            num_heads=2,
            dropout=do,
        ).to(args.device)
    raise ValueError("Unsupported model type.")


def prepare_window_defs(raw_windows, n_nodes):
    """Pass-through for window definitions (indices already compact int32 from compute_windows_from_graph).

    This function is kept for backward compatibility but no longer performs conversions.
    compute_windows_from_graph already produces memory-efficient int32 indices.
    """
    if not raw_windows:
        return []
    # Windows are already optimized, return as-is
    return raw_windows


def resolve_n_nodes(first_graph, df):
    if df is not None:
        return len(df)
    try:
        return int(first_graph["flight"].x.size(0))
    except Exception:
        return None


def save_model_state(model, args, model_path):
    save_obj = {"model_state_dict": model.state_dict()}
    try:
        ns = getattr(args, "norm_stats", None)
        if ns:
            save_obj["norm_stats"] = ns
    except Exception:
        pass
    torch.save(save_obj, model_path)


def main():
    # get command line arguments
    args = get_args()

    verbosity = getattr(args, "verbosity", 0)
    logger = setup_logging(verbosity)
    logger.info("Starting run with args: %s", vars(args))
    ensure_dir(args.graph_dir)
    ensure_dir(args.model_dir)

    # Lazy import torch to not slow down the help command
    # import torch
    # (nevermind it gets imported through all the other stuff)

    ####################################### GRAPH LOADING/BUILDING/SAVING ###########################################
    graph = None
    graph_loaded = False
    first_graph = None
    df = None
    window_splits = None

    if args.load_graph:
        graph_filename = (
            args.load_graph
            if args.load_graph.endswith(".pt")
            else f"{args.load_graph}.pt"
        )
        graph_path = os.path.join(args.graph_dir, graph_filename)
        if os.path.exists(graph_path):
            logger.info(f"Loading graph from {graph_path}...")
            graph = torch.load(graph_path, weights_only=False)
            graph_loaded = True
            # Restore norm_stats if saved with graph
            try:
                if hasattr(graph, "norm_stats"):
                    setattr(args, "norm_stats", graph.norm_stats)
                    logger.info("Loaded normalization stats from graph")
            except Exception as e:
                logger.warning(f"Could not restore norm_stats from graph: {e}")
            print_graph_stats(graph)
        else:
            logger.info(f"Graph file not found at {graph_path}. Building from data...")

    if graph is None:
        logger.info("Loading data...")
        df, norm_config = load_data(
            args.data_path,
            max_rows=getattr(args, "rows", None),
            normalize_cols_file=(
                "data/normalize.txt" if getattr(args, "normalize", True) else None
            ),
            normalize=getattr(args, "normalize", True),
        )

        # Compute splits and normalization on dataframe first
        split_indices, norm_stats = compute_split_and_normalize(
            df,
            split_ratios=tuple(x / 100 for x in args.data_split),
            norm_config=norm_config,
        )
        setattr(args, "norm_stats", norm_stats)

        # Build graph with temporary equal splits (will be overwritten)
        logger.info("Building graph...")
        temp_idx = np.arange(len(df))
        first_graph = build_graph(args, df, temp_idx, temp_idx, temp_idx, norm_stats)
        print_graph_stats(first_graph)
    else:
        first_graph = graph

    # Compute in_channels_dict
    in_channels_dict = {
        nodeType: first_graph[nodeType].x.size(1)
        for nodeType in first_graph.metadata()[0]
    }

    # Always compute splits from graph (whether loaded or built)
    # This stores split boundaries (2 integers) instead of full index arrays or masks
    logger.info("Computing splits from graph...")
    split_info = compute_splits_from_graph(
        first_graph,
        split_ratios=tuple(x / 100 for x in args.data_split),
    )
    logger.info(
        f"Split boundaries stored on graph: train=[0,{split_info['train_end']}), "
        f"val=[{split_info['train_end']},{split_info['val_end']}), test=[{split_info['val_end']},{split_info['n_flights']})"
    )
    logger.info(
        "Memory-efficient: only 2 integers stored instead of full masks/indices"
    )

    norm_stats = getattr(args, "norm_stats", None)
    n_nodes = resolve_n_nodes(first_graph, df)

    # Always recompute windows from graph if sliding windows are enabled
    # compute_windows_from_graph respects graph.flight.{train/val/test}_mask
    if getattr(args, "use_sliding_windows", False) and first_graph is not None:
        window_splits = compute_windows_from_graph(
            first_graph,
            unit=getattr(args, "unit", 60),
            learn_window=getattr(args, "learn_window", 10),
            pred_window=getattr(args, "pred_window", 1),
            window_stride=getattr(args, "window_stride", 1),
        )
        logger.info(
            f"Windows generated per split: train={len(window_splits['train'])}, "
            f"val={len(window_splits['val'])}, test={len(window_splits['test'])}"
        )
    elif not getattr(args, "use_sliding_windows", False):
        window_splits = None

    if args.save_graph and not graph_loaded and first_graph is not None:
        graph_filename = (
            args.save_graph
            if args.save_graph.endswith(".pt")
            else f"{args.save_graph}.pt"
        )
        graph_path = os.path.join(args.graph_dir, graph_filename)
        logger.info(f"Saving graph to {graph_path}...")
        # Only persist norm_stats; windows are recomputed on load
        first_graph.norm_stats = norm_stats
        torch.save(first_graph, graph_path)
        logger.info("Graph saved successfully (with norm_stats).")

    # Proceed to device checks

    logger.info("Checking for GPUs...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info(f"Using device: {device}")
    print_available_memory()
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    metadata = (
        first_graph.metadata()
    )  # metadata[0] are node types, metadata[1] are edge types
    logger.info(f"Node feature sizes: {in_channels_dict}")

    ################################## MODEL AND MODE SELECTION ################################################

    if args.model_type == "none":
        logger.info("Not using any model; exiting (used for testing graph building)")
        return
    model = build_model(args, metadata, in_channels_dict)

    model_basename = getattr(args, "model_file", None) or args.model_type
    model_filename = (
        model_basename if model_basename.endswith(".pt") else f"{model_basename}.pt"
    )
    model_path = os.path.join(args.model_dir, model_filename)

    if args.mode == "train":
        # Get train windows and prepare window_defs
        raw_train_windows = (
            window_splits.get("train", [])
            if isinstance(window_splits, dict)
            else (window_splits if window_splits else [])
        )
        window_defs = prepare_window_defs(raw_train_windows, n_nodes)

        working_graph = first_graph
        # When using sliding windows with GPU, keep graph on GPU for zero-copy subgraph building
        # Otherwise keep on CPU and move only subgraphs to device
        if not use_neighbor_sampling:
            if window_defs and device.type == "cuda":
                # GPU-resident mode: keep full graph on GPU
                working_graph = move_graph_to_device(working_graph, device)
                logger.info(
                    "GPU-resident windowing: graph kept on GPU for zero-copy subgraph building"
                )
            elif not window_defs:
                # No windows: move to device
                working_graph = move_graph_to_device(working_graph, device)

        if not window_defs:
            # No sliding windows, train on full graph
            train(model, working_graph, args, window_defs=None)
        else:
            logger.info(f"Preparing {len(window_defs)} training windows")
            train(model, working_graph, args, window_defs=window_defs)

        logger.info(f"Saving final model to {model_path}...")
        save_model_state(model, args, model_path)

    elif args.mode in ("test", "val"):
        # Load model for evaluation
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}...")
            loaded = torch.load(model_path)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
                if "norm_stats" in loaded:
                    setattr(args, "norm_stats", loaded["norm_stats"])
                    logger.info("Restored normalization stats from model file.")
            else:
                model.load_state_dict(loaded)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Get windows for the appropriate split and prepare defs FIRST (used to decide device move)
        raw_eval_windows = (
            window_splits.get(args.mode, []) if isinstance(window_splits, dict) else []
        )
        eval_window_defs = prepare_window_defs(raw_eval_windows, n_nodes)

        # Use the graph we already built
        if first_graph is not None:
            graph = first_graph
            # When using sliding windows, keep full graph on CPU and move only subgraphs to device
            if not use_neighbor_sampling and not eval_window_defs:
                graph = move_graph_to_device(graph, device)
        elif graph_loaded and graph is not None:
            if not use_neighbor_sampling and not eval_window_defs:
                graph = move_graph_to_device(graph, device)
        else:
            raise RuntimeError("No graph available for evaluation")

        if not eval_window_defs:
            # No windows, evaluate on full split
            logger.info(f"Evaluating on full {args.mode} split (no windows)")
            test(model, graph, args, window_defs=None)
        else:
            logger.info(f"Evaluating on {len(eval_window_defs)} {args.mode} windows")
            test(model, graph, args, window_defs=eval_window_defs)


if __name__ == "__main__":
    main()
