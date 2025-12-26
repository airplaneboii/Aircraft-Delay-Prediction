import os
import torch
from src.config import get_args
from src.graph.base import BaseGraph
from src.graph.hetero1 import HeteroGraph1
from src.graph.hetero2 import HeteroGraph2
from src.graph.hetero3 import HeteroGraph3
from src.graph.hetero4 import HeteroGraph4
from src.graph.not_very_hetero import NotVeryHetero
from src.graph.homo import HomoGraph
from src.graph.hetero2nodes import Hetero2Nodes
from src.models.dummy import DummyModel
from src.models.heterosage import HeteroSAGE
from src.models.hgt import HGT
from src.models.rgcn import RGCN
from src.models.leakyrgcn import LeakyRGCN
from src.train import train
from src.test import test
from src.utils import setup_logging, ensure_dir, move_graph_to_device, print_graph_stats, print_available_memory
from data.data_loader import load_data

GRAPH_BUILDERS = {
    "base": BaseGraph,
    "hetero1": HeteroGraph1,
    "hetero2": HeteroGraph2,
    "hetero3": HeteroGraph3,
    "hetero4": HeteroGraph4,
    "not_very_hetero": NotVeryHetero,
    "homo": HomoGraph,
    "hetero2nodes": Hetero2Nodes,
}


def build_graph(args, dataframe, train_idx, val_idx, test_idx, norm_stats):
    builder_cls = GRAPH_BUILDERS.get(args.graph_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported graph type: {args.graph_type}")
    return builder_cls(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()


def build_model(args, metadata, in_channels_dict):
    out_channels = 1
    mt = args.model_type
    if mt == "none":
        return None
    if mt == "dummy":
        return DummyModel(metadata=metadata, in_channels_dict=in_channels_dict, hidden_channels=64, out_channels=out_channels).to(args.device)
    if mt == "heterosage":
        return HeteroSAGE(metadata=metadata, in_channels_dict=in_channels_dict, hidden_channels=64, out_channels=out_channels, num_layers=2, dropout=0.2).to(args.device)
    if mt == "rgcn":
        return RGCN(metadata=metadata, in_channels_dict=in_channels_dict, hidden_channels=64, out_channels=out_channels, num_layers=2, dropout=0.2).to(args.device)
    if mt == "leakyrgcn":
        return LeakyRGCN(metadata=metadata, in_channels_dict=in_channels_dict, hidden_channels=128, out_channels=out_channels, num_layers=2, dropout=0.2).to(args.device)
    if mt == "hgt":
        return HGT(metadata=metadata, in_channels_dict=in_channels_dict, hidden_channels=64, out_channels=out_channels, num_layers=2, num_heads=2, dropout=0.2).to(args.device)
    raise ValueError("Unsupported model type.")


def prepare_window_defs(raw_windows, n_nodes):
    import numpy as np
    if not raw_windows or n_nodes is None or n_nodes == 0:
        return []
    defs = []
    for w in raw_windows:
        learn_mask = np.zeros(n_nodes, dtype=bool)
        pred_mask = np.zeros(n_nodes, dtype=bool)
        learn_mask[w['learn_indices']] = True
        pred_mask[w['pred_indices']] = True
        defs.append({
            'window_id': w['window_id'],
            'learn_mask': learn_mask,
            'pred_mask': pred_mask,
            'learn_indices': w['learn_indices'],
            'pred_indices': w['pred_indices'],
            'learn_count': w.get('learn_count', int(len(w['learn_indices']))),
            'pred_count': w.get('pred_count', int(len(w['pred_indices']))),
        })
    return defs

def resolve_n_nodes(first_graph, df):
    if df is not None:
        return len(df)
    try:
        return int(first_graph['flight'].x.size(0))
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
        graph_filename = args.load_graph if args.load_graph.endswith(".pt") else f"{args.load_graph}.pt"
        graph_path = os.path.join(args.graph_dir, graph_filename)
        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}...")
            graph = torch.load(graph_path, weights_only=False)
            graph_loaded = True
            # Restore window_splits and norm_stats if saved with graph
            try:
                if hasattr(graph, "window_splits"):
                    window_splits = graph.window_splits
                    print(f"Loaded window splits from graph (train: {len(window_splits.get('train', []))}, "
                          f"val: {len(window_splits.get('val', []))}, test: {len(window_splits.get('test', []))})")
                if hasattr(graph, "norm_stats"):
                    setattr(args, "norm_stats", graph.norm_stats)
                    print("Loaded normalization stats from graph")
            except Exception as e:
                print(f"Warning: Could not restore window_splits/norm_stats from graph: {e}")
            print_graph_stats(graph)
        else:
            print(f"Graph file not found at {graph_path}. Building from sliding windows...")

    if graph is None:
        print(f"Loading data...")
        df, split_indices, norm_stats, loaded_windows = load_data(
            args.data_path,
            mode=args.mode,
            task_type=args.prediction_type,
            max_rows=getattr(args, "rows", None),
            split_ratios=tuple(x/100 for x in args.data_split),
            normalize_cols_file=("data/normalize.txt" if getattr(args, "normalize", True) else None),
            unit=getattr(args, "unit", 60),
            learn_window=getattr(args, "learn_window", 10),
            pred_window=getattr(args, "pred_window", 1),
            window_stride=getattr(args, "window_stride", 1),
            normalize=getattr(args, "normalize", True),
            use_sliding_windows=getattr(args, "use_sliding_windows", False),
        )
        # Only set window_splits if not already restored from graph and if loader provided windows
        if window_splits is None and loaded_windows is not None:
            window_splits = loaded_windows
        setattr(args, "norm_stats", norm_stats)

    if graph is None and df is not None:
        # Get split indices (loader now returns chronological split index arrays)
        train_idx = split_indices['train_idx']
        val_idx = split_indices['val_idx']
        test_idx = split_indices['test_idx']

        first_graph = build_graph(args, df, train_idx, val_idx, test_idx, norm_stats)
        print_graph_stats(first_graph)
        in_channels_dict = { nodeType: first_graph[nodeType].x.size(1) for nodeType in first_graph.metadata()[0]}
    else:
        first_graph = graph
        in_channels_dict = { nodeType: first_graph[nodeType].x.size(1) for nodeType in first_graph.metadata()[0]}

    norm_stats = getattr(args, "norm_stats", None)
    n_nodes = resolve_n_nodes(first_graph, df)

    if args.save_graph and not graph_loaded and first_graph is not None:
        graph_filename = args.save_graph if args.save_graph.endswith(".pt") else f"{args.save_graph}.pt"
        graph_path = os.path.join(args.graph_dir, graph_filename)
        print(f"Saving graph to {graph_path}...")
        # Attach window_splits and norm_stats to graph for persistence
        first_graph.window_splits = window_splits
        first_graph.norm_stats = norm_stats
        torch.save(first_graph, graph_path)
        print("Graph saved successfully (with windows and norm_stats).")
    
    #####################################################################################0
    

    print("checking for GPUs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Using device: {device}")
    print_available_memory()
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    metadata = first_graph.metadata()  # metadata[0] are node types, metadata[1] are edge types
    print(f"Node feature sizes: {in_channels_dict}")


    ################################## MODEL AND MODE SELECTION ################################################

    if args.model_type == "none":
        print("Not using any model, this is just for testing the surrounding code and graph building")
        return
    model = build_model(args, metadata, in_channels_dict)

    model_basename = getattr(args, "model_file", None) or args.model_type
    model_filename = model_basename if model_basename.endswith(".pt") else f"{model_basename}.pt"
    model_path = os.path.join(args.model_dir, model_filename)

    if args.mode == "train":
        # Get train windows and prepare window_defs
        raw_train_windows = window_splits.get('train', []) if isinstance(window_splits, dict) else (window_splits if window_splits else [])
        window_defs = prepare_window_defs(raw_train_windows, n_nodes)

        working_graph = first_graph
        if not use_neighbor_sampling:
            working_graph = move_graph_to_device(working_graph, device)

        if not window_defs:
            # No sliding windows, train on full graph
            train(model, working_graph, args, window_defs=None)
        else:
            print(f"Preparing {len(window_defs)} training windows")
            train(model, working_graph, args, window_defs=window_defs)

        print(f"Saving final model to {model_path}...")
        save_model_state(model, args, model_path)
        
    elif args.mode in ("test", "val"):
        # Load model for evaluation
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            loaded = torch.load(model_path)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
                if "norm_stats" in loaded:
                    setattr(args, "norm_stats", loaded["norm_stats"])
                    print("Restored normalization stats from model file.")
            else:
                model.load_state_dict(loaded)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Use the graph we already built
        if first_graph is not None:
            graph = first_graph
            if not use_neighbor_sampling:
                graph = move_graph_to_device(graph, device)
        elif graph_loaded and graph is not None:
            if not use_neighbor_sampling:
                graph = move_graph_to_device(graph, device)
        else:
            raise RuntimeError("No graph available for evaluation")
        
        # Get windows for the appropriate split and prepare defs
        raw_eval_windows = window_splits.get(args.mode, []) if isinstance(window_splits, dict) else []
        eval_window_defs = prepare_window_defs(raw_eval_windows, n_nodes)

        if not eval_window_defs:
            # No windows, evaluate on full split
            print(f"Evaluating on full {args.mode} split (no windows)")
            test(model, graph, args, window_defs=None)
        else:
            print(f"Evaluating on {len(eval_window_defs)} {args.mode} windows")
            test(model, graph, args, window_defs=eval_window_defs)

if __name__ == "__main__":
    main()