from src.config import get_args
import numpy as np
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
import os

def main():
    args = get_args()    
    verbosity = getattr(args, "verbosity", 0)
    logger = setup_logging(verbosity)
    logger.info("Starting run with args: %s", vars(args))
    ensure_dir(args.graph_dir)
    ensure_dir(args.model_dir)
    
    import torch  # lazy import for quicker help

    graph = None
    graph_loaded = False
    data_windows = None
    first_graph = None

    if args.load_graph:
        graph_filename = args.load_graph if args.load_graph.endswith(".pt") else f"{args.load_graph}.pt"
        graph_path = os.path.join(args.graph_dir, graph_filename)
        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}...")
            graph = torch.load(graph_path, weights_only=False)
            graph_loaded = True
            try:
                if getattr(graph, "norm_stats", None):
                    setattr(args, "norm_stats", graph.norm_stats)
            except Exception:
                pass
            print_graph_stats(graph)
        else:
            print(f"Graph file not found at {graph_path}. Building from sliding windows...")

    df = None
    norm_stats = None
    window_splits = None
    
    if graph is None:
        print(f"Loading data...")
        df, norm_stats, window_splits = load_data(
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
        )
        setattr(args, "norm_stats", norm_stats)

    def build_graph_from_df(dataframe, train_idx, val_idx, test_idx, norm_stats):

        if args.graph_type == "base":
            graph = BaseGraph(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "hetero1":
            graph = HeteroGraph1(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "hetero2":
            graph = HeteroGraph2(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "hetero3":
            graph = HeteroGraph3(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "hetero4":
            graph = HeteroGraph4(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "not_very_hetero":
            graph = NotVeryHetero(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "homo":
            graph = HomoGraph(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        elif args.graph_type == "hetero2nodes":
            graph = Hetero2Nodes(dataframe, args, train_idx, val_idx, test_idx, norm_stats).build()
        else:
            raise ValueError(f"Unsupported graph type: {args.graph_type}")
        return graph

    if graph is None and df is not None:
        # Get split indices
        train_idx = np.where(df["split"] == "train")[0]
        val_idx = np.where(df["split"] == "val")[0]
        test_idx = np.where(df["split"] == "test")[0]
        
        first_graph = build_graph_from_df(df, train_idx, val_idx, test_idx, norm_stats)
        print_graph_stats(first_graph)
        in_channels_dict = { nodeType: first_graph[nodeType].x.size(1) for nodeType in first_graph.metadata()[0]}
    else:
        first_graph = graph
        in_channels_dict = { nodeType: first_graph[nodeType].x.size(1) for nodeType in first_graph.metadata()[0]}

    if args.save_graph and not graph_loaded and first_graph is not None:
        graph_filename = args.save_graph if args.save_graph.endswith(".pt") else f"{args.save_graph}.pt"
        graph_path = os.path.join(args.graph_dir, graph_filename)
        print(f"Saving graph to {graph_path}...")
        torch.save(first_graph, graph_path)
        print("Graph saved successfully.")
    
    print("checking for GPUs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_available_memory()
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))

    metadata = first_graph.metadata()  # metadata[0] are node types, metadata[1] are edge types
    print(f"Node feature sizes: {in_channels_dict}")

    if args.model_type == "none":
        print("Not using any model, this is just for testing the surrounding code and graph building")
        return
    elif args.model_type == "dummy":
        print("Using model: dummy")
        out_channels = 1
        model = DummyModel(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
        ).to(device)
    elif args.model_type == "heterosage":
        print("Using model: heterosage")
        out_channels = 1
        model = HeteroSAGE(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif args.model_type == "rgcn":
        print("Using model: rgcn")
        out_channels = 1
        print("out_channels:", out_channels)
        model = RGCN(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif args.model_type == "leakyrgcn":
        print("Using model: leakyrgcn")
        out_channels = 1
        model = LeakyRGCN(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=128,
            out_channels=out_channels,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif args.model_type == "hgt":
        print("Using model: hgt")
        out_channels = 1
        model = HGT(
            metadata=metadata,
            in_channels_dict = in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
            num_layers=2,
            num_heads=2,
            dropout=0.2
        ).to(device)
    else:
        raise ValueError("Unsupported model type.")

    model_basename = getattr(args, "model_file", None) or args.model_type
    model_filename = model_basename if model_basename.endswith(".pt") else f"{model_basename}.pt"
    model_path = os.path.join(args.model_dir, model_filename)

    if args.mode == "train":
        if window_splits is None or len(window_splits) == 0:
            # No sliding windows, just train on full graph
            working_graph = first_graph
            if not use_neighbor_sampling:
                working_graph = move_graph_to_device(working_graph, device)
            train(model, working_graph, args, window_defs=None)
        else:
            # Sliding window training with learn/pred splits
            working_graph = first_graph
            if not use_neighbor_sampling:
                working_graph = move_graph_to_device(working_graph, device)
                print("Graph moved to device for sliding window training")
            
            print(f"\nPreparing {len(window_splits)} windows for training")
            
            # Prepare window definitions with ARR_DELAY masking for pred windows
            window_defs = []
            for w in window_splits:
                # Create masks as boolean arrays
                learn_mask = np.zeros(len(df), dtype=bool)
                pred_mask = np.zeros(len(df), dtype=bool)
                learn_mask[w['learn_indices']] = True
                pred_mask[w['pred_indices']] = True
                
                window_defs.append({
                    'window_id': w['window_id'],
                    'learn_mask': learn_mask,
                    'pred_mask': pred_mask,
                    'learn_indices': w['learn_indices'],
                    'pred_indices': w['pred_indices'],
                    'learn_count': w['learn_count'],
                    'pred_count': w['pred_count'],
                })
            
            # Train with sliding windows
            train(model, working_graph, args, window_defs=window_defs)
            
            # After training, evaluate on val/test
            if working_graph["flight"].val_mask.sum() > 0:
                print("\n=== Validation on held-out validation data ===")
                prev_mode = args.mode
                args.mode = "val"
                test(model, working_graph, args, window_id=0, total_windows=1)
                args.mode = prev_mode
            
            if working_graph["flight"].test_mask.sum() > 0:
                print("\n=== Testing on held-out test data ===")
                prev_mode = args.mode
                args.mode = "test"
                test(model, working_graph, args, window_id=0, total_windows=1)
                args.mode = prev_mode

        print(f"Saving final model to {model_path}...")
        save_obj = {"model_state_dict": model.state_dict()}
        try:
            norm_stats = getattr(args, "norm_stats", None)
            if norm_stats:
                save_obj["norm_stats"] = norm_stats
        except Exception:
            pass
        torch.save(save_obj, model_path)
    elif args.mode in ("test", "val"):
        # For test/val mode, use the full graph we already built
        if first_graph is not None:
            graph = first_graph
            if not use_neighbor_sampling:
                graph = move_graph_to_device(graph, device)
        elif graph_loaded and graph is not None:
            if not use_neighbor_sampling:
                graph = move_graph_to_device(graph, device)
        else:
            raise RuntimeError("No graph available for evaluation")

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            loaded = torch.load(model_path)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
                if "norm_stats" in loaded:
                    try:
                        setattr(args, "norm_stats", loaded["norm_stats"])
                        print("Restored normalization stats from model file.")
                    except Exception:
                        pass
            else:
                model.load_state_dict(loaded)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        test(model, graph, args, window_id=0, total_windows=1)

if __name__ == "__main__":
    main()