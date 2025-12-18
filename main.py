from src.config import get_args
from src.graph.base import BaseGraph
from src.graph.heteroNew import HeteroNewGraph
from src.graph.heteroNew2 import HeteroNewGraph2
from src.graph.heteroNew3 import HeteroNewGraph3
from src.graph.not_very_hetero import NotVeryHetero
from src.models.dummymodel import DummyModel
from src.models.heterosage import HeteroSAGE
from src.models.hgtmodel import HGT
from src.models.rgcnmodel import RGCN
from src.models.rgcn_norelu import RGCNNoReLU
from src.train import train
from src.test import test
from src.utils import setup_logging, ensure_dir, move_graph_to_device, print_graph_stats, print_available_memory
from data.data_loader import load_data
import torch
import os

def main():
    args = get_args()    
    # Configure logging as early as possible
    verbosity = getattr(args, "verbosity", 0)
    logger = setup_logging(verbosity)
    logger.info("Starting run with args: %s", vars(args))
    ensure_dir(args.graph_dir)
    ensure_dir(args.model_dir)
    
    # Load or build graph
    graph = None
    if args.load_graph:
        graph_path = os.path.join(args.graph_dir, args.load_graph)
        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}...")
            # weights_only=False required for PyG HeteroData objects
            graph = torch.load(graph_path, weights_only=False)
            print_graph_stats(graph)
        else:
            print(f"Graph file not found at {graph_path}. Building from scratch...")
    
    # Build graph if not loaded
    if graph is None:
        print("Loading data...")
        df, train_index, val_index, test_index, norm_stats = load_data(
            args.data_path,
            mode=args.mode,
            task_type=args.prediction_type,
            max_rows=getattr(args, "rows", None),
            normalize_cols_file="data/normalize.txt",
        )
        # attach normalization stats to args for downstream use/printing
        setattr(args, "norm_stats", norm_stats)
        print("Building graph...")
        if args.graph_type == "base":
            graph = BaseGraph(df, args, train_index, val_index, test_index, norm_stats).build()
            print_graph_stats(graph)
        elif args.graph_type == "heteroNew":
            graph = HeteroNewGraph(df, args, train_index, val_index, test_index, norm_stats).build()
            print_graph_stats(graph)
        elif args.graph_type == "heteroNew2":
            graph = HeteroNewGraph2(df, args, train_index, val_index, test_index, norm_stats).build()
            print_graph_stats(graph)
        elif args.graph_type == "heteroNew3":
            graph = HeteroNewGraph3(df, args, train_index, val_index, test_index, norm_stats).build()
            print_graph_stats(graph)
        elif args.graph_type == "not_very_hetero":
            graph = NotVeryHetero(df, args, train_index, val_index, test_index, norm_stats).build()
            print_graph_stats(graph)
        else:
            print(args.graph_type)
            raise ValueError("Unsupported graph type.")
    
    # Save graph if requested
    if args.save_graph:
        graph_path = os.path.join(args.graph_dir, args.save_graph)
        print(f"Saving graph to {graph_path}...")
        torch.save(graph, graph_path)
        print("Graph saved successfully.")
    
    # Move to GPU if available
    print("checking for GPUs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Display available memory before graph building
    print_available_memory()
    use_neighbor_sampling = bool(getattr(args, "neighbor_sampling", False))
    if use_neighbor_sampling and args.mode == "train":
        print("Neighbor sampling enabled: keeping full graph on CPU; mini-batches will be moved per step.")
    else:
        graph = move_graph_to_device(graph, device)
        print("Graph moved to device.")
        
    metadata = graph.metadata() #metadata[0] are node types, metadata[1] are edge types
    in_channels_dict = { nodeType: graph[nodeType].x.size(1) for nodeType in metadata[0]}
    print(f"Node feature sizes: {in_channels_dict}")

    # Select model
    if args.model_type == "none":
        print("Not using any model, this is just for testing the surrounding code and graph building")
        return
    elif args.model_type == "dummymodel":
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
    elif args.model_type == "rgcnmodel":
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
    elif args.model_type == "rgcn_norelu":
        print("Using model: rgcn_norelu")
        out_channels = 1
        model = RGCNNoReLU(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=128,
            out_channels=out_channels,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif args.model_type == "hgtmodel":
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

    # Determine model file path (allow user-provided filename without extension)
    model_basename = getattr(args, "model_file", None) or args.model_type
    model_filename = model_basename if model_basename.endswith(".pt") else f"{model_basename}.pt"
    model_path = os.path.join(args.model_dir, model_filename)

    if args.mode == "train":
        train(model, graph, args)
        print(f"Saving model to {model_path}...")
        # Save model state plus normalization stats (if present) for reproducible inference
        save_obj = {"model_state_dict": model.state_dict()}
        try:
            norm_stats = getattr(args, "norm_stats", None)
            if norm_stats:
                save_obj["norm_stats"] = norm_stats
        except Exception:
            pass
        torch.save(save_obj, model_path)
    elif args.mode == "test" or args.mode == "val":
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            loaded = torch.load(model_path)
            # Backwards compatible: file may contain only a state_dict
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
                # restore normalization stats if saved with the model
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
        test(model, graph, args)

if __name__ == "__main__":
    main()