from src.config import get_args
from src.graph.base import BaseGraph
from src.models.dummymodel import DummyModel
from src.models.heterosage import HeteroSAGE
from src.models.hgtmodel import HGT
from src.models.rgcnmodel import RGCN
from src.train import train
from src.test import test
from src.utils import ensure_dir, move_graph_to_device, print_graph_stats
from data.data_loader import load_data
import torch
import os

def main():
    args = get_args()    
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
        set_train, set_val, set_test = load_data(args.data_path, mode=args.mode, task_type=args.prediction_type, development=args.development)
        print("Building graph...")
        data = set_train if args.mode == "train" else (set_val if args.mode == "val" else set_test)
        if args.graph_type == "base":
            graph = BaseGraph(data, args).build()
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

    graph = move_graph_to_device(graph, device)
    metadata = graph.metadata() #metadata[0] are node types, metadata[1] are edge types
    in_channels_dict = { nodeType: graph[nodeType].x.size(1) for nodeType in metadata[0]}

    # Select model
    if args.model_type == "none":
        print("Not using any model, this is just for testing the surrounding code and graph building")
        return
    elif args.model_type == "dummymodel":
        print("Using model: dummy")
        out_channels = 2 if args.prediction_type == "classification" else 1
        model = DummyModel(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
        ).to(device)
    elif args.model_type == "heterosage":
        print("Using model: heterosage")
        out_channels = 2 if args.prediction_type == "classification" else 1
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
        out_channels = 2 if args.prediction_type == "classification" else 1
        model = RGCN(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            out_channels=out_channels,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif args.model_type == "hgtmodel":
        print("Using model: hgt")
        out_channels = 2 if args.prediction_type == "classification" else 1
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

    # Run mode
    if args.mode == "develop":
        print(graph)
    elif args.mode == "train":
        train(model, graph, args)
        torch.save(model.state_dict(), f"{args.model_dir}/{args.model_type}.pt")
    elif args.mode == "test" or args.mode == "val":
        model.load_state_dict(torch.load(f"{args.model_dir}/{args.model_type}.pt"))
        test(model, graph, args)

if __name__ == "__main__":
    main()