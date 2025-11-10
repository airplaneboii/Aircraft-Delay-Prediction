from src.config import get_args
from src.data_loader import load_data
from src.graph.base import BaseGraph
from src.models.dummymodel import DummyModel
from src.models.rgcnmodel import RGCN
from src.models.hgtmodel import HGT
from src.train import train
from src.test import test
from src.utils import ensure_dir, move_graph_to_device
import torch

def main():
    args = get_args()
    df = load_data(args.data_path, mode=args.mode, task_type=args.prediction_type, development=args.development)
    ensure_dir(args.graph_dir)
    ensure_dir(args.model_dir)

    # Build graph
    if args.graph_type == "base":
        graph = BaseGraph(df, args).build()
    else:
        print(args.graph_type)
        raise ValueError("Unsupported graph type.")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    graph = move_graph_to_device(graph, device)
    metadata = graph.metadata() #metadata[0] are node types, metadata[1] are edge types
    in_channels_dict = { nodeType: graph[nodeType].x.size(1) for nodeType in metadata[0]}

    # Select model
    if args.model_type == "dummymodel":
        out_channels = 2 if args.prediction_type == "classification" else 1
        model = DummyModel(
            in_channels=graph["airport"].x.shape[1],
            hidden_channels=64,
            out_channels=out_channels,
        ).to(device)
    elif args.model_type == "rgcnmodel":
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
        print("Using model hgt")
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
    elif args.mode == "test":
        model.load_state_dict(torch.load(f"{args.model_dir}/{args.model_type}.pt"))
        test(model, graph, args)

if __name__ == "__main__":
    main()