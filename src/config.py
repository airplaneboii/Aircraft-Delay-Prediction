import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Configuration")

    # General
    parser.add_argument("--mode", type=str, choices=["develop", "train", "test"], default="develop", help="Mode of operation")
    parser.add_argument("--graph_type", type=str, choices=["base"], default="base", help="Type of graph to build")             # TODO: Add graph types when implemented
    parser.add_argument("--model_type", type=str, choices=["dummymodel"], default="dummymodel", help="Type of model to use")   # TODO: Add models when implemented
    parser.add_argument("--data_path", type=str, default="data/dataset.csv", help="Path to the dataset file")
    parser.add_argument("--graph_dir", type=str, default="src/graph/", help="Directory to save or load graphs")
    parser.add_argument("--model_dir", type=str, default="src/models/", help="Directory to save or load models")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")

    parser.add_argument("--prediction_type", type=str, choices=["regression", "classification"], default="regression", help="Type of prediction task")

    # Graph parameters
    parser.add_argument("--time_window", type=int, default=6, help="Time window (hours) for temporal edges")

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.graph_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    return args
