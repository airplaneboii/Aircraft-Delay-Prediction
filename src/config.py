import argparse
import os
import glob

# Default values for CLI
DEFAULT_MODE = "train"
DEFAULT_GRAPH_TYPE = "base"
DEFAULT_MODEL_TYPE = "hgtmodel"
DEFAULT_DATA_PATH = "data/datasets/"
DEFAULT_GRAPH_DIR = "src/graph/"
DEFAULT_MODEL_DIR = "src/models/"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001
DEFAULT_PREDICTION_TYPE = "regression"
DEFAULT_TIME_WINDOW = 6

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Configuration")

    # General
    parser.add_argument("-m", "--mode", type=str, choices=["train", "val", "test"], default=DEFAULT_MODE,
                        help=f"Mode of operation (default: {DEFAULT_MODE})")
    parser.add_argument("-d", "--development", action="store_true",
                        help=f"Toggle development mode with smaller dataset ONLY for quick code testing")    
    
    # Graphs and graph parameters 
    parser.add_argument("-g", "--graph_type", type=str, choices=["base"], default=DEFAULT_GRAPH_TYPE,
                        help=f"Type of graph to build (default: {DEFAULT_GRAPH_TYPE})")   #TODO: Add more graph types when implemented
    parser.add_argument("-s", "--save_graph", type=str, default=None,
                        help="Filename to save the built graph to (e.g., 'graph.pt'; default: None, don't save)")
    parser.add_argument("-l", "--load_graph", type=str, default=None,
                        help="Filename to load a pre-built graph from (e.g., 'graph.pt'; default: None, build from scratch)")
    parser.add_argument("-w", "--time_window", type=int, default=DEFAULT_TIME_WINDOW,
                        help=f"Time window (hours) for temporal edges (default: {DEFAULT_TIME_WINDOW})")
    
    # Available models
    parser.add_argument("-t", "--model_type", type=str, choices=["none", "dummymodel", "rgcnmodel", "hgtmodel", "heterosage"],
                        default=DEFAULT_MODEL_TYPE, help=f"Type of model to use (default: {DEFAULT_MODEL_TYPE})")
    parser.add_argument("-D", "--data_path", type=str, default=None,
                        help=f"Path to a specific dataset file (default: auto-select latest from {DEFAULT_DATA_PATH})")
    parser.add_argument("-G", "--graph_dir", type=str, default=DEFAULT_GRAPH_DIR,
                        help=f"Directory to save or load graphs (default: {DEFAULT_GRAPH_DIR})")
    parser.add_argument("-M", "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save or load models (default: {DEFAULT_MODEL_DIR})")    
    
    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("-r", "--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate for the optimizer (default: {DEFAULT_LR})")
    parser.add_argument("-p", "--prediction_type", type=str, choices=["regression", "classification"],
                        default=DEFAULT_PREDICTION_TYPE, help=f"Type of prediction task (default: {DEFAULT_PREDICTION_TYPE})")

    args = parser.parse_args()

    # If no data_path provided, auto-select the latest dataset in DEFAULT_DATA_PATH
    if args.data_path is None:
        pattern = os.path.join(DEFAULT_DATA_PATH, "*.csv")
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            args.data_path = latest
            print(f"Auto-selected latest dataset: {latest}")
        else:
            raise FileNotFoundError(f"No CSV files found in {DEFAULT_DATA_PATH}. Please provide a dataset or run merge.py.")

    # Ensure directories exist
    os.makedirs(args.graph_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    return args
