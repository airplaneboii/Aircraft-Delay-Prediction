import argparse
import os
import glob

# ========================================
# Default Configuration Values
# ========================================

# --- General ---
DEFAULT_MODE = "train"
DEFAULT_VERBOSITY_LEVEL = 1
DEFAULT_ROWS = None

# --- Data ---
DEFAULT_DATA_PATH = "data/datasets/"
DEFAULT_NORMALIZE = True
DEFAULT_DATA_SPLIT = "80,10,10"

# --- Sliding Window ---
DEFAULT_UNIT = 60  # minutes per time unit
DEFAULT_LEARN_WINDOW = 10  # number of units to learn from
DEFAULT_PRED_WINDOW = 1  # number of units to predict
DEFAULT_WINDOW_STRIDE = 1  # number of units to slide by

# --- Graph ---
DEFAULT_GRAPH_TYPE = "hetero3"
DEFAULT_SAVE_GRAPH = None
DEFAULT_LOAD_GRAPH = None

# --- Model ---
DEFAULT_MODEL_TYPE = "rgcn"
DEFAULT_MODEL_FILENAME = None
DEFAULT_PREDICTION_TYPE = "regression"

# --- Training ---
DEFAULT_EPOCHS = 500
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_BATCH_SIZE = 150000
DEFAULT_CRITERION = "huber"
DEFAULT_CLASS_BORDER = 0.45

# --- Neighbor Sampling ---
DEFAULT_NEIGHBOR_SAMPLING = False
DEFAULT_NEIGHBOR_FANOUTS = "15,10"

# --- Performance ---
DEFAULT_AMP = False
DEFAULT_COMPILE = False
DEFAULT_COMPILE_BACKEND = "inductor"
DEFAULT_COMPILE_MODE = "default"
DEFAULT_RESUME = False

# --- Directories ---
DEFAULT_GRAPH_DIR = "pretrained/graphs/"
DEFAULT_MODEL_DIR = "pretrained/models/"
DEFAULT_LOG_DIR = "logs/"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_fanouts(fanout_str: str):
    parts = [p.strip() for p in fanout_str.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Fanouts must be a comma-separated list of integers") from exc

def parse_split(split_str: str):
    parts = [p.strip() for p in split_str.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Split must have three comma-separated percentages: train,val,test")
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Split values must be integers") from exc
    if sum(values) != 100:
        raise argparse.ArgumentTypeError("Split percentages must sum to 100")
    return tuple(values)

def get_args():
    parser = argparse.ArgumentParser(description="Aircraft Delay Prediction with GNNs")

    # ========================================
    # Configuration File
    # ========================================
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to a YAML or JSON config file to load default argument values from")

    # ========================================
    # General
    # ========================================
    parser.add_argument("-m", "--mode", type=str, choices=["train", "val", "test"], default=DEFAULT_MODE,
                        help=f"Mode of operation (default: {DEFAULT_MODE})")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=DEFAULT_VERBOSITY_LEVEL,
                        help=f"Verbosity level: 0=warning, 1=info, 2=debug (default: {DEFAULT_VERBOSITY_LEVEL})")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help="Optional: limit the number of rows to load from CSV for quick tests")

    # ========================================
    # Data
    # ========================================
    parser.add_argument("-D", "--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help=f"Path to a dataset file (default: auto-select latest from {DEFAULT_DATA_PATH})")
    parser.add_argument("--normalize", type=str2bool, default=DEFAULT_NORMALIZE,
                        help=f"Enable normalization of numeric features (default: {DEFAULT_NORMALIZE})")
    parser.add_argument("--data_split", type=parse_split, default=DEFAULT_DATA_SPLIT,
                        help=f"Train/val/test split percentages, e.g., '80,10,10' (default: {DEFAULT_DATA_SPLIT})")

    # ========================================
    # Sliding Window Parameters
    # ========================================
    parser.add_argument("--unit", type=int, default=DEFAULT_UNIT,
                        help=f"Time unit in minutes (default: {DEFAULT_UNIT})")
    parser.add_argument("--learn_window", type=int, default=DEFAULT_LEARN_WINDOW,
                        help=f"Learning window: number of time units to learn from (default: {DEFAULT_LEARN_WINDOW})")
    parser.add_argument("--pred_window", type=int, default=DEFAULT_PRED_WINDOW,
                        help=f"Prediction window: number of time units to predict (default: {DEFAULT_PRED_WINDOW})")
    parser.add_argument("--window_stride", type=int, default=DEFAULT_WINDOW_STRIDE,
                        help=f"Stride: number of time units to slide between windows (default: {DEFAULT_WINDOW_STRIDE})")

    # ========================================
    # Graph
    # ========================================
    parser.add_argument("-g", "--graph_type", type=str, 
                        choices=["base", "hetero1", "hetero2", "hetero3", "hetero4", "not_very_hetero", "homo", "hetero2nodes"], 
                        default=DEFAULT_GRAPH_TYPE,
                        help=f"Type of graph to build (default: {DEFAULT_GRAPH_TYPE})")
    parser.add_argument("-s", "--save_graph", type=str, default=DEFAULT_SAVE_GRAPH,
                        help="Filename to save the built graph to (e.g., 'graph.pt'; default: None)")
    parser.add_argument("-l", "--load_graph", type=str, default=DEFAULT_LOAD_GRAPH,
                        help="Filename to load a pre-built graph from (e.g., 'graph.pt'; default: None)")

    # ========================================
    # Model
    # ========================================
    parser.add_argument("-t", "--model_type", type=str, 
                        choices=["none", "dummy", "rgcn", "leakyrgcn", "hgt", "heterosage"],
                        default=DEFAULT_MODEL_TYPE, 
                        help=f"Type of model to use (default: {DEFAULT_MODEL_TYPE})")
    parser.add_argument("-F", "--model_file", type=str, default=DEFAULT_MODEL_FILENAME,
                        help="Filename (without extension) to save or load the model (default: auto-generated from model/graph type)")
    parser.add_argument("-p", "--prediction_type", type=str, choices=["regression", "classification"],
                        default=DEFAULT_PREDICTION_TYPE, 
                        help=f"Type of prediction task (default: {DEFAULT_PREDICTION_TYPE})")
    parser.add_argument("--border", type=float, default=DEFAULT_CLASS_BORDER,
                        help=f"Border value for classification tasks (default: {DEFAULT_CLASS_BORDER})")

    # ========================================
    # Training
    # ========================================
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("-r", "--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate for the optimizer (default: {DEFAULT_LR})")
    parser.add_argument("-a", "--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help=f"Weight decay (L2 regularization) for the optimizer (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--criterion", type=str, choices=["mse", "huber", "l1"], default=DEFAULT_CRITERION,
                        help=f"Loss criterion to use for training (default: {DEFAULT_CRITERION})")

    # ========================================
    # Neighbor Sampling
    # ========================================
    parser.add_argument("-n", "--neighbor_sampling", action="store_true", default=DEFAULT_NEIGHBOR_SAMPLING,
                        help=f"Enable mini-batch neighbor sampling instead of full-batch training (default: {DEFAULT_NEIGHBOR_SAMPLING})")
    parser.add_argument("-f", "--neighbor_fanouts", type=parse_fanouts, default=DEFAULT_NEIGHBOR_FANOUTS,
                        help=f"Comma-separated neighbors to sample per layer, e.g., '15,10' (default: {DEFAULT_NEIGHBOR_FANOUTS})")

    # ========================================
    # Performance
    # ========================================
    parser.add_argument("--amp", action="store_true", default=DEFAULT_AMP,
                        help=f"Enable PyTorch AMP (mixed precision) on CUDA (default: {DEFAULT_AMP})")
    parser.add_argument("--compile", action="store_true", default=DEFAULT_COMPILE,
                        help=f"Compile the model with torch.compile for speed (default: {DEFAULT_COMPILE})")
    parser.add_argument("--compile_backend", type=str, default=DEFAULT_COMPILE_BACKEND,
                        help=f"Backend for torch.compile (default: {DEFAULT_COMPILE_BACKEND})")
    parser.add_argument("--compile_mode", type=str, default=DEFAULT_COMPILE_MODE,
                        help=f"Mode for torch.compile (default: {DEFAULT_COMPILE_MODE})")
    parser.add_argument("--resume", action="store_true", default=DEFAULT_RESUME,
                        help=f"Resume training from a .bk.pt checkpoint if available (default: {DEFAULT_RESUME})")

    # ========================================
    # Directories
    # ========================================
    parser.add_argument("-G", "--graph_dir", type=str, default=DEFAULT_GRAPH_DIR,
                        help=f"Directory to save or load graphs (default: {DEFAULT_GRAPH_DIR})")
    parser.add_argument("-M", "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save or load models (default: {DEFAULT_MODEL_DIR})")    
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                        help=f"Directory to write training logs and predictions (default: {DEFAULT_LOG_DIR})")

    # First parse known args to see if a config file was specified
    preliminary, remaining = parser.parse_known_args()
    if preliminary.config:
        # Load config file (JSON or YAML if available)
        cfg_path = preliminary.config
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg_text = open(cfg_path, "r", encoding="utf-8").read()
        cfg = None
        try:
            import json
            cfg = json.loads(cfg_text)
        except Exception:
            # Try YAML if json failed
            try:
                import yaml
                cfg = yaml.safe_load(cfg_text)
            except Exception as e:
                raise RuntimeError("Failed to parse config file. Install pyyaml or provide JSON.") from e

        if not isinstance(cfg, dict):
            raise RuntimeError("Config file must contain a top-level object/dictionary of options")

        # Set defaults on the parser so CLI can still override them
        parser.set_defaults(**cfg)

    # Now parse final args (with config defaults applied)
    args = parser.parse_args()

    # Normalize neighbor fanouts: allow None to mean full-neighbor sampling
    if isinstance(args.neighbor_fanouts, str):
        args.neighbor_fanouts = parse_fanouts(args.neighbor_fanouts)

    if isinstance(args.data_split, str):
        args.data_split = parse_split(args.data_split)

    # Derive default model filename: prefer provided graph filename (save first, then load)
    # otherwise fall back to the graph type. Do not override if user explicitly set model_file.
    if not args.model_file:
        graph_basename = None
        if getattr(args, "save_graph", None):
            graph_basename = os.path.splitext(os.path.basename(args.save_graph))[0]
        elif getattr(args, "load_graph", None):
            graph_basename = os.path.splitext(os.path.basename(args.load_graph))[0]
        else:
            graph_basename = args.graph_type
        args.model_file = f"{args.model_type}_{graph_basename}"

    # If data_path is not provided or points to a directory, auto-select the latest CSV in that directory
    # Treat empty string as not provided (useful for YAML configs that set data_path: "").
    if not args.data_path or os.path.isdir(args.data_path):
        # choose directory to search: provided dir or DEFAULT_DATA_PATH
        search_dir = args.data_path if args.data_path and os.path.isdir(args.data_path) else DEFAULT_DATA_PATH
        pattern = os.path.join(search_dir, "*.csv")
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            args.data_path = latest
            print(f"Auto-selected latest dataset: {latest}")
        else:
            raise FileNotFoundError(f"No CSV files found in {search_dir}. Please provide a dataset or run merge.py.")
    else:
        # If a file path was provided, validate it exists
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Provided data_path does not exist: {args.data_path}")

    # Ensure directories exist
    os.makedirs(args.graph_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args
