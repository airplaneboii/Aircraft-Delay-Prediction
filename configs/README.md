Configs for Aircraft-Delay-Prediction

Location: `configs/`

Usage:

  - Provide a path to a config file with `-c`/`--config` when running `main.py`.
  - CLI options override config values.

Examples:

  - RGCN full-batch (no sampling):
    `python main.py -c configs/rgcn_full.yaml --data_path data/datasets/your.csv`
    
    Uses the RGCN model over the full graph. Good baseline before optimizations.

  - RGCN with neighbor sampling:
    `python main.py -c configs/rgcn_sampling.yaml --data_path data/datasets/your.csv`
    
    Trains RGCN using PyG NeighborLoader with `neighbor_fanouts` from the config.

  - HGT example (sampling enabled):
    `python main.py -c configs/hgt_example.yaml --data_path data/datasets/your.csv`
    
    Trains the HGT model; demonstrates hetero fanouts and sampling.

  - Use the defaults file as a quick config reference or as a config itself:
    `python main.py -c configs/defaults.yaml --data_path data/datasets/your.csv`
    
    `configs/defaults.yaml` lists the repository defaults (useful to copy/modify).

  - Dummy full-batch example (no sampling):
    `python main.py -c configs/dummy_nosample.yaml --data_path data/datasets/your.csv`
    
    Runs the lightweight `dummy` on the full graph (no NeighborLoader sampling).

Notes:

  - `data_path` is intentionally left blank in example configs to avoid committing dataset paths.
    You can either edit the config to set your dataset location, or pass it through `--data_path` on the command line.
    You can also put in a directory which will automatically choose latest dataset from that directory.
    You can also leave it empty, in which case the latest dataset from the default directory (hardcoded in `config.py` to be `data/datasets`) will be used
  - `neighbor_sampling: true|false` toggles NeighborLoader usage. When `false`, the full graph is used.
  - `neighbor_fanouts` is a comma-separated string (e.g. "30,20"). The parser converts it to a list.
  - `verbosity: 0|1|2` controls logging detail (0=minimal, 1=info, 2=debug). CLI `--verbosity` overrides config.
  - CLI flags always override config values: for example, `--batch_size`, `--epochs`, `--model_type`, etc.