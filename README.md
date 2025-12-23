# Aircraft Delay Prediction
Group project for MLG course 2025.

This is still work in progress.

---

## Requirements
Python 3.10 - 3.13 is recommended. The operating system shouldn't matter. 
You can clone the repository with git or simply download the zip and unzip it.
While you can technically run the models on the CPU, it's best to use a GPU.
Currently only NVIDIA GPUs are supported. While pytorch already supports AMD GPUs on Linux, PyG does not (although this might change in the future). 
In the meantime you can use a custom build found here: https://github.com/Looong01/pyg-rocm-build, though in that case you'll need to install everything manually or tweak the `setup_env.py` script.

## Environment Setup
Clone the project and use our python script to create a venv and install dependencies:
```bash
git clone https://github.com/airplaneboii/Aircraft-Delay-Prediction.git
cd Aircraft-Delay-Prediction
python setup_env.py
```
The script prints the activation command for your OS. You can customize the env dir, Torch and CUDA versions, see:
`python setup_env.py -h`

## Data Pipeline
We provide two helpers: `parser.py` (downloads monthly TranStats ZIPs) and `merge.py` (unzips, filters, cleans, converts dtypes, and merges CSVs).

### Download ZIPs (from project root)
The parser has multiple modes:
- `data`: download flight data ZIPs
- `lookup`: download lookup tables into `data/lookup/`
- `md`: emit fields + descriptions as Markdown
- `ids`: emit available field IDs in chosen format

For a full list of available options, run:
```bash
python parser.py -h
```

Example (Nov 2017 → Jan 2018, using a field list):
```bash
python parser.py -m data -Y1 2017 -M1 11 -Y2 2018 -M2 1 -F data/fields1.txt
```
Optional zip filename prefix (helps later filtering):
```bash
python parser.py -m data -Y1 2017 -M1 11 -Y2 2018 -M2 1 -F data/fields1.txt -p SF
```

### Unzip and Merge CSVs
`merge.py` merges selected data and writes merged datasets to `data/datasets/`.

Usage modes and options:
- Normal mode: unzip matching ZIPs from `data/zipped/` → merge available CSVs from `data/unzipped/`.
- `--unzip-only` or `--merge-only` to run a single phase.
- `--dry-run` to preview what would be processed.
- Date range selection via `--start-year/--start-month/--end-year/--end-month`.
- Prefix filtering via `--prefix PREFIX`: when provided, only files starting with `PREFIX_` or `PREFIX-` are considered at both unzip and merge stages.
- Optional cleaning via `--essential-cols` and dtype conversion via `--dtypes-file`.
For a full list of available options, run:
```bash
python parser.py -h
```
Examples:
```bash
# Preview
python merge.py --dry-run

# Merge a date range
python merge.py -Y1 2017 -M1 11 -Y2 2018 -M2 1

# Only merge files that start with "SF_" in the date range
python merge.py -Y1 2017 -M1 11 -Y2 2018 -M2 1 --prefix SF

# Apply dtype conversions
python merge.py --dtypes-file data/dtypes.yaml
```

---
## Training and Evaluation
Run `main.py` directly or via a config file. It's recommended to use a config file for ease of use. For all available options run:
```bash
python main.py -h
```

### Using configs (recommended)
Config files are stored in `configs/`, the config with all the default values being `configs/defaults.yaml`.
```bash
python main.py -c configs/hetero3_rgcn_class.yaml
```
You can override any option from CLI:
```bash
python main.py -c configs/hetero3_rgcn_class.yaml --mode test --neighbor_sampling
```

### Graph and model files
- Graphs directory: `pretrained/graphs/`
- Models directory: `pretrained/models/`
- Provide graph/model names without extension (the code appends `.pt`).

Examples:
```bash
# Build a graph and save it
python main.py -c configs/hetero3_rgcn_class.yaml -s hetero3_class

# Load a saved graph and train a model
python main.py -c configs/hetero3_rgcn_class.yaml -l hetero3_class --mode train

# Evaluate using the latest dataset in data/datasets/
python main.py -c configs/hetero3_rgcn_class.yaml --mode test
```

### Available options
- Graph types: `base`, `hetero1`, `hetero2`, `hetero3`, `not_very_hetero`, `homo`, `hetero2nodes`
- Model types: `dummy`, `heterosage`, `rgcn`, `leakyrgcn`, `hgt`
- Neighbor sampling: `--neighbor_sampling` with `--neighbor_fanouts 15,10`
- Classification threshold: `--border` (see configs)

Logs and predictions are written to `logs/`. By default, `data_path` auto-selects the most recent file in `data/datasets/`, 
unless a specific path is specified.

---
## Repository Structure
Key directories and files:

- configs/: YAML configs for main.py
- data/: data storage (datasets/, zipped/, unzipped/, lookup/) and data loading and conversion specifications (normalize.txt, essential.txt, dtypes.yaml etc.)
- logs/: training logs and predictions
- pretrained/: pretrained graphs and models
- src/: most of the code (config.py, train.py, test.py, utils.py, graph/, models/)
- data parsing and merging: merge.py, parser.py
- main program: main.py, 
- environment setup: setup_env.py
- container setup files: apptainer.def, setup_env.sh, build_apptainer.sh
- SLURM batch jobs: hetero1.sbatch, hetero3.sbatch, run.sbatch

---
## Cluster/Container Notes
- SLURM: see the provided `*.sbatch` scripts for examples.
- Apptainer: see `apptainer.def` and `build_apptainer.sh` to build a container that matches the Python/CUDA stack.

---
## Contributing
To add a model, place a new module under `src/models/` and register it in `main.py`. To add a graph builder, add a module under `src/graph/` and register it similarly. Also expose new CLI options in `src/config.py` as needed.

Please ensure code runs cleanly before submitting (no errors/warnings).