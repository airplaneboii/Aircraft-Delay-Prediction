#!/bin/bash
#SBATCH --job-name=hetero3          # Job name
#SBATCH --partition=gpu               # Partition (queue)
#SBATCH --gres=gpu                    # Request GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # CPU cores per task
#SBATCH --mem=100G                     # Memory
#SBATCH --time=2-00:00:00               # Walltime
#SBATCH --output="logs/%x_%j.log"       # Stdout
#SBATCH --error="logs/%x_%j.log"        # Stderr

# --- Environment setup ---
#module purge
#module load CUDA/12.8.0
#module load Python/3.12.3-GCCcore-13.3.0

# --- System diagnostics ---
echo "===== SLURM JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per Node: $SLURM_TASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs requested: $SLURM_GPUS"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Time limit: $SLURM_TIMELIMIT"

echo "===== SYSTEM INFO ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo "OS: $(uname -a)"
echo "CPU Info:"
lscpu
echo "Memory Info:"
free -h
echo "Project Disk Usage:"
srun du -sh ./*

echo "===== GPU INFO ====="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
srun nvidia-smi

# --- Run training (prefer Apptainer SIF, fallback to venv or system python) ---
SIF_PATH="${SIF_PATH:-aircraft-delay.sif}"
APP_BIND="$PWD:/app"
DATA_BIND="$PWD/data:/app/data"
PRETRAINED_BIND="$PWD/pretrained:/app/pretrained"
LOGS_BIND="$PWD/logs:/app/logs"
CONFIGS_BIND="$PWD/configs:/app/configs"

echo "===== STARTING TRAINING (Apptainer only) ====="
# Ensure host dirs exist so bind points are present
mkdir -p logs pretrained data configs

# Helper function to run commands inside the SIF with standard binds.
# Usage: RUN <command> [args...]
RUN() {
	apptainer exec --nv \
		--bind "$APP_BIND","$DATA_BIND","$PRETRAINED_BIND","$LOGS_BIND","$CONFIGS_BIND" \
		"$SIF_PATH" python3 /app/main.py "$@"
}

# Example: run training (you can replace or pass different args)
RUN -c /app/configs/hetero3_hgt_reg.yaml
RUN -c /app/configs/hetero3_hgt_reg.yaml -m val
RUN -c /app/configs/hetero3_hgt_reg.yaml -m test

RUN -c /app/configs/hetero3_hgt_cls.yaml
RUN -c /app/configs/hetero3_hgt_cls.yaml -m val
RUN -c /app/configs/hetero3_hgt_cls.yaml -m test

RUN -c /app/configs/hetero3_rgcn_reg.yaml
RUN -c /app/configs/hetero3_rgcn_reg.yaml -m val
RUN -c /app/configs/hetero3_rgcn_reg.yaml -m test

RUN -c /app/configs/hetero3_rgcn_cls.yaml
RUN -c /app/configs/hetero3_rgcn_cls.yaml -m val
RUN -c /app/configs/hetero3_rgcn_cls.yaml -m test
