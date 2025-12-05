import os
import zipfile
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from splitter import split_file_to_list
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama (needed on Windows)
init(autoreset=True)

# Suppress FutureWarning about fillna downcasting
pd.set_option('future.no_silent_downcasting', True)

# Default values for CLI (used in help strings)
DEFAULT_INPUT_DIR = "zipped"
DEFAULT_UNZIP_DIR = "unzipped"
DEFAULT_MERGED_DIR = "datasets"
DEFAULT_OUTPUT_PREFIX = "DATASET"
DEFAULT_CHUNKSIZE = 150000
DEFAULT_ESSENTIAL_COLS = "essential.txt"
DEFAULT_FLOAT2INT_FILE = "float2int_fields.txt"

def unzip_all_in_directory(input_dir, unzip_dir):
    """Unzip all .zip files in input_dir, renaming CSVs to avoid collisions, saving into unzip_dir."""
    os.makedirs(unzip_dir, exist_ok=True)
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))

    for zip_path in tqdm(zip_files, desc=f"{Fore.CYAN}Unzipping ZIPs{Style.RESET_ALL}", unit="zip"):
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc=f"{Fore.BLUE}Extracting {zip_name}{Style.RESET_ALL}", unit="file", leave=False):
                if member.lower().endswith(".csv"):
                    new_name = f"{zip_name}.csv"
                    target_path = os.path.join(unzip_dir, new_name)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    tqdm.write(f"{Fore.CYAN}Unzipped:{Style.RESET_ALL} {member} → {new_name}")
                else:
                    zip_ref.extract(member, unzip_dir)
                    tqdm.write(f"{Fore.YELLOW}Unzipped other file:{Style.RESET_ALL} {member}")

def clean_chunk(chunk, essential_cols=None):
    """Modular cleaning function: drop rows missing essential fields, fill all NaNs with 0.
    
    Args:
        chunk: DataFrame chunk
        essential_cols: List of column names that must not be NaN (rows with NaN in these are dropped)
    
    Returns:
        Cleaned DataFrame chunk with optimized dtypes
    """
    if essential_cols:
        # Drop rows missing essential columns (only consider columns that exist in this chunk)
        subset = [c for c in essential_cols if c in chunk.columns]
        if subset:
            before = len(chunk)
            chunk = chunk.dropna(subset=subset)
            dropped = before - len(chunk)
            if dropped:
                tqdm.write(f"{Fore.YELLOW}Dropped {dropped} rows missing essential columns{Style.RESET_ALL}")
    
    # Fill all remaining NaNs with 0 in one operation
    chunk = chunk.fillna(0)
    
    return chunk


def convert_float_to_int(chunk, cols):
    """Convert specified columns from float to int (in-place optimization).

    Assumes NaNs have already been filled (via clean_chunk).
    Columns not present in the chunk are skipped silently.
    Reports any conversion errors that occur.
    """
    if not cols:
        return chunk

    for col in cols:
        if col not in chunk.columns or chunk[col].dtype == 'int32':
            continue
        try:
            chunk[col] = chunk[col].astype('int32')
            #tqdm.write(f"{Fore.CYAN}Converted {col} to int {Style.RESET_ALL}")
        except Exception as e:
            tqdm.write(f"{Fore.RED}Error converting {col} to int: {e}{Style.RESET_ALL}")
    tqdm.write(f"{Fore.CYAN}Converted selected columns to int{Style.RESET_ALL}")            
    return chunk

def merge_csvs(unzip_dir, merged_dir, output_prefix, chunksize, essential_cols=None, float2int_cols=None):
    """Merge CSVs with row-level progress using chunked reading.

    While merging, applies modular cleaning: drops rows with missing essential fields
    and fills all remaining NaNs with 0. Then optionally converts selected columns to int.
    """
    os.makedirs(merged_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(unzip_dir, "*.csv")))
    
    if not csv_files:
        tqdm.write(f"{Fore.RED}No CSV files found in {unzip_dir}{Style.RESET_ALL}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_prefix}_{timestamp}.csv"
    output_path = os.path.join(merged_dir, output_file)

    first = True
    total_rows = 0

    essential_cols = essential_cols or []
    float2int_cols = float2int_cols or []

    for csv_file in tqdm(csv_files, desc=f"{Fore.MAGENTA}Merging CSVs{Style.RESET_ALL}", unit="csv"):
        try:
            # Count rows for progress bar
            row_count = sum(1 for _ in open(csv_file, encoding="utf-8", errors="ignore")) - 1
            with tqdm(total=row_count, desc=f"{Fore.CYAN}Rows in {os.path.basename(csv_file)}{Style.RESET_ALL}", unit="row", leave=False) as pbar:
                for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
                    chunk = clean_chunk(chunk, essential_cols=essential_cols)
                    chunk = convert_float_to_int(chunk, float2int_cols)
                    chunk.to_csv(output_path, mode="a", header=first, index=False)
                    first = False
                    total_rows += len(chunk)
                    pbar.update(len(chunk))
            tqdm.write(f"{Fore.MAGENTA}Loaded:{Style.RESET_ALL} {csv_file} ({row_count} rows)")
        except Exception as e:
            tqdm.write(f"{Fore.RED}Skipping:{Style.RESET_ALL} {csv_file} → {e}")

    tqdm.write(f"{Fore.GREEN}Merged {len(csv_files)} CSVs into {output_path} ({total_rows} rows).{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Unzip and merge TranStats CSVs with optional cleaning")
    parser.add_argument("-I", "--input-dir", default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing ZIP files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("-U", "--unzip-dir", default=DEFAULT_UNZIP_DIR,
                        help=f"Directory to extract CSVs into (default: {DEFAULT_UNZIP_DIR})")
    parser.add_argument("-M", "--merged-dir", default=DEFAULT_MERGED_DIR,
                        help=f"Directory to write merged CSV into (default: {DEFAULT_MERGED_DIR})")
    parser.add_argument("-o", "--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help=f"Prefix for merged output file (default: {DEFAULT_OUTPUT_PREFIX})")
    parser.add_argument("-c", "--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                        help=f"Rows per chunk when reading CSVs (default: {DEFAULT_CHUNKSIZE})")
    parser.add_argument("-e", "--essential-cols", type=str, default=DEFAULT_ESSENTIAL_COLS,
                        help=f"Path to file listing essential columns one-per-line (default: {DEFAULT_ESSENTIAL_COLS})")
    parser.add_argument("--float2int-file", type=str, default=DEFAULT_FLOAT2INT_FILE,
                        help=f"Path to file listing columns to convert from float->int (one-per-line) (default: {DEFAULT_FLOAT2INT_FILE})")
    parser.add_argument("--skip-unzip", action="store_true",
                        help="Skip unzipping; assume CSVs are already extracted in --unzip-dir")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    UNZIP_DIR = args.unzip_dir
    MERGED_DIR = args.merged_dir
    OUTPUT_PREFIX = args.output_prefix
    CHUNKSIZE = args.chunksize

    # essential_cols may be provided as filename (one name per line) or comma-separated list
    if os.path.isfile(args.essential_cols):
        essential_cols = split_file_to_list(args.essential_cols)
    else:
        essential_cols = [c.strip() for c in args.essential_cols.split(",") if c.strip()]

    # Float->int columns: read file or accept comma-separated list
    float2int_cols = []
    if args.float2int_file:
        if os.path.isfile(args.float2int_file):
            float2int_cols = split_file_to_list(args.float2int_file)
        else:
            float2int_cols = [c.strip() for c in args.float2int_file.split(",") if c.strip()]

    if not args.skip_unzip:
        unzip_all_in_directory(INPUT_DIR, UNZIP_DIR)
    
    merge_csvs(UNZIP_DIR, MERGED_DIR, OUTPUT_PREFIX, CHUNKSIZE, essential_cols=essential_cols, float2int_cols=float2int_cols)

if __name__ == "__main__":
    main()
