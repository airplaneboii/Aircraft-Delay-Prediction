import os
import zipfile
import glob
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama (needed on Windows)
init(autoreset=True)

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

def merge_csvs(unzip_dir, merged_dir, output_prefix, chunksize):
    """Merge CSVs with row-level progress using chunked reading."""
    os.makedirs(merged_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(unzip_dir, "*.csv"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_prefix}_{timestamp}.csv"
    output_path = os.path.join(merged_dir, output_file)

    first = True
    total_rows = 0

    for csv_file in tqdm(csv_files, desc=f"{Fore.MAGENTA}Merging CSVs{Style.RESET_ALL}", unit="csv"):
        try:
            # Count rows for progress bar
            row_count = sum(1 for _ in open(csv_file, encoding="utf-8", errors="ignore")) - 1
            with tqdm(total=row_count, desc=f"{Fore.CYAN}Rows in {os.path.basename(csv_file)}{Style.RESET_ALL}", unit="row", leave=False) as pbar:
                for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
                    chunk.to_csv(output_path, mode="a", header=first, index=False)
                    first = False
                    total_rows += len(chunk)
                    pbar.update(len(chunk))
            tqdm.write(f"{Fore.MAGENTA}Loaded:{Style.RESET_ALL} {csv_file} ({row_count} rows)")
        except Exception as e:
            tqdm.write(f"{Fore.RED}Skipping:{Style.RESET_ALL} {csv_file} → {e}")

    tqdm.write(f"{Fore.GREEN}Merged {len(csv_files)} CSVs into {output_path} ({total_rows} rows).{Style.RESET_ALL}")

def main():
    INPUT_DIR = "zipped"       # raw ZIP files
    UNZIP_DIR = "unzipped"     # extracted CSVs
    MERGED_DIR = "datasets"    # final merged CSV
    OUTPUT_PREFIX = "DATASET"  # prefix for merged file
    CHUNKSIZE = 150000  # chunk size (rows per batch) for loading into the merged CSV

    unzip_all_in_directory(INPUT_DIR, UNZIP_DIR)
    merge_csvs(UNZIP_DIR, MERGED_DIR, OUTPUT_PREFIX, CHUNKSIZE)

if __name__ == "__main__":
    main()
