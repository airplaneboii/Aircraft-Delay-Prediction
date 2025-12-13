import os
import glob
import yaml
import re
import zipfile
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from splitter import split_file_to_list
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
DEFAULT_CHUNKSIZE = 0
DEFAULT_ESSENTIAL_COLS = "essential.txt"
DEFAULT_DTYPE_FILE = "dtypes.yaml"


def fast_count_rows(path, buf_size=1024*1024):
    """Fast approximate row count by counting newlines in binary mode.

    Returns number of data rows (excludes header) when possible, or None on error.
    """
    try:
        with open(path, 'rb') as f:
            count = 0
            while True:
                b = f.read(buf_size)
                if not b:
                    break
                count += b.count(b"\n")
        # subtract one for header if file non-empty
        if count <= 0:
            return None
        return max(0, count - 1)
    except Exception:
        return None
    

def select_by_date(file_paths, start_year=None, start_month=None, end_year=None, end_month=None):
    """Select files whose basename starts with YEAR[_-]MONTH using provided range.

    Args:
        file_paths: iterable of file paths to test.
        start_year/start_month/end_year/end_month: optional ints defining inclusive range.

    Returns:
        Sorted list of file paths that match the pattern and fall inside the range.
    """
    selected = []
    for p in file_paths:
        bn = os.path.basename(p)
        m = re.match(r"^(\d{4})[_-]?(\d{1,2})", bn)
        if not m:
            continue
        year = int(m.group(1))
        month = int(m.group(2))
        if start_year is not None and start_month is not None:
            if (year, month) < (start_year, start_month):
                continue
        if end_year is not None and end_month is not None:
            if (year, month) > (end_year, end_month):
                continue
        selected.append(p)
    return sorted(selected)


def unzip_files(zip_paths, unzip_dir):
    """Extract the provided list of zip file paths into `unzip_dir`.

    Args:
        zip_paths: list of paths to .zip files to extract.
        unzip_dir: directory where extracted CSVs will be placed.

    Returns:
        Sorted list of CSV file paths now present in `unzip_dir`.
    """
    os.makedirs(unzip_dir, exist_ok=True)
    if not zip_paths:
        return sorted(glob.glob(os.path.join(unzip_dir, "*.csv")))

    for zip_path in tqdm(zip_paths, desc=f"{Fore.CYAN}Unzipping ZIPs{Style.RESET_ALL}", unit="zip"):
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        try:
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
        except Exception as e:
            tqdm.write(f"{Fore.RED}Failed to extract {zip_path}: {e}{Style.RESET_ALL}")

    # Return the list of CSVs now available in unzip_dir (caller will select/filter)
    return sorted(glob.glob(os.path.join(unzip_dir, "*.csv")))


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


def convert_dtypes(chunk, dtype_map):
    """Convert columns according to dtype_map (col -> dtype string).

    - dtype_map may be a dict mapping column->dtype (e.g. 'int32','float16','category','datetime').
    - Missing columns are skipped silently.
    - If the column contains only empty/missing values after cleaning, conversion is skipped.
    """
    if not dtype_map:
        return chunk

    # map common aliases to concrete pandas dtypes
    alias_map = {
        'short': 'int16',
        'long': 'int64',
        'half': 'float16',
        'double': 'float64',
        'integer': 'int32',
        'tiny': 'int8'
    }

    converted_cols = []
    for col, target in dtype_map.items():
        if not col or col not in chunk.columns:
            continue
        if target is None:
            continue

        # Skip entirely-empty columns
        try:
            if chunk[col].isnull().all():
                continue
        except Exception:
            pass

        t = str(target).lower()
        # normalize aliases like 'long','short','half','double'
        if t in alias_map:
            t = alias_map[t]
        s = chunk[col]
        try:
            # Pre-clean stringy numeric values (commas, percent signs, surrounding whitespace)
            if s.dtype == object or str(s.dtype).startswith('string'):
                ser = s.astype('string').str.strip()
                ser = ser.str.replace(',', '', regex=False)
                ser = ser.str.replace('%', '', regex=False)
            else:
                ser = s

            # Integers: use pandas nullable integer dtypes (Int8/16/32/64)
            if 'int' in t:
                num = pd.to_numeric(ser, errors='coerce')
                if num.dropna().empty:
                    continue
                m = re.search(r'int(8|16|32|64)', t)
                pdtype = f"Int{m.group(1)}" if m else 'Int32'
                chunk[col] = num.astype(pdtype)

            # Floats
            elif 'float' in t:
                num = pd.to_numeric(ser, errors='coerce')
                if num.dropna().empty:
                    continue
                m = re.search(r'float(16|32|64)', t)
                pdtype = f"float{m.group(1)}" if m else 'float32'
                chunk[col] = num.astype(pdtype)

            # Booleans: map common textual values and use pandas nullable boolean
            elif t in ('bool', 'boolean'):
                if hasattr(ser, 'str'):
                    lower = ser.str.lower()
                    true_set = {'true', '1', 'y', 'yes', 't'}
                    false_set = {'false', '0', 'n', 'no', 'f'}
                    mapped = lower.map(lambda x: True if x in true_set else (False if x in false_set else pd.NA))
                    chunk[col] = mapped.astype('boolean')
                else:
                    # numeric-ish
                    num = pd.to_numeric(ser, errors='coerce')
                    if num.dropna().empty:
                        continue
                    chunk[col] = (~num.isna()) & (num != 0)
                    chunk[col] = chunk[col].astype('boolean')

            elif t == 'category':
                chunk[col] = chunk[col].astype('category')

            elif t in ('datetime', 'datetime64'):
                chunk[col] = pd.to_datetime(chunk[col], errors='coerce', infer_datetime_format=True)

            elif t in ('string', 'str'):
                chunk[col] = chunk[col].astype('string')

            else:
                # Best-effort: try numeric coercion first, else fallback to astype
                num = pd.to_numeric(ser, errors='coerce')
                if not num.dropna().empty:
                    chunk[col] = num
                else:
                    chunk[col] = chunk[col].astype(target)

            converted_cols.append(col)

        except Exception as e:
            tqdm.write(f"{Fore.RED}Error converting {col} to {target}: {e}{Style.RESET_ALL}")

    if converted_cols:
        tqdm.write(f"{Fore.CYAN}Converted columns: {', '.join(converted_cols)}{Style.RESET_ALL}")
    return chunk

def merge_csvs(csv_files, merged_dir, output_prefix, chunksize, essential_cols=None, dtypes_map=None):
    """Merge the provided list of CSV file paths into a single output CSV.

    The caller is responsible for selecting and filtering `csv_files` before
    calling this function.
    """
    os.makedirs(merged_dir, exist_ok=True)
    
    if not csv_files:
        tqdm.write(f"{Fore.RED}No CSV files provided to merge{Style.RESET_ALL}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_prefix}_{timestamp}.csv"
    output_path = os.path.join(merged_dir, output_file)

    first = True
    total_rows = 0

    essential_cols = essential_cols or []
    dtypes_map = dtypes_map or {}

    for csv_file in tqdm(csv_files, desc=f"{Fore.MAGENTA}Merging CSVs{Style.RESET_ALL}", unit="csv"):
        try:
            # If chunksize is 0 or None, load the entire file at once.
            if not chunksize:
                df = pd.read_csv(csv_file, low_memory=False)
                original_rows = len(df)
                df = clean_chunk(df, essential_cols=essential_cols)
                df = convert_dtypes(df, dtypes_map)

                # Sort by DAY_OF_MONTH then CRS_DEP_TIME if available
                sort_cols = []
                if "DAY_OF_MONTH" in df.columns:
                    sort_cols.append("DAY_OF_MONTH")
                if "CRS_DEP_TIME" in df.columns:
                    sort_cols.append("CRS_DEP_TIME")
                if sort_cols:
                    df = df.sort_values(by=sort_cols, ascending=[True] * len(sort_cols))
                else:
                    tqdm.write(f"{Fore.YELLOW}Sort columns missing; skipping sort for {os.path.basename(csv_file)}{Style.RESET_ALL}")

                df.to_csv(output_path, mode="a", header=first, index=False)
                first = False
                total_rows += len(df)
                tqdm.write(f"{Fore.MAGENTA}Loaded:{Style.RESET_ALL} {csv_file} (original={original_rows}, written={len(df)})")
            else:
                # Chunked read path
                # Count rows for progress bar using a fast binary reader
                try:
                    row_count = fast_count_rows(csv_file)
                except Exception:
                    row_count = None

                with tqdm(total=row_count, desc=f"{Fore.CYAN}Rows in {os.path.basename(csv_file)}{Style.RESET_ALL}", unit="row", leave=False) as pbar:
                    for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
                        chunk = clean_chunk(chunk, essential_cols=essential_cols)
                        chunk = convert_dtypes(chunk, dtypes_map)
                        chunk.to_csv(output_path, mode="a", header=first, index=False)
                        first = False
                        total_rows += len(chunk)
                        pbar.update(len(chunk))
                if row_count is None:
                    tqdm.write(f"{Fore.MAGENTA}Loaded:{Style.RESET_ALL} {csv_file}")
                else:
                    tqdm.write(f"{Fore.MAGENTA}Loaded:{Style.RESET_ALL} {csv_file} ({row_count} rows)")
        except Exception as e:
            tqdm.write(f"{Fore.RED}Skipping:{Style.RESET_ALL} {csv_file} → {e}")

    tqdm.write(f"{Fore.GREEN}Merged {len(csv_files)} CSVs into {output_path} ({total_rows} rows).{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Unzip and merge TranStats CSVs with optional cleaning")
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing ZIP files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("-u", "--unzip-dir", default=DEFAULT_UNZIP_DIR,
                        help=f"Directory to extract CSVs into (default: {DEFAULT_UNZIP_DIR})")
    parser.add_argument("-m", "--merged-dir", default=DEFAULT_MERGED_DIR,
                        help=f"Directory to write merged CSV into (default: {DEFAULT_MERGED_DIR})")
    parser.add_argument("-o", "--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help=f"Prefix for merged output file (default: {DEFAULT_OUTPUT_PREFIX})")
    parser.add_argument("-c", "--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                        help=f"Rows per chunk when reading CSVs (default: {DEFAULT_CHUNKSIZE})")
    parser.add_argument("-e", "--essential-cols", type=str, default=DEFAULT_ESSENTIAL_COLS,
                        help=f"Path to file listing essential columns one-per-line (default: {DEFAULT_ESSENTIAL_COLS})")
    parser.add_argument("--dtypes-file", type=str, default=DEFAULT_DTYPE_FILE,
                        help="Path to YAML file mapping fields to dtypes (one mapping or empty to skip conversion)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List CSVs that would be merged (after filtering) and exit")

    # mutually exclusive: unzip-only (do not merge), merge-only (do not unzip)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unzip-only", action="store_true",
                       help="Only unzip files and exit (do not merge)")
    group.add_argument("--merge-only", action="store_true",
                       help="Only merge existing CSVs in --unzip-dir (do not unzip)")

    parser.add_argument("-Y1", "--start-year", type=int, default=None,
                        help="Start year (e.g. 2017) for filtering files by name")
    parser.add_argument("-M1", "--start-month", type=int, default=None,
                        help="Start month (1-12) for filtering files by name")
    parser.add_argument("-Y2", "--end-year", type=int, default=None,
                        help="End year (e.g. 2018) for filtering files by name")
    parser.add_argument("-M2", "--end-month", type=int, default=None,
                        help="End month (1-12) for filtering files by name")
    args = parser.parse_args()


    # essential_cols may be provided as filename (one name per line) or comma-separated list
    if os.path.isfile(args.essential_cols):
        essential_cols = split_file_to_list(args.essential_cols)
    else:
        essential_cols = [c.strip() for c in args.essential_cols.split(",") if c.strip()]

    # Dtype conversions: only load if a filepath is provided. Empty => skip conversion
    dtypes_map = {}
    if args.dtypes_file:
        if os.path.isfile(args.dtypes_file):
            try:
                with open(args.dtypes_file, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh) or {}
                    if isinstance(data, dict):
                        # Keep YAML values as-is; conversion will skip None/empty targets
                        dtypes_map = data
            except Exception as e:
                tqdm.write(f"{Fore.RED}Failed to load dtypes YAML ({args.dtypes_file}): {e}{Style.RESET_ALL}")
                dtypes_map = {}
        else:
            tqdm.write(f"{Fore.YELLOW}Dtypes file not found at {args.dtypes_file}; skipping dtype conversions{Style.RESET_ALL}")
            dtypes_map = {}

    # Validate explicit year/month args and prepare values for merging
    start_year = None
    start_month = None
    end_year = None
    end_month = None
    try:
        if (args.start_year is None) ^ (args.start_month is None):
            raise ValueError("Both --start-year and --start-month must be provided together")
        if (args.end_year is None) ^ (args.end_month is None):
            raise ValueError("Both --end-year and --end-month must be provided together")

        if args.start_year is not None and args.start_month is not None:
            if not (1 <= args.start_month <= 12):
                raise ValueError("--start-month must be between 1 and 12")
            start_year = int(args.start_year)
            start_month = int(args.start_month)

        if args.end_year is not None and args.end_month is not None:
            if not (1 <= args.end_month <= 12):
                raise ValueError("--end-month must be between 1 and 12")
            end_year = int(args.end_year)
            end_month = int(args.end_month)

        if start_year is not None and end_year is not None and (start_year, start_month) > (end_year, end_month):
            raise ValueError("Start year/month must be <= end year/month")
    except Exception as e:
        tqdm.write(f"{Fore.RED}Date parsing/validation error: {e}{Style.RESET_ALL}")
        return

    # Collect zip paths, perform selection
    zip_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.zip")))
    selected_zip_paths = select_by_date(zip_paths, start_year, start_month, end_year, end_month)
    # Collect existing and expected CSV files, perform selection
    existing_csvs = sorted(glob.glob(os.path.join(args.unzip_dir, "*.csv")))
    selected_existing_csvs = select_by_date(existing_csvs, start_year, start_month, end_year, end_month)
    expected_csvs = [os.path.join(args.unzip_dir, os.path.splitext(os.path.basename(z))[0] + ".csv") for z in selected_zip_paths]
    # Union while preserving order: existing selected first, then expected new ones
    display_csvs = list(selected_existing_csvs)
    for e in expected_csvs:
        if e not in display_csvs:
            display_csvs.append(e)

    # 1) unzip-only: show ZIPs
    if args.unzip_only and not args.merge_only:
        tqdm.write(f"{Fore.CYAN}ZIP(s) to extract:{Style.RESET_ALL}")
        for z in selected_zip_paths:
            tqdm.write(f"  {os.path.basename(z)}")
        # After display, run or return
        if args.dry_run:
            return
        if not selected_zip_paths:
            tqdm.write(f"{Fore.YELLOW}No ZIP files to extract after filtering.{Style.RESET_ALL}")
            return
        unzip_files(selected_zip_paths, args.unzip_dir)
        tqdm.write(f"{Fore.GREEN}Unzip-only requested; extracted selected ZIPs.{Style.RESET_ALL}")
        return

    # 2) merge-only: show existing CSVs only (no expected CSVs from ZIPs)
    if args.merge_only and not args.unzip_only:
        tqdm.write(f"{Fore.CYAN}CSV(s) to merge:{Style.RESET_ALL}")
        for p in selected_existing_csvs:
            rows = fast_count_rows(p)
            rows_text = "unknown" if rows is None else str(rows)
            tqdm.write(f"  {os.path.basename(p)} - rows: {rows_text}")
        if args.dry_run:
            return
        if not selected_existing_csvs:
            tqdm.write(f"{Fore.YELLOW}No CSV files to merge after filtering.{Style.RESET_ALL}")
            return
        merge_csvs(selected_existing_csvs, args.merged_dir, args.output_prefix, args.chunksize,
                   essential_cols=essential_cols, dtypes_map=dtypes_map)
        return

    # 3) normal mode: show both
    tqdm.write(f"{Fore.CYAN}ZIP(s) to extract:{Style.RESET_ALL}")
    for z in selected_zip_paths:
        tqdm.write(f"  {os.path.basename(z)}")
    tqdm.write(f"{Fore.CYAN}CSV(s) to merge:{Style.RESET_ALL}")
    for p in display_csvs:
        if os.path.exists(p):
            rows = fast_count_rows(p)
            rows_text = "unknown" if rows is None else str(rows)
            tqdm.write(f"  {os.path.basename(p)} - rows: {rows_text}")
        else:
            tqdm.write(f"  {os.path.basename(p)} - missing (will be created from ZIP)")

    # After display, if dry-run, return; else perform unzip+merge
    if args.dry_run:
        return

    # Unzip phase
    if selected_zip_paths:
        unzip_files(selected_zip_paths, args.unzip_dir)

    # Recompute available CSVs post-unzip and filter
    available_csvs = sorted(glob.glob(os.path.join(args.unzip_dir, "*.csv")))
    selected_csvs = select_by_date(available_csvs, start_year, start_month, end_year, end_month)

    if not selected_csvs:
        tqdm.write(f"{Fore.YELLOW}No CSV files to merge after filtering.{Style.RESET_ALL}")
        return

    merge_csvs(selected_csvs, args.merged_dir, args.output_prefix, args.chunksize,
               essential_cols=essential_cols, dtypes_map=dtypes_map)

if __name__ == "__main__":
    main()
