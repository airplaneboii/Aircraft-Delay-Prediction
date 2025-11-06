import os
import zipfile
import glob
import pandas as pd

def unzip_all_in_directory(directory, extract_dir=None):
    """Unzip all .zip files in the given directory, renaming CSVs to avoid collisions."""
    if extract_dir is None:
        extract_dir = directory

    for zip_path in glob.glob(os.path.join(directory, "*.zip")):
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.lower().endswith(".csv"):
                    # Build a new filename: zipname_original.csv
                    new_name = f"{zip_name}_{os.path.basename(member)}"
                    target_path = os.path.join(extract_dir, new_name)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    print(f"Extracted {member} → {new_name}")
                else:
                    zip_ref.extract(member, extract_dir)

def merge_csvs(directory, output_file="merged.csv"):
    """Merge all uniquely named CSV files in the given directory into one file."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"Loaded: {csv_file} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_file, index=False)
        print(f"Merged {len(csv_files)} CSVs into {output_file} ({len(merged)} rows).")

def main():
    directory = "mergetest"  # <-- change this
    unzip_all_in_directory(directory)
    merge_csvs(directory, output_file=f"{directory}/all_merged.csv")

if __name__ == "__main__":
    main()
