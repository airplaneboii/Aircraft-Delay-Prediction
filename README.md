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

## Setting up the environment
Clone the repository:
```bash
git clone https://github.com/airplaneboii/Aircraft-Delay-Prediction.git
```
It's recommended to use a python virtual environment (some operating systems don't even support global python environments anymore).
We have a script that will create it automatically and install all needed packages, simply run:
```bash
python setup_env.py 
```
There are some pretty large packages so this could take a while.

To use a custom directory name, torch version or CUDA version, you can set them through command line arguments. See `python setup_env.py -h`.

Depending on your OS, activating the environment may be different. The script will tell you how to do it, 
but for more details you can follow the instructions here: https://docs.python.org/3/library/venv.html#how-venvs-work

## Fetching data
Since we can only download one month of data at a time from [Transtats](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=),
we wrote a dedicated parser to easily get the data in bulk. The script has multiple modes:
1) data - parsing the flight data
2) lookup - parsing lookup tables
3) md - parsing available fields and their descriptions in markdown format
4) ids - parsing a list of available fields

The modes can be configured via arguments, to see all options change into the data directory (important!) and run:
```bash
cd data
python parser.py --help
```
```
usage: parser.py [-h] [-m {data,lookup,md,ids}] [-is] [-f {newline,quoted-newline-comma,comma,quoted-comma}] [-u URL] [-Y1 START_YEAR] [-M1 START_MONTH]
                 [-Y2 END_YEAR] [-M2 END_MONTH] [-g GEOGRAPHY] [-i INTERVAL] [-F DATA_FIELDS]

TranStats bulk downloader and field extractor

options:
  -h, --help            show this help message and exit
  -m, --mode {data,lookup,md,ids}
                        Select 'data' to download ZIPs, 'lookup' for lookup tables, 'md' for Markdown field table, or 'ids' for formatted ID list
  -is, --include-separators
                        Include separator rows in Markdown output (only applies to mode=md)
  -f, --format {newline,quoted-newline-comma,comma,quoted-comma}
                        Formatting style for ID list (only applies to mode=ids)
  -u, --url URL         URL of the page to scrape for field metadata
  -Y1, --start-year START_YEAR
                        Start year (default: 2017)
  -M1, --start-month START_MONTH
                        Start month (default: 1)
  -Y2, --end-year END_YEAR
                        End year (default: 2017)
  -M2, --end-month END_MONTH
                        End month (default: 1)
  -g, --geography GEOGRAPHY
                        Geography filter (default: All)
  -i, --interval INTERVAL
                        Request interval in seconds between downloads to avoid rate-limiting (default: 60)
  -F, --data-fields DATA_FIELDS
                        Path to file containing comma- or newline-separated field names (default: fields2.txt)

```
The list of all available fields (`fields_all.txt`), the description table (`legend.md`) and the lookup tables (in `lookup`) are already in the repository.
The datasets however are too large to store on GitHub so they either have to be merged from the zip files provided in `zipped`, or parsed again.

For parsing data you have to provide a date range as descriped in help, and a text file that contains a list of fields you want to parse. Currently there are 3 templates, the default being `fields2.txt`. The fields in the provided files are separated by newlines for readability, but multiple separators are supported and should be detected automatically. As an example here's how to parse fields `fields1.txt` from November 2017 to January 2018 (inclusive):
```bash
python parser.py -m data -Y1 2017 -M1 11 -Y2 2018 -M2 1 -F fields1.txt
```
To use the data, the CSVs need to extracted and merged. The `merge.py` helper unzips TranStats ZIPs, optionally filters by date, cleans and converts dtypes, and merges CSVs into a single dataset.

Usage modes and options:

- Normal (default): unzip any matching ZIPs from `data/zipped/` into `data/unzipped/`, then merge available CSVs into `data/datasets/`.
- `--unzip-only`: only extract ZIP files (no merge).
- `--merge-only`: only merge existing CSVs in `--unzip-dir` (no extraction).
- `--dry-run`: display which ZIPs/CSVs would be processed, then exit.
- Date filtering: use `--start-year`, `--start-month`, `--end-year`, `--end-month` to select a contiguous inclusive range of files named like `YYYY_MM...`.
- `--essential-cols PATH`: drop rows missing the listed essential columns (file with one column name per line)
- `--dtypes-file PATH`: YAML mapping of column -> dtype to apply conversions (skipped if missing or empty).
- Output naming: merged file is written to `--merged-dir` (default `data/datasets/`) with prefix `--output-prefix` and a timestamp (so multiple runs produce distinct files).

For a full list of available arguments, run `python merge.py -h`.

Examples:

```bash
# Default: unzip+merge everything found in data/zipped/ and data/unzipped/
python merge.py

# Only show what would be done (no changes):
python merge.py --dry-run

# Only extract ZIPs in a given input dir:
python merge.py --input-dir my_zips --unzip-only

# Merge only existing CSVs (skip unzipping):
python merge.py --merge-only --unzip-dir data/unzipped

# Merge a specific date range (Nov 2017 - Jan 2018) and apply dtype mapping:
python merge.py -Y1 2017 -M1 11 -Y2 2018 -M2 1 --dtypes-file data/dtypes.yaml
```

**Don't forget to change back to the project directory when you're done dealing with data!**

---
## Running the main program

Model training, validation and testing is all performed by running `main.py` with command line arguments to configure parameters. To get a comprehensive list, run:
```bash
python main.py -h
```
Example:
```bash
python main.py --mode train --model_type rgcn --epochs 100 --lr 0.0005
```
However it is recommended to use config files instead, as there are a lot of arguments and it becomes difficult to manage. There are examples in the `configs` folder. To use a config file, run:
```bash
python main.py -c path/to/config/file
```
You don't have to set all available arguments in your config file, the unused arguments will be set to their default values, which can be seen in `configs/defaults.yaml`. Both JSON and YAML files are supported. If you want to override some config file arguments, you can still do that:
```bash
python main.py -c path/to/config/file --mode test --neighbor_sampling
```
This example would take all values from the config file, but replace the mode and neighbor sampling values with the ones provided in the command line.

---
## Editing
To add model, add new `.py` code of your model to `src/models/` folder. To add graph, add new `.py` code of your model to `src/graph/` folder. Following python structure of other models/graphs is strongly encouraged. Additionally, filename should be added to argument list in `src/config.py`. Simmilarly, `main.py` should be updated (new `elif` option should be properly added).

Before submitting new model or graph, make sure code runs without errors, failures and warnings.