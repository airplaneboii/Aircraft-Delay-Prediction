import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import re
import time
from tqdm import tqdm, trange
from colorama import Fore, Style, init
from datetime import datetime, timedelta
import argparse
from splitter import split_file_to_list, format_list

# Initialize colorama (needed on Windows)
init(autoreset=True)

def fetch_initial_page(base_url, query_params):
    """Stage 1: Perform the initial GET request to the TranStats page."""
    full_get_url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
    print(f"{Fore.CYAN}➜ Performing initial GET request to: {full_get_url}{Style.RESET_ALL}")
    try:
        session = requests.Session()
        response = session.get(full_get_url)
        response.raise_for_status()
        print(f"{Fore.GREEN}✔ Initial GET request successful{Style.RESET_ALL}")
        return BeautifulSoup(response.text, "html.parser"), session, full_get_url
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}✖ Error during initial GET request: {e}{Style.RESET_ALL}")
        return None, None, None

def extract_aspnet_state_fields(soup):
    """Stage 2: Extract ASP.NET hidden state fields required for POST requests."""
    aspnet_fields = {}
    field_names = [
        "__EVENTTARGET", "__EVENTARGUMENT", "__LASTFOCUS",
        "__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"
    ]
    for name in field_names:
        tag = soup.find("input", {"id": name}) or soup.find("input", {"name": name})
        aspnet_fields[name] = tag.get("value", "") if tag else ""
    print(f"{Fore.GREEN}✔ Extracted ASP.NET state fields{Style.RESET_ALL}")
    return aspnet_fields

def prepare_post_payload(aspnet_fields, year, month, geography, data_fields, soup):
    """Stage 3: Construct the POST payload."""
    payload = aspnet_fields.copy()

    # Geography
    geo = soup.find("select", id=re.compile("cboGeography$")) or soup.find("select", {"name": re.compile("cboGeography$")})
    payload[geo["name"] if geo else "ctl00$ContentPlaceHolder1$cboGeography"] = geography

    # Year
    yr = soup.find("select", id=re.compile("cboYear$")) or soup.find("select", {"name": re.compile("cboYear$")})
    payload[yr["name"] if yr else "ctl00$ContentPlaceHolder1$cboYear"] = str(year)

    # Month
    mo = soup.find("select", id=re.compile("cboPeriod$")) or soup.find("select", {"name": re.compile("cboPeriod$")})
    payload[mo["name"] if mo else "ctl00$ContentPlaceHolder1$cboPeriod"] = str(month)

    # Initialize all checkboxes to off
    for cb in soup.find_all("input", {"type": "checkbox"}):
        if cb.get("name"):
            payload[cb["name"]] = ""

    # Turn on requested data fields
    for short in data_fields:
        for cb in soup.find_all("input", {"type": "checkbox"}):
            if short.lower() in cb.get("name", "").lower():
                payload[cb["name"]] = "on"
                break

    # Download button
    btn = soup.find("input", {"type": "submit", "id": re.compile("btnDownload$")}) \
          or soup.find("input", {"type": "submit", "name": re.compile("btnDownload$")})
    payload[btn["name"] if btn else "ctl00$ContentPlaceHolder1$btnDownload"] = btn.get("value", "Download") if btn else "Download"

    print(f"{Fore.YELLOW}⚙ Prepared POST payload for {year}-{month}{Style.RESET_ALL}")
    return payload

def send_post_request_and_download(base_url, payload, session, referer_url, 
                                   output_filename, file_index, total_files, data_dir):
    """Stage 4: Send POST request and download the resulting ZIP file."""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": referer_url}
    os.makedirs(data_dir, exist_ok=True)
    full_path = os.path.join("data", output_filename)

    start_ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{Fore.CYAN}[File {file_index}/{total_files}] Starting download: {output_filename}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}⏱ Start time: {start_ts}{Style.RESET_ALL}")

    start_time = time.time()
    resp = session.post(base_url, data=payload, headers=headers, stream=True)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"{Fore.RED}✖ Error during POST request: {e}{Style.RESET_ALL}")
        return 0

    if resp.headers.get("Content-Type") == "application/zip":
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(full_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
            desc=output_filename,
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=120
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        elapsed = time.time() - start_time
        size_mb = os.path.getsize(full_path) / 1024 / 1024
        end_ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"{Fore.GREEN}✔ Completed {output_filename} in {elapsed:.1f}s ({size_mb:.2f} MB){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⏱ End time: {end_ts}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}➜ Progress: {file_index}/{total_files} files complete{Style.RESET_ALL}")
        return size_mb
    else:
        print(f"{Fore.RED}✖ Response was not a ZIP file. Possible error page returned.{Style.RESET_ALL}")
        print(resp.text[:500])
        return 0

def month_year_iter(start_year, start_month, end_year, end_month):
    """Yield (year, month) pairs from start to end inclusive."""
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1

def run_downloads(soup, session, full_url, base_url, data_dir,
                  start_year, start_month, end_year, end_month,
                  geography, data_fields, request_interval):
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}📅 Selected Date Range: {start_year}-{start_month} ➜ {end_year}-{end_month}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")

    aspnet_fields = extract_aspnet_state_fields(soup)
    months = list(month_year_iter(start_year, start_month, end_year, end_month))
    total_files = len(months)

    total_size = 0
    overall_start = time.time()

    for idx, (year, month) in enumerate(months, start=1):
        payload = prepare_post_payload(aspnet_fields, year, month, geography, data_fields, soup)
        filename = f"Transtats_Data_{year}_{month}_{geography}.zip"
        size_mb = send_post_request_and_download(
            base_url, payload, session, full_url,
            filename, idx, total_files, data_dir
        )
        total_size += size_mb

        if idx < total_files:
            print(f"{Fore.YELLOW}⏳ Waiting {request_interval} seconds before next request...{Style.RESET_ALL}")
            for remaining in trange(request_interval, 0, -1,
                                    desc="⏳ Waiting",
                                    unit="s",
                                    ncols=60,
                                    bar_format="{desc} ({remaining}s left)"):
                time.sleep(1)
            print()

    overall_elapsed = time.time() - overall_start
    td = timedelta(seconds=int(overall_elapsed))
    print(f"\n{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}✔ All downloads complete: {total_files} files, {total_size:.2f} MB total{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}⏱ Total elapsed time: {td} seconds{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")


def download_lookup_tables(soup, session, full_url, base_url, lookup_dir):
    """Scrape and download all lookup tables into data/lookup using server filenames."""
    print(f"{Fore.CYAN}➜ Fetching lookup table links from: {full_url}{Style.RESET_ALL}")

    os.makedirs(lookup_dir, exist_ok=True)
    links = soup.find_all("a", class_="dataTDRight")
    print(f"{Fore.YELLOW}⚙ Found {len(links)} candidate links{Style.RESET_ALL}")

    for link in links:
        href = link.get("href")
        if not href:
            continue

        lookup_url = urllib.parse.urljoin(base_url, href)
        try:
            resp = session.get(lookup_url)
            resp.raise_for_status()

            cd_header = resp.headers.get("Content-Disposition", "")
            match = re.search(r'filename="?([^"]+)"?', cd_header)
            filename = match.group(1) if match else "unknown.csv"
            full_path = os.path.join("data/lookup", filename)

            with open(full_path, "wb") as f:
                f.write(resp.content)
            print(f"{Fore.GREEN}✔ Saved: {full_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✖ Failed to download {lookup_url}: {e}{Style.RESET_ALL}")

def extract_from_mytable(soup):
    """Scrape table#myTable from an already-fetched soup and extract field metadata."""
    table = soup.find("table", {"id": "myTable"})
    if table is None:
        raise RuntimeError('Could not find table with id="myTable"')

    results = []
    rows = table.find_all("tr")

    # Skip header row
    for tr in rows[1:]:
        tds = tr.find_all("td")

        # Separator row
        if len(tds) == 1 or "lightcyan" in tr.get("style", "").lower():
            sep_text = tds[0].get_text(" ", strip=True)
            results.append({
                "field_id": "",
                "label": sep_text,
                "description": "",
                "separator": True
            })
            continue

        # Normal field row
        if len(tds) >= 2:
            first_td, second_td = tds[0], tds[1]
            inp = first_td.find("input", {"type": "checkbox"})
            field_id = (inp.get("id", "").strip().upper() if inp else "")

            lbl = first_td.find("label", {"for": field_id}) if field_id else first_td.find("label")
            label = lbl.get_text(strip=True) if lbl else ""
            description = second_td.get_text(" ", strip=True)

            results.append({
                "field_id": field_id,
                "label": label,
                "description": description,
                "separator": False
            })

    return results

def print_markdown_table(rows, include_separators=False):
    """Print extracted rows as a Markdown table."""
    print("| FIELD ID | Label | Description |")
    print("|----------|-------|-------------|")
    for r in rows:
        if r["separator"]:
            if include_separators:
                sep_text = f"<div align='center'>**{r['label']}**</div>"
                print(f"| {sep_text} |   |   |")
        else:
            print(f"| {r['field_id']} | {r['label']} | {r['description']} |")


# ─── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
QUERY_PARAMS = {"gnoyr_VQ": "FGJ", "QO_fu146_anzr": ""}
DEFAULT_START_YEAR = 2017
DEFAULT_START_MONTH = 1
DEFAULT_END_YEAR = 2017
DEFAULT_END_MONTH = 1
DEFAULT_GEOGRAPHY = "All"
DEFAULT_INTERVAL = 60
DEFAULT_FIELD_FILE = "fields1.txt"
DATA_DIR = "data"
LOOKUP_DIR = "data/lookup"

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TranStats bulk downloader and field extractor")
    parser.add_argument("-m", "--mode",
                        choices=["data", "lookup", "md", "ids"],
                        default="data",
                        help="Select 'data' to download ZIPs, 'lookup' for lookup tables, "
                            "'md' for Markdown field table, or 'ids' for formatted ID list")
    parser.add_argument("-is", "--include-separators",
                        action="store_true",
                        help="Include separator rows in Markdown output (only applies to mode=md)")
    parser.add_argument("-f", "--format",
                        choices=["newline", "quoted-newline-comma", "comma", "quoted-comma"],
                        default="comma",
                        help="Formatting style for ID list (only applies to mode=ids)")
    parser.add_argument("-u", "--url",
                        default="https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=",
                        help="URL of the page to scrape for field metadata")
    parser.add_argument("-y1", "--start-year", type=int, default=DEFAULT_START_YEAR,
                        help="Start year (default: 2017)")
    parser.add_argument("-M1", "--start-month", type=int, default=DEFAULT_START_MONTH,
                        help="Start month (default: 1)")
    parser.add_argument("-y2", "--end-year", type=int, default=DEFAULT_END_YEAR,
                        help="End year (default: 2017)")
    parser.add_argument("-M2", "--end-month", type=int, default=DEFAULT_END_MONTH,
                        help="End month (default: 1)")
    parser.add_argument("-g", "--geography", type=str, default=DEFAULT_GEOGRAPHY,
                        help="Geography filter (default: All)")
    parser.add_argument("-i", "--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Request interval in seconds (default: 60)")
    parser.add_argument("-F", "--data-fields", type=str, default=DEFAULT_FIELD_FILE,
                        help="Path to file containing comma- or newline-separated field names (default: field1.txt)")

    args = parser.parse_args()

    # Always fetch initial page first
    soup, session, full_url = fetch_initial_page(BASE_URL, QUERY_PARAMS)
    if not soup:
        print(f"{Fore.RED}✖ Failed to fetch initial page. Exiting.{Style.RESET_ALL}")
        exit(1)

    if args.mode == "lookup":
        download_lookup_tables(soup, session, full_url, BASE_URL, LOOKUP_DIR)

    elif args.mode == "md":
        rows = extract_from_mytable(soup)
        print_markdown_table(rows, include_separators=args.include_separators)

    elif args.mode == "ids":
        rows = extract_from_mytable(soup)
        ids = [r["field_id"] for r in rows if not r["separator"] and r["field_id"]]
        formatted = format_list(ids, args.format)
        print(formatted)

    elif args.mode == "data":
        DATA_FIELDS = split_file_to_list(args.data_fields)
        run_downloads(
            soup, session, full_url,
            BASE_URL,
            DATA_DIR,
            args.start_year,
            args.start_month,
            args.end_year,
            args.end_month,
            args.geography,
            DATA_FIELDS,
            args.interval
        )