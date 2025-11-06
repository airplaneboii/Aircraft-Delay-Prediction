import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import re
import time
from tqdm import tqdm, trange
from colorama import Fore, Style, init
from datetime import datetime, timedelta

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
                                   output_filename, file_index, total_files):
    """Stage 4: Send POST request and download the resulting ZIP file."""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": referer_url}
    os.makedirs("data", exist_ok=True)
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

def run_downloads(base_url, query_params, start_year, start_month, end_year, end_month,
                  geography, data_fields, request_interval):
    # Print selected date range banner
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}📅 Selected Date Range: {start_year}-{start_month} ➜ {end_year}-{end_month}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")

    soup, session, full_url = fetch_initial_page(base_url, query_params)
    if not soup:
        print(f"{Fore.RED}✖ Failed to fetch initial page. Exiting.{Style.RESET_ALL}")
        return

    aspnet_fields = extract_aspnet_state_fields(soup)
    months = list(month_year_iter(start_year, start_month, end_year, end_month))
    total_files = len(months)

    total_size = 0
    overall_start = time.time()

    for idx, (year, month) in enumerate(months, start=1):
        payload = prepare_post_payload(aspnet_fields, year, month, geography, data_fields, soup)
        filename = f"Transtats_Data_{year}_{month}_{geography}.zip"
        size_mb = send_post_request_and_download(base_url, payload, session, full_url,
                                                 filename, idx, total_files)
        total_size += size_mb

        if idx < total_files:
            print(f"{Fore.YELLOW}⏳ Waiting {request_interval} seconds before next request...{Style.RESET_ALL}")
            # Countdown using tqdm (Option 1)
            for remaining in trange(request_interval, 0, -1,
                                    desc="⏳ Waiting",
                                    unit="s",
                                    ncols=60,
                                    bar_format="{desc} ({remaining}s left)"):
                time.sleep(1)
            print()  # move to new line after countdown


    overall_elapsed = time.time() - overall_start
    td = timedelta(seconds=int(overall_elapsed))
    print(f"\n{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}✔ All downloads complete: {total_files} files, {total_size:.2f} MB total{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}⏱ Total elapsed time: {td} seconds{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")


# --- Main configuration only ---
if __name__ == "__main__":
    BASE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    QUERY_PARAMS = {"gnoyr_VQ": "FGJ", "QO_fu146_anzr": ""}
    REQUEST_INTERVAL = 60

    # REMEMBER THAT 2025 IS ONLY AVAILABLE UNTIL JULY!
    START_YEAR, START_MONTH = 2017, 1
    END_YEAR, END_MONTH = 2017, 1
    TARGET_GEOGRAPHY = "All"
    
    # Fields originally selected by domen:
    # row format
    # DATA_FIELDS = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_DELAY', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_DELAY', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    # column format
    DATA_FIELDS = [
        'YEAR', 
        'QUARTER', 
        'MONTH', 
        'DAY_OF_WEEK', 
        'FL_DATE', 
        'OP_UNIQUE_CARRIER', 
        'OP_CARRIER_AIRLINE_ID', 
        'OP_CARRIER', 
        'TAIL_NUM', 
        'OP_CARRIER_FL_NUM', 
        'ORIGIN_AIRPORT_ID', 
        'ORIGIN_AIRPORT_SEQ_ID', 
        'ORIGIN_CITY_MARKET_ID', 
        'ORIGIN', 
        'ORIGIN_CITY_NAME', 
        'ORIGIN_WAC', 
        'DEST_AIRPORT_ID', 
        'DEST_AIRPORT_SEQ_ID', 
        'DEST_CITY_MARKET_ID', 
        'DEST', 
        'DEST_CITY_NAME', 
        'DEST_WAC', 
        'CRS_DEP_TIME', 
        'DEP_DELAY', 
        'DEP_DEL15', 
        'DEP_DELAY_GROUP', 
        'TAXI_OUT', 
        'WHEELS_OFF', 
        'WHEELS_ON', 
        'TAXI_IN', 
        'CRS_ARR_TIME', 
        'ARR_DELAY', 
        'ARR_DEL15', 
        'ARR_DELAY_GROUP', 
        'CANCELLED', 
        'CANCELLATION_CODE', 
        'DIVERTED', 
        'CRS_ELAPSED_TIME', 
        'ACTUAL_ELAPSED_TIME', 
        'AIR_TIME', 
        'FLIGHTS', 
        'DISTANCE', 
        'DISTANCE_GROUP', 
        'CARRIER_DELAY', 
        'WEATHER_DELAY', 
        'NAS_DELAY', 
        'SECURITY_DELAY', 
        'LATE_AIRCRAFT_DELAY'
    ]

    run_downloads(BASE_URL, QUERY_PARAMS, START_YEAR, START_MONTH,
                  END_YEAR, END_MONTH, TARGET_GEOGRAPHY, DATA_FIELDS, REQUEST_INTERVAL)
