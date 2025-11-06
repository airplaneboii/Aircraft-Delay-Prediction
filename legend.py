from bs4 import BeautifulSoup
import requests
import argparse

def extract_from_mytable(url):
    """
    Fetch the page at `url`, find table#myTable, and extract:
    - FIELD ID: from the <input id="..."> in the first <td>
    - Label: from the <label for="..."> in the first <td>
    - Description: text content of the second <td>
    - Separator rows: detected by bgcolor/style (e.g. LightCyan) or by having only one <td>
    Returns a list of dicts: {"field_id": ..., "label": ..., "description": ..., "separator": bool}
    """
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", {"id": "myTable"})
    if table is None:
        raise RuntimeError('Could not find table with id="myTable"')

    results = []
    rows = table.find_all("tr")

    # Skip the very first row
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

            label = ""
            if field_id:
                lbl = first_td.find("label", {"for": field_id})
                label = lbl.get_text(strip=True) if lbl else ""
            else:
                lbl = first_td.find("label")
                label = lbl.get_text(strip=True) if lbl else ""

            description = second_td.get_text(" ", strip=True)

            results.append({
                "field_id": field_id,
                "label": label,
                "description": description,
                "separator": False
            })

    return results

def print_markdown_table(rows):
    """Print extracted rows as a Markdown table."""
    print("| FIELD ID | Label | Description |")
    print("|----------|-------|-------------|")
    for r in rows:
        if r["separator"]:
            sep_text = f"<div align='center'>**{r['label']}**</div>"
            print(f"| {sep_text} |   |   |")
        else:
            print(f"| {r['field_id']} | {r['label']} | {r['description']} |")

def print_id_array(rows):
    """Print extracted field IDs as a Python list (horizontal)."""
    ids = [r["field_id"] for r in rows if not r["separator"] and r["field_id"]]
    print(ids)

def print_id_array_vertical(rows):
    """Print extracted field IDs one per line (vertical)."""
    ids = [r["field_id"] for r in rows if not r["separator"] and r["field_id"]]
    for fid in ids:
        print(f"{fid},")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract fields from TranStats myTable and print them in different formats."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=",
        help="URL of the page to scrape (default: TranStats DL_SelectFields)"
    )
    parser.add_argument(
        "--mode",
        choices=["md", "ids", "ids-vertical"],
        default="md",
        help="Output mode: 'md' for Markdown table, 'ids' for Python array, 'ids-vertical' for one ID per line (default: md)"
    )
    args = parser.parse_args()

    rows = extract_from_mytable(args.url)

    if args.mode == "ids":
        print_id_array(rows)
    elif args.mode == "ids-vertical":
        print_id_array_vertical(rows)
    else:
        print_markdown_table(rows)
