import requests
import json
import os
import csv

# SEC requires a descriptive User-Agent string on all API calls
HEADERS = {
    "User-Agent": "Your Name your_email@example.com",
    "Accept": "application/json"
}

def get_cik_for_ticker(ticker: str) -> str:
    """
    Look up a ticker in the SEC's mapping file and return its zero-padded 10-digit CIK.
    """
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(mapping_url, headers=HEADERS)
    resp.raise_for_status()
    
    data = resp.json()
    ticker_lower = ticker.lower()
    for entry in data.values():
        if entry["ticker"].lower() == ticker_lower:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC mapping")

def fetch_xbrl_facts(cik: str) -> dict:
    """
    Fetch all XBRL facts for the given 10-digit CIK via the Company Facts API.
    """
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(facts_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def save_facts_to_csv(facts: dict, filename: str):
    """
    Flatten the XBRL facts JSON into CSV rows and save to filename.
    Columns: taxonomy, concept, unit, value, start, end, filed, accn, form, frame, fy, fp
    """
    fieldnames = [
        'taxonomy','concept','unit','value',
        'start','end','filed','accn','form',
        'frame','fy','fp'
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for taxonomy, concepts in facts.get('facts', {}).items():
            for concept, details in concepts.items():
                for unit, entries in details.get('units', {}).items():
                    for entry in entries:
                        writer.writerow({
                            'taxonomy': taxonomy,
                            'concept':   concept,
                            'unit':      unit,
                            'value':     entry.get('val'),
                            'start':     entry.get('start', ''),
                            'end':       entry.get('end', ''),
                            'filed':     entry.get('filed', ''),
                            'accn':      entry.get('accn', ''),
                            'form':      entry.get('form', ''),
                            'frame':     entry.get('frame', ''),
                            'fy':        entry.get('fy', ''),
                            'fp':        entry.get('fp', ''),
                        })
    print(f"Saved flattened facts to {os.path.abspath(filename)}")

if __name__ == "__main__":
    # ← Change this to any ticker you need
    ticker = "CAT"
    
    # 1) Convert ticker → CIK
    cik = get_cik_for_ticker(ticker)
    print(f"CIK for {ticker.upper()}: {cik}")
    
    # 2) Fetch XBRL facts
    facts = fetch_xbrl_facts(cik)
    
    # 3) Save everything out as CSV
    csv_filename = f"{ticker.upper()}_xbrl_facts.csv"
    save_facts_to_csv(facts, csv_filename)
