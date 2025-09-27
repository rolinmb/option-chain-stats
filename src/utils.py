from consts import TABLEHEADERS, TRADINGDAYS, BASEURL, URLP2, URLP3, URLP4, MODES, HEADERS
from options import OptionContract, OptionExpiry, OptionChain, UnderlyingAsset
import os
import re
import sys
import csv
import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument("--headless")  # run without opening browser window

def scrapeUnderlyingInfo(ticker, url, csvname):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    pairs = []
    table = soup.find("table", class_="js-snapshot-table snapshot-table2 screener_snapshot-table-body")
    if table:
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
            # cells come in [label, value, label, value, ...]
            for i in range(0, len(cells), 2):
                label = cells[i]
                value = cells[i+1] if i+1 < len(cells) else ""
                pairs.append([label, value])

    spans = soup.find_all("span", class_=[
        "table-row", "w-full", "items-baseline",
        "justify-end", "whitespace-nowrap",
        "text-negative", "text-muted-2"
    ])

    if not spans:
        print(f"src/utils.py :: Ticker {ticker} is invalid")
        sys.exit()

    dollar_change = spans[0].get_text(strip=True)
    dollar_change_clean = re.search(r"[-+]?[\d.,]+", dollar_change) # Keep only numbers, sign, decimal, and percent
    dollar_change = dollar_change_clean.group(0) if dollar_change_clean else dollar_change
    pairs.append(["$ Change", dollar_change])

    with open(csvname, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Value"])
        writer.writerows(pairs)
    
    response.close()
    print(f"src/utils.py :: Successfully webscraped finviz.com for {ticker} to create {csvname}")


def scrapeChainStats(ticker, url, csvname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=OPTIONS)
    driver.get(url)
    # Give time for JavaScript to load
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table table-sm table-hover table-bordered table-responsive optioncharts-table-styling")
    if not table:
        print("src/util.py :: No HTML table found")
        driver.quit()
        return
    
    driver.quit()
    data_rows = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        text_vals = [col.get_text(strip=True).replace("\xa0", " ") for col in cols]
        data_rows.append(text_vals)
    
    with open(csvname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TABLEHEADERS)
        writer.writerows(data_rows)
    
    print(f"src/utils.py :: Successfully webscraped {ticker} option data table to {csvname}")
    
def scrapeEntireChain(ticker, url, csvname, underlying_csvname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=OPTIONS)
    driver.get(url)
    # Give time for JavaScript to load
    time.sleep(3)
    try:
        button = driver.find_element(By.ID, "expiration-dates-form-button-1")
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        driver.execute_script("arguments[0].click();", button)
        wait = WebDriverWait(driver, 1)
        wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "label.tw-ml-3.tw-min-w-0.tw-flex-1.tw-text-gray-600")
            )
        )
    except Exception as e:
        print(f"src/utils.py :: Error clicking expiration button or waiting for labels: {e}")
        driver.quit()
        sys.exit(1)

    spans = driver.find_elements(By.CSS_SELECTOR, ".tw-text-3xl.tw-font-semibold")
    price_string = spans[1].text.strip()
    labels = driver.find_elements(
        By.CSS_SELECTOR, "label.tw-ml-3.tw-min-w-0.tw-flex-1.tw-text-gray-600"
    )
    expirations_text = [lbl.text for lbl in labels if lbl.text.strip()]
    formatted_expiration_dates = []
    exp_in_years = []
    wms = []
    for text in expirations_text:
        parts = text.split("(")

        date_str = parts[0].strip()
        exp_dt = datetime.strptime(date_str, "%b %d, %Y")
        final_dt = exp_dt.strftime("%Y-%m-%d")
        formatted_expiration_dates.append(final_dt)

        days_part = parts[1].split()[0]
        days = int(days_part)
        years = days / TRADINGDAYS
        exp_in_years.append(years)

        if len(formatted_expiration_dates) != len(exp_in_years):
            print(f"src/utils.py :: len(formatted_expiration_dates) != len(exp_in_years)")
            sys.exit(1)

        if "w" in parts[2]:
            wms.append("w")
        elif "m" in parts[2]:
            wms.append("m")

    print(f"src/utils.py :: Successfully parsed expiration dates and calculated yte for each expiry/contract")
    time.sleep(1.0)
    expiries = []
    for i in range(0, len(exp_in_years)):
        url = f"{BASEURL}{ticker}{URLP2}{URLP3}{formatted_expiration_dates[i]}:{wms[i]}{URLP4}"
        driver.get(url)
        time.sleep(3.0)
        shortdate_parts = formatted_expiration_dates[i].split("-")
        shortdate = shortdate_parts[0][2:]+shortdate_parts[1]+shortdate_parts[2]
        calls = []
        puts = []
        table = driver.find_element(By.CSS_SELECTOR, "table.table.table-sm.table-hover")
        rows = table.find_elements(By.TAG_NAME, "tr")[1:]
        for row in rows:
            cbid = cask = cvol = c_oi = strike = 0
            pbid = pask = pvol = p_oi = 0
            cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
            if len(cols) != 11:
                continue
            
            clast = cols[0] if cols[0] != "-" else "0.00"
            cbid = cols[1]
            cask = cols[2]
            cvol = cols[3]
            c_oi = cols[4]
            strike = cols[5]
            plast = cols[6] if cols[6] != "-" else "0.00"
            pbid = cols[7]
            pask = cols[8]
            pvol = cols[9]
            p_oi = cols[10]

            # Make symbols like CHGG251121C00002000 = CHGG 2025-11-21 $2.00 call
            strike_val = float(strike)            # e.g. "2.00" → 2.0
            strike_int = int(strike_val * 1000)   # scale by 1000 → 2000
            strike_str = f"{strike_int:08d}"
            csymbol = f"{ticker}{shortdate}C{strike_str}"
            psymbol = f"{ticker}{shortdate}P{strike_str}"

            calls.append(OptionContract(ticker, csymbol, price_string, strike, exp_in_years[i], clast, cbid, cask, cvol, c_oi, True))
            puts.append(OptionContract(ticker, psymbol, price_string, strike, exp_in_years[i], plast, pbid, pask, pvol, p_oi, False))

        print(f"src/utils.py :: Processed Calls and Puts for expiration {formatted_expiration_dates[i]}")
        expiries.append(OptionExpiry(ticker, formatted_expiration_dates[i], exp_in_years[i], calls, puts))

    underlying_asset = UnderlyingAsset(ticker, f"data/{ticker}info.csv")
    option_chain = OptionChain(ticker, underlying_asset, expiries)
    print(f"src/utils.py :: Underlying Asset ({option_chain.underlying_asset.underlying}) Price: {option_chain.underlying_asset.price}")
    print(f"src/utils.py :: Dividend Yield: {option_chain.underlying_asset.divyield} %Change: {option_chain.underlying_asset.pchange} $Change: {option_chain.underlying_asset.change}")
    with open(csvname, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "expiry", "yte", "underlying", "symbol", "underlying_price", "strike", "call_or_put",
            "last", "bid", "ask", "volume", "open_interest",
            "bs_iv", "baw_iv", "bin_iv", "delta", "gamma", "vega", "theta", "rho",
            "charm", "vanna", "vomma", "veta", "speed", "zomma", "color", "ultima",
            "intrinsic_value", "time_value"
        ])
        for expiry in option_chain.expiries:
            for c in expiry.calls:
                writer.writerow([
                    expiry.date, f"{c.yte:.4f}", c.underlying, c.symbol, c.underlying_price, c.strike, "Call",
                    c.midprice, c.bidprice, c.askprice, c.volume, c.openinterest,
                    c.bsiv, c.bawiv, c.biniv, c.delta, c.gamma, c.vega, c.theta, c.rho,
                    c.charm, c.vanna, c.vomma, c.veta, c.speed, c.zomma, c.color, c.ultima,
                    c.intrinsic_value, c.time_value
                ])
            for p in expiry.puts:
                writer.writerow([
                    expiry.date, f"{p.yte:.4f}", p.underlying, p.symbol, p.underlying_price, p.strike, "Put",
                    p.midprice, p.bidprice, p.askprice, p.volume, p.openinterest,
                    p.bsiv, p.bawiv, p.biniv, p.delta, p.gamma, p.vega, p.theta, p.rho,
                    p.charm, p.vanna, p.vomma, p.veta, p.speed, p.zomma, p.color, p.ultima,
                    p.intrinsic_value, p.time_value
                ])

    print(f"scr/utils.py :: Successfully saved {ticker} option chain to {csvname}")
    return option_chain

def plotChainIvCurve(ticker, csvname, pngname):
    if not os.path.exists(csvname):
        print(f"src/utils.py :: {csvname} does not exist")
        return
    
    if "chain" in csvname:
        print(f"src/utils.py :: {csvname} cannot be used because it is an option chain csv")
        return

    df = pd.read_csv(csvname)
    df = df[df["IV"].notna() & (df["IV"] != "")]  # skip missing
    df["IV"] = df["IV"].str.replace("%", "", regex=False).astype(float)
    df["Expiration"] = df["Expiration"].str.extract(r'([A-Za-z]+ \d{2}, \d{4})')[0]
    df["ExpirationDate"] = pd.to_datetime(df["Expiration"], format="%b %d, %Y")

    plt.figure(figsize=(10, 6))
    plt.plot(df["ExpirationDate"], df["IV"], marker="o", linestyle="-", color="blue")
    plt.xlabel("Expiration")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"{ticker} Implied Volatility Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pngname, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created IV curve and saved to {pngname}")

def plotChainSurface(ticker, mode, csvname, pngnamec, pngnamep):
    if not os.path.exists(csvname):
        print(f"src/utils.py :: {csvname} does not exist")
        return
    
    if "chain" not in csvname:
        print(f"src/utils.py :: {csvname} cannot be used because it is not an option chain csv")
        return

    if mode not in MODES:
        print(f"src/utils.py :: Mode {mode} is invalid; valid modes are: {MODES}")
        return
    
    df = pd.read_csv(csvname)
    df = df[df[mode].notna()]
    df["strike"] = df["strike"].astype(float)
    df[mode] = df[mode].astype(float)
    df["yte"] = df["yte"].astype(float)
    
    df_calls = df[df["call_or_put"] == "Call"]
    df_puts = df[df["call_or_put"] == "Put"]

    strikesc = sorted(df_calls["strike"].unique())
    expiriesc = sorted(df_calls["yte"].unique())
    strikesp = sorted(df_puts["strike"].unique())
    expiriesp = sorted(df_puts["yte"].unique())

    XC, YC = np.meshgrid(strikesc, expiriesc)
    XP, YP = np.meshgrid(strikesp, expiriesp)
    ZC = np.zeros_like(XC, dtype=float)
    ZP = np.zeros_like(XP, dtype=float)

    for i, t in enumerate(expiriesc):
        for j, k in enumerate(strikesc):
            data_row = df_calls[(df_calls["yte"] == t) & (df_calls["strike"] == k)]
            if not data_row.empty:
                ZC[i, j] = data_row[mode].values[0]
            else:
                ZC[i, j] = 0.000001  # in case some strike-expiry combinations don't exist

    for i, t in enumerate(expiriesp):
        for j, k in enumerate(strikesp):
            data_row = df_puts[(df_puts["yte"] == t) & (df_puts["strike"] == k)]
            if not data_row.empty:
                ZP[i, j] = data_row[mode].values[0]
            else:
                ZP[i, j] = 0.000001  # in case some strike-expiry combinations don't exist

    figc = plt.figure(figsize=(12,8))
    axc = figc.add_subplot(111, projection="3d")
    surfc = axc.plot_surface(XC, YC, ZC, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)
    axc.set_xlabel("Strike")
    axc.set_ylabel("Time to Expiry (Years)")
    axc.set_title(f"{ticker} {mode} Surface (Calls)")
    figc.colorbar(surfc, shrink=0.5, aspect=10, label=mode)
    plt.tight_layout()
    plt.savefig(pngnamec, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created {mode} call surface and saved to {pngnamec}")

    figp = plt.figure(figsize=(12,8))
    axp = figp.add_subplot(111, projection="3d")
    surfp = axp.plot_surface(XP, YP, ZP, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)
    axp.set_xlabel("Strike")
    axp.set_ylabel("Time to Expiry (Years)")
    axp.set_title(f"{ticker} {mode} Surface (Puts)")
    figp.colorbar(surfp, shrink=0.5, aspect=10, label=mode)
    plt.tight_layout()
    plt.savefig(pngnamep, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created {mode} put surface and saved to {pngnamep}")

def parseExpectedMove(val):
    if pd.isna(val) or not isinstance(val, str):
        return 0.0
    match = re.search(r"±([\d\.]+)", val)  # extract digits after ±
    if match:
        return float(match.group(1))
    return 0.0

def getProjectionChart(ticker, ohlc_csvname, stats_csvname, pngname):
    ohlc = pd.read_csv(ohlc_csvname, parse_dates=["Date"])
    stats = pd.read_csv(stats_csvname)

    stats["Expiration"] = stats["Expiration"].str.extract(r"([A-Za-z]{3} \d{2}, \d{4})")
    stats["Expiration"] = pd.to_datetime(stats["Expiration"], format="%b %d, %Y")

    stats["Expected Move"] = stats["Expected Move"].apply(parseExpectedMove)
    stats = stats.dropna(subset=["Expected Move"])

    stats["Max Pain"] = pd.to_numeric(stats["Max Pain"], errors="coerce")

    stats = stats.sort_values("Expiration")

    last_close = ohlc.sort_values("Date").iloc[-1]["Close"]

    projection = stats.copy()
    projection["Base Price"] = last_close
    projection["Upper Band"] = last_close + projection["Expected Move"]
    projection["Lower Band"] = last_close - projection["Expected Move"]
        
    plt.figure(figsize=(12,6))
    plt.plot(ohlc["Date"], ohlc["Close"], label="Close (History)", color="black")
    plt.plot(projection["Expiration"], projection["Upper Band"], label="Upper Band", color="green", linestyle="--", marker="o")
    plt.plot(projection["Expiration"], projection["Lower Band"], label="Lower Band", color="red", linestyle="--", marker="o")
    plt.fill_between(projection["Expiration"], projection["Lower Band"], projection["Upper Band"], color="gray", alpha=0.2)
    plt.plot(projection["Expiration"], projection["Max Pain"], label="Max Pain", color="orange", linestyle="-", marker="x")
    plt.axhline(last_close, color="blue", linestyle="-.", label=f"Last Close {last_close:.2f}")
    plt.title(f"{ticker} History & Projection")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(pngname, dpi=150)
    plt.close()
    print(f"src/polygon.py :: Successfully created projection chart {pngname}")

def getOptionSymbols(ticker, expiration_date, price):
    closest_strike = round(price)
    closest_strike_str = f"{int(closest_strike * 1000):08d}"
    expiration_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    expiration_str = expiration_dt.strftime("%y%m%d")
    option_symbolc = f"{ticker}{expiration_str}C{closest_strike_str}"
    option_symbolp = f"{ticker}{expiration_str}P{closest_strike_str}"
    return option_symbolc, option_symbolp

if __name__ == "__main__":
    pass